# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, xavier_init
from mmcv.runner import BaseModule, ModuleList

from ..builder import NECKS, build_backbone
from .fpn import FPN


class ASPP(BaseModule):
    """ASPP (Atrous Spatial Pyramid Pooling)

    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,   # 256
                 out_channels,  # 64
                 dilations=(1, 3, 6, 1),
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        super().__init__(init_cfg)
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)): #4
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x  #最后一层用平均池花层
            out.append(F.relu_(self.aspp[aspp_idx](inp)))       #1X1卷集全局池花层
        out[-1] = out[-1].expand_as(out[-2])   #
        out = torch.cat(out, dim=1)
        return out


@NECKS.register_module()
class RFP(FPN):
    """RFP (Recursive Feature Pyramid)

    This is an implementation of RFP in `DetectoRS
    <https://arxiv.org/pdf/2006.02334.pdf>`_. Different from standard FPN, the
    input of RFP should be multi level features along with origin input image
    of backbone.

    Args:
        rfp_steps (int): Number of unrolled steps of RFP.
        rfp_backbone (dict): Configuration of the backbone for RFP.
        aspp_out_channels (int): Number of output channels of ASPP module.
        aspp_dilations (tuple[int]): Dilation rates of four branches.
            Default: (1, 3, 6, 1)
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 rfp_steps,
                 rfp_backbone,
                 aspp_out_channels,
                 aspp_dilations=(1, 3, 6, 1),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.rfp_steps = rfp_steps
        # Be careful! Pretrained weights cannot be loaded when use
        # nn.ModuleList
        self.rfp_modules = ModuleList()
        for rfp_idx in range(1, rfp_steps):    #（1，2）
            rfp_module = build_backbone(rfp_backbone)  #在这又重新初始化了backbone
            self.rfp_modules.append(rfp_module)
        self.rfp_aspp = ASPP(self.out_channels, aspp_out_channels, #创建ASPP（256，64）
                             aspp_dilations)
        self.rfp_weight = nn.Conv2d(    #（256，1）
            self.out_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def init_weights(self):
        # Avoid using super().init_weights(), which may alter the default
        # initialization of the modules in self.rfp_modules that have missing
        # keys in the pretrained checkpoint.
        for convs in [self.lateral_convs, self.fpn_convs]:
            for m in convs.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
        for rfp_idx in range(self.rfp_steps - 1):
            self.rfp_modules[rfp_idx].init_weights()
        constant_init(self.rfp_weight, 0)

    def forward(self, inputs):
        inputs = list(inputs)  #转成列表方便操作，这里的inputs有5个长度
        assert len(inputs) == len(self.in_channels) + 1  # +1 for input image
        img = inputs.pop(0)   #是一个就地操作，inputs变成4个了
        # FPN forward
        x = super().forward(tuple(inputs))      #通过FPN得到out:x
        for rfp_idx in range(self.rfp_steps - 1):   #2-1=1 :0
            rfp_feats = [x[0]] + list(          #[C2+ASPP(C3,C4,C5,C6)]
                self.rfp_aspp(x[i]) for i in range(1, len(x)))
            x_idx = self.rfp_modules[rfp_idx].rfp_forward(img, rfp_feats) #返回反馈操作后的
            # FPN forward
            x_idx = super().forward(x_idx)  #将其再次使用FPN
            x_new = []
            for ft_idx in range(len(x_idx)):
                add_weight = torch.sigmoid(self.rfp_weight(x_idx[ft_idx]))  #将输出的进行sigmoid
                x_new.append(add_weight * x_idx[ft_idx] +
                             (1 - add_weight) * x[ft_idx])  #分配原来的和经过反馈的各自占的权重
            x = x_new
        return x

# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS
#
import torch
from mmcv.cnn import xavier_init,normal_init
from antialiased_cnns.Up import Ups

@NECKS.register_module()
class AugFPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 pool_ratios=[0.1, 0.2, 0.3],#
                 add_extra_convs=False,
                 train_with_auxiliary=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(AugFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        #
        self.train_with_auxiliary = train_with_auxiliary

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'
        #
        # self.sam1 = SAM(kernel_size=7)
        # self.sam2 = SAM(kernel_size=7)
        #self.aspp = ASPP(in_channel=2048,depth=256)
        # self.ups = nn.ModuleList()

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)


            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)


        # add lateral conv for features generated by rato-invariant scale adaptive pooling
        if self.train_with_auxiliary:
            self.adaptive_pool_output_ratio = pool_ratios
            self.high_lateral_conv = nn.ModuleList()
            self.high_lateral_conv.extend( #[2048,256,k=(1,1)]X3
                [nn.Conv2d(in_channels[-1], out_channels, 1) for k in range(len(self.adaptive_pool_output_ratio))])
            self.high_lateral_conv_attention = nn.Sequential( #[256*3,256,1],RELU,[256,3,K=(3,3)]
                nn.Conv2d(out_channels * (len(self.adaptive_pool_output_ratio)), out_channels, 1), nn.ReLU(),
                nn.Conv2d(out_channels, len(self.adaptive_pool_output_ratio), 3, padding=1))

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        for m in self.high_lateral_conv_attention.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        # for m in self.high_lateral_conv_attention.modules():
        #     if isinstance(m, nn.Conv2d):
        #         xavier_init(m, distribution='uniform')
        # for m in self.ups.modules():
        #     if isinstance(m, nn.Conv2d):
        #         normal_init(m,std=0.01)


    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        #
        #正常加
        # input1 = inputs[1]+inputs[1]*self.sam1(inputs[1])
        # input2 = inputs[2]+inputs[2]*self.sam2(inputs[2])
        # inputs_m =[inputs[0],input1,input2,inputs[3]]
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)  #if i<2
        ]
        #
        if self.train_with_auxiliary:
            # Residual Feature Augmentation
            h, w = inputs[-1].size(2), inputs[-1].size(3)
            # Ratio Invariant Adaptive Pooling #自适应池花层，针对的是C5层，对应着那张图
            # 先平均池花（有要求），再卷集成通道为256，最后双线性插值上采样尺寸（H，W）
            AdapPool_Features = [F.upsample(self.high_lateral_conv[j](F.adaptive_avg_pool2d(inputs[-1],
            output_size=(max(1, int(h *self.adaptive_pool_output_ratio[j])),max(1, int(w *self.adaptive_pool_output_ratio[j]))))),
            size=(h, w), mode='bilinear', align_corners=True) for j in range(len(self.adaptive_pool_output_ratio))]
            #concat
            Concat_AdapPool_Features = torch.cat(AdapPool_Features, dim=1)
            #注意力机制
            fusion_weights = self.high_lateral_conv_attention(Concat_AdapPool_Features)
            fusion_weights = F.sigmoid(fusion_weights)
            adap_pool_fusion = 0
            for i in range(len(self.adaptive_pool_output_ratio)): #取出后变为3维了，在unsqueeze一维
                adap_pool_fusion += torch.unsqueeze(fusion_weights[:,i, :,:], dim=1) * AdapPool_Features[i]
            raw_laternals = [laterals[i].clone() for i in range(len(laterals))]  # 复制

            laterals[-1] += adap_pool_fusion  #加在M5上
        #另一种方法,#直接相加的，可以考虑是否可以直接替换
        # laterals.append(adap_pool_fusion)
        # aspp = self.aspp(inputs[3])
        # laterals[-1] += aspp
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                #p = self.sam(laterals[i - 1])
                laterals[i - 1] +=F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
                # lateral = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg) #X
                # mask = laterals[i - 1].detach() #
                # laterals[i - 1] = (laterals[i - 1] +self.ups[i](lateral,mask))
                # p = self.sam(lateral)
                # laterals[i - 1] += (lateral*p)  #正常的CAM

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        if self.train_with_auxiliary:
            return tuple(outs), tuple(raw_laternals)
        else:
            return tuple(outs)


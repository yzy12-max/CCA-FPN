# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS
#import matplotlib.pyplot as plt
# torch.autograd.set_detect_anomaly(True)
from mmcv.cnn import xavier_init,kaiming_init,normal_init
# from mmcv.ops.carafe import CARAFEPack
from antialiased_cnns.DCCnn1 import DCCcnn1
from antialiased_cnns.DCCcnn import CAM_Module,DCCcnn
from antialiased_cnns.pac import PacConv2d   #
#from antialiased_cnns.context import CFEM
from mmcv.cnn.bricks import NonLocal2d
from antialiased_cnns.CTdc import CTdc


@NECKS.register_module()
class SIFPN(BaseModule):
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
                 add_extra_convs=False,
                 train_with_auxiliary = False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=None
                 # init_cfg=dict(
                 #     type='Xavier', layer='Conv2d', distribution='uniform')
                                            ):
        super(SIFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        # 修改亚像素和额外的卷积层（替换成赤化层试试）
        # self.Subconv = nn.PixelShuffle(2)
        #self.k1 = Downsample_PASA_group_softmax(512,kernel_size=3,stride=1,pad_type='reflect',group=2)  #7600-5900=1700
        # self.k2 = Downsample_PASA_group_softmax(1024, kernel_size=3, stride=1, pad_type='reflect', group=2)
        # self.k3 = Downsample_PASA_group_softmax(2048, kernel_size=3, stride=1, pad_type='reflect', group=1)
        # self.k3 = Downsampleimprove1(2048, kernel_size=3, stride=1, group=1,dilation=1)
        # self.k1 = Downsample_PASA_group_softmax(256, kernel_size=3, stride=1, pad_type='reflect', group=2) #用于latary层
        # self.k2 = Downsample_PASA_group_softmax(256, kernel_size=3, stride=1, pad_type='reflect', group=2)
        # self.k3 = Downsample_PASA_group_softmax(256, kernel_size=3, stride=1, pad_type='reflect', group=1)
        # self.k1 = FEM(in_channel=512,group=8,kernel_size=3)
        # self.k2 = FEM(in_channel=1024,group=8,kernel_size=3)
        #self.k3 = FEM(in_channel=2048,group=64,kernel_size=3)
        # self.k1 = SpatialGroupEnhance(64)
        # self.k2 = SpatialGroupEnhance(64)
        # self.k3 = SpatialGroupEnhance(64)
        # self.k1 = AClayer(in_channels=512, p=4, kernel_size=3, group=16)
        # self.k2 = AClayer(in_channels=1024, p=4, kernel_size=3, group=16)
        # self.k3 = AClayer(in_channels=2048, p=4, kernel_size=3, group=16)
        #self.k1 = nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1,bias=False) #直接一个3X3卷集，占内存500M，6400
        #self.k3 = Downsample(2048, kernel_size=3, stride=1, group=2)
        #self.k3_0 = NonLocal2d(2048,mode='dot_product')
        #self.k3 = SpatialGroupEnhance(64)
        #self.k3 = SACNN(2048,3)
        #self.k3 = Downsampleimprove(2048, kernel_size=3, stride=1, pad_type='reflect', group=1)
        #self.k3_0 = CAM_Module(2048)
        #self.k1 = DCCcnn(512)      #显存暴了
        #self.k2 = DCCcnn(1024)      #增加了700M显存
        #self.k3 = DCCcnn(2048)
        self.k3 = DCCcnn1(2048)
        self.train_with_auxiliary = train_with_auxiliary
        # self.k2 = CAM_Module(1024) #增加了6077-5911=166M显存
        # self.k1 = CAM_Module(512)
        #self.k3 = FEM()
        #self.k3 = ReDCCcnn()
        # self.dcm = DCM(2048)
        #self.k3 = CFEM(2048)
        # self.se = SEModule(2048)
        #self.k3 = CTdc()
        # self.pac = PacConv2d(2048,256,1)

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
        assert isinstance(add_extra_convs, (str, bool))  # 可以是 bool 或 str
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        #
        #self.pacs = nn.ModuleList()
        #self.cams = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],  # 修改成固定值256
                out_channels,
                1,  # 修改
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

            #sge = SpatialGroupEnhance(64)  # 加在3X3卷集的后面
            #cam = CAM_Module(in_channels[i])
            #pac = PacConv2d(256, 256, 3, padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            #self.cams.append(cam)
        # self.lateral_convs = self.lateral_convs

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

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        # for convs in [self.lateral_convs, self.fpn_convs]:
        #     for m in convs.modules():
        #         if isinstance(m, nn.Conv2d):
        #             xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        if isinstance(self.k3, DCCcnn1):
            self.k3.init_weights()
                #m.init_weight()   #初始化很重要，不然会不收敛
                #kaiming_init(m, mode='fan_out', nonlinearity='relu')
                #xavier_init(m, distribution='uniform')
            #     normal_init(m,std=0.01)
            # if isinstance(m, NonLocal2d):
            #     m.init_weights()

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)  # input:四层，in_channel[256,512,1024,2048]
        # from tools.feature_visualization import feature_single, draw_feature_map, draw_feature_map1
        # feature_single(inputs[1])
        #se
        # input3 = self.se(inputs[-1])
        # 自适应卷集和学习，进行特征增强,两种方法增强
        # 1.因为通道间的相似性，采用相同的卷集和
        #inputs_m = [inputs[0], inputs[1], inputs[2], input3]
        # build laterals  使用1X1卷集
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs) if i<2  #注意如果是用C2就不同了<3
        ]
        # laterals = [
        #     cam(inputs[i + self.start_level])
        #     for i, cam in enumerate(self.cams)  # if i<2  #注意如果是用C2就不同了
        # ]
        #laterals[-1] = self.k3(laterals[-1])
        #laterals.append(self.k3_0(inputs[-1]))
        laterals.append(self.k3(inputs[-1]))   #替换,5911
        # laterals.append(self.pac(inputs[-1],inputs[-1]))
        # addlayer = self.k3(inputs[-1])  #改成残茶的,相加
        # laterals[-1] += addlayer
        # laterals.append(self.k3(self.k3_0(inputs[-1]))) #1X1替换成CAM，5865
        # laterals = [input1, input2]
        # laterals.append(self.lateral_convs[2](inputs[3]))
        # draw_feature_map1(laterals[2], imgpath, name='laterals3_')
        if self.train_with_auxiliary:
            raw_laternals = [laterals[i].clone() for i in range(len(laterals))]  # 复制
        # build top-down path
        # laterals[1]=laterals[1]+laterals[2]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):  # 变为2，1层要进行上采样
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                #p = self.sam(laterals[i-1])
                laterals[i - 1] += (F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg))
                # ups = F.interpolate(laterals[i], size=prev_shape, **self.upsample_cfg)  # 6003
                # guid = self.pacs[i]( ups,laterals[i - 1])
                # laterals[i - 1] += guid


        # laterals[0] = self.k1(laterals[0])
        # laterals[1] = self.k2(laterals[1])
        # laterals[2] = self.k3(laterals[2])

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)  # 修改让他输出2层，-1就型
        ]
        # outs = [
        #     self.sges[i](self.fpn_convs[i](laterals[i])) for i in range(used_backbone_levels)  # 修改让他输出2层，-1就型
        # ]
        #outs[2] = self.k3(outs[2])
        # ]   #直接预测，不经过3X3卷及。注意看下显存占用情况
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
                    extra_source = inputs[self.backbone_end_level - 1]  # zhuyi
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
        # from tools.feature_visualization import draw_feature_map
        # draw_feature_map(outs)
        # from tools.feature_visualization import feature_map
        # feature_map(outs)
        if self.train_with_auxiliary:
            return tuple(outs), tuple(raw_laternals)
        else:
            return tuple(outs)


# class SPP(nn.Module):
#     '''
#     Spatial pyramid pooling layer used in YOLOv3-SPP
#     '''
#     def __init__(self, kernels=[1, 3, 5]):
#         super(SPP, self).__init__()
#         self.maxpool_layers = nn.ModuleList([
#             nn.Conv2d(256,256,kernel_size=kernel, stride=1, padding=kernel // 2) #保持尺寸
#             for kernel in kernels
#         ])
#
#     def forward(self, x):
#         out = [layer(x) for layer in self.maxpool_layers],
#         out1 =  out[0][0]+out[0][1]+out[0][2]                 #维度拼接512X4
#
#         return out1
# CBAM
class SAM(nn.Module):
    def __init__(self,kernel_size=7):
        super(SAM, self).__init__()
        assert kernel_size in (3,7) , 'kernel size must be 3 or 7'
        padding=3 if kernel_size==7 else 1
        self.conv1 = nn.Conv2d(2,1,kernel_size,padding=padding,stride=1,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x): #x:[n,c,h,w]
        avg_out = torch.mean(x, dim=1, keepdim=True)   #这个就是[n,1,h,w]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([max_out,avg_out],dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        out = out*x  # 自己加的
        return out

def get_pad_layer(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Downsample_PASA_group_softmax(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=2):
        super(Downsample_PASA_group_softmax, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size // 2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        # 2*K*K，先得到g*k*k个通道的maps。接着每个位置(i,j)都对应了一个k*k的向量，reshape为k x k，作为一个核
        self.conv = nn.Conv2d(in_channels, group * kernel_size * kernel_size, kernel_size=kernel_size, stride=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size * kernel_size) #不进行BN
        self.softmax = nn.Softmax(dim=1)  # 初始化
        #self.sigmoid = nn.Sigmoid()  #不能指定通道
        #nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu') #初始化方法改进
        # nn.init.xavier_normal_(self.conv.weight,gain=1.0)
        #nn.init.normal_(self.conv.weight,std=0.01)   #相当于开始高斯
        #nn.init.kaiming_uniform_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        #nn.init.xavier_uniform_(self.conv.weight, gain=1.0)

    def forward(self, x):
        #print(x.mean(), x.std())
        sigma = self.conv(self.pad(x))  # 用一个卷集层，维持HW不变的PAD后进行卷集，将通道变成C-》2*K*K
        #print(sigma.mean(), sigma.std())
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)
        # sigma = self.sigmoid(sigma)     #通道上进行softmax， K*K值和为1，保证地通且不改变亮度
        #print(sigma.mean(), sigma.std())
        n, c, h, w = sigma.shape  # C=2*K*K  18

        sigma = sigma.reshape(n, 1, c, h * w)  # [n,1,18,h*w]：1表示同一组的特征的不同channel，都使用相同的weight
        # 重新改变了值                    #有的是用N个weight【n,c,】
        n, c, h, w = x.shape  # 将这个特征图连续的在分辨率维度（H和W）维度取出特征, C=256  K*K多的是  【N，C，K*K，H‘*W’】
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(
            (n, c, self.kernel_size * self.kernel_size, h * w))
        # unfold函数的输入数据是四维，但输出是三维的。假设输入数据是[B, C, H, W], 那么输出数据是 [B, C* k * k, L]，L表示卷集输出的尺寸H‘*W’
        n, c1, p, q = x.shape  # p=k*k,q=h*w [N,256,9,h*w] - [256，N,9,h*w]-[2,256/2,n,9,h*w]-[n,2,256/2,9,h*w] 分两组
        x = x.permute(1, 0, 2, 3).reshape(self.group, c1 // self.group, n, p, q).permute(2, 0, 1, 3,
                                                                                         4)  # 为什么步直接reshape成[n,2,256/2,9,p*q]
        # x = x.reshape(n,self.group,c1//self.group,p,q)
        # 感觉上面的unfold不应该是针对学习的参数进行麻，怎么对X进行了
        n, c2, p, q = sigma.shape  # n,c2,p,q-[n,1,18,h*w] c2=1,c=18   [18,N,1,h*w]-[18/9,9,n,1,h*w]-[n,18/9,1,9,h*w] 分两组，  1变成对应的256/2就好了
        sigma = sigma.permute(2, 0, 1, 3).reshape(
            (p // (self.kernel_size * self.kernel_size), self.kernel_size * self.kernel_size, n, c2, q)).permute(2, 0,
                                                                                                                 3, 1,
                                                                                                                 4)
        # x*sigma：输入乘以对输入学习到的参数  X[n,256,H,W]
        x = torch.sum(x * sigma, dim=3).reshape(n, c1, h, w)  # 在卷集和的维度上对x*sigma求和相当于卷集操作
        # return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0] #下采样操作
        #print(x.mean(), x.std())
        #x = F.relu_(x)
        return x
        # x:是通过取的一块3X3的区域，sigma:是自己手动设置的将通道（1，K*K）reshape成（K，K）


# class DCMLayer(nn.Module):
#     def __init__(self, k, channel):
#         super(DCMLayer, self).__init__()
#         self.k = k   #卷集和大小
#         self.channel = channel
#         self.conv1 = nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True) #2048，256
#         self.conv2 = nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True) #2048-256
#         self.fuse = nn.Conv2d(channel // 8, channel//8, 1, padding=0, bias=True) #256，2048  #256，256，groups=256
#         self.dw_conv = nn.Conv2d(channel // 8, channel // 8, self.k, padding=(self.k-1) // 2, groups=channel // 8)
#         self.pool = nn.AdaptiveAvgPool2d(k)
#
#     def forward(self, x):
#         N, C, H, W = x.shape
#         # [N * C/4 * H * W]
#         f = self.conv1(x)
#         # [N * C/4 * K * K]            #这里有问题，共享卷集参数了
#         g = self.conv2(self.pool(x))  #这里自适应池花成1X1，直接就是卷集河的尺寸了
#         #torch.split()作用将tensor分成块结构，维度DIM=0
#         f_list = torch.split(f, 1, 0)
#         g_list = torch.split(g, 1, 0)
#
#         out = []
#         for i in range(N):
#             #[1* C/4 * H * W]
#             f_one = f_list[i]
#             # [C/4 * 1 * K * K]
#             g_one = g_list[i].squeeze(0).unsqueeze(1)
#             self.dw_conv.weight = nn.Parameter(g_one)  #在前向处理的时候将卷集作为卷集和参数的操作，分了2048//4=512组，所以参数【512，1，3，3】
#                                                         #正常是【512，512，3，3】
#             # [1* C/4 * H * W]
#             o = self.dw_conv(f_one)
#             out.append(o)
#
#         # [N * C/4 * H * W]
#         y = torch.cat(out, dim=0)
#         y = self.fuse(y)
#
#         return y
#
# class DCM(nn.Module):
#     def __init__(self, channel):
#         super(DCM, self).__init__()
#         self.DCM1 = DCMLayer(1, channel)
#         self.DCM3 = DCMLayer(3, channel)
#         self.DCM5 = DCMLayer(5, channel)
#         self.conv = nn.Conv2d(2816 , channel//8, 1, padding=0, bias=True) #4096，1024，
#
#     def forward(self, x):
#         dcm1 = self.DCM1(x)
#         dcm3 = self.DCM3(x)
#         dcm5 = self.DCM5(x)
#
#         out = torch.cat([x, dcm1, dcm3, dcm5], dim=1)
#         out = self.conv(out)
#         return out


# class FEM(nn.Module):
#     def __init__(self, in_channel, group, kernel_size, pad_type='reflect'):
#         super(FEM, self).__init__()
#         self.in_channel = in_channel
#         self.group = group
#         self.kernel_size = kernel_size
#         self.pad = get_pad_layer(pad_type)(self.kernel_size // 2)
#         self.sge = SpatialGroupEnhance(self.group)  # 不用还原成X，返回分组的注意力
#         self.adcnn = Downsample_PASA_group_sigmoid(in_channels=self.in_channel, kernel_size=self.kernel_size,
#                                                    group=self.group)  # 返回分组的sigma
#
#     def forward(self, x):
#         sigma = x
#         n, c0, h0, w0 = x.size()
#         x = self.sge(x)  # (b*32, c'(c/g), h, w)
#         b, c, h, w = x.size()
#         # n=b//32   【B，C*KH*KW，L】-[b,c,k*k,h*w]
#         x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(
#             (b, c, self.kernel_size * self.kernel_size, h * w))  # [b*32,c/g,9,h*w]
#         # n, c1, p, q = x.shape  # [256，N,9,p*q]-[2,256/2,n,9,p*q]-[n,2,256/2,9,p*q] 分两组
#         # x = x.permute(1, 0, 2, 3).reshape(self.group, c1 // self.group, n, p, q).permute(2, 0, 1, 4, 3)
#         x = x.reshape(n, self.group, c, 9, h * w)  # 改成(n,self.group, c, h * w，9)更好理解
#         sigma = self.adcnn(sigma)  # (n,self.group, c, h * w，9)
#         x = torch.sum(x * sigma, dim=3).reshape(n, c0, h0, w0)
#
#         return x
#
#     # SGENET


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups):
        super(SpatialGroupEnhance, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))  # 自定义权重参数 初始化为 0
        self.bias = nn.Parameter(torch.ones(1, groups, 1, 1))  # 1 （科学习的两个参数，还原规范花的图）就是BN操作
        self.sig = nn.Sigmoid()
        # print('add one SGE!')

    def forward(self, x):  # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)  # (b*32, c', h, w)   #划分为C//32组
        xn = x * self.avg_pool(x)  # 自适应平均池花 ，x * self.avg_pool(x)点乘能够利用每组空间上的相似性(b*32, c', h, w)
        xn = xn.sum(dim=1, keepdim=True)  # (b*32, 1, h, w) 求和操作，加强空间位置的语义信息
        t = xn.view(b * self.groups, -1)  # (b*32， h * w)
        t = t - t.mean(dim=1, keepdim=True)  # 做减均值除标准差的操作,BN,不同样本在同一组上分布差异很大
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)  # (b，32， h ， w)
        t = t * self.weight + self.bias  # 乘以科学习的参数 (b，32， h ， w) 32组，每组用一个卷集和参数学习
        t = t.view(b * self.groups, 1, h, w)  # (b*32， 1，h ， w)
        x = x * self.sig(t)  # (b*32, c', h, w) * (b*32， 1，h ， w)=(b*32, c', h, w)
        x = x.view(b, c, h, w)      #还原成原来的X

        return x


# 值适应卷集核的生成

class Downsampleimprove1(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, group=1,dilation=1):
        super(Downsampleimprove1, self).__init__()
        #self.pad = get_pad_layer(pad_type)(dilation)  #
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.dilation = dilation
        # 2*K*K，先得到g*k*k个通道的maps。接着每个位置(i,j)都对应了一个k*k的向量，reshape为k x k，作为一个核
        self.conv = nn.Conv2d(in_channels, group * kernel_size * kernel_size, kernel_size=kernel_size, stride=1,padding=dilation,dilation=dilation,
                              bias=False)
        # self.conv = nn.Conv2d(in_channels, group * kernel_size * kernel_size, kernel_size=kernel_size, stride=1,
        #                        dilation=dilation,bias=False)  #用镜像PADDING
        self.bn = nn.BatchNorm2d(group * kernel_size * kernel_size)
        self.softmax = nn.Softmax(dim=1)  # 初始化
        self.weight = nn.Parameter(torch.Tensor(256, 2048, kernel_size * kernel_size))
        self.bias = nn.Parameter(torch.Tensor(256))
        #self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)  # 将通道压缩成256
        #self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU(inplace=True)
        #nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.weight,std=0.01)

    def forward(self, x):
        sigma = self.conv(x)  # 用一个卷集层，维持HW不变的PAD后进行卷集，将通道变成C-》2*K*K
        #sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)  # 32*9=288
        sigma = self.softmax(sigma)
        # sigma = self.sigmoid(sigma)     #通道上进行softmax， K*K值和为1，保证地通且不改变亮度
        # sigma = self.relu(sigma)

        n, c, h, w = sigma.shape  # C=2*K*K  18

        sigma = sigma.reshape(n, 1, c, h * w)  # [n,1,18,h*w]：1表示同一组的特征的不同channel，都使用相同的weight

        #                    #有的是用N个weight【n,c,】
        # x = self.conv1(x)  # 先压缩成256
        n,c,h0,w0 = x.shape  #将这个特征图连续的在分辨率维度（H和W）维度取出特征, C=256  K*K多的是  【N，C，K*K，H‘*W’】
        x = F.unfold(x, kernel_size=self.kernel_size,padding=self.dilation,dilation=self.dilation).reshape((n,c,self.kernel_size*self.kernel_size,h*w))
        #unfold函数的输入数据是四维，但输出是三维的。假设输入数据是[B, C, H, W], 那么输出数据是 [B, C* K*K, L]，L表示卷集输出的尺寸H‘*W’
        n,c1,p,q = x.shape      #[256，N,9,p*q]-[2,256/2,n,9,p*q]-[n,2,256/2,9,p*q] 分两组
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)
        # 感觉上面的unfold不应该是针对学习的参数进行麻，怎么对X进行了
        n, c2, p, q = sigma.shape  # [n,1,18,h*w]-[18,N,1,h*w]-[18/9,9,n,1,h*w]-[n,18/9,1,9,h*w] 分两组，  1变成对应的256/2就好了
        sigma = sigma.permute(2, 0, 1, 3).reshape(
            (p // (self.kernel_size * self.kernel_size), self.kernel_size * self.kernel_size, n, c2, q)).permute(2, 0,3, 1, 4)  # permute(2,0,3,1,4)
        # x*sigma：输入乘以对输入学习到的参数  [n,1,256,h*w]-[n,256,H,W]
        # x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)  #在卷集和的维度上对x*sigma求和相当于卷集操作
        x = torch.einsum('ijklm,okl->ijom', (x * sigma, self.weight))
        x = x.squeeze(1).reshape(n,self.weight.shape[0],h,w)
        # x = x+self.bias.view(1, -1, 1, 1)
        #x = F.relu_(x)
        #
        return x


#不PADING，直接上采样的
class Downsample(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, group=32):
        super(Downsample, self).__init__()
        #self.pad = get_pad_layer(pad_type)(kernel_size // 2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        # 2*K*K，先得到g*k*k个通道的maps。接着每个位置(i,j)都对应了一个k*k的向量，reshape为k x k，作为一个核
        self.conv = nn.Conv2d(in_channels, group * kernel_size * kernel_size, kernel_size=kernel_size, stride=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size * kernel_size)
        self.softmax = nn.Softmax(dim=1)  # 初始化
        # self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU(inplace=True)
        #nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        #sigma = self.conv(self.pad(x))  # 用一个卷集层，维持HW不变的PAD后进行卷集，将通道变成C-》2*K*K
        sigma = self.conv(x)    #不填充，后面上采样
        sigma = self.bn(sigma)  # 32*9=288
        sigma = self.softmax(sigma)
        # sigma = self.sigmoid(sigma)     #通道上进行softmax， K*K值和为1，保证地通且不改变亮度
        # sigma = self.relu(sigma)

        n, c, h, w = sigma.shape  # C=2*K*K  18

        sigma = sigma.reshape(n, 1, c, h * w)  # [n,1,18,h*w]：1表示同一组的特征的不同channel，都使用相同的weight
        # 重新改变了值                    #有的是用N个weight【n,c,】
        n,c,h0,w0 = x.shape  #将这个特征图连续的在分辨率维度（H和W）维度取出特征, C=256  K*K多的是  【N，C，K*K，H‘*W’】
        #x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))
        x = F.unfold(x, kernel_size=self.kernel_size).reshape((n, c, self.kernel_size * self.kernel_size, h*w)) #h*w
        #unfold函数的输入数据是四维，但输出是三维的。假设输入数据是[B, C, H, W], 那么输出数据是 [B, C* K*K, L]，L表示卷集输出的尺寸H‘*W’
        n,c1,p,q = x.shape      #[256，N,9,p*q]-[2,256/2,n,9,p*q]-[n,2,256/2,9,p*q] 分两组
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)
        # 感觉上面的unfold不应该是针对学习的参数进行麻，怎么对X进行了
        n, c2, p, q = sigma.shape  # [n,1,18,h*w]-[18,N,1,h*w]-[18/9,9,n,1,h*w]-[n,18/9,1,9,h*w] 分两组，  1变成对应的256/2就好了
        sigma = sigma.permute(2, 0, 1, 3).reshape(
            (p // (self.kernel_size * self.kernel_size), self.kernel_size * self.kernel_size, n, c2, q)).permute(2, 0,3, 1, 4)  # permute(2,0,3,1,4)
        # x*sigma：输入乘以对输入学习到的参数  X[n,256,H,W]
        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)  #在卷集和的维度上对x*sigma求和相当于卷集操作
        x = F.upsample(x,size=(h0,w0),mode='bilinear', align_corners=True)

        return x

#SE模块
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  #
        return input * x

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
            self.aspp.append(conv)  # 1x1 3x3 3x3 1x1
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)): #0,1,2,3
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x  #最后一层用平均池花层
            out.append(F.relu_(self.aspp[aspp_idx](inp)))       #1X1卷集全局池花层
        out[-1] = out[-1].expand_as(out[-2])   #
        out = torch.cat(out, dim=1)
        return out
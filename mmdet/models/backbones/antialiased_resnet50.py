# This code is built from the PyTorch examples repository: https://github.com/pytorch/vision/tree/master/torchvision/models.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobes modifications are Copyright 2019 Adobe. All rights reserved.
# Adobes modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
#
# ==========================================================================================
#
# BSD-3 License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from antialiased_cnns import *
from ..builder import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import BaseModule

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
#            'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']
#
# model_urls = {
#     'resnet18_lpf2': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf2-6e2ee76f.pth',
#     'resnet18_lpf3': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf3-449351b9.pth',
#     'resnet18_lpf4': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf4-8c77af40.pth',
#     'resnet18_lpf5': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf5-c1eed0a1.pth',
#     'resnet34_lpf2': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet34_lpf2-4707aed9.pth',
#     'resnet34_lpf3': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet34_lpf3-16aa6c48.pth',
#     'resnet34_lpf4': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet34_lpf4-55747267.pth',
#     'resnet34_lpf5': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet34_lpf5-85283561.pth',
#     'resnet50_lpf2': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf2-f0f7589d.pth',
#     'resnet50_lpf3': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf3-a4e868d2.pth',
#     'resnet50_lpf4': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf4-994b528f.pth',
#     'resnet50_lpf5': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf5-9953c9ad.pth',
#     'resnet101_lpf2': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet101_lpf2-3d00941d.pth',
#     'resnet101_lpf3': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet101_lpf3-928f1444.pth',
#     'resnet101_lpf4': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet101_lpf4-f8a116ff.pth',
#     'resnet101_lpf5': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet101_lpf5-1f3745af.pth',
#     'resnet18_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet18_lpf4_finetune-8cc58f59.pth',
#     'resnet34_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet34_lpf4_finetune-db622952.pth',
#     'resnet50_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet50_lpf4_finetune-cad66808.pth',
#     'resnet101_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet101_lpf4_finetune-9280acb0.pth',
#     'resnet152_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnet152_lpf4_finetune-7f67d9ae.pth',
#     'resnext50_32x4d_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnext50_32x4d_lpf4_finetune-9106e549.pth',
#     'resnext101_32x8d_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/resnext101_32x8d_lpf4_finetune-8f13a25d.pth',
#     'wide_resnet50_2_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/wide_resnet50_2_lpf4_finetune-02a183f7.pth',
#     'wide_resnet101_2_lpf4_finetune': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/wide_resnet101_2_lpf4_finetune-da4eae04.pth',
# }


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, filter_size=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if (stride == 1):
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = nn.Sequential(BlurPool(planes, filt_size=filter_size, stride=stride),
                                       conv3x3(planes, planes), )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, filter_size=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        width = int(planes * (base_width / 64.)) * groups  #planes * 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, groups=groups,
                             dilation=dilation)  # Conv(stride2)-Norm-Relu --> #Conv-Norm-Relu-BlurPool(stride2)
        self.bn2 = norm_layer(width)
        if (stride == 1):
            self.conv3 = conv1x1(width, planes * self.expansion)
        else:   #让最后的1X1卷基层下采样
            self.conv3 = nn.Sequential(BlurPool(width, filt_size=filter_size, stride=stride),
                                       conv1x1(width, planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

@BACKBONES.register_module()
class AResNet(BaseModule):  #BaseModule，要想加载初始化与训练，要继承他

    def __init__(self,
                frozen_stages=-1,  #一般为1
                norm_eval=True,
                pretrained=None,
                _force_nonfinetuned=False,
                init_cfg=None,
                deep_stem=False,
                block=Bottleneck,      #这里开始都是原来的
                layers=[3, 4, 6, 3],
                #num_classes=1000,
                zero_init_residual=False,
                groups=1,
                width_per_group=64,
                norm_layer=None,
                filter_size=1,
                pool_only=True,
                replace_stride_with_dilation=None):
        super(AResNet, self).__init__(init_cfg)  #init_cfg：初始化训练权重
        # if depth==50:
        #     resnet50(pretrained=True, filter_size=4)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        #为了匹配MMDET，增加的
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        #与训练权重
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:  # 先看是否有预训练模型，有在大数据集上训练的最好
            if init_cfg is None:  # 再看RESNET的与训练
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups   #1
        self.base_width = width_per_group   #64

        if (pool_only):
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)   #将stride=2的maxpool采用BlurPool
            self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=1),
                                           BlurPool(self.inplanes, filt_size=filter_size, stride=2, )])
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=1, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Sequential(*[BlurPool(self.inplanes, filt_size=filter_size, stride=2, ),
                                           nn.MaxPool2d(kernel_size=2, stride=1),
                                           BlurPool(self.inplanes, filt_size=filter_size, stride=2, )])

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                       filter_size=filter_size)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                       filter_size=filter_size)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                       filter_size=filter_size)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        #
        self.deep_stem=deep_stem
        self._freeze_stages()   #加了个冻结层


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if (m.in_channels != m.out_channels or m.out_channels != m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    print('Not initializing')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, filter_size=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # since this is just a conv1x1 layer (no nonlinearity),
            # conv1x1->blurpool is the same as blurpool->conv1x1; the latter is cheaper
            downsample = [BlurPool(filt_size=filter_size, stride=stride, channels=self.inplanes), ] if (
                        stride != 1) else []
            downsample += [conv1x1(self.inplanes, planes * block.expansion, 1),
                           norm_layer(planes * block.expansion)]
            downsample = nn.Sequential(*downsample)     #初始化不长为2的参擦卷基层

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, filter_size=filter_size))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, filter_size=filter_size))

        return nn.Sequential(*layers)
    # 按照MMDET要求改的
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:   # 是否开启了将7X7卷集，替换成3个3X3卷集
                self.stem.eval() #
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:   # 私有属性切换为评估模式要加(self)
                self.bn1.eval()  # 如果网络模型model中含有BN层，则在预测时应当将模式切换为评估模式，即model.eval()
                for m in [self.conv1, self.bn1]:  # 评估模拟下BN层的均值和方差应该是整个训练集的均值和方差，即 moving mean/variance。
                    for param in m.parameters():    # 训练模式下BN层的均值和方差为mini-batch的均值和方差，因此应当特别注意
                        param.requires_grad = False  # 第一个卷集层前的层全部冻结

        for i in range(1, self.frozen_stages + 1):  #（1，2）
            m = getattr(self, f'layer{i}')   # getattr() 函数用于返回一个对象属性值。
            m.eval()
            for param in m.parameters():
                param.requires_grad = False  # layer1层冻结，不怎么用


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return tuple([x1,x2,x3,x4])
    #匹配MMDET，增加切换模式
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(AResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()  #eval()时，框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
#训练时是针对每个min-batch的，但是在测试中往往是针对单张图片，即不存在min-batch的概念。由于网络训练完毕后参数都是固定的，
# 因此每个批次的均值和方差都是不变的，因此直接结算所有batch的均值和方差。

# def resnet18(pretrained=False, filter_size=4, pool_only=True, _force_nonfinetuned=False, **kwargs):
#     """Constructs a ResNet-18 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         filter_size (int): Antialiasing filter size
#         pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], filter_size=filter_size, pool_only=pool_only, **kwargs)
#     if pretrained:
#         if (filter_size == 4 and not _force_nonfinetuned):
#             model.load_state_dict(
#                 model_zoo.load_url(model_urls['resnet18_lpf4_finetune'], map_location='cpu', check_hash=True)[
#                     'state_dict'])
#         else:
#             model.load_state_dict(
#                 model_zoo.load_url(model_urls['resnet18_lpf%i' % filter_size], map_location='cpu', check_hash=True)[
#                     'state_dict'])
#     return model
#
#
# def resnet34(pretrained=False, filter_size=4, pool_only=True, _force_nonfinetuned=False, **kwargs):
#     """Constructs a ResNet-34 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         filter_size (int): Antialiasing filter size
#         pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
#         _force_nonfinetuned (bool): [False] If True, load the trained-from scratch pretrained model (if available)
#     """
#     model = ResNet(BasicBlock, [3, 4, 6, 3], filter_size=filter_size, pool_only=pool_only, **kwargs)
#     if pretrained:
#         if (filter_size == 4 and not _force_nonfinetuned):
#             model.load_state_dict(
#                 model_zoo.load_url(model_urls['resnet34_lpf4_finetune'], map_location='cpu', check_hash=True)[
#                     'state_dict'])
#         else:
#             model.load_state_dict(
#                 model_zoo.load_url(model_urls['resnet34_lpf%i' % filter_size], map_location='cpu', check_hash=True)[
#                     'state_dict'])
#     return model
#
#
# def resnet50(pretrained=False, filter_size=4, pool_only=True, _force_nonfinetuned=False, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         filter_size (int): Antialiasing filter size
#         pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
#         _force_nonfinetuned (bool): [False] If True, load the trained-from scratch pretrained model (if available)
#     """
#     model = AResNet(Bottleneck, [3, 4, 6, 3], filter_size=filter_size, pool_only=pool_only, **kwargs)
#     if pretrained:
#         if (filter_size == 4 and not _force_nonfinetuned):
#             model.load_state_dict(
#                 model_zoo.load_url(model_urls['resnet50_lpf4_finetune'], map_location='cpu', check_hash=True)[
#                     'state_dict'])
#         else:
#             model.load_state_dict(
#                 model_zoo.load_url(model_urls['resnet50_lpf%i' % filter_size], map_location='cpu', check_hash=True)[
#                     'state_dict'])
#     return model
#
#
# def resnet101(pretrained=False, filter_size=4, pool_only=True, _force_nonfinetuned=False, **kwargs):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         filter_size (int): Antialiasing filter size
#         pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
#         _force_nonfinetuned (bool): [False] If True, load the trained-from scratch pretrained model (if available)
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], filter_size=filter_size, pool_only=pool_only, **kwargs)
#     if pretrained:
#         if (filter_size == 4 and not _force_nonfinetuned):
#             model.load_state_dict(
#                 model_zoo.load_url(model_urls['resnet101_lpf4_finetune'], map_location='cpu', check_hash=True)[
#                     'state_dict'])
#         else:
#             model.load_state_dict(
#                 model_zoo.load_url(model_urls['resnet101_lpf%i' % filter_size], map_location='cpu', check_hash=True)[
#                     'state_dict'])
#     return model
#
#
# def resnet152(pretrained=False, filter_size=4, pool_only=True, _force_nonfinetuned=False, **kwargs):
#     """Constructs a ResNet-152 model.
#     Args:
#         filter_size (int): Antialiasing filter size
#         pool_only (bool): [True] don't antialias the first downsampling operation (which is costly to antialias)
#         _force_nonfinetuned (bool): [False] If True, load the trained-from scratch pretrained model (if available)
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], filter_size=filter_size, pool_only=pool_only, **kwargs)
#     if pretrained:
#         if (filter_size == 4):
#             model.load_state_dict(
#                 model_zoo.load_url(model_urls['resnet152_lpf4_finetune'], map_location='cpu', check_hash=True)[
#                     'state_dict'])
#         else:
#             raise ValueError('No pretrained model available')
#     return model
#
#
# def resnext50_32x4d(pretrained=False, filter_size=4, pool_only=True, _force_nonfinetuned=False, **kwargs):
#     model = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, filter_size=filter_size, pool_only=pool_only,
#                    **kwargs)
#     if pretrained:
#         if (filter_size == 4):
#             model.load_state_dict(
#                 model_zoo.load_url(model_urls['resnext50_32x4d_lpf4_finetune'], map_location='cpu', check_hash=True)[
#                     'state_dict'])
#         else:
#             raise ValueError('No pretrained model available')
#     return model
#
#
# def resnext101_32x8d(pretrained=False, filter_size=4, pool_only=True, _force_nonfinetuned=False, **kwargs):
#     model = ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8, filter_size=filter_size,
#                    pool_only=pool_only, **kwargs)
#     if pretrained:
#         if (filter_size == 4):
#             model.load_state_dict(
#                 model_zoo.load_url(model_urls['resnext101_32x8d_lpf4_finetune'], map_location='cpu', check_hash=True)[
#                     'state_dict'])
#         else:
#             raise ValueError('No pretrained model available')
#     return model
#
#
# def wide_resnet50_2(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
#     """Wide ResNet-50-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
#
#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], width_per_group=64 * 2, filter_size=filter_size, **kwargs)
#     if pretrained:
#         if (filter_size == 4):
#             model.load_state_dict(
#                 model_zoo.load_url(model_urls['wide_resnet50_2_lpf4_finetune'], map_location='cpu', check_hash=True)[
#                     'state_dict'])
#         else:
#             raise ValueError('No pretrained model available')
#     return model
#
#
# def wide_resnet101_2(pretrained=False, filter_size=4, _force_nonfinetuned=False, **kwargs):
#     """Wide ResNet-101-2 model from
#     `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
#
#     The model is the same as ResNet except for the bottleneck number of channels
#     which is twice larger in every block. The number of channels in outer 1x1
#     convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
#     channels, and in Wide ResNet-50-2 has 2048-1024-2048.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], width_per_group=64 * 2, filter_size=filter_size, **kwargs)
#     if pretrained:
#         if (filter_size == 4):
#             model.load_state_dict(
#                 model_zoo.load_url(model_urls['wide_resnet101_2_lpf4_finetune'], map_location='cpu', check_hash=True)[
#                     'state_dict'])
#         else:
#             raise ValueError('No pretrained model available')
#     return model


# a = AResNet(frozen_stages=1,filter_size=4)
# input = [torch.randn(1,64,800,800)]
# #print(input)
# out = a(input)
#b = resnet50(pretrained=True,filter_size=4)

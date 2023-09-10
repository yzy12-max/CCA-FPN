# This code is built from the PyTorch examples repository: https://github.com/pytorch/vision/tree/master/torchvision/models.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe’s modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
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

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
#from models_lpf import *
from antialiased_cnns import *
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import BaseModule
from ..builder import BACKBONES

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                 padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, filter_size=1, pasa_group=2):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if(stride==1):
            self.conv2 = conv3x3(planes,planes)
        else:
            self.conv2 = nn.Sequential(Downsample_PASA_group_softmax(kernel_size=filter_size, stride=stride, in_channels=planes, group=pasa_group),
                conv3x3(planes, planes),)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, filter_size=1, pasa_group=2):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, groups) # stride moved
        self.bn2 = norm_layer(planes)
        if(stride==1):
            self.conv3 = conv1x1(planes, planes * self.expansion)
        else:
            self.conv3 = nn.Sequential(Downsample_PASA_group_softmax(kernel_size=filter_size, stride=stride, in_channels=planes, group=pasa_group),
                conv1x1(planes, planes * self.expansion))
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
class PASAResNet(BaseModule):

    def __init__(self,
                 frozen_stages=1,
                 norm_eval=True,
                 pretrained=None,
                 init_cfg=None,
                 deep_stem=False,
                 block=Bottleneck,
                 layers=[3,4,6,3],
                 #num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 norm_layer=None,
                 filter_size=1,
                 pool_only=True,
                 pasa_group=2):
        super(PASAResNet, self).__init__(init_cfg)
        #修改匹配MMDET
        self.deep_stem = deep_stem
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):  #
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
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]  #输入通道数
        self.inplanes = planes[0]

        if(pool_only):
            self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(planes[0])  #
        self.relu = nn.ReLU(inplace=True)

        if(pool_only):
            self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=1),
                Downsample_PASA_group_softmax(kernel_size=filter_size, stride=2, in_channels=planes[0], group=pasa_group)])
        else:
            self.maxpool = nn.Sequential(*[Downsample_PASA_group_softmax(kernel_size=filter_size, stride=2, in_channels=planes[0], group=pasa_group),
                nn.MaxPool2d(kernel_size=2, stride=1),
                Downsample_PASA_group_softmax(kernel_size=filter_size, stride=2, in_channels=planes[0], group=pasa_group)])

        self.layer1 = self._make_layer(block, planes[0], layers[0], groups=groups, norm_layer=norm_layer, pasa_group=pasa_group)
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size, pasa_group=pasa_group)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size, pasa_group=pasa_group)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size, pasa_group=pasa_group)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(planes[3] * block.expansion, num_classes) #检测不用

        self._freeze_stages()  # 注意顺序，要用到一些参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
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
                #pasa_group=2
    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None, filter_size=1, pasa_group=2):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  #64！=256
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride, filter_size=filter_size),
            #     norm_layer(planes * block.expansion),
            # )

            downsample = [Downsample_PASA_group_softmax(kernel_size=filter_size, stride=stride, in_channels=self.inplanes, group=pasa_group),] if(stride !=1) else []
            downsample += [conv1x1(self.inplanes, planes * block.expansion, 1),
                norm_layer(planes * block.expansion)]
            # print(downsample)
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer, filter_size=filter_size, pasa_group=pasa_group))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer, filter_size=filter_size, pasa_group=pasa_group))

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.bn1.eval()   #如果网络模型model中含有BN层，则在预测时应当将模式切换为评估模式，即model.eval()
                for m in [self.conv1, self.bn1]:  #评估模拟下BN层的均值和方差应该是整个训练集的均值和方差，即 moving mean/variance。
                    for param in m.parameters():    #训练模式下BN层的均值和方差为mini-batch的均值和方差，因此应当特别注意
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):   #第一层
            m = getattr(self, f'layer{i}')   #getattr() 函数用于返回一个对象属性值。
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


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

        # 匹配MMDET，增加切换模式
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(PASAResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

# def resnet50(pretrained=False, filter_size=1, pool_only=True, **kwargs):
#     """Constructs a ResNet-50 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], filter_size=filter_size, pool_only=pool_only, **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return model
#
#
# def resnet101(pretrained=False, filter_size=1, pool_only=True, **kwargs):
#     """Constructs a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], filter_size=filter_size, pool_only=pool_only, **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
#     return model
#
#
# def resnet152(pretrained=False, filter_size=1, pool_only=True, **kwargs):
#     """Constructs a ResNet-152 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 8, 36, 3], filter_size=filter_size, pool_only=pool_only, **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
#     return model
#
#
# def resnext50_32x4d(pretrained=False, filter_size=1, pool_only=True, **kwargs):
#     model = ResNet(Bottleneck, [3, 4, 6, 3], groups=4, width_per_group=32, filter_size=filter_size, pool_only=pool_only, **kwargs)
#     # if pretrained:
#     #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return model


# def resnext101_32x8d(pretrained=False, filter_size=1, pool_only=True, **kwargs):
#     model = ResNet(Bottleneck, [3, 4, 23, 3], groups=8, width_per_group=32, filter_size=filter_size, pool_only=pool_only, **kwargs)
#     # if pretrained:
#     #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#     return model

# if __name__ == '__main__':
#     model = resnet50(filter_size=3,groups=1,num_classes=4) #他这个后面有pasa_group=2的，所以groups=1
#     input = torch.randn([1,3,512,512])
#     out = model(input)
#     pass
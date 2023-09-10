import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import json
path = open('/data1/yzycode/datasets/coco/annotations/train_new.json', 'r', encoding='utf-8')
data = json.load(path)

def adv_cnn():
    x = torch.arange(0, 1 * 9 * 2 * 2).float()
    x = x.view(1, 9, 2, 2)
    print(x)
    x1 = F.unfold(x, kernel_size=1, dilation=1, stride=1)   #[B, C* kH * kW, L]
    print(x1.shape)
    B, C_kh_kw, L = x1.size()
    x1 = x1.permute(0, 2, 1)        #(b,h*w,c*k*k)
    x1 = x1.view(B, L, -1, 3, 3)    #(b,h*w,c,k,k)  通道K*K作为卷集和参数
    print(x1)
    x2 = x1.view(x1.size(0),x1.size(1),x1.size(2),-1).permute(0,2,3,1)  #【B，L，1，9】 通过F.unfold取通道K^2作为卷集和参数

    #[n,g,c,l,k*k]：表示输入N个特征图，那个特征图分G组，每个组里的特征层通道数为C，一个通道上特征层通过K*K划分的数量，最后是K*K的区域
    #[B, C* kH * kW, L]，获取Ck个K*K，主要是看C，G看L，G更容易控制，G控制了。CK也就控制了
    #可以尝试卷集和用kH * kW来作为参数，这样可以引入更大的感受也，通过周围邻近的信息来做加权，这样的化C就作为CK了，G=H*W/K*K

    #直接自己手动通过通道取K^2作为卷集和参数
    sigma = x
    n,c,h,w = sigma.shape
    sigma = sigma.reshape(n,1,c,h*w)                                         # n,g,1,k*k,h*w  (1,k*k)-(k,k)
    n, c2, p, q = sigma.shape  #n=1,c2=1,c=9=p,q=2*2=4 #[9,1,1,9]-[1,9,1,1,9]-[1,1,1,9,4]
    sigma = sigma.permute(2, 0, 1, 3).reshape(1, 9, n, c2, q).permute(2, 0, 3, 1, 4)
    sigma = sigma.permute(0,4,1,2,3).reshape(1,4,1,3,3)
    sigma.view(1,4,1,-1)

    #输入X，怎么与输入相卷集
    #1.求和再相加
    input = torch.arange(0, 2 * 4 * 3 * 3).float()
    input = input.reshape(2, 4, 3, 3)  #[b,c,h,w]  获取（H，W）的范围不好弄，现在只能通过F.unfold先填充再获取
    input = input.reshape(1,4,2,-1)
    out = torch.sum(input*sigma,dim=3)

    #2.通过将X作为W参数，给DW卷集做卷集
    dw_conv = nn.Conv2d(2048 // 4, 4, kernel_size=3, padding=(3-1) // 2, groups= 4)
    x = torch.randn(1,512,3,3)
    dw_conv.weight = nn.Parameter(x)

#STN
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定位网络-卷积层
        self.localization_convs = nn.Sequential(  #[1,1,28,28]
            nn.Conv2d(1, 8, kernel_size=7),         #[1,8,22,22]
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),          #[1,8,11,11]
            nn.Conv2d(8, 10, kernel_size=5),    #[1,10,7,7]
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),          #[1,10,3,3]
        )
        # 定位网络-线性层
        self.localization_linear = nn.Sequential(
            nn.Linear(in_features=10 * 3 * 3, out_features=32),  #
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2 * 3)   #
        )
        # 初始化定位网络仿射矩阵的权重/偏置，即是初始化θ值。使得图片的空间转换从原始图像开始。
        self.localization_linear[2].weight.data.zero_()
        self.localization_linear[2].bias.data.copy_(torch.tensor([1, 0, 0,
                                                                  0, 1, 0], dtype=torch.float))

        # 图片分类-卷积层
        self.convs = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
        )
        # 图片分类-线性层
        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=10),
        )

    # 空间变换器网络，转发图片张量
    def stn(self, x):
        # 使用CNN对图像结构定位，生成变换参数矩阵θ（2*3矩阵）
        x2 = self.localization_convs(x)
        x2 = x2.view(x2.size()[0], -1)  #[1,90],这里变成线性层的输入
        x2 = self.localization_linear(x2)
        theta = x2.view(x2.size()[0], 2, 3)  # [1, 2, 3]
        # print(theta)
        '''
        2D空间变换初始θ参数应置为tensor([[[1., 0., 0.],
                                        [0., 1., 0.]]])
        '''
        # 网格生成器，根据θ建立原图片的坐标仿射矩阵
        grid = nn.functional.affine_grid(theta, x.size(), align_corners=True)  # [1, 28, 28, 2]
        # 采样器，根据网格对原图片进行转换，转发给CNN分类网络
        x = nn.functional.grid_sample(x, grid, align_corners=True)  # [1, 1, 28, 28]
        return x

    def forward(self, x):
        x = self.stn(x)
        # print(x.size())
        x = self.convs(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        # print(x.size())
        return x


# if __name__ == '__main__':
#     x = torch.rand(1, 1, 28, 28)
#     model = Model()
#     print(model)
#     y = model(x)
#     print(y)

#softmax ,sigmod de 区别
# x = torch.arange(0, 1 * 3 * 2 * 2).float()
# x = x.view(1, 3, 2, 2)
# sd = nn.Sigmoid()
# a = sd(x)
# print(a)
# sx = nn.Softmax(dim=1)
# b = sx(x)


def linear(x, w, b): return x @ w + b

def relu(x): return x.clamp_min(0.)

nh = 50
x_train = torch.randn(100,784)
W1 = torch.randn(784, nh)
b1 = torch.zeros(nh)
W2 = torch.randn(nh, 1)
b2 = torch.zeros(1)

z1 = linear(x_train, W1, b1)
print(z1.mean(), z1.std())

#tensor(-0.3165) tensor(27.8031)
W1 = torch.randn(784, nh) * math.sqrt(1 / 784)
b1 = torch.zeros(nh)
W2 = torch.randn(nh, 1) * math.sqrt(1 / nh)
b2 = torch.zeros(1)

z1 = linear(x_train, W1, b1)
print(z1.mean(), z1.std())

#tensor(0.1031) tensor(0.9458)

a1 = relu(z1)
a1.mean(), a1.std()

#(tensor(0.4272), tensor(0.5915))
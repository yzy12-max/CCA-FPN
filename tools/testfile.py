import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from mmcv.ops.carafe import CARAFEPack

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


#使用CARAFE上采样
# carafe = CARAFEPack(channels=256,scale_factor=2)
# out = carafe(x)   #收敛太慢了
#特征增强方案：
class FEM(nn.Module):
    def __init__(self,in_channel,group,kernel_size, pad_type='reflect'):
        super(FEM, self).__init__()
        self.in_channel = in_channel
        self.group = group
        self.kernel_size = kernel_size
        self.pad = get_pad_layer(pad_type)(self.kernel_size // 2)
        self.sge = SpatialGroupEnhance(self.group)  #不用还原成X，返回分组的注意力
        self.adcnn = Downsample_PASA_group_softmax(in_channels=self.in_channel,kernel_size=self.kernel_size,group=self.group) #返回分组的sigma


    def forward(self,x):
        sigma = x
        n,c0,h0,w0 = x.size()  #
        x = self.sge(x)     #(b*g, c'(c/g), h, w),[b,c,h,w]
        b, c, h, w = x.size()
        #n=b//32                [B, C* kH * kW, L]---[b*g, c'*k*k, h* w],[b,c*k*k,h*w]
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(
            (b, c, self.kernel_size * self.kernel_size, h * w))  #[b*32,g,9,h*w].[b,c,9,h*w]
        #x = x.reshape(n,self.group, c, 9, h * w) #[b,g,c,9,h*w]
        n,c1,p,q = x.shape      #[256，N,9,p*q]-[2,256/2,n,9,p*q]-[n,2,256/2,9,p*q] 分两组
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)
        sigma = self.adcnn(sigma)     #[b,g,1,9,h*w]
        x = torch.sum(x * sigma, dim=3).reshape(n, c0, h0, w0)

        return x

    #SGENET
class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = nn.Parameter(torch.zeros(1, groups, 1, 1))  #自定义权重参数 初始化为 0
        self.bias     = nn.Parameter(torch.ones(1, groups, 1, 1))  # 1 （科学习的两个参数，还原规范花的图）就是BN操作
        self.sig      = nn.Sigmoid()
        #print('add one SGE!')

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w) # (b*32, c', h, w)   #划分为C//32组，关注每一组
        xn = x * self.avg_pool(x)        #自适应平均池花 ，x * self.avg_pool(x)点乘能够利用每组空间上的相似性(b*32, c', h, w)
        xn = xn.sum(dim=1, keepdim=True) # (b*32, 1, h, w) 求和操作，加强空间位置的语义信息
        t = xn.view(b * self.groups, -1)    #(b*32， h * w)
        t = t - t.mean(dim=1, keepdim=True) #做减均值除标准差的操作,BN,不同样本在同一组上分布差异很大
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)    #(b，32， h ， w)  #32组
        t = t * self.weight + self.bias     #乘以科学习的参数 (b，32， h ， w) 32组，每组用一个卷集和参数学习
        t = t.view(b * self.groups, 1, h, w)    #(b*32， 1，h ， w) 又关注每组
        x = x * self.sig(t)             #(b*32, c', h, w) * (b*32， 1，h ， w)=(b*32, c', h, w) 每组乘以
        x = x.view(b, c, h, w)      #还原成原来的X

        #x = x.view(b, self.groups,c, h, w)  #(b,32，c'，h ， w)
        return x

#值适应卷集核的生成


class Downsample_PASA_group_softmax(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=2):
        super(Downsample_PASA_group_softmax, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
           #2*K*K，先得到g*k*k个通道的maps。接着每个位置(i,j)都对应了一个k*k的向量，reshape为k x k，作为一个核
        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)        #初始化
        #self.sigmoid = nn.Sigmoid()        #换激活函数，RELU
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        sigma = self.conv(self.pad(x))   #用一个卷集层，维持HW不变的PAD后进行卷集，学习参数，将通道变成C-》2*K*K
        sigma = self.bn(sigma)          # 32*9=288
        sigma = self.softmax(sigma)     #[b,g*k*k,h,w]
        #sigma = self.sigmoid(sigma)     #通道上进行softmax， K*K值和为1，保证地通且不改变亮度

        n,c,h,w = sigma.shape       #C=2*K*K  18

        sigma = sigma.reshape(n,1,c,h*w) #[n,1,18,h*w]：1表示同一组的特征的不同channel，都使用相同的weight
        # 重新改变了值                    #有的是用N个weight【n,c,】
        # n,c,h,w = x.shape  #将这个特征图连续的在分辨率维度（H和W）维度取出特征, C=256  K*K多的是  【N，C，K*K，H‘*W’】
        # x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))
        # #unfold函数的输入数据是四维，但输出是三维的。假设输入数据是[B, C, H, W], 那么输出数据是 [B, C* kH * kW, L]，L表示卷集输出的尺寸H‘*W’
        # n,c1,p,q = x.shape      #[256，N,9,p*q]-[2,256/2,n,9,p*q]-[n,2,256/2,9,p*q] 分两组
        # x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)
        #感觉上面的unfold不应该是针对学习的参数进行麻，怎么对X进行了
        n,c2,p,q = sigma.shape  #[18,N,1,h*w]-[18/9,9,n,1,h*w]-[n,18/9,1,9,h*w] 分两组，  1变成对应的256/2就好了
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)
        #x*sigma：输入乘以对输入学习到的参数  X[n,256,H,W]
        # x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)  #在卷集和的维度上对x*sigma求和相当于卷集操作
        #return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0] #下采样操作
        # return x
        return sigma

# fem = FEM(1024,32,3)
# input = torch.randn(1,1024,200,320)
# out = fem(input)

#自适应卷集和
class AClayer(nn.Module):
    def __init__(self,in_channels,p,kernel_size,group):
        super(AClayer, self).__init__()
        self.kernel_size = kernel_size                #
        #self.out_channel = out_channel
        #out_channel1 = in_channels//4
        self.group = group
        self.p = p
        Ck = in_channels//self.group                #512/256=2,1224/256=4,2048/256=8
        out_channel1 = Ck * self.kernel_size * self.kernel_size     #out_channel1 = Ck*k*k,这个是学习参数的输出
        #Cw = self.out_channel * Ck * self.kernel_size * self.kernel_size // (self.p * self.p)
        self.adpool = nn.AdaptiveAvgPool2d((16,16))     #G=16*16=256
        self.conv1 = nn.Conv2d(in_channels,out_channel1,kernel_size=1,stride=1,padding=0,bias=False)#要用3X3的
        self.bn = nn.BatchNorm2d(out_channel1)
        #self.dw_conv = nn.Conv2d(in_channels , in_channels , self.k, padding=(self.k - 1) // 2, groups=256)
        #self.conv2 = nn.Conv2d(out_channel1,Cw,kernel_size=3,stride=1,bias=False)


    def forward(self,x):
        sigma = self.conv1(self.adpool(x))      #[n,18,16,16]  分成16*16=256个组
        sigma = self.bn(sigma)
        n,c,h,w = x.shape                       #[n,512,h,w]
        #x = self.conv2(x) #unfold函数的输入数据是四维，但输出是三维的。假设输入数据是[B, C, H, W], 那么输出数据是 [B, C* K*K, L]，L表示卷集输出的尺寸H‘*W’
        x = F.unfold(x, kernel_size=3,padding=1)          #[n,c,k*k,l] x要PAD
        x =x.reshape(n, c, self.kernel_size * self.kernel_size, h * w)  #【n,512,9,h * w】
        n, c1, p, q = x.shape  # [512，N,9,p*q]-[256,512/256,n,9,h*w]-[n,256,512/256,9,h*w] 分两组，一个组两个通道
        x = x.permute(1, 0, 2, 3).reshape(self.group, c1 // self.group, n, p, q).permute(2, 0, 1, 3, 4)
        #
        n, cw, h1, w1 = sigma.shape
        sigma = sigma.reshape(n, 1, cw, h1*w1)          #Ck = 18/9
        n, c2, cw, q = sigma.shape      #[n,1,18,p*p] [18,n,1,h*w]-[18/9,9,n,1,h*w]-[n,18/9,1,9,h*w]        ps*ps=g
        sigma = sigma.permute(2, 0, 1, 3).reshape(                              #最后[n,p*p,2,9,1],p*p=g c=2
            (cw // (self.kernel_size * self.kernel_size), self.kernel_size * self.kernel_size, n, c2, q)).permute(2,4,0,1,3)
        # x*sigma：输入乘以对输入学习到的参数  X[n,256,H,W]
        x = torch.sum(x * sigma, dim=3).reshape(n, c1, h, w)
        return x

# aclay = AClayer(1024,16,3,256)
# input = torch.randn(1,1024,200,320)
# out = aclay(input)

#改进成多尺度的

class SACNN(nn.Module):
    def __init__(self,in_channels, kernel_size, stride=1, group=1):
        super(SACNN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k1 = Downsampleimprove(in_channels, kernel_size[0], pad_type='reflect', stride=1, group=1)
        self.k2 = Downsampleimprove(in_channels, kernel_size[1], pad_type='reflect', stride=1, group=1)
        self.k3 = Downsampleimprove(in_channels, kernel_size[2], pad_type='reflect', stride=1, group=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels * 4, 2048, kernel_size=1, stride=1)

    def forward(self,x):
        x0 = x*self.avg_pool(x)
        x0 = self.bn(x0)
        x1 = self.k1(x)
        x2 = self.k2(x)
        x3 = self.k3(x)
        x = torch.cat([x0,x1,x2,x3],dim=1)
        x = self.conv(x)
        return x


#1.1X1结合注意力机制的
class SAM(nn.Module):
    def __init__(self,kernel_size=1):
        super(SAM, self).__init__()
        assert kernel_size in (1,3,7) , 'kernel size must be 1,3 or 7'
        padding=3 if kernel_size==7 else 1
        self.conv1 = nn.Conv2d(2,1,kernel_size,padding=padding,stride=1,bias=False) #这个也可以用自适应卷集和去卷集
        #self.encode = Downsampleimprove(2048,3,pad_type='reflect', stride=1, group=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x): #x:[n,c,h,w]
        avg_out = torch.mean(x, dim=1, keepdim=True)   #这个就是[n,1,h,w]
        max_out, _ = torch.max(x, dim=1, keepdim=True)   #不用池花层来做，而是用1X1的自适应卷集层做，这样的话，可以尝试再BACKBONE里使用
        x = torch.cat([max_out,avg_out],dim=1)
        #x = self.conv1(x)

        x = self.sigmoid(x)

        return x

#3X3+空间注意力机制
class Downsampleimprove(nn.Module):

    def __init__(self, in_channels, kernel_size, pad_type='reflect', stride=1, group=1):
        super(Downsampleimprove, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size // 2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        # 2*K*K，先得到g*k*k个通道的maps。接着每个位置(i,j)都对应了一个k*k的向量，reshape为k x k，作为一个核
        self.conv = nn.Conv2d(in_channels, group * kernel_size * kernel_size, kernel_size=kernel_size, stride=1,
                              bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size * kernel_size)
        self.softmax = nn.Softmax(dim=1)  # 初始化
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels,256,kernel_size=1,stride=1)  #将通道压缩成256
        # self.relu = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        #nn.init.xavier_normal_(self.conv.weight,gain=1.0)
        #nn.init.normal_(self.conv.weight,std=0.01)
        #nn.init.kaiming_uniform_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        #nn.init.xavier_uniform_(self.conv.weight,gain=1.0)

    def forward(self, x):
        #print(x.mean(), x.std())
        sigma = self.conv(self.pad(x))  # 用一个卷集层，维持HW不变的PAD后进行卷集，将通道变成C-》2*K*K
        #print(sigma.mean(), sigma.std())
        sigma = self.bn(sigma)  # 32*9=288
        #print(sigma.mean(), sigma.std())
        sigma = self.softmax(sigma)
        #sigma = self.sigmoid(sigma)     #通道上进行softmax， K*K值和为1，保证地通且不改变亮度
        #sigma = self.relu(sigma)
        #print(sigma.mean(), sigma.std())

        n, c, h, w = sigma.shape  # C=2*K*K  18

        sigma = sigma.reshape(n, 1, c, h * w)  # [n,1,18,h*w]：1表示同一组的特征的不同channel，都使用相同的weight
        # 重新改变了值                    #有的是用N个weight【n,c,】
        # 加空间注意力机制
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 这个就是[n,1,h,w]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 不用池花层来做，而是用1X1的自适应卷集层做，这样的话，可以尝试再BACKBONE里使用
        x1 = torch.cat([max_out, avg_out], dim=1)
        n, c0, h, w = x1.shape  #c=2
        x1 = F.unfold(self.pad(x1), kernel_size=self.kernel_size).reshape((n, c0, self.kernel_size * self.kernel_size, h * w))
        x1 = x1.reshape(n,self.group,c0//self.group,self.kernel_size * self.kernel_size,h * w)
        #将通道压缩成256
        x = self.conv1(x)

        n,c,h0,w0 = x.shape  #将这个特征图连续的在分辨率维度（H和W）维度取出特征, C=256  K*K多的是  【N，C，K*K，H‘*W’】
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))
        #print(x.mean(), x.std())
        #unfold函数的输入数据是四维，但输出是三维的。假设输入数据是[B, C, H, W], 那么输出数据是 [B, C* K*K, L]，L表示卷集输出的尺寸H‘*W’
        n,c1,p,q = x.shape      #[256，N,9,p*q]-[2,256/2,n,9,p*q]-[n,2,256/2,9,p*q] 分两组
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)
        # 感觉上面的unfold不应该是针对学习的参数进行麻，怎么对X进行了
        n, c2, p, q = sigma.shape  # [n,1,18,h*w]-[18,N,1,h*w]-[18/9,9,n,1,h*w]-[n,18/9,1,9,h*w] 分两组，  1变成对应的256/2就好了
        sigma = sigma.permute(2, 0, 1, 3).reshape(
            (p // (self.kernel_size * self.kernel_size), self.kernel_size * self.kernel_size, n, c2, q)).permute(2, 0,3, 1, 4)  # permute(2,0,3,1,4)
        # x*sigma：输入乘以对输入学习到的参数  X[n,256,H,W]
        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)  #在卷集和的维度上对x*sigma求和相当于卷集操作
        x1 = torch.sum(x1*sigma, dim=3).reshape(n, c0, h, w)
        x1 = torch.sum(x1, dim=1,keepdim=True)
        x1 = self.sigmoid(x1)
        x = x*x1
        #print(x.mean(), x.std())
        #x = self.bn(x)
        #print(x.mean(), x.std())
        return x     #sigma 用来作为SAM的卷集和  多个不同感受也的信息，用了注意力可以考虑相加了CAT后处理的方式也可以不同了

# aclay = Downsampleimprove(2048,3)
# input = torch.randn(1,2048,20,32)
# out = aclay(input)

#5X5，用空洞卷集尝试
class Downsampleimprove1(nn.Module):

    def __init__(self, in_channels, kernel_size, pad_type='reflect', stride=1, group=1):
        super(Downsampleimprove1, self).__init__()
        #self.pad = get_pad_layer(pad_type)(kernel_size // 2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        # 2*K*K，先得到g*k*k个通道的maps。接着每个位置(i,j)都对应了一个k*k的向量，reshape为k x k，作为一个核
        self.conv = nn.Conv2d(in_channels, group * kernel_size * kernel_size, kernel_size=kernel_size, stride=1,padding=3,dilation=3,
                              bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size * kernel_size)
        self.softmax = nn.Softmax(dim=1)  # 初始化
        # self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        #sigma = self.conv(self.pad(x))  # 用一个卷集层，维持HW不变的PAD后进行卷集，将通道变成C-》2*K*K
        sigma = self.conv(x)
        sigma = self.bn(sigma)  # 32*9=288
        sigma = self.softmax(sigma)
        # sigma = self.sigmoid(sigma)     #通道上进行softmax， K*K值和为1，保证地通且不改变亮度
        # sigma = self.relu(sigma)


        n, c, h, w = sigma.shape  # C=2*K*K  18

        sigma = sigma.reshape(n, 1, c, h * w)  # [n,1,18,h*w]：1表示同一组的特征的不同channel，都使用相同的weight
        # 重新改变了值                    #有的是用N个weight【n,c,】
        n,c,h0,w0 = x.shape  #将这个特征图连续的在分辨率维度（H和W）维度取出特征, C=256  K*K多的是  【N，C，K*K，H‘*W’】
        x = F.unfold(x, kernel_size=self.kernel_size,padding=3,dilation=3).reshape((n,c,self.kernel_size*self.kernel_size,h*w))
        #unfold函数的输入数据是四维，但输出是三维的。假设输入数据是[B, C, H, W], 那么输出数据是 [B, C* K*K, L]，L表示卷集输出的尺寸H‘*W’
        n,c1,p,q = x.shape      #[256，N,9,p*q]-[2,256/2,n,9,p*q]-[n,2,256/2,9,p*q] 分两组
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)
        # 感觉上面的unfold不应该是针对学习的参数进行麻，怎么对X进行了
        n, c2, p, q = sigma.shape  # [n,1,18,h*w]-[18,N,1,h*w]-[18/9,9,n,1,h*w]-[n,18/9,1,9,h*w] 分两组，  1变成对应的256/2就好了
        sigma = sigma.permute(2, 0, 1, 3).reshape(
            (p // (self.kernel_size * self.kernel_size), self.kernel_size * self.kernel_size, n, c2, q)).permute(2, 0,3, 1, 4)  # permute(2,0,3,1,4)
        # x*sigma：输入乘以对输入学习到的参数  X[n,256,H,W]
        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)  #在卷集和的维度上对x*sigma求和相当于卷集操作

        return x

aclay = Downsampleimprove1(2048,3)
input = torch.randn(1,2048,20,32)
out = aclay(input)

#不先PADING，直接上采样
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
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

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

# aclay = Downsample(2048,3)
# input = torch.randn(1,2048,20,32)
# out = aclay(input)

#将FPN的C2层由融合后各层rsize到相同大小后（BFP）上采样得到



#将特征尺寸区域作为卷集和参数





class DCMLayer(nn.Module):
    def __init__(self, k, channel):
        super(DCMLayer, self).__init__()
        self.k = k   #卷集和大小
        self.channel = channel
        self.conv1 = nn.Conv2d(channel, channel // 4, 1, padding=0, bias=True) #1024，256
        self.conv2 = nn.Conv2d(channel, channel // 4, 1, padding=0, bias=True)
        self.fuse = nn.Conv2d(channel // 4, channel, 1, padding=0, bias=True) #256，1024  #256，256，groups=256
        self.dw_conv = nn.Conv2d(channel // 4, channel // 4, self.k, padding=(self.k-1) // 2, groups=channel // 4)
        #self.dw_conv1 = nn.Conv2d(channel // 4, channel // 4, self.k, padding=(self.k - 1) // 2, groups=4)
        self.pool = nn.AdaptiveAvgPool2d(k)

    def forward(self, x):
        N, C, H, W = x.shape
        # [N * C/4 * H * W]
        f = self.conv1(x)
        # [N * C/4 * K * K]            #这里有问题，共享卷集参数了
        g = self.conv2(self.pool(x))  #这里自适应池花成1X1，直接就是卷集河的尺寸了
        #torch.split()作用将tensor分成块结构，维度DIM=0
        f_list = torch.split(f, 1, 0)   #【N，512，H，W】
        g_list = torch.split(g, 1, 0)

        out = []
        for i in range(N):
            #[1* C/4 * H * W]
            f_one = f_list[i]   #原来的
            # [C/4 * 1 * K * K]
            g_one = g_list[i].squeeze(0).unsqueeze(1)   #【512，1，1，1】
            self.dw_conv.weight = nn.Parameter(g_one)  #在前向处理的时候将卷集作为卷集和参数的操作，分了2048//4=512组，所以参数【512，1，3，3】
                                                        #正常是【512，512，3，3】
            # [1* C/4 * H * W]
            o = self.dw_conv(f_one)
            out.append(o)

        # [N * C/4 * H * W]
        y = torch.cat(out, dim=0)
        y = self.fuse(y)

        return y

class DCM(nn.Module):
    def __init__(self, channel):
        super(DCM, self).__init__()
        self.DCM1 = DCMLayer(1, channel)
        self.DCM3 = DCMLayer(3, channel)
        self.DCM5 = DCMLayer(5, channel)
        self.conv = nn.Conv2d(channel * 4, channel, 1, padding=0, bias=True) #4096，1024，

    def forward(self, x):
        dcm1 = self.DCM1(x)
        dcm3 = self.DCM3(x)
        dcm5 = self.DCM5(x)

        out = torch.cat([x, dcm1, dcm3, dcm5], dim=1)
        out = self.conv(out)
        return out

# dcm = DCM(2048)
# input = torch.randn(1,2048,200,320)
# out = dcm(input)

#给HFPN里加一个全局信息
# FC = F.avg_pool2d(x,1,stride=2) #利用卷集将尺寸变为1X1
# conv = nn.Conv2d(2048,256,1,1)  #压缩通道

# ASPP ：without bn version
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12) #pading为了保持输入与输出一致
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

# aspp = ASPP(512)
# input = torch.randn(1,512,200,320)
# out = aspp(input)



# class SubConv(nn.Module):
#     # 卷积+ReLU函数
#     def __init__(self, in_channels, r, kernel_sizes, paddings, dilations):
#         super().__init__()
#         self.r = int(r**0.5)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels*r, kernel_size=kernel_sizes , padding = paddings, dilation = dilations),
#             nn.BatchNorm2d(in_channels*r),
#             #nn.LayerNorm([out_channels,3199]),
#         )
#
#     def forward(self, x):
#         B,C,H,W = x.size()
#         print(B,C,H,W)
#         x = self.conv(x)
#         x = torch.reshape(x,(B,C,self.r*H,self.r*W)) ###变形
#         return x
#
# x=torch.randn(5, 1, 320,400)
# net = SubConv(1,4,1,0,1)
# outputs = net(x)
# print(outputs.size())


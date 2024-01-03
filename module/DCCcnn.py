import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import NonLocal2d
from mmcv.cnn import xavier_init,kaiming_init,normal_init,constant_init



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

class DCCcnn(nn.Module):
    def __init__(self,in_channel,mid_channel=256,kernel_size=3,scale=4):
        super(DCCcnn, self).__init__()
        # self.pad1 = get_pad_layer(pad_type1)(3)
        # self.pad2 = get_pad_layer(pad_type2)(6)
        # self.pad0 = get_pad_layer(pad_type0)(1)
        self.scale = scale
        self.kernel_size = kernel_size
        self.cam = CAM_Module(in_channel)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  #
        self.adcnn0 = ADCnn(mid_channel, kernel_size=self.kernel_size, stride=1,  group=1,dilation=1)
        self.adcnn1 = ADCnn(mid_channel, kernel_size=self.kernel_size, stride=1,  group=1,dilation=3)
        self.adcnn2 = ADCnn(mid_channel, kernel_size=self.kernel_size, stride=1,  group=1,dilation=6)
        self.adcnn3 = ADCnn(mid_channel, kernel_size=self.kernel_size, stride=1,  group=1,dilation=12)
        #bias
        #self.bias0 = nn.Conv2d(64,1,kernel_size=1)
        self.reduce = nn.Conv2d(in_channel,mid_channel,kernel_size=1,stride=1)
        self.conv = nn.Conv2d(mid_channel*3,mid_channel,kernel_size=1,stride=1)
        #self.gn = nn.GroupNorm(num_groups=4, num_channels=256, eps=1e-5, affine=True)
        # self.se = SEModule(256, reduction=16)
        # self.nonlcoal = NonLocal2d(256)

        #self.init_weights()

    def init_weights(self):
        for convs in [self.cam,self.adcnn0,self.adcnn1,self.adcnn2]:
            for m in convs.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
                    #kaiming_init(m, mode='fan_out', nonlinearity='relu')
                    #xavier_init(m, distribution='uniform')
                # if isinstance(m, nn.BatchNorm2d):
                #     nn.init.constant_(m.weight, 1)
                #     nn.init.constant_(m.bias, 0)
        # if isinstance(self.nonlcoal, NonLocal2d):
        #     self.nonlcoal.init_weights()
        #constant_init(self.cam.gamma,0)   
        #kaiming_init(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self,x):
        x_input = x
        x = self.cam(x)  #2048-256
        # size = x.shape[2:]
        # x0 = self.avg_pool(x)
        # x0 = F.interpolate(x0, size=size, mode='bilinear', align_corners=True)
        sigma0 = self.adcnn0(x)         #k=3,d=1  sigma:[n,1,9,h,w]
        sigma1 = self.adcnn1(x)
        sigma2 = self.adcnn2(x)
        sigma3 = self.adcnn3(x)   #

        x_reduce = x
        n, cx, h, w = x.shape
        x = x.reshape(n,self.scale,cx//self.scale,h,w)
        x0 = F.unfold(x[:,0,:,:], kernel_size=3, padding=1, dilation=1).reshape(
            (n, cx//self.scale, self.kernel_size*self.kernel_size, h * w))
        # x0 = F.unfold(self.pad0(x[:, 0, :, :]), kernel_size=3,  dilation=1).reshape(
        #     (n, cx // 4, 9, h * w))
        x0 = torch.sum(x0*sigma0,dim=2).reshape(n,cx//self.scale,h,w)  
        #x0 = self.bn0(x0)
        # avg_out0 = torch.mean(x[:, 0, :, :], dim=1, keepdim=True)
        # bias0 = F.sigmoid(avg_out0)
        # x0 +=  bias0
        #x0 = F.relu_(x0)
        x1 = F.unfold(x[:, 1, :, :,:], kernel_size=3,  padding=3,dilation=3).reshape(
            (n, cx // self.scale, self.kernel_size*self.kernel_size, h * w))  #[B, C* K*K, L]——【B，64，9，HW】
        x1 = torch.sum(x1 * sigma1, dim=2).reshape(n,cx//self.scale,h,w)
        #x1 = self.bn1(x1)
        #x1 = F.relu_(x1)
        x2 = F.unfold(x[:, 2, :, :, :], kernel_size=3, padding=6,  dilation=6).reshape(
            (n, cx // self.scale, self.kernel_size*self.kernel_size, h * w))
        x2 = torch.sum(x2 * sigma2, dim=2).reshape(n,cx//self.scale,h,w)  
        #x2 = self.bn2(x2)
        #x2 = F.relu_(x2)
        x3 = F.unfold(x[:, 3, :, :, :], kernel_size=3, padding=12, dilation=12).reshape(
            (n, cx // self.scale, self.kernel_size * self.kernel_size, h * w))
        x3 = torch.sum(x3 * sigma3, dim=2).reshape(n, cx // self.scale, h, w)  
        #x3 = self.bn3(x3)
        #x3 = F.relu_(x3)
        x4 = self.avg_pool(x_input)   #houjia 1X1
        x4 = self.reduce(x4)   #
        x4 = F.interpolate(x4, size=(h,w), mode='bilinear', align_corners=True)
        x = torch.cat([x0,x1,x2,x3,x4,x_reduce],dim=1) 
        x = self.conv(x)
        # x = self.nonlcoal(x)  
        #x = self.gn(x)
        # x/ = x_input + x  
        #x = self.relu(x)
        #x = self.leakrelu(x)
        #x  = self.cam0(x)   
        # x = self.se(x)   
        #x = self.bn(x)
        # activate[0] = x0   
        # for i,a in activate.items():
        #     a = a.cpu().detach().numpy()
        #     plt.figure()
        #     plt.title('0')
        #     plt.hist(a.flatten(),30,range=(0,1))
        # plt.show()
        # plt.figure()
          

        return x


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # 先经过3个卷积层生成3个新特征图B C D （尺寸不变）
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # α尺度系数初始化为0，并逐渐地学习分配到更大的权重

        self.softmax = nn.Softmax(dim=-1)  # 对每一行进行softmax

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × (H×W) × (H×W)
        """
        m_batchsize, C, height, width = x.size()
        # B -> (N,C,HW) -> (N,HW,C)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # C -> (N,C,HW)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        # BC，空间注意图 -> (N,HW,HW)
        energy = torch.bmm(proj_query, proj_key)
        # S = softmax(BC) -> (N,HW,HW)
        attention = self.softmax(energy)
        # D -> (N,C,HW)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        # DS -> (N,C,HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class CAM_Module(nn.Module):
    def __init__(self,in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        scale = in_dim//256  #2048/256=8 1024/256=4
        self.conv = nn.Conv2d(in_dim,in_dim//scale,kernel_size=1,stride=1)      
        #self.conv1 = nn.Conv2d(in_dim, in_dim // scale, kernel_size=1, stride=1)  
        self.gamma = nn.Parameter(torch.zeros(1))  
        #self.gamma = nn.Parameter(torch.randn(in_channels * kernel_size ** 2) * std, requires_grad = True)
        #self.gamma = nn.Parameter(torch.ones(1,256,1,1))
        self.softmax = nn.Softmax(dim=-1)  



    def forward(self, x):
        """
            inputs :
                x : input feature maps( B × C × H × W)
            returns :
                out : attention value + input feature
                attention: B × C × C
        """
        m_batchsize, C, height, width = x.size()
        # Q -> (N,C',HW),C'=256
        proj_query = self.conv(x)   #增加的
        xres = proj_query
        C1 = proj_query.size(1)
        proj_query = proj_query.view(m_batchsize, C1, -1)
        # K -> (N,HW,C)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1).contiguous()
        proj_key = proj_key.reshape(m_batchsize, height*width,8,C//8)
        energy = torch.einsum('ijk,iklm->ijlm', (proj_query,proj_key))
        energy = energy.view(m_batchsize, C1, -1)
        energy /= (proj_query.shape[-1]**0.5)

        attention = self.softmax(energy)   


        # V -> (N,C,HW)
        proj_value = x.view(m_batchsize, C, -1)
        # QKV -> （N,C',HW）
        out = torch.bmm(attention, proj_value)
        # output -> (N,C',H,W)
        out = out.view(m_batchsize, C1, height, width)
        #x = self.conv(x)
        out = self.gamma*out + xres          
        return out

class CAM0_Module(nn.Module):
        """ Channel attention module"""

        def __init__(self, in_dim):
            super(CAM0_Module, self).__init__()
            self.chanel_in = in_dim

            self.gamma = nn.Parameter(torch.zeros(1))  
            self.softmax = nn.Softmax(dim=-1)  

        def forward(self, x):
            """
                inputs :
                    x : input feature maps( B × C × H × W)
                returns :
                    out : attention value + input feature
                    attention: B × C × C
            """
            m_batchsize, C, height, width = x.size()
            # A -> (N,C,HW)
            proj_query = x.view(m_batchsize, C, -1)
            # A -> (N,HW,C)
            proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
            # 矩阵乘积，通道注意图：X -> (N,C,C)
            energy = torch.bmm(proj_query, proj_key)

            attention = self.softmax(energy) 
            # A -> (N,C,HW)
            proj_value = x.view(m_batchsize, C, -1)
            # XA -> （N,C,HW）
            out = torch.bmm(attention, proj_value)
            # output -> (N,C,H,W)
            out = out.view(m_batchsize, C, height, width)

            out = self.gamma * out + x
            return out

#filter_height = heght+(height-1)*(rate-1)
class ADCnn(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, group=1,dilation=1):
        super(ADCnn, self).__init__()
        #self.pad = get_pad_layer(pad_type)(dilation)  #
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.dilation = dilation
        # 2*K*K，先得到g*k*k个通道的maps。接着每个位置(i,j)都对应了一个k*k的向量，reshape为k x k，作为一个核
        self.conv = nn.Conv2d(in_channels, group * kernel_size * kernel_size, kernel_size=kernel_size, stride=1,padding=dilation,dilation=dilation,
                              bias=False)
        # self.conv = nn.Conv2d(in_channels, group * kernel_size * kernel_size, kernel_size=kernel_size, stride=1,
        #                        dilation=dilation,bias=False) 
        self.bn = nn.BatchNorm2d(group * kernel_size * kernel_size)
        self.softmax = nn.Softmax(dim=1)  
        #self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)  # 将通道压缩成256
        #self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU(inplace=True)
        #nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        sigma = self.conv(x)  
        #sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)  # 
        sigma = self.softmax(sigma)
        # sigma = self.sigmoid(sigma)     
        # sigma = self.relu(sigma)

        n, c, h, w = sigma.shape  

        sigma = sigma.reshape(n, 1, c, h * w)  

        return sigma

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out 
        out = self.sigmoid(out)
        x = x*out
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

if __name__ == '__main__':
    cam = CAM_Module(2048)
    input = torch.randn(1,2048,20,32)
    out = cam(input)
    # adc = DCCcnn(2048)
    # input = torch.randn(2,2048,20,32)
    # out = adc(input)

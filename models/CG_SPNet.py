import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from models.block.TCRPN import MLSF,TST
from models.block.MSSP import MSSP
#from thop import profile
from models.hrnet import HRNet18_cg
from utils import initialize_weights


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class CGFE(nn.Module):
    # codes derived from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super(CGFE, self).__init__()
        self.chanel_in = in_dim

        self.query_conv= nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv= nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)

        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x1,x2, change):
        ''' inputs :
                x : input feature maps( B X C X H X W)
                change : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x1.size()

        q1 = self.query_conv(change).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k1 = self.key_conv(change).view(m_batchsize, -1, width * height)

        v1 = self.value_conv1(x1).view(m_batchsize, -1, width * height)
        v2 = self.value_conv2(x2).view(m_batchsize, -1, width * height)

        energy = torch.bmm(q1, k1)
        attention= self.softmax(energy)
        out1 = torch.bmm(v1, attention.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)

        out2 = torch.bmm(v2, attention.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)

        out1 = x1 + self.gamma1 * out1
        out2 = x2 + self.gamma2 * out2


        # out1 = self.gamma1 * out1
        # out2 = self.gamma2 * out2
        return out1,out2

class Conv(nn.Module):
    def __init__(self, inch, outch, kernel, s, pad):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(inch, outch, kernel_size=kernel, stride=s, padding=pad),
            nn.BatchNorm2d(outch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv3(x)

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

class CG_SPNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7,drop_rate=0.2):
        super(CG_SPNet, self).__init__()
        #Encoder
        self.FCN = HRNet18_cg(True)
        #MSPA
        self.siam_FPR=MSSP(128,128,5)
        #SFCD
        self.down_sample=nn.Conv2d(256,128,3,2,padding=1)#SFCD
        self.CGFE=CGFE(128)#CGM
        self.fusion=MLSF(128)#SFCD
        self.TF=TST(256)#SFCD
        self.transit=nn.Conv2d(512,256,1,1)#SFCD
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)#SFCD
        self.down=nn.Conv2d(768,128,1)#SFCD

        classifier1=[]
        classifier2=[]
        classifierCD=[]
        if drop_rate>0:
            classifier1.append(nn.Dropout(p=drop_rate,inplace=False))
            classifier2.append(nn.Dropout(p=drop_rate,inplace=False))
            classifierCD.append(nn.Dropout(p=drop_rate,inplace=False))
        classifier1.append(nn.Conv2d(128, num_classes, kernel_size=1))
        classifier2.append(nn.Conv2d(128, num_classes, kernel_size=1))
        classifierCD.append(nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                          nn.Conv2d(64, 1, kernel_size=1)))
        self.classifier1 = nn.Sequential(*classifier1)
        self.classifier2 = nn.Sequential(*classifier2)
        self.classifierCD = nn.Sequential(*classifierCD)
        initialize_weights(self.resCD, self.classifierCD, self.classifier1, self.classifier2)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def base_forward(self, x,x_2):

        out1 ,x1_low=self.FCN(x)
        out2, x2_low=self.FCN(x_2)
        xc_low=self.TF(self.down_sample(x1_low),self.down_sample(x2_low))


        return out1,out2,xc_low
    def res_cd_forward(self,x1,x2,xc_low):
        b,c,h,w = x1.size()
        xc_list=[]
        #xc = torch.cat([x1,x2], 1)
        xc=self.fusion(x1,x2)
        xc=self.transit(torch.cat([xc,xc_low],dim=1))
        #xc+=xc_low
        for level in self.resCD:
            xc = level(xc)
            xc_list.append(xc)
        x1, x2, x3, x4,x5,x6= xc_list[0], xc_list[1], xc_list[2], xc_list[3],xc_list[4],xc_list[5]
        xc = torch.cat([x1, x2, x3, x4,x5,x6], dim=1)
        xc = self.down(xc)
        return xc
    def forward(self, input):
        dim = len(input.size())
        if (dim != 4):
            input = torch.unsqueeze(input, dim=0)
        (x1, x2) = torch.chunk(input, 2, dim=1)


        x_size = x1.size()
        x1,x2 ,xc_low= self.base_forward(x1,x2)


        x1 = self.siam_FPR(x1)
        x2 = self.siam_FPR(x2)

        change = self.res_cd_forward(x1, x2,xc_low)


        x1,x2=self.CGFE(x1,x2,change)
        out1 = self.classifier1(x1)
        out2 = self.classifier2(x2)
        change=self.classifierCD(change)

        return F.upsample(change, x_size[2:], mode='bilinear'), F.upsample(out1, x_size[2:],
                                                                           mode='bilinear'), F.upsample(out2,
                                                                                                        x_size[2:],
                                                                                                        mode='bilinear')



# net=CG_SPNet()
#
# x2=torch.randn((1,3,512,512))
# x1=torch.randn((1,3,512,512))
#
# #获取浮点数总数
# # stat(net, input_size=(6,512,512))
# print("----------------------------------------------------")
# print("----------------------------------------------------")
# # # summary(net,input_size=(6,512,512),device="cpu")
# # # print("----------------------------------------------------")
# # # print("----------------------------------------------------")
# flops, params = profile(net,(torch.cat([x1,x2],dim=1)))
# print("Total parameters: {:.2f}Mb".format(params / 1e6))
# print("Total flops: {:.2f}Gbps".format(flops / 1e9))
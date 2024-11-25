import torch
from torch import nn
#from thop  import profile
from models.block.TST import TST,TSF
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        #out = self.spatial_attention(out)
        return out

class MLSF(nn.Module):
    def __init__(self, inplane):
        super(MLSF, self).__init__()
        self.SSAE1 = CBAM(inplane)
        self.SSAE2 = CBAM(inplane)
        self.TF = TST(inplane*2)
        #self.TSF = TSF(1*2)
        self.TSF = TST(inplane*2)

        self.relu=nn.ReLU(True)

    def forward(self, x1, x2):
        identity = self.TF(x1, x2)
        x1 = self.SSAE1(x1)
        x2 = self.SSAE2(x2)
        TSF = self.TSF(x1, x2)
        out = identity * TSF
        out += identity
        cam=self.relu(out)
        #return self.relu(out)
        return out

class TCRPN_Siam(nn.Module):
    def __init__(self, inplane):
        super(TCRPN_Siam, self).__init__()
        self.SSAE = CBAM(inplane)
        self.TF = TST(inplane*2)
        #self.TSF = TSF(1*2)
        self.TSF = TST(inplane*2)

        self.relu=nn.ReLU(True)

    def forward(self, x1, x2):
        identity = self.TF(x1, x2)
        x1 = self.SSAE(x1)
        x2 = self.SSAE(x2)
        TSF = self.TSF(x1, x2)
        out = identity * TSF
        out += identity
        cam=self.relu(out)
        #return self.relu(out)
        return out

class TCRPN_WOSF(nn.Module):
    def __init__(self, inplane):
        super(TCRPN_WOSF, self).__init__()
        self.SSAE1 = CBAM(inplane)
        self.SSAE2 = CBAM(inplane)
        self.TF = nn.Sequential(nn.Conv2d(inplane*2, inplane*2, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(inplane*2), nn.ReLU())

        self.TSF =nn.Sequential(nn.Conv2d(inplane*2, inplane*2, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(inplane*2), nn.ReLU())

        self.relu=nn.ReLU(True)

    def forward(self, x1, x2):
        identity = self.TF(torch.cat([x1,x2],dim=1))
        x1 = self.SSAE1(x1)
        x2 = self.SSAE2(x2)
        TSF = self.TSF(torch.cat([x1,x2],dim=1))
        out = identity * TSF
        out += identity
        cam=self.relu(out)
        #return self.relu(out)
        return out

#cbam=CBAM(128)
# x1 = torch.randn((1, 128, 64, 64))
# x2 = torch.randn((1, 128, 64, 64))
#
# net = TCRPN(128)
# net.cuda()
# params_num=sum(p.numel()for p in net.parameters())
# print("\nParams: %1fM"%(params_num/1e6))
# out = net(x1.cuda(), x2.cuda())
# print(out)

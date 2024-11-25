import torch
from torch import nn

class PAM(nn.Module):
    '''Spatial reasoning module'''

    # codes from DANet 'Dual attention network for scene segmentation'
    def __init__(self, in_dim):
        super(PAM, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        ''' inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = x + self.gamma * out

        return out

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


class MSSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(MSSP, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, 0)
        self.cv2 = Conv(c1, c_, 1, 1, 0)
        self.cv3 = Conv(c_, c_, 3, 1, 1)
        self.cv4 = Conv(c_, c_, 1, 1, 0)
        self.SR1 = PAM(c_)
        self.SR2 = PAM(c_)
        self.SR3 = PAM(c_)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1, 0)
        self.cv6 = Conv(c_, c_, 3, 1, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1, 0)

    def forward(self, x):
        # test param

        x1 = self.cv4(self.cv3(self.cv1(x)))

        x2 = self.m(x1)
        x2 = self.SR1(x2)

        x3 = self.m(x2)
        x3 = self.SR2(x3)

        x4 = self.m(x3)
        x4 = self.SR3(x4)

        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, x4), 1)))
        y2 = self.cv2(x)

        out = self.cv7(torch.cat((y1, y2), dim=1))
        return out



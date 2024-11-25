import torch
from torch import nn
class TST(nn.Module):
    def __init__(self,inch):
        super(TST, self).__init__()
        self.ST1=nn.Conv2d(in_channels=inch,out_channels=inch,kernel_size=3,stride=1,padding=1)
        self.ST2=nn.Conv2d(in_channels=inch,out_channels=inch,kernel_size=3,stride=1,padding=1)
        self.BN1=nn.BatchNorm2d(inch)
        self.BN2=nn.BatchNorm2d(inch)
        self.relu1=nn.ReLU(2*inch)
        self.relu2=nn.ReLU(2*inch)

        self.relu3=nn.ReLU(2*inch)
    def forward(self,x1,x2):
        change1=self.ST1(torch.cat([x1,x2],dim=1))
        change1 = self.BN1(change1)
        change1=self.relu1(change1)

        change2=self.ST2(torch.cat([x2,x1],dim=1))
        change2 = self.BN2(change2)
        change2=self.relu2(change2)

        cam=self.relu3(change1*change2)

        # return self.relu3(change1*change2)
        return change1*change2
        #return change1+change2
class TSF(nn.Module):
    def __init__(self,inch):
        super(TSF, self).__init__()
        self.ST1=nn.Conv2d(in_channels=inch,out_channels=inch//2,kernel_size=3,stride=1,padding=1)
        self.ST2=nn.Conv2d(in_channels=inch,out_channels=inch//2,kernel_size=3,stride=1,padding=1)
        self.BN1=nn.BatchNorm2d(inch//2)
        self.BN2=nn.BatchNorm2d(inch//2)
        self.relu1=nn.ReLU(2*inch)
        self.relu2=nn.ReLU(2*inch)
    def forward(self,x1,x2):
        change1=self.ST1(torch.cat([x1,x2],dim=1))
        change1 = self.BN1(change1)
        change1=self.relu1(change1)

        change2=self.ST2(torch.cat([x2,x1],dim=1))
        change2 = self.BN2(change2)
        change2=self.relu2(change2)
        return change1*change2

class FF(nn.Module):
    def __init__(self,inch):
        super(FF, self).__init__()
        self.ST1=nn.Conv2d(in_channels=inch,out_channels=inch,kernel_size=3,stride=1,padding=1)
        self.BN1=nn.BatchNorm2d(inch)
        self.relu1=nn.ReLU(2*inch)
        self.relu3=nn.ReLU(2*inch)
    def forward(self,x1,x2):
        change1=self.ST1(torch.cat([x1,x2],dim=1))
        change1 = self.BN1(change1)
        change1=self.relu1(change1)
        cam=self.relu3(change1)
        # return self.relu3(change1*change2)
        return change1
        #return change1+change2
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def cal_score(score, target):
    if target == 1:
        return 1 - score
    else:
        return max(0, score)
def person_compute(X, Y):
    XY = X * Y
    EX = X.mean()
    EY = Y.mean()
    EX2 = (X ** 2).mean()
    EY2 = (Y ** 2).mean()
    EXY = XY.mean()
    numerator = EXY - EX * EY  # 分子
    denominator = math.sqrt(EX2 - EX ** 2) * math.sqrt(EY2 - EY ** 2)  # 分母
    if denominator == 0:
        return 'NaN'
    return numerator / denominator

class PearsonLoss(nn.Module):
    def __init__(self,reduction='mean'):
        super(PearsonLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x1, x2, margin):
        b, c, h, w = x1.size()
        scores = torch.zeros(b)
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b *h * w, c])
        x2 = torch.reshape(x2, [b* h * w, c])

        label_unchange = ~margin.bool()
        target = label_unchange.float()
        target = target - margin.float()
        target = torch.reshape(target, [b * h * w])
        for i in range(b*h*w):
            score = person_compute(x1[i], x2[i])
            scores[i] = cal_score(score, target[i].item())
        if self.reduction == 'mean':
            return scores.mean()
        elif self.reduction == 'sum':
            return scores.sum()
class PearsonLoss2(nn.Module):
    def __init__(self, reduction='mean'):
        super(PearsonLoss2, self).__init__()
        self.reduction = reduction

    def forward(self, x1, x2, margin):
        b, c, h, w = x1.size()

        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1).reshape(b * h * w, c)
        x2 = x2.permute(0, 2, 3, 1).reshape(b * h * w, c)

        XY = torch.mul(x1, x2)
        EX = x1.mean(dim=1, keepdim=True)
        EY = x2.mean(dim=1, keepdim=True)
        EX2 = torch.mean(x1**2, dim=1, keepdim=True)
        EY2 = torch.mean(x2**2, dim=1, keepdim=True)
        EXY = XY.mean(dim=1, keepdim=True)

        numerator = EXY - EX * EY
        denominator = torch.sqrt(EX2 - EX**2) * torch.sqrt(EY2 - EY**2)
        denominator[denominator == 0] = float('inf')

        scores = numerator / denominator
        scores[denominator == float('inf')] = float('nan')

        target = (~margin.bool()).float()
        s=torch.unique(target)
        target = torch.reshape(target, [b * h * w]).unsqueeze(dim=1)
        changed=torch.clamp(scores,0)
        unchanged=torch.clamp(1-scores,0)

        scores =  torch.mul(1-target, changed)+torch.mul(target,unchanged)

        if self.reduction == 'mean':
            return scores.mean()
        elif self.reduction == 'sum':
            return scores.sum()

class PearsonLoss3(nn.Module):
    def __init__(self, reduction='mean'):
        super(PearsonLoss3, self).__init__()
        self.reduction = reduction

    def forward(self, x1, x2, margin):
        b, c, h, w = x1.size()

        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1).reshape(b * h * w, c)
        x2 = x2.permute(0, 2, 3, 1).reshape(b * h * w, c)

        XY = torch.mul(x1, x2)
        EX = x1.mean(dim=1, keepdim=True)
        EY = x2.mean(dim=1, keepdim=True)
        EX2 = torch.mean(x1**2, dim=1, keepdim=True)
        EY2 = torch.mean(x2**2, dim=1, keepdim=True)
        EXY = XY.mean(dim=1, keepdim=True)

        numerator = EXY - EX * EY
        denominator = torch.sqrt(EX2 - EX**2) * torch.sqrt(EY2 - EY**2)
        denominator[denominator == 0] = float('inf')

        scores = numerator / denominator
        scores[denominator == float('inf')] = float('nan')

        target = (~margin.bool()).float()
        s=torch.unique(target)
        target = torch.reshape(target, [b * h * w]).unsqueeze(dim=1)
        changed=torch.exp(scores)
        unchanged= torch.exp(1-scores)

        scores =  torch.mul(1-target, changed)+torch.mul(target,unchanged)
        scores=0.5*scores

        if self.reduction == 'mean':
            return scores.mean()
        elif self.reduction == 'sum':
            return scores.sum()
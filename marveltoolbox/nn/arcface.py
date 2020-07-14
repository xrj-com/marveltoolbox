import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np



class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=4.5, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.th = math.pi - 1.01*m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        print('arcface with hyparam s:{} and m:{}'.format(self.s, m))

    def forward(self, z, y=None):
        N = len(z)
        z = z.view(N, -1)
        cosine = F.linear(F.normalize(z), F.normalize(self.weight))
        if y is None:
            return cosine
        zeros = torch.zeros(cosine.size(), device=cosine.device)
        theta = torch.acos(cosine*0.9999)
        delta = torch.where(theta >= self.th, zeros, self.m+zeros)
        one_hot = F.one_hot(y.long(), self.out_features).type_as(z)
        scores = self.s*torch.cos(theta + one_hot*delta)
        return scores

class ArcMarginProductPlus(ArcMarginProduct):
    def __init__(self, in_features, out_features, m=0.5, s_lower=0.5, s_upper=4.5):
        super().__init__(in_features, out_features, m=m)
        self.v = nn.Parameter(torch.Tensor([0]))
        self.lower = s_lower
        self.upper = s_upper
        print('arcface set trainable s:({}, {})'.format(self.lower, self.upper))

    def forward(self, z, y=None):
        N = len(z)
        z = z.view(N, -1)
        cosine = F.linear(F.normalize(z), F.normalize(self.weight))
        if y is None:
            return cosine
        zeros = torch.zeros(cosine.size(), device=cosine.device)
        theta = torch.acos(cosine*0.9999)
        delta = torch.where(theta >= self.th, zeros, self.m+zeros)
        one_hot = F.one_hot(y.long(), self.out_features).type_as(z)
        s = 1/(1+(self.v*-1).exp()) 
        s = s*(self.upper - self.lower) + self.lower
        self.s = s.item()
        scores = s*torch.cos(theta + one_hot*delta)
        return scores









if __name__ == '__main__':
    z = torch.Tensor([[-0.8, 0.8], [0.9, -0.6]])
    y = torch.Tensor([1, 0])
    clf = ArcFace(2, 2)
    clf.classifier.weight.data = torch.Tensor([[-1, 1],[1, -1]])
    print(clf(z))
    print(clf(z, y))



import torch
import torch.nn as nn
import torch.nn.functional as F

class IGAN(nn.Module):
    def __init__(self, realnvp, deq, nc, img_size, cond_label_size):
        super().__init__()
        self.model = realnvp
        self.deq = deq
        self.nc = nc
        self.img_size = img_size
        self.cond_label_size = cond_label_size

    def forward(self, z, y=None):
        N = len(z)
        labels = None
        if not y is None:
            labels = torch.zeros(N, self.cond_label_size).to(z.device)
            labels[:,y] = 1

        x, _ = self.model.inverse(z, labels)
        x = self.deq.inverse(x).view(-1, self.nc, self.img_size, self.img_size)
        return x

    def inverse(self, x, y=None):
        N = len(x)
        x = self.deq(x)
        x = x.view(x.shape[0], -1)
        if not y is None:
            labels = torch.zeros(N, self.cond_label_size).to(x.device)
            labels[:,y] = 1
        z, _ = self.model(x, labels)
        return z

    def log_prob(self, x, y=None):
        N = len(x)
        x = self.deq(x)
        x = x.view(N, -1)
        if not y is None:
            labels = torch.zeros(N, self.cond_label_size).to(x.device)
            labels[:,y] = 1
        return self.model.log_prob(x, labels)/(self.nc * self.img_size * self.img_size)
        
    
    
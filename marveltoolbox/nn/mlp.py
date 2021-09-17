import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalized_model import NormalizedModel


class MLP(nn.Module):
    def __init__(self, layers=[10, 1024, 512, 256, 10]):
        super().__init__()
        self.main = nn.ModuleList()
        self.layers = layers

        for i in range(len(self.layers)-1):
            self.main.append(
                nn.Linear(self.layers[i], self.layers[i+1])
                )
            if i < (len(self.layers) - 2):
                self.main.append(
                    nn.Sequential(
                        nn.BatchNorm1d(self.layers[i+1]),
                        nn.ReLU()
                    )
                )
                
    def forward(self, x):
        B = len(x)
        x = x.flatten().view(B, -1)
        for i in range(len(self.main)):
            x = self.main[i](x)
        return x
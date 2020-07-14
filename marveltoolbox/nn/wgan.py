import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalized_model import NormalizedModel

class Gnet32(nn.Module):
    def __init__(self, nc, nz):
        super().__init__()
        self.main = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=nz, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=nc, kernel_size=4, stride=2, padding=1))
        # self.output = nn.Tanh()
        self.output = nn.Sigmoid()

    def forward(self, z):
        N = len(z)
        out = self.main(z.view(N, -1, 1, 1))
        out2 = self.output(out)
        return out2
    
    
class Enet32(nn.Module):
    def __init__(self, nc, nz):
        super().__init__()
        self.main= nn.Sequential(
            # Image (Cx32x32)
            NormalizedModel(mean=torch.Tensor([0.5]), std=torch.Tensor([0.5])),
            nn.Conv2d(in_channels=nc, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.output = nn.Sequential(
            nn.Conv2d(1024, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, nz*2, 1)
        )   

    def forward(self, x):
        out =  self.main(x)
        out2 = self.output(out).view(len(x), -1)
        return out2 

    def from_begin(self, x):
        return self.main(x)
    
    def to_end(self, x):
        out = self.output(x).view(len(x), -1)
        return out

class Dnet32(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.main= nn.Sequential(
            # Image (Cx32x32)
            NormalizedModel(mean=torch.Tensor([0.5]), std=torch.Tensor([0.5])),
            nn.Conv2d(in_channels=nc, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.output = nn.Sequential(
            nn.Conv2d(1024, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 1)
        )   

    def forward(self, x):
        out =  self.main(x)
        out = self.output(out).view(len(x), -1)
        return out
import torch
import torch.distributions as D
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MOG(Dataset):
    def __init__(self, dataset_size=25000):
        self.dataset_size = dataset_size
        self.resample()
    
    def resample(self):
        self.data = torch.Tensor([(i,j) for i in range(-4, 5, 2) for j in range(-4, 5, 2)] * self.dataset_size)
        index = torch.randperm(self.data.size(0))
        self.data = self.data[index]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.data[i] + 0.1 * torch.randn(*self.data[i].shape)

if __name__ == "__main__":
    test_data = MOG()
    testloader = DataLoader(test_data, batch_size=10)
    print(next(iter(testloader)) )
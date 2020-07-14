import torch
import torch.distributions as D
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ADV_SPHERE_ONLINE(Dataset):
    def __init__(self, dim=500, r=1, R=1.3, dataset_size=100000):
        self.dim = 500
        self.R = R
        self.r = r
        self.dataset_size = dataset_size
    
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        x = torch.nn.functional.normalize(torch.randn(1, self.dim))
        if torch.rand(1).item() > 0.5:
            return x * self.r, 0
        else:
            return x * self.R, 1

class ADV_SPHERE_FIXED(Dataset):
    def __init__(self, dim=500, r=1, R=1.3, dataset_size=100000):
        self.dim = 500
        self.R = R
        self.r = r
        self.dataset_size = dataset_size
        self.resample()
    
    def resample(self):
        self.data = torch.nn.functional.normalize(torch.randn(self.dataset_size, self.dim))
        self.labels = torch.rand(self.dataset_size) > 0.5
        self.data[self.labels] *= self.R
        self.data[~self.labels] *= self.r
        self.labels = self.labels.long()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.data[i], self.labels[i]

class ADV_SPHERE_batch():
    def __init__(self, dim=500, r=1, R=1.3, batch_size=500):
        self.dim = 500
        self.R = R
        self.r = r
        self.batch_size = batch_size
        
    def __next__(self):
        data = torch.nn.functional.normalize(torch.randn(self.batch_size, self.dim))
        labels = torch.rand(self.batch_size) > 0.5
        data[labels] *= self.R
        data[~labels] *= self.r
        labels = labels.long()
        return data, labels


if __name__ == "__main__":
    test_data = ADV_SPHERE_batch(batch_size=10)
    # testloader = DataLoader(test_data, batch_size=10)
    
    # print(next(iter(testloader)) )
    print(next(test_data))
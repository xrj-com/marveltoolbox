from .configs import DATASET
import torch.utils.data as data
from PIL import Image
from typing import Dict, Tuple
import collections
import os
import torch
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from random import shuffle


class CelebA_SELECT(torchvision.datasets.ImageFolder):
    
    def __init__(self, transform=None):
        CelebA_Image_root = os.path.join(DATASET.CELEBA_ROOT, 'Img')
        super().__init__(root=CelebA_Image_root, transform=transform)
        self.labels = self.get_labels()
                
    def get_labels(self):
        CelebA_Attr_file = os.path.join(DATASET.CELEBA_ROOT, 'Anno/list_attr_celeba.txt')
        Attr_type = 21   
        labels = []
        with open(CelebA_Attr_file, "r") as Attr_file:
            Attr_info = Attr_file.readlines()
            Attr_info = Attr_info[2:]
            for line in Attr_info:
                info = line.split()
                id = int(info[0].split('.')[0])
                label = int(info[Attr_type])
                labels.append(int((label+1)/2))
        return labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img = super().__getitem__(index)[0]
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)
    
    
def load_celeba(
    downsample_pct: float = 0.5, train_pct: float = 0.8, batch_size: int = 50, img_size: int = 28, label_list: list = None
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Load CelebA dataset (download if necessary) and split data into training,
        validation, and test sets.
    Args:
        downsample_pct: the proportion of the dataset to use for training,
            validation, and test
        train_pct: the proportion of the downsampled data to use for training
    Returns:
        DataLoader: training data
        DataLoader: validation data
        DataLoader: test data
    """

    transform = transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = CelebA_SELECT(transform=transform)
    N = len(dataset)
    n = int(N*downsample_pct)
    n_train = int(n*0.80*train_pct)
    n_val = int(n*0.80*(1-train_pct))
    n_test = int(n*0.20)
    n_other = N-(n_train+n_val+n_test)
    
    train_set, val_set, test_set, _ = torch.utils.data.random_split(
        dataset,
        lengths=[n_train, n_val, n_test, n_other],
    )
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

    return train_loader, val_loader, test_loader, test_loader

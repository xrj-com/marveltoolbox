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
import copy



class COIL_SELECT(torchvision.datasets.ImageFolder):
    def __init__(self, label_list=None, transform=None, target_transform=None):
        super().__init__(DATASET.COIL_ROOT, transform, target_transform) 
        self.label_list = label_list
        self.class_num = 20
        if self.label_list is not None:
            self.remap_dict = {}
            for i, label in enumerate(label_list):
                self.remap_dict[label] = i
            self.preprocess()
            self.class_num = len(label_list)
        
        self.shuffle_targets()
        
    def shuffle_targets(self):
        temp_samples = []
        self.origin_samples = self.samples
        if self.class_num > 1:
            for i in range(len(self.samples)):
                target = self.samples[i][1]
                target_list = [i for i in range(self.class_num)]
                target_list.remove(int(target))
                rand_target = np.random.choice(target_list)
                temp_samples.append((self.samples[i][0], rand_target))
            self.attack_samples = temp_samples

    def target_remap(self, target):
        return self.remap_dict[target]

    def set_attack(self, flag=False):
        if flag:
            self.samples = self.attack_samples
        else:
            self.samples = self.origin_samples

    def preprocess(self):
        temp_samples = []
        
        for i in range(len(self.samples)):
            if self.samples[i][1] in self.label_list:
                temp_samples.append((self.samples[i][0], self.remap_dict[self.samples[i][1]]))
        
        self.samples = temp_samples

def load_coil(
    downsample_pct: float = 0.5, train_pct: float = 0.8, batch_size: int = 50, img_size: int = 28, label_list: list = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load MNIST dataset (download if necessary) and split data into training,
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
    # Specify transforms
    # pyre-fixme[16]: Module `transforms` has no attribute `Compose`.
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    # )
    transform = transforms.Compose(
        [transforms.Resize(img_size),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]
    ) 
    # Load training set
    # pyre-fixme[16]: Module `datasets` has no attribute `MNIST`.
    train_test_set = COIL_SELECT(
        label_list=label_list, transform=transform
    )

    # Partition into training/validation
    downsampled_num_examples = int(downsample_pct * len(train_test_set))
    n_train_examples = int(train_pct * downsampled_num_examples)
    n_test_examples = downsampled_num_examples - n_train_examples

    train_set, test_set, _ = torch.utils.data.random_split(
        train_test_set,
        lengths=[
            n_train_examples,
            n_test_examples,
            len(train_test_set) - downsampled_num_examples,
        ],
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    valid_loader = None
    
 
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    targeted_test_set = copy.deepcopy(test_set)
    targeted_test_set.dataset.set_attack(True)

    targeted_test_loader = DataLoader(targeted_test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, valid_loader, test_loader, targeted_test_loader

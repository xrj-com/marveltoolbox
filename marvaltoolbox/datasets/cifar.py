from .configs import DATASET

from PIL import Image
from typing import Dict, Tuple
import collections
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from random import shuffle


class saturation:
    def __init__(self, p=265):
        self.p=p

    def __call__(self, x):
        xcen = x*2-1
        out = torch.sign(xcen)* torch.abs(xcen)**(2/self.p) /2 + 1/2
        return out

def get_suffle_index(data_len, seed=0):
    subset_index = [i for i in range(data_len)]
    random.seed(seed)
    shuffle(subset_index)
    return subset_index

class CIFAR_SELECT(torchvision.datasets.CIFAR10):
    def __init__(self, label_list=None, train=True, transform=None, target_transform=None, download=False, is_target_attack=False, is_pair=False):
        super().__init__(DATASET.CIFAR10_ROOT, train, transform, target_transform, download)
        self.label_list = label_list
        self.class_num = 10
        self.is_target_attack = is_target_attack
        self.is_pair = is_pair
        if self.label_list is not None:
            self.remap_dict = {}
            for i, label in enumerate(label_list):
                self.remap_dict[label] = i
            self.preprocess()
            self.class_num = len(label_list)
        
        if self.is_target_attack:
            self.shuffle_targets()
        
        if self.is_pair:
            self.shuffle_data()
            # self.remap_dict = self.one_vs_all(label_list[0], label_list)
    def shuffle_targets(self):
        shuffle_targets = []
        
        for target in self.targets:
            target_list = [i for i in range(self.class_num)]
            target_list.remove(int(target))
            rand_target = np.random.choice(target_list)
            shuffle_targets.append(rand_target)
        self.targets = shuffle_targets

    def shuffle_data(self):
        shuffle_data = []
        
        for i, (sample, target) in enumerate(zip(self.data, self.targets)):
            index_list = [j for j in range(len(self.data))]
            index_list.remove(i)
            while 1:
                rand_index = np.random.choice(index_list)
                if self.targets[rand_index] != target:
                    shuffle_data.append(self.data[rand_index])
                    break
        self.target_data = shuffle_data
        self.targets = [0 for i in range(len(self.data))]


    def one_vs_all(self, target, label_list):
        remap_dict = {}
        for i, label in enumerate(label_list):
            if target == label:
                remap_dict[label] = 1
            else:
                remap_dict[label] = 0
        self.class_num = 2
        return remap_dict

    def target_remap(self, target):
        return self.remap_dict[target]

    def preprocess(self):
        selected_data = []
        selected_targets = []
        
        for i in range(len(self.data)):
            if self.targets[i] in self.label_list:
                selected_data.append(self.data[i])
                selected_targets.append(self.target_remap(self.targets[i]))
        
        self.data = selected_data
        self.targets = selected_targets
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        
        if self.is_pair:
            target_img = self.target_data[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.is_pair:
            target_img = Image.fromarray(target_img)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.is_pair:
            target_img = self.transform(target_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_pair:
            return img, target_img, target

        return img, target


def load_cifar10(
    downsample_pct: float = 0.5, train_pct: float = 0.8, batch_size: int = 50, img_size: int = 32, label_list: list = None, is_norm=False
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
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
    if is_norm:
        train_transform = transforms.Compose(
            [
            # transforms.RandomCrop(32, padding=4),
            transforms.Resize(img_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # saturation(p=256)
            ]
                            
        ) 
        eval_transform = transforms.Compose(
            [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # saturation(p=256)
        ]
        ) 
    else:
        train_transform = transforms.Compose(
            [
            # transforms.RandomCrop(32, padding=4),
            transforms.Resize(img_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # saturation(p=256)
            ]
                            
        ) 
        eval_transform = transforms.Compose(
            [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            # saturation(p=256)
        ]
        ) 
    # Load training set
    # pyre-fixme[16]: Module `datasets` has no attribute `MNIST`.

    train_valid_set = CIFAR_SELECT(label_list, train=True, transform=train_transform, download=True)
    # train_set = torch.utils.data.Subset(CIFAR_SELECT(label_list, train=True, transform=train_transform, download=True), list(range(45000)))
    # valid_set = torch.utils.data.Subset(CIFAR_SELECT(label_list, train=True, transform=eval_transform, download=True),
    #                     list(range(45000, 50000)))
    # Partition into training/validation
    # downsampled_num_examples = int(downsample_pct * len(train_valid_set))
    # n_train_examples = int(train_pct * downsampled_num_examples)
    # n_valid_examples = downsampled_num_examples - n_train_examples
    # train_set, valid_set, _ = torch.utils.data.random_split(
    #     train_valid_set,
    #     lengths=[
    #         n_train_examples,
    #         n_valid_examples,
    #         len(train_valid_set) - downsampled_num_examples,
    #     ],
    # )
        # Partition into training/validation
    downsampled_num_examples = int(downsample_pct * len(train_valid_set))
    n_train_examples = int(train_pct * downsampled_num_examples)
    n_valid_examples = downsampled_num_examples - n_train_examples

    train_set, valid_set, _ = torch.utils.data.random_split(
        train_valid_set,
        lengths=[
            n_train_examples,
            n_valid_examples,
            len(train_valid_set) - downsampled_num_examples,
        ],
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    if train_pct < 1.0:
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        valid_loader = None

    # if train_pct < 1.0:
    #     valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2)
    # else:

    # Load test set
    # pyre-fixme[16]: Module `datasets` has no attribute `MNIST`.
    test_set_all = CIFAR_SELECT(
        label_list=label_list, train=False, download=True, transform=eval_transform
    )
    subset_index = get_suffle_index(len(test_set_all))

    downsampled_num_test_examples = int(downsample_pct * len(test_set_all))
    test_set = torch.utils.data.Subset(
        test_set_all, indices=subset_index[0:downsampled_num_test_examples]
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    targeted_attack_test_set_all = CIFAR_SELECT(
        label_list=label_list, train=False, download=True, transform=eval_transform, is_target_attack=False
    )
    subset_index = get_suffle_index(len(targeted_attack_test_set_all))

    downsampled_num_test_examples = int(downsample_pct * len(test_set_all))
    targeted_test_set = torch.utils.data.Subset(
        targeted_attack_test_set_all, indices=subset_index[0:downsampled_num_test_examples]
    )
    targeted_test_loader = DataLoader(targeted_test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, valid_loader, test_loader, targeted_test_loader


def load_cifar100(
    downsample_pct: float = 0.5, train_pct: float = 0.8, batch_size: int = 50
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
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1,))]
    ) 
    # Load training set
    # pyre-fixme[16]: Module `datasets` has no attribute `MNIST`.

    train_valid_set = torchvision.datasets.CIFAR100(
        root=DATASET.CIFAR100_ROOT, train=True, download=True, transform=transform
    )

    # Partition into training/validation
    downsampled_num_examples = int(downsample_pct * len(train_valid_set))
    n_train_examples = int(train_pct * downsampled_num_examples)
    n_valid_examples = downsampled_num_examples - n_train_examples
    train_set, valid_set, _ = torch.utils.data.random_split(
        train_valid_set,
        lengths=[
            n_train_examples,
            n_valid_examples,
            len(train_valid_set) - downsampled_num_examples,
        ],
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    if train_pct < 1.0:
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=1)
    else:
        valid_loader = None

    # Load test set
    # pyre-fixme[16]: Module `datasets` has no attribute `MNIST`.
    test_set_all = torchvision.datasets.CIFAR100(
        root=DATASET.CIFAR100_ROOT, train=False, download=True, transform=transform
    )
    downsampled_num_test_examples = int(downsample_pct * len(test_set_all))
    test_set, _ = torch.utils.data.random_split(
        test_set_all,
        lengths=[
            downsampled_num_test_examples,
            len(test_set_all) - downsampled_num_test_examples,
        ],
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, valid_loader, test_loader
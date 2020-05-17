from torchvision.datasets import SVHN
from .configs import DATASET
import torch.utils.data as data
from typing import Dict, Tuple
from PIL import Image
from random import shuffle
import collections
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import random
from random import shuffle

def get_suffle_index(data_len, seed=0):
    subset_index = [i for i in range(data_len)]
    random.seed(seed)
    shuffle(subset_index)
    return subset_index

class SVHN_SELECT(SVHN):
    def __init__(self, label_list=None, split='train', transform=None, target_transform=None, download=False, is_target_attack=False, is_pair=False):
        super().__init__(DATASET.SVHN_ROOT, split, transform, target_transform, download)
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
            # self.remap_dict = self.one_vs_all(label_list[0], label_list)
        if self.is_target_attack:
            self.shuffle_targets()

        if self.is_pair:
            self.shuffle_data()

    def shuffle_targets(self):
        shuffle_targets = []
        if self.class_num > 1:
            for target in self.labels:
                target_list = [i for i in range(self.class_num)]
                target_list.remove(target)
                rand_target = np.random.choice(target_list)
                shuffle_targets.append(rand_target)
            self.labels = shuffle_targets
        else:
            pass

    def shuffle_data(self):
        shuffle_data = []
        
        for i, (sample, target) in enumerate(zip(self.data, self.labels)):
            index_list = [j for j in range(len(self.data))]
            index_list.remove(i)
            while 1:
                rand_index = np.random.choice(index_list)
                if self.labels[rand_index] != target:
                    shuffle_data.append(self.data[rand_index])
                    break
        self.target_data = shuffle_data
        self.labels = [0 for i in range(len(self.data))]

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
        selected_labels = []
        
        for i in range(len(self.data)):
            # print(self.labels[i])
            if self.labels[i] in self.label_list:
                selected_data.append(self.data[i])
                selected_labels.append(self.target_remap(self.labels[i].item()))
        
        self.data = selected_data
        self.labels = selected_labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])
        
        if self.is_pair:
            target_img = self.target_data[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.is_pair:
            target_img = Image.fromarray(np.transpose(target_img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)
        
        if self.is_pair:
            target_img = self.transform(target_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_pair:
            return img, target_img, target

        return img, target

def load_svhn(
    downsample_pct: float = 0.5, train_pct: float = 0.8, batch_size: int = 50, img_size: int = 32, label_list: list = None, is_norm=False
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
    if is_norm:
        print('image range [-1, 1]')
        transform = transforms.Compose(
            [transforms.Resize(img_size),transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ) 
    else:
        print('image range [0, 1]')
        transform = transforms.Compose(
            [transforms.Resize(img_size),transforms.ToTensor()]
        ) 
    # Load training set
    # pyre-fixme[16]: Module `datasets` has no attribute `MNIST`.
    train_valid_set = SVHN_SELECT(
        label_list=label_list, split='train', download=True, transform=transform
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
    print('trainset len: {}'.format(len(train_set)))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    if train_pct < 1.0:
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=4)
        print('valset len: {}'.format(len(valid_set)))
    else:
        valid_loader = None
    
    # Load test set
    # pyre-fixme[16]: Module `datasets` has no attribute `MNIST`.
    test_set_all = SVHN_SELECT(
        label_list=label_list, split='test', download=True, transform=transform
    )
    subset_index = get_suffle_index(len(test_set_all))
    downsampled_num_test_examples = int(downsample_pct * len(test_set_all))
    test_set = torch.utils.data.Subset(
        test_set_all, indices=subset_index[0:downsampled_num_test_examples]
    )
    print('testset len: {}'.format(len(test_set)))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    targeted_attack_test_set_all = SVHN_SELECT(
        label_list=label_list, split='test', download=True, transform=transform, is_target_attack=True
    )
    downsampled_num_test_examples = int(downsample_pct * len(test_set_all))
    targeted_test_set = torch.utils.data.Subset(
        targeted_attack_test_set_all, indices=subset_index[0:downsampled_num_test_examples]
    )
    targeted_test_loader = DataLoader(targeted_test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader, targeted_test_loader

    # return train_loader, valid_loader, test_loader

def load_svhn_pairs(
    downsample_pct: float = 0.5, batch_size: int = 50, img_size: int = 32, label_list: list = None
) -> Tuple[DataLoader]:
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
    transform = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ) 

    test_set_all = SVHN_SELECT(
        label_list=label_list, split='test', download=True, transform=transform, is_pair=True
    )
    subset_index = get_suffle_index(len(test_set_all))
    downsampled_num_test_examples = int(downsample_pct * len(test_set_all))
    test_set = torch.utils.data.Subset(
        test_set_all, indices=subset_index[0:downsampled_num_test_examples]
    )
    
    pair_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    print(len(pair_loader.dataset))

    return pair_loader




        

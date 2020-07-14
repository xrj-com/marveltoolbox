from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
import bcolz
import pickle
import torch
# import mxnet as mx
from tqdm import tqdm
from .configs import DATASET
import os

# class DATASET:
#     # CASIA_ROOT = '/media/D/DATASET/casia-webface' 
#     # LFW_ROOT = '/media/D/DATASET/lfw/LFW' 
#     # MNIST_ROOT = '/media/D/DATASET/MNIST'
#     EMORE_ROOT = Path('/root/DATASET/faces_emore')
#     WEBFACE_ROOT = Path('/root/DATASET/faces_webface_112x112')
#     CASIA_ROOT = '/root/DATASET/casia-webface' 
#     LFW_ROOT = '/root/DATASET/lfw/LFW' 
#     MNIST_ROOT = '/root/DATASET/MNIST'
#     FASHION_MNIST_ROOT = '/root/DATASET/FashionMNIST'
#     CIFAR10_ROOT = '/root/DATASET/CIFAR10'
#     CIFAR100_ROOT = '/root/DATASET/CIFAR100'
#     SVHN_ROOT = '/root/DATASET/SVHN'
#     CELEBA_ROOT = '/root/DATASET/CelebA'
#     COIL_ROOT = '/root/DATASET/coil-20-proc'

#     CASIA_LANDMARK = os.path.join(CASIA_ROOT, 'casia_landmark.txt')
#     CASIA_DATA_ROOT = os.path.join(CASIA_ROOT, 'CASIA-WebFace')
#     PREPROCESS_CASIA_ROOT = os.path.join(CASIA_ROOT, 'PRE-CASIA-WebFace')

#     LFW_DATA_ROOT = os.path.join(LFW_ROOT, 'lfw')
#     LFW_LANDMARK = os.path.join(LFW_ROOT, 'lfw_landmark.txt')
#     LFW_PAIRS = os.path.join(LFW_ROOT, 'pairs.txt')
# class DATASET:
#     # CASIA_ROOT = '/media/D/DATASET/casia-webface' 
#     # LFW_ROOT = '/media/D/DATASET/lfw/LFW' 
#     WEBFACE_ROOT = '/media/D/DATASET/MNIST'
#     EMORE_ROOT = Path('/root/DATASET/faces_emore')
#     CASIA_ROOT = '/root/DATASET/casia-webface' 
#     LFW_ROOT = '/root/DATASET/lfw/LFW' 
#     MNIST_ROOT = '/root/DATASET/MNIST'
#     CIFAR10_ROOT = '/root/DATASET/CIFAR10'
#     CIFAR100_ROOT = '/root/DATASET/CIFAR100'




class conf:
    data_mode = 'emore'
    emore_folder = DATASET.EMORE_ROOT
    webface_folder = DATASET.WEBFACE_ROOT
    pin_memory = True
    num_workers = 4


def de_preprocess(tensor):
    return tensor*0.5 + 0.5
    
def get_subsample_indices(ds, fract=0.15, min_num=30):
    class_num = ds[-1][1] + 1
    if fract >= 1.0:
        return ds, class_num

    max_class_num = int(class_num * fract)
    class_counts = {}
    class2Index = {}
    for i in range(len(ds)):
        y = ds.targets[i]
        class_counts[y] = class_counts.setdefault(y, 0) + 1

    indices = []
    class_dict = {}
    t = 0
    for i in range(len(ds)):
        y = ds.targets[i]
        if class_counts[y] > min_num:
            if class_dict.setdefault(y, -1) != -1:
                indices.append(i)
            elif t < max_class_num:
                class_dict[y] = t
                indices.append(i)
                t +=1
            else:
                pass

    for i in range(len(ds)):
        y = ds.targets[i]
        if class_counts[y] > min_num:
            if class_dict[y] != -1:
                ds.targets[i] = class_dict[y]
                temp_sample = ds.samples[i]
                ds.samples[i] = (temp_sample[0], class_dict[y])

    ds = Subset(ds, indices)
    class_num = ds[-1][1] + 1
    return ds, class_num


def get_train_dataset(imgs_folder, fract=0.25, min_num=30):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.CenterCrop((112, 112)),
        trans.ToTensor(),
        # trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    ds = ImageFolder(imgs_folder, train_transform)
    ds, class_num = get_subsample_indices(ds, fract, min_num)
    return ds, class_num

def get_face_trainloader(batch_size, fract=1, min_num=0, mode='webface',conf=conf):
    # if conf.data_mode in ['ms1m', 'concat']:
    #     ms1m_ds, ms1m_class_num = get_train_dataset(conf.ms1m_folder/'imgs')
    #     print('ms1m loader generated')
    # if conf.data_mode in ['vgg', 'concat']:
    #     vgg_ds, vgg_class_num = get_train_dataset(conf.vgg_folder/'imgs')
    #     print('vgg loader generated')        
    # if conf.data_mode == 'vgg':
    #     ds = vgg_ds
    #     class_num = vgg_class_num
    # elif conf.data_mode == 'ms1m':
    #     ds = ms1m_ds
    #     class_num = ms1m_class_num
    # elif conf.data_mode == 'concat':
    #     for i,(url,label) in enumerate(vgg_ds.imgs):
    #         vgg_ds.imgs[i] = (url, label + ms1m_class_num)
    #     ds = ConcatDataset([ms1m_ds,vgg_ds])
    #     class_num = vgg_class_num + ms1m_class_num
    if mode == 'webface':
        ds, class_num = get_train_dataset(conf.webface_folder/'imgs', fract=fract, min_num=min_num)
    elif mode == 'emore':
        ds, class_num = get_train_dataset(conf.emore_folder/'imgs', fract=fract, min_num=min_num)
 
    loader = DataLoader(
        ds, batch_size=batch_size, 
        shuffle=True,
        pin_memory=conf.pin_memory, 
        num_workers=conf.num_workers)
    return loader, class_num 
    
def load_bin(path, rootdir, transform, image_size=[112,112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir)+'_list', np.array(issame_list))
    return data, issame_list

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir = path/name, mode='r')
    issame = np.load(path/'{}_list.npy'.format(name))
    return carray, issame

def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame


hflip = trans.Compose([
            trans.ToPILImage(),
            trans.functional.hflip,
            trans.ToTensor(),
        ])

class VAL_DATASET(Dataset):

    def __init__(self, path, name, transform=None, target_transform=None,
                 is_flip=False):
        
        self.data, self.issame = get_val_pair(path, name)
        self.img1s = self.data[0::2]
        self.img2s = self.data[1::2]
        self.transform = transform
        self.target_transform = target_transform
        self.is_flip = is_flip

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img1 = de_preprocess(torch.tensor(self.img1s[index]))
        img2 = de_preprocess(torch.tensor(self.img2s[index]))
        target = int(self.issame[index])
        if self.is_flip:
            img1_flip = hflip(img1)
            img2_flip = hflip(img2)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            if self.is_flip:
                img1_flip = self.transform(img1_flip)
                img2_flip = self.transform(img2_flip)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_flip:
            return (img1, img1_flip, img2, img2_flip), target
        return (img1, img2), target

    def __len__(self):
        return len(self.issame)


def load_lfw(batch_size, is_flip=True):
    transform = trans.Compose([
            trans.ToPILImage(),
            trans.CenterCrop((112, 112)),
            trans.ToTensor(),
        ])
    test_loader = DataLoader(
        VAL_DATASET(DATASET.EMORE_ROOT, 'lfw', transform=transform, is_flip=is_flip), 
        batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return test_loader


# def get_val_loader(path, name):
#     data, issame = get_val_pair(path, name)
    

if __name__ == "__main__":
    # test =  load_LFW(5).dataset
    # print(test[1:4].shape)
    dataloader, class_num = get_train_loader(10, fract=0.5, min_num=30, mode='webface',conf=conf)
    
    # dataset = VAL_DATASET(DATASET.EMORE_ROOT, 'lfw', is_filp=True)
    # print(dataset[1:4])
    # print([type(dataset[i][-1]) for i in [1,4,5,6]])
    # b = torch.squeeze(a)
    # print(b)
    # print(lfw.shape)
    # print(issame.shape)
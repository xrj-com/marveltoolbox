import os
import cv2
from PIL import Image 
from tqdm import tqdm
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from .configs import DATASET
from .matlab_cp2tform import get_similarity_transform_for_cv2


def alignment(src_img, src_pts):
    of = 2
    ref_pts = [[30.2946 + of, 51.6963 + of], [65.5318 + of, 51.5014 + of],
               [48.0252 + of, 71.7366 + of], [33.5493 + of, 92.3655 + of], [62.7299 + of, 92.2041 + of]]
    # crop_size = (96 + of * 2, 112 + of * 2)
    crop_size = (112 + of * 2, 112 + of * 2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

def img_loader(path, landmark_dict, is_filp=False, pre_root=None):
    if pre_root is not None:
        try:
            tail_path = os.path.join(*path.split('/')[-2:])
            pre_path = os.path.join(pre_root, tail_path)
            pil_img = Image.open(pre_path)
            if is_filp:
                return pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            return pil_img
        except:
            tail_root = path.split('/')[-2]
            pre_folder = os.path.join(pre_root, tail_root)
            if not os.path.exists(pre_folder):
                os.makedirs(pre_folder)

    cv2_img = cv2.imread(path)
    src_pts = landmark_dict[path]
    cv2_img = alignment(cv2_img, src_pts)
    cv2_img = np.asarray(cv2_img)[:, :, ::-1].copy()
    pil_img = Image.fromarray(cv2_img)
    if pre_root is not None:
        tail_path = os.path.join(*path.split('/')[-2:])
        pre_path = os.path.join(pre_root, tail_path)
        pil_img.save(pre_path)

    if is_filp:
        return pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    return pil_img


def lfw_pairs(data_root, pairs_root, file_ext='jpg'):
    pairs = []
    with open(pairs_root, 'r') as file:
        pbar = tqdm(file.readlines()[1:])
        pbar.set_description(
            "LFW PAIRS Processing")
        for line in pbar:
            line_split = line.split()
            if len(line_split) == 3:
                img1 = os.path.join(
                    data_root, line_split[0], line_split[0] + '_' + '%04d' % int(line_split[1]) + '.' + file_ext).replace('\\', '/')
                img2 = os.path.join(
                    data_root, line_split[0], line_split[0] + '_' + '%04d' % int(line_split[2]) + '.' + file_ext).replace('\\', '/')
                target = 1
            elif len(line_split) == 4:
                img1 = os.path.join(
                    data_root, line_split[0], line_split[0] + '_' + '%04d' % int(line_split[1]) + '.' + file_ext).replace('\\', '/')
                img2 = os.path.join(
                    data_root, line_split[2], line_split[2] + '_' + '%04d' % int(line_split[3]) + '.' + file_ext).replace('\\', '/')
                target = 0
            pairs.append([img1, img2, target])
    return pairs


def lfw_landmark(data_root, landmark_root):
    landmark_dict = dict()
    with open(landmark_root, 'r') as file:
        pbar = tqdm(file.readlines())
        pbar.set_description(
            "Landmark dict Processing")
        for line in pbar:
            line_split = line.split()
            path = os.path.join(data_root, line_split[0])
            src_pts = []
            for i in range(5):
                src_pts.append([int(line_split[2 * i + 1]),
                                int(line_split[2 * i + 2])])
            landmark_dict[path] = src_pts

    return landmark_dict

class LFWpairs_DATASET(data.Dataset):

    def __init__(self, root=DATASET.LFW_DATA_ROOT, 
                 landmark_root=DATASET.LFW_LANDMARK, 
                 pairs_root=DATASET.LFW_PAIRS,
                 transform=None, target_transform=None,
                 loader=img_loader, is_filp=False):
        self.cache_path = os.path.join(root, 'cache_dict.pt.tar') 
        if not os.path.isfile(self.cache_path):
            self.landmark_dict = lfw_landmark(root, landmark_root)
            self.pairs = lfw_pairs(root, pairs_root)
            torch.save(
                {
                    'landmark_dict': self.landmark_dict,
                    'pairs': self.pairs
                }, self.cache_path)
        else:
            cache_dict = torch.load(self.cache_path)
            self.landmark_dict = cache_dict['landmark_dict']
            self.pairs = cache_dict['pairs']
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.is_filp = is_filp

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path1, path2, target = self.pairs[index]
        img1 = self.loader(path1, self.landmark_dict)
        img2 = self.loader(path2, self.landmark_dict)
        if self.is_filp:
            img1_flip = self.loader(path1, self.landmark_dict, is_filp=True)
            img2_flip = self.loader(path2, self.landmark_dict, is_filp=True)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            if self.is_filp:
                img1_flip = self.transform(img1_flip)
                img2_flip = self.transform(img2_flip)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_filp:
            return img1, img1_flip, img2, img2_flip, target
        return img1, img2, target

    def __len__(self):
        return len(self.pairs)


def load_lfw(batch_size):
    test_transform = transforms.Compose([
                transforms.CenterCrop((112, 112)),
                transforms.ToTensor(),
            ])
    test_loader = data.DataLoader(
        LFWpairs_DATASET(transform=test_transform, is_filp=True), 
        batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return test_loader

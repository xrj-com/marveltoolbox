import os
import cv2
from PIL import Image 
from tqdm import tqdm
from .matlab_cp2tform import get_similarity_transform_for_cv2
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from .configs import DATASET


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

def casia_imgs_landmark(data_root, landmark_root):
    landmark_dict = dict()
    imgs = []
    class_list = []
    cache_path = os.path.join(os.path.split(landmark_root)[0], 'cache_dict.pt.tar')
    if not os.path.isfile(cache_path):
        with open(landmark_root, 'r') as file:
            pbar = tqdm(file.readlines())
            for line in pbar:
                line_split = line.split()
                pbar.set_description(
                    "Landmark dict Processing {}".format(line_split[0]))
                path = os.path.join(data_root, line_split[0])
                target = int(line_split[1])
                class_list.append(target)
                imgs.append([path, target])
                src_pts = []
                for i in range(5):
                    src_pts.append([int(line_split[2 * i + 2]),
                                    int(line_split[2 * i + 3])])
                landmark_dict[path] = src_pts
        torch.save(
            {
                "imgs": imgs,
                "landmark_dict" : landmark_dict,
                "class_list" : class_list
            }, cache_path)
    else: 
        cache_dict = torch.load(cache_path)
        imgs = cache_dict['imgs']
        landmark_dict = cache_dict['landmark_dict']
        class_list = cache_dict['class_list']

    return imgs, landmark_dict, len(set(class_list))

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

class CASIAWebFace_DATASET(data.Dataset):

    def __init__(self, origin_root=DATASET.CASIA_DATA_ROOT, pre_root=DATASET.PREPROCESS_CASIA_ROOT,
                 landmark_root=DATASET.CASIA_LANDMARK,
                 transform=None, target_transform=None,
                 loader=img_loader):

        self.imgs, self.landmark_dict, self.classes = casia_imgs_landmark(
            origin_root, landmark_root)
        self.origin_root = origin_root
        self.pre_root = pre_root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path, self.landmark_dict, pre_root=self.pre_root)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def load_casia(batch_size):
    train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop((112, 112)),
                transforms.CenterCrop((112, 112)),
                transforms.ToTensor(),
            ])
    train_loader = data.DataLoader(
        CASIAWebFace_DATASET(transform=train_transform), 
        batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return train_loader









import os 
from pathlib import Path

class DATASET:
    # CASIA_ROOT = '/media/D/DATASET/casia-webface' 
    # LFW_ROOT = '/media/D/DATASET/lfw/LFW' 
    # MNIST_ROOT = '/media/D/DATASET/MNIST'
    EMORE_ROOT = Path('/workspace/DATASET/faces_emore')
    WEBFACE_ROOT = Path('/workspace/DATASET/faces_webface_112x112')
    CASIA_ROOT = '/workspace/DATASET/casia-webface' 
    LFW_ROOT = '/workspace/DATASET/lfw/LFW' 
    MNIST_ROOT = '/workspace/DATASET/MNIST'
    FASHION_MNIST_ROOT = '/workspace/DATASET/FashionMNIST'
    CIFAR10_ROOT = '/workspace/DATASET/CIFAR10'
    CIFAR100_ROOT = '/workspace/DATASET/CIFAR100'
    SVHN_ROOT = '/workspace/DATASET/SVHN'
    CELEBA_ROOT = '/workspace/DATASET/CelebA'
    COIL_ROOT = '/workspace/DATASET/coil-20-proc'

    CASIA_LANDMARK = os.path.join(CASIA_ROOT, 'casia_landmark.txt')
    CASIA_DATA_ROOT = os.path.join(CASIA_ROOT, 'CASIA-WebFace')
    PREPROCESS_CASIA_ROOT = os.path.join(CASIA_ROOT, 'PRE-CASIA-WebFace')

    LFW_DATA_ROOT = os.path.join(LFW_ROOT, 'lfw')
    LFW_LANDMARK = os.path.join(LFW_ROOT, 'lfw_landmark.txt')
    LFW_PAIRS = os.path.join(LFW_ROOT, 'pairs.txt')


class EXP:
    ROOT = '/workspace/EXP-xrj'
    TB_ROOT = '/workspace/runs'


if __name__ == "__main__":
    print(DATASET.CASIA_DATA_ROOT)
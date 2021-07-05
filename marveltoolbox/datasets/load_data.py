from .mnist import load_mnist, load_mnist_pairs
from .cifar import load_cifar10, load_cifar100
from .svhn import load_svhn, load_svhn_pairs
from .fashion_mnist import load_fmnist, load_fmnist_pairs
from .toy import load_toy
from .moons import load_moons


def load_data(dataset, all_frac, train_frac, batch_size, img_size, label_list, is_norm=False, data_root='./DATASET'): 
    if dataset == 'mnist':
        return load_mnist(all_frac, train_frac, batch_size, img_size=img_size,  label_list=label_list, is_norm=is_norm, data_root='./DATASET')
    elif dataset == 'svhn':
        return load_svhn(all_frac, train_frac, batch_size, img_size=img_size,  label_list=label_list, is_norm=is_norm, data_root='./DATASET')
    elif dataset == 'fmnist':
        return load_fmnist(all_frac, train_frac, batch_size, img_size=img_size,  label_list=label_list, is_norm=is_norm, data_root='./DATASET')
    elif dataset == 'cifar10':
        return load_cifar10(all_frac, train_frac, batch_size,img_size=img_size,  label_list=label_list, is_norm=is_norm, data_root='./DATASET')
    else:
        return [None, None, None, None]
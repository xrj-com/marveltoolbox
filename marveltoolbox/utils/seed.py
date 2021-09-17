import torch
import numpy as np
import random

def set_seed(seed=None):
    if not seed is None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic=True
        random.seed(seed)
        print('Set random seed to: {}'.format(seed))
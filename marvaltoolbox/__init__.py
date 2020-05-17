import os

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as f:
    __version__ = f.read().strip()

from . import utils
from . import nn
from . import inn
from . import datasets
from .trainer import BaseTrainer, HvdTrainer
from .configs import BaseConfs
import sys
sys.path.append('..')
import mytoolbox as mt
import torch

if __name__ == '__main__':
    confs = mt.BaseConfs()
    trainer = mt.BaseTrainer(confs)
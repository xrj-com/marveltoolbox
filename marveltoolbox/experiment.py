import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.utils import save_image
import numpy as np
import os
import logging
import traceback
from marveltoolbox import utils
import shutil

class BaseExperiment():
    def __init__(self, confs):
        self.flag = confs.exp_flag
        self.save_flag = confs.exp_flag
        self.log_path = confs.exp_path
        self.exp_path = confs.exp_path
        self.datasets = {}
        self.dataloaders = {}
        self.trainers = {}
        self.results = {}
        self.logs = {}
        self.logger = None

        if not os.path.exists(self.exp_path):
            print(self.exp_path, 'dose not exist')
            os.makedirs(self.exp_path)

    
    def preprocessing(self):
        kwargs = {'num_workers': 0, 'drop_last': True, 'pin_memory': True} if torch.cuda.is_available() else {}
        for key in self.datasets.keys():
            self.dataloaders[key] = torch.utils.data.DataLoader(
                self.datasets[key], batch_size=self.batch_size, **kwargs)

        for trainer in self.trainers.keys():
            self.results[trainer] = {}
            for dataset in self.datasets.keys():
                self.results[trainer][dataset] = {}

    def set_logger(self):
        self.logger = logging.getLogger(__name__) 
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        log_file_name = '{}.log'.format(self.save_flag)
        log_file = os.path.join(self.log_path, log_file_name)
        print('Log file save at: ', log_file)
        hfile = logging.FileHandler(log_file)
        self.logger.addHandler(hfile)

    def print_logs(self):
        print_str = 'Results: '
        for key, value in self.logs.items():
            if type(value) == str:
                print_str += '{}:{} '.format(key, value)
            else:
                print_str += '{}:{:4f} '.format(key, value)
        print(print_str)
        if self.logger is not None:
            self.logger.info(print_str)

    def main(self):
        pass
                

    def run(self, *args, **kwargs):
        try:
            utils.set_seed(self.seed)
            self.set_logger()
            temp_results = self.load() 
            if kwargs['is_rerun'] or (temp_results is None):
                self.main(*args, **kwargs)
                self.save()
            else:
                self.results = temp_results
            if kwargs['is_del_loger']:
                del self.logger
                self.logger = None

        except Exception as e:
            print_str = traceback.format_exc()
            print(print_str)
            if self.logger is not None:
                self.logger.info(print_str)

    def save(self):
        file_name = os.path.join(self.exp_path, 'Exp_{}.pth.tar'.format(self.save_flag))
        torch.save(self.results, file_name)

        
    def load(self):
        results = None
        result_file = os.path.join(self.exp_path, 'Exp_{}.pth.tar'.format(self.save_flag))
        if os.path.isfile(result_file):
            print("=> loading results '{}'".format(result_file))
            results = torch.load(result_file)
        else:
            print("=> no results found at '{}'".format(result_file))
        return results

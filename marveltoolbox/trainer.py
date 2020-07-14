import torch 
import torch.nn as nn
import torch.nn.functional as F
import horovod.torch as hvd
import torchvision as tv
from torchvision.utils import save_image
import numpy as np
import os
import logging
import traceback
from marveltoolbox import utils
import shutil


def save_checkpoint(state, is_best, file_path='./', flag=''):
    file_name = os.path.join(file_path, 'checkpoint_{}.pth.tar'.format(flag))
    torch.save(state, file_name)
    if is_best:
        best_file_name = os.path.join(file_path, 'model_best_{}.pth.tar'.format(flag))
        shutil.copyfile(file_name, best_file_name)

def load_checkpoint(is_best, file_path='./', flag=''):
    checkpoint = None
    if is_best:
        chkpt_file = os.path.join(file_path, 'model_best_{}.pth.tar'.format(flag))
    else:
        chkpt_file = os.path.join(file_path, 'checkpoint_{}.pth.tar'.format(flag))

    if os.path.isfile(chkpt_file):
        print("=> loading checkpoint '{}'".format(chkpt_file))
        checkpoint = torch.load(chkpt_file, lambda storage, loc: storage)
    else:
        print("=> no checkpoint found at '{}'".format(chkpt_file))
    return checkpoint


class BaseTrainer():
    def __init__(self, confs):
        self.flag = confs.flag
        self.save_flag = confs.flag
        self.chkpt_path = confs.chkpt_path
        self.log_path = confs.log_path
        self.device = confs.device
        self.batch_size = confs.batch_size
        self.epochs = confs.epochs
        self.seed = confs.seed
        self.device_ids = confs.device_ids
        self.start_epoch = 0
        self.schedulers = {}
        self.models = {}
        self.optims = {}
        self.datasets = {}
        self.dataloaders = {}
        self.records = {}
        self.logs = {}
        self.model_names = self.models.keys()
        self.logger = None

        if not os.path.exists(self.chkpt_path):
            print(self.chkpt_path, 'dose not exist')
            os.makedirs(self.chkpt_path)
        
        if not os.path.exists(self.log_path):
            print(self.log_path, 'dose not exist')
            os.makedirs(self.log_path)

    def set_logger(self):
        self.logger = logging.getLogger(__name__) 
        self.logger.handlers = []
        self.logger.setLevel(logging.INFO)
        log_file_name = '{}.log'.format(self.save_flag)
        log_file = os.path.join(self.log_path, log_file_name)
        print('Log file save at: ', log_file)
        hfile = logging.FileHandler(log_file)
        self.logger.addHandler(hfile)
        
    def train(self, epoch):
        return 0.0
                
    def eval(self, epoch):
        return False

    def scheduler_step(self):
        pass

    def print_logs(self, epoch, step):
        print_str = 'Epoch/Iter:{:0>3d}/{:0>4d} '.format(epoch, step)
        for key, value in self.logs.items():
            print_str += '{}:{:4f} '.format(key, value)
        print(print_str)
        if self.logger is not None:
            self.logger.info(print_str)

    def main(self, load_best=False, retrain=False):
        utils.set_seed(self.seed)
        self.set_logger()
        if not retrain:
            self.load()
        timer = utils.Timer(self.epochs-self.start_epoch, self.logger)
        timer.init()
        for epoch in range(self.start_epoch, self.epochs):
            loss = self.train(epoch)
            is_best = self.eval(epoch)
            timer.step()
            self.save(is_best=is_best)
            self.start_epoch += 1
            self.scheduler_step()

        if load_best:
            self.load(is_best=True)
            print_str = 'Best epoch: {:0>3d} \n'.format(self.start_epoch)
            print(print_str)
            if self.logger is not None:
                self.logger.info(print_str)
        del self.logger
        self.logger = None

    def run(self, *args, **kwargs):
        try:
            self.main(*args, **kwargs)
        except Exception as e:
            print_str = traceback.format_exc()
            print(print_str)
            if self.logger is not None:
                self.logger.info(print_str)
            
    def save(self, is_best=False):
        state_dict = {}
        state_dict['epoch'] = self.start_epoch + 1
        state_dict['records'] = self.records

        for name, optim in self.optims.items():
            state_dict['optim_{}'.format(name)] = self.optims[name].state_dict()
        
        for name, model in self.models.items():
            state_dict['model_{}'.format(name)] = self.models[name].state_dict()

        for name, scheduler in self.schedulers.items():
            state_dict['scheduler_{}'.format(name)] = self.schedulers[name].state_dict()

        save_checkpoint(state_dict, is_best, file_path=self.chkpt_path, flag=self.save_flag)
        
    def load(self, is_best=False):
        chkpt = load_checkpoint(is_best, file_path=self.chkpt_path, flag=self.save_flag)
        if chkpt:
            self.start_epoch = chkpt['epoch']
            self.records = chkpt['records']

            for name, optim in self.optims.items():
                self.optims[name].load_state_dict(chkpt['optim_{}'.format(name)])
            
            for name, model in self.models.items():
                self.models[name].load_state_dict(chkpt['model_{}'.format(name)])

            for name, scheduler in self.schedulers.items():
                self.schedulers[name].load_state_dict(chkpt['scheduler_{}'.format(name)])
            print_str = "=> loaded checkpoint (epoch {})".format(chkpt['epoch'])
            print(print_str)
            if self.logger is not None:
                self.logger.info(print_str)


class HvdTrainer(BaseTrainer):
    def __init__(self, confs):
        super().__init__(confs)
        self.samplers = {}
        self.lr_scaler = 1
        hvd.init()
        torch.set_num_threads(1)
        torch.cuda.set_device(hvd.local_rank())
        self.device = torch.device(
            "cuda:{}".format(hvd.local_rank()))
        self.hvd_param_scaling()

    
    def hvd_param_scaling(self):
        if hvd.nccl_built():
            self.lr_scaler = hvd.local_size()
            print('Rescale lr = {} * lr'.format(self.lr_scaler))
            # self.batch_size = int(self.batch_size / hvd.local_size())
            # print('batchsize = 1/{} * batchsize'.format(hvd.local_size()))

    def hvd_preprocessing(self, op=hvd.Adasum):
        kwargs = {'num_workers': 1, 'drop_last': True, 'pin_memory': True} if torch.cuda.is_available() else {}

        for key in self.datasets.keys():
            self.samplers[key] = torch.utils.data.distributed.DistributedSampler(
                self.datasets[key], num_replicas=hvd.size(), rank=hvd.rank())
            self.dataloaders[key] = torch.utils.data.DataLoader(
                self.datasets[key], batch_size=self.batch_size, sampler=self.samplers[key], **kwargs)

        self.hvd_broadcast()

        for key in self.optims.keys():
            self.optims[key] = hvd.DistributedOptimizer(self.optims[key],
                                         named_parameters=self.models[key].named_parameters(),
                                         op=op)

    def hvd_broadcast(self):
        for key in self.optims.keys():
            hvd.broadcast_parameters(self.models[key].state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self.optims[key], root_rank=0)

    def metric_average(self, val, name):
        tensor = torch.tensor(val)
        avg_tensor = hvd.allreduce(tensor, name=name)
        return avg_tensor.item()

    def main(self, load_best=False, retrain=False):
        utils.set_seed(hvd.rank())
        if not retrain:
            self.load()

        if hvd.rank() == 0:
            self.set_logger()
            timer = utils.Timer(self.epochs-self.start_epoch, self.logger)
            timer.init()
        for epoch in range(self.start_epoch, self.epochs):
            loss = self.train(epoch)
            is_best = self.eval(epoch)
            self.start_epoch += 1
            self.scheduler_step()

            if hvd.rank() == 0:
                timer.step()
                self.save(is_best=is_best)

        if hvd.rank() == 0:
            if load_best:
                self.load(is_best=True)
                self.hvd_broadcast()
                print_str = 'Best epoch: {:0>3d} \n'.format(self.start_epoch)
                print(print_str)
                if self.logger is not None:
                    self.logger.info(print_str)
            del self.logger
            self.logger = None



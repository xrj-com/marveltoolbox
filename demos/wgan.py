import sys
sys.path.append('..')
import marveltoolbox as mt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import horovod.torch as hvd
import math
import os


class Confs(mt.BaseConfs):
    def __init__(self):
        super().__init__()
    
    def get_dataset(self):
        self.dataset = 'mnist'
        self.nc = 1
        self.nz = 8
        self.img_size = 32

    def get_flag(self):
        self.batch_size = 64
        self.critic_iter = 5
        self.max_iterations = 30000
        self.lr = 5e-5
        self.iters_per_epoch = int(self.max_iterations/self.epochs)
        self.plot_path = './temp'
        self.flag = 'demo-{}-wgan'.format(self.dataset)


    def get_device(self):
        self.device_ids = [0]
        self.ngpu = len(self.device_ids)
        self.device = torch.device(
            "cuda:{}".format(self.device_ids[0]) if \
            (torch.cuda.is_available() and self.ngpu > 0) else "cpu")



class Trainer(mt.BaseTrainer, Confs):
    def __init__(self):
        Confs.__init__(self)
        mt.BaseTrainer.__init__(self, self)

        self.models['G'] = mt.nn.wgan.Gnet32(self.nc, self.nz).to(self.device)
        self.models['D'] = mt.nn.wgan.Dnet32(self.nc).to(self.device)

        self.optims['G'] = torch.optim.RMSprop(
            self.models['G'].parameters(), 
            lr=self.lr)

        self.optims['D'] = torch.optim.RMSprop(
            self.models['D'].parameters(), 
            lr=self.lr)
        
        self.dataloaders['train'], self.dataloaders['val'], self.dataloaders['test'], _ = \
            mt.datasets.load_data(self.dataset, 1.0, 1.0, self.batch_size, self.img_size, None, False)

        self.dataiter = self.get_infinite_batches(self.dataloaders['train'])

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def train(self, epoch):
        for i in range(self.iters_per_epoch):
            self.models['G'].train()
            self.models['D'].train()
            # Training D
            # for real data
            # Requires grad, Generator requires_grad = False
            for p in self.models['D'].parameters():
                p.requires_grad = True
                
            Wasserstein_D = 0.0
            for _ in range(self.critic_iter):
                batch_data = next(self.dataiter)
                x_real = batch_data
                x_real = x_real.to(self.device)
                bs = x_real.size(0)
                
                self.optims['D'].zero_grad()
                
                probs_real = self.models['D'](x_real)
                loss_real = probs_real.mean()
                
                # for fake data
                z = torch.randn(bs, self.nz, device=self.device)

                x_fake = self.models['G'](z).clone().detach()
                probs_fake = self.models['D'](x_fake)
                loss_fake = probs_fake.mean()
                
                D_loss = loss_fake - loss_real
                D_loss.backward()
                self.optims['D'].step()
                for p in self.models['D'].parameters(): p.data.clamp_(-0.01, 0.01)

                Wasserstein_D = (loss_real-loss_fake).detach().item()

            # Training G
            for p in self.models['D'].parameters():
                p.requires_grad = False  # to avoid computation

            self.optims['G'].zero_grad()

            z = torch.randn(bs, self.nz, device=self.device)
            x_fake = self.models['G'](z)
            probs_fake = self.models['D'](x_fake)
            G_loss = -probs_fake.mean()

            G_loss.backward()
            self.optims['G'].step()

            if i % 100 == 0:
                self.logs['D loss'] = D_loss.item()
                self.logs['G loss'] = G_loss.item()
                self.logs['WD'] = Wasserstein_D
                self.print_logs(epoch, i)

    @torch.no_grad()
    def eval(self, epoch):
        self.models['G'].eval()
        z = torch.randn(64, self.nz, device=self.device)
        x_fake = self.models['G'](z)
        filename = os.path.join(self.plot_path, 'wgan_x.png')
        mt.utils.plot_tensor(x_fake, figsize=(20, 20), filename=filename)
        return False

class Trainer_hvd(mt.HvdTrainer, Confs):
    def __init__(self):
        Confs.__init__(self)
        mt.HvdTrainer.__init__(self, self)

        self.models['G'] = mt.nn.wgan.Gnet32(self.nc, self.nz).to(self.device)
        self.models['D'] = mt.nn.wgan.Dnet32(self.nc).to(self.device)

        self.optims['G'] = torch.optim.RMSprop(
            self.models['G'].parameters(), 
            lr=self.lr * self.lr_scaler)

        self.optims['D'] = torch.optim.RMSprop(
            self.models['D'].parameters(), 
            lr=self.lr * self.lr_scaler)
        
        train_loader, _, test_loader, _ = \
            mt.datasets.load_data(self.dataset, 1.0, 1.0, self.batch_size, self.img_size, None, False)
        
        self.datasets['train'] = train_loader.dataset
        self.datasets['test'] = test_loader.dataset

        self.hvd_preprocessing(op=hvd.Average)

        self.dataiter = self.get_infinite_batches(self.dataloaders['train'])

    def hvd_param_scaling(self):
        if hvd.nccl_built():
            self.batch_size = int(self.batch_size/hvd.local_size())
            self.iters_per_epoch = int(self.max_iterations/self.epochs/hvd.local_size())
            self.lr_scaler = hvd.local_size()

    def get_infinite_batches(self, data_loader):
        while True:
            for i, (images, _) in enumerate(data_loader):
                yield images

    def train(self, epoch):
        for i in range(self.iters_per_epoch):
            self.models['G'].train()
            self.models['D'].train()
            # Training D
            # for real data
            # Requires grad, Generator requires_grad = False
            for p in self.models['D'].parameters():
                p.requires_grad = True
                
            Wasserstein_D = 0.0
            for _ in range(self.critic_iter):
                batch_data = next(self.dataiter)
                x_real = batch_data
                x_real = x_real.to(self.device)
                bs = x_real.size(0)
                
                self.optims['D'].zero_grad()
                
                probs_real = self.models['D'](x_real)
                loss_real = probs_real.mean()
                
                # for fake data
                z = torch.randn(bs, self.nz, device=self.device)

                x_fake = self.models['G'](z).clone().detach()
                probs_fake = self.models['D'](x_fake)
                loss_fake = probs_fake.mean()
                
                D_loss = loss_fake - loss_real
                D_loss.backward()
                self.optims['D'].step()
                for p in self.models['D'].parameters(): p.data.clamp_(-0.01, 0.01)

                Wasserstein_D = (loss_real-loss_fake).detach().item()

            # Training G
            for p in self.models['D'].parameters():
                p.requires_grad = False  # to avoid computation

            self.optims['G'].zero_grad()

            z = torch.randn(bs, self.nz, device=self.device)
            x_fake = self.models['G'](z)
            probs_fake = self.models['D'](x_fake)
            G_loss = -probs_fake.mean()

            G_loss.backward()
            self.optims['G'].step()

            if i % 100 == 0:
                self.logs['D loss'] = D_loss.item()
                self.logs['G loss'] = G_loss.item()
                self.logs['WD'] = Wasserstein_D
                self.print_logs(epoch, i)

    @torch.no_grad()
    def eval(self, epoch):
        self.models['G'].eval()
        z = torch.randn(64, self.nz, device=self.device)
        x_fake = self.models['G'](z)
        if hvd.rank() == 0:
            filename = os.path.join(self.plot_path, 'wgan_x.png')
            mt.utils.plot_tensor(x_fake, figsize=(20, 20), filename=filename)
        return False

if __name__ == '__main__':
    trainer = Trainer_hvd()
    trainer.run(load_best=False, retrain=True)
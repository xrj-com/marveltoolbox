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
        self.lambda_term = 10
        self.max_iterations = 30000
        self.lr = 1e-4
        self.beta1 = 0.5
        self.beta2 = 0.9
        self.iters_per_epoch = int(self.max_iterations/self.epochs)
        self.plot_path = './temp'
        self.flag = 'demo-{}-wgan-gp'.format(self.dataset)


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

        self.models['G'] = mt.nn.wgangp.Gnet32(self.nc, self.nz).to(self.device)
        self.models['D'] = mt.nn.wgangp.Dnet32(self.nc).to(self.device)

        self.optims['G'] = torch.optim.Adam(
            self.models['G'].parameters(), 
            lr=self.lr, betas=(self.beta1, self.beta2))

        self.optims['D'] = torch.optim.Adam(
            self.models['D'].parameters(), 
            lr=self.lr, betas=(self.beta1, self.beta2))
        
        self.train_loader, self.val_loader, self.test_loader, _ = \
            mt.datasets.load_data(self.dataset, 1.0, 1.0, self.batch_size, self.img_size, None, False)

        self.dataiter = self.get_infinite_batches(self.train_loader)

    def calculate_gradient_penalty(self, real_images, fake_images):
        bs = real_images.shape[0]
        eta = torch.FloatTensor(bs,1,1,1).uniform_(0,1)
        eta = eta.expand(*(real_images.shape))
        eta = eta.to(self.device)

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        # define it to calculate gradient
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.models['D'](interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size(), device=self.device),
                               create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

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
                
                gradient_penalty = self.calculate_gradient_penalty(x_real, x_fake)
                D_loss = loss_fake - loss_real + gradient_penalty
                D_loss.backward()
                self.optims['D'].step()
                
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
                self.logs['GP'] = gradient_penalty.item()
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

        self.models['G'] = mt.nn.wgangp.Gnet32(self.nc, self.nz).to(self.device)
        self.models['D'] = mt.nn.wgangp.Dnet32(self.nc).to(self.device)

        self.optims['G'] = torch.optim.Adam(
            self.models['G'].parameters(), 
            lr=self.lr, betas=(self.beta1, self.beta2))

        self.optims['D'] = torch.optim.Adam(
            self.models['D'].parameters(), 
            lr=self.lr, betas=(self.beta1, self.beta2))
        
        train_loader, _, test_loader, _ = \
            mt.datasets.load_data(self.dataset, 1.0, 1.0, self.batch_size, self.img_size, None, False)
        
        self.datasets['train'] = train_loader.dataset
        self.datasets['test'] = test_loader.dataset

        self.hvd_preprocessing()

        self.dataiter = self.get_infinite_batches(self.dataloaders['train'])

    def hvd_param_scaling(self):
        if hvd.nccl_built():
            self.batch_size = int(self.batch_size/hvd.local_size())
            self.iters_per_epoch = int(self.max_iterations/self.epochs/hvd.local_size())

    def calculate_gradient_penalty(self, real_images, fake_images):
        bs = real_images.shape[0]
        eta = torch.FloatTensor(bs,1,1,1).uniform_(0,1)
        eta = eta.expand(*(real_images.shape))
        eta = eta.to(self.device)

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        # define it to calculate gradient
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.models['D'](interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size(), device=self.device),
                               create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty

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
                
                gradient_penalty = self.calculate_gradient_penalty(x_real, x_fake)
                D_loss = loss_fake - loss_real + gradient_penalty
                D_loss.backward()
                self.optims['D'].step()
                
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
                self.logs['GP'] = gradient_penalty.item()
                self.print_logs(epoch, i)

    @torch.no_grad()
    def eval(self, epoch):
        self.models['G'].eval()
        z = torch.randn(64, self.nz, device=self.device)
        x_fake = self.models['G'](z)
        filename = os.path.join(self.plot_path, 'wgan_x.png')
        mt.utils.plot_tensor(x_fake, figsize=(20, 20), filename=filename)
        return False

if __name__ == '__main__':
    trainer = Trainer_hvd()
    trainer.run(load_best=False, retrain=False)
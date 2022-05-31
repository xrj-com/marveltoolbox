import marveltoolbox as mt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import matplotlib.pyplot as plt


class Confs(mt.BaseConfs):
    def __init__(self):
        super().__init__()
    
    def get_dataset(self):
        self.dataset = 'mog'
        self.input_size = 2
        self.nz = 10
        self.nz_m = 3
        self.img_size = 32

    def get_flag(self):
        self.epochs = 300
        self.batch_size = 100
        self.critic_iter = 5
        self.lambda_term = 10
        self.max_iterations = 300000
        self.lr = 2e-4
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.iters_per_epoch = int(self.max_iterations/self.epochs)
        self.plot_path = './temp'
        self.flag = 'demo-{}-mine-gan'.format(self.dataset)


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

        self.models['G'] = mt.nn.mine.Gnet(self.nz, output_size=self.input_size).to(self.device)
        self.models['D'] = mt.nn.mine.Dnet(self.input_size, output_size=1).to(self.device)
        self.models['M'] = mt.nn.mine.Mine(noise_size=self.nz_m, sample_size=self.input_size, output_size=1).to(self.device)

        self.optims['G'] = torch.optim.Adam(
            self.models['G'].parameters(), 
            lr=self.lr, betas=(self.beta1, self.beta2))

        self.optims['D'] = torch.optim.Adam(
            self.models['D'].parameters(), 
            lr=self.lr, betas=(self.beta1, self.beta2))

        self.optims['M'] = torch.optim.Adam(
            self.models['M'].parameters(), 
            lr=self.lr, betas=(self.beta1, self.beta2))
        
        self.train_loader, self.val_loader, self.test_loader, _ = \
            mt.datasets.load_data(self.dataset, 1.0, 1.0, self.batch_size, self.img_size, None, False)

        self.dataiter = self.get_infinite_batches(self.train_loader)

        self.ET = None

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
            self.models['M'].train()
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
            z_b = torch.randn(bs, self.nz_m, device=self.device)
            x_fake = self.models['G'](z)
            probs_fake = self.models['D'](x_fake)
            mi_loss = self.MI_object(z[:, 0:self.nz_m], z_b, x_fake, is_train_mi=False)
            G_loss = -probs_fake.mean() - 0.01 * mi_loss

            G_loss.backward()
            self.optims['G'].step()

            # Training M
            self.optims['M'].zero_grad()
            z = torch.randn(bs, self.nz, device=self.device)
            z_b = torch.randn(bs, self.nz_m, device=self.device)
            x_fake = self.models['G'](z)
            mi_loss = self.MI_object(z[:, 0:self.nz_m], z_b, x_fake, is_train_mi=True)
            M_loss = - mi_loss
            M_loss.backward()
            self.optims['M'].step()

            if i % 100 == 0:
                self.logs['D loss'] = D_loss.item()
                self.logs['G loss'] = G_loss.item()
                self.logs['M loss'] = M_loss.item()
                self.logs['WD'] = Wasserstein_D
                self.logs['GP'] = gradient_penalty.item()
                self.print_logs(epoch, i)

    @torch.no_grad()
    def eval(self, epoch):
        self.models['G'].eval()
        z = torch.randn(64, self.nz, device=self.device)
        x_fake = self.models['G'](z)
        filename = os.path.join(self.plot_path, 'mine_gan_x.png')
        mt.utils.plot_tensor(x_fake, figsize=(20, 20), filename=filename)
        return False

def update_target(ma_net, net, update_rate=1e-1):
    # update moving average network parameters using network
    for ma_net_param, net_param in zip(ma_net.parameters(), net.parameters()):
        ma_net_param.data.copy_((1.0 - update_rate) \
                                * ma_net_param.data + update_rate*net_param.data)




if __name__ == '__main__':
    trainer = Trainer()
    trainer.run(load_best=False, retrain=True)
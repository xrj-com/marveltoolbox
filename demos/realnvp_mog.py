import sys
sys.path.append('..')
import marveltoolbox as mt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import horovod.torch as hvd
import math
import os
import matplotlib.pyplot as plt


class Confs(mt.BaseConfs):
    def __init__(self):
        super().__init__()
    
    def get_dataset(self):
        self.epochs = 3000
        self.dataset = 'mog'
        self.input_size = 2

    def get_flag(self):
        self.n_blocks = 10
        self.n_components = 1
        self.hidden_size = 100
        self.n_hidden = 1
        self.cond_label_size = None
        self.is_batch_norm = False
        self.plot_path = './temp'
        self.flag = 'demo-{}-realnvp'.format(self.dataset)


    def get_device(self):
        self.device_ids = [0]
        self.ngpu = len(self.device_ids)
        self.device = torch.device(
            "cuda:{}".format(self.device_ids[0]) if \
            (torch.cuda.is_available() and self.ngpu > 0) else "cpu")



class Trainer(mt.HvdTrainer, Confs):
    def __init__(self):
        Confs.__init__(self)
        mt.BaseTrainer.__init__(self, self)

        self.models['realnvp'] = mt.inn.RealNVP(
            self.n_blocks, self.input_size, 
            self.hidden_size, self.n_hidden,
            self.cond_label_size, batch_norm=self.is_batch_norm).to(self.device)

        self.optims['realnvp'] = torch.optim.Adam(
            self.models['realnvp'].parameters(), 
            lr=1e-6, betas=(0.9, 0.99), weight_decay=1e-6)
        
        self.datasets['train'] = mt.datasets.MOG(dataset_size=25000)
        self.datasets['test'] = mt.datasets.MOG(dataset_size=2500)

        self.hvd_preprocessing()

        self.records['best_logprob'] = float('-inf')

    def train(self, epoch):
        self.models['realnvp'].train()
        for i, x in enumerate(self.train_loader):
            x = x.to(self.device)
            x = self.models['deq'](x)
            x = x.view(x.shape[0], -1)
            loss = - self.models['realnvp'].log_prob(
                x, y if self.cond_label_size else None).mean(0)
            self.optims['realnvp'].zero_grad()
            loss.backward()
            self.optims['realnvp'].step()
            if i % 100 == 0:
                self.logs['Train Loss'] = loss.item()
                self.print_logs(epoch, i)

        return loss.item()

    @torch.no_grad()            
    def eval(self, epoch):
        self.models['realnvp'].eval()
        # conditional model
        if self.cond_label_size is not None:
            logprior = torch.tensor(1 / self.cond_label_size).log().to(self.device)
            loglike = [[] for _ in range(self.cond_label_size)]

            for i in range(self.cond_label_size):
                # make one-hot labels
                labels = torch.zeros(self.batch_size, self.cond_label_size).to(self.device)
                labels[:,i] = 1

                for x, y in self.val_loader:
                    x = self.models['deq'](x)
                    x = x.view(x.shape[0], -1).to(self.device)
                    loglike[i].append(self.models['realnvp'].log_prob(x, labels))

                loglike[i] = torch.cat(loglike[i], dim=0)   # cat along data dim under this label
            loglike = torch.stack(loglike, dim=1)           # cat all data along label dim

            # log p(x) = log ∑_y p(x,y) = log ∑_y p(x|y)p(y)
            # assume uniform prior      = log p(y) ∑_y p(x|y) = log p(y) + log ∑_y p(x|y)
            logprobs = logprior + loglike.logsumexp(dim=1)
            # TODO -- measure accuracy as argmax of the loglike

        # unconditional model
        else:
            logprobs = []
            for x, y in self.val_loader:
                x = self.models['deq'](x)
                x = x.view(x.shape[0], -1).to(self.device)
                logprobs.append(self.models['realnvp'].log_prob(x))
            logprobs = torch.cat(logprobs, dim=0).to(self.device)

        logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len(self.val_loader.dataset))
        is_best = False
        if hvd.rank() == 0:
            if logprob_mean.item() >= self.records['best_logprob']:
                is_best = True
                self.records['best_logprob'] = logprob_mean.item()
            output = 'Evaluate (epoch {}) -- '.format(epoch) + 'logp(x) = {:.3f} +/- {:.3f}'.format(logprob_mean, logprob_std)
            print(output)
            if self.logger is not None:
                self.logger.info(output)

            points = x_fake.data.cpu().numpy()
            plt.title('Epoch {0}'.format(epoch))
            plt.scatter(points[:,0], points[:,1], s=2.0)
            plt.xlim((-10, 10))
            plt.ylim((-10, 10))
            plt.show()
            plt.savefig(filename)
            plt.close()
        return is_best

class Trainer_hvd(mt.HvdTrainer, Confs):
    def __init__(self):
        Confs.__init__(self)
        mt.HvdTrainer.__init__(self, self)

        self.models['realnvp'] = mt.inn.RealNVP(
            self.n_blocks, self.input_size, 
            self.hidden_size, self.n_hidden,
            self.cond_label_size, batch_norm=self.is_batch_norm).to(self.device)
        self.models['deq'] = mt.inn.DequantizeLayer().to(self.device)

        self.optims['realnvp'] = torch.optim.Adam(
            self.models['realnvp'].parameters(), 
            lr=1e-6, betas=(0.9, 0.99), weight_decay=1e-6)
        
        self.datasets['train'] = mt.datasets.MOG(dataset_size=25000)
        self.datasets['test'] = mt.datasets.MOG(dataset_size=2500)

        self.hvd_preprocessing(op=hvd.Adasum)

        self.z_rand = torch.randn(10000, self.input_size, device=self.device)

        self.records['best_logprob'] = float('-inf')

    def hvd_param_scaling(self):
        if hvd.nccl_built():
            # self.batch_size = int(self.batch_size/hvd.local_size())
            self.lr_scaler = 1,0

    def train(self, epoch):
        self.models['realnvp'].train()
        self.models['deq'].train()
        for i, x in enumerate(self.dataloaders['train']):
            x = x.to(self.device)
            x = x.view(x.shape[0], -1)
            loss = - self.models['realnvp'].log_prob(
                x, y if self.cond_label_size else None).mean(0)
            self.optims['realnvp'].zero_grad()
            loss.backward()
            self.optims['realnvp'].step()
            if i % 100 == 0:
                self.logs['Train Loss'] = loss.item()
                self.print_logs(epoch, i)

        return loss.item()

    @torch.no_grad()            
    def eval(self, epoch):
        self.models['realnvp'].eval()

        # conditional model
        if self.cond_label_size is not None:
            logprior = torch.tensor(1 / self.cond_label_size).log().to(self.device)
            loglike = [[] for _ in range(self.cond_label_size)]

            for i in range(self.cond_label_size):
                # make one-hot labels
                labels = torch.zeros(self.batch_size, self.cond_label_size).to(self.device)
                labels[:,i] = 1

                for x in self.dataloaders['test']:
                    x = x.view(x.shape[0], -1).to(self.device)
                    loglike[i].append(self.models['realnvp'].log_prob(x, labels))

                loglike[i] = torch.cat(loglike[i], dim=0)   # cat along data dim under this label
            loglike = torch.stack(loglike, dim=1)           # cat all data along label dim

            # log p(x) = log ∑_y p(x,y) = log ∑_y p(x|y)p(y)
            # assume uniform prior      = log p(y) ∑_y p(x|y) = log p(y) + log ∑_y p(x|y)
            logprobs = logprior + loglike.logsumexp(dim=1)
            # TODO -- measure accuracy as argmax of the loglike

        # unconditional model
        else:
            logprobs = []
            for x in self.dataloaders['test']:
                x = x.view(x.shape[0], -1).to(self.device)
                logprobs.append(self.models['realnvp'].log_prob(x))
            logprobs = torch.cat(logprobs, dim=0).to(self.device)

        logprob_mean, logprob_std = logprobs.mean(0), 2 * logprobs.var(0).sqrt() / math.sqrt(len(self.samplers['test']))
        is_best = False
        x = self.datasets['test'].data.to(self.device)
        x += 0.1 * torch.randn(*x.shape, device=self.device)
        z_real = self.models['realnvp'](x)[0]
        x_fake = self.models['realnvp'].inverse(self.z_rand)[0]
        x_real = self.models['realnvp'].inverse(z_real)[0]
        
        if hvd.rank() == 0:
            if logprob_mean.item() >= self.records['best_logprob']:
                is_best = True
                self.records['best_logprob'] = logprob_mean.item()
            output = 'Evaluate (epoch {}) -- '.format(epoch) + 'logp(x) = {:.3f} +/- {:.3f}'.format(logprob_mean, logprob_std)
        
            print(output)
            if self.logger is not None:
                self.logger.info(output)

            filename = os.path.join(self.plot_path, 'real_nvp_x_rand.png')
            points = x_fake.data.cpu().numpy()
            plt.title('Epoch {0}'.format(epoch))
            plt.scatter(points[:,0], points[:,1], s=2.0)
            plt.xlim((-10, 10))
            plt.ylim((-10, 10))
            plt.show()
            plt.savefig(filename)
            plt.close()

            filename = os.path.join(self.plot_path, 'real_nvp_x_real.png')
            points = x_real.data.cpu().numpy()
            plt.title('Epoch {0}'.format(epoch))
            plt.scatter(points[:,0], points[:,1], s=2.0)
            plt.xlim((-10, 10))
            plt.ylim((-10, 10))
            plt.show()
            plt.savefig(filename)
            plt.close()
        return is_best

if __name__ == '__main__':
    trainer = Trainer_hvd()
    trainer.run(load_best=False, retrain=True)
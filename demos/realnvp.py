import sys
sys.path.append('..')
import mytoolbox as mt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Confs(mt.BaseConfs):
    def __init__(self):
        super().__init__()
    
    def get_dataset(self):
        self.dataset = 'mnist'
        self.nc = 1
        self.img_size = 28
        self.input_size = self.img_size * self.img_size * self.nc

    def get_flag(self):
        self.n_blocks = 5
        self.n_components = 1
        self.hidden_size = 100
        self.n_hidden = 1
        self.cond_label_size = None
        self.is_batch_norm = False
        self.flag = 'demo-{}-realnvp'.format(self.dataset)


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

        self.models['realnvp'] = mt.inn.RealNVP(
            self.n_blocks, self.input_size, 
            self.hidden_size, self.n_hidden,
            self.cond_label_size, batch_norm=self.is_batch_norm).to(self.device)
        self.models['deq'] = mt.inn.DequantizeLayer().to(self.device)

        self.optims['realnvp'] = torch.optim.Adam(
            self.models['realnvp'].parameters(), 
            lr=1e-6, betas=(0.9, 0.99), weight_decay=1e-6)
        
        self.train_loader, self.val_loader, self.test_loader, _ = \
            mt.datasets.load_data(self.dataset, 1.0, 0.8, self.batch_size, self.img_size, None, False)

        self.records['best_logprob'] = float('-inf')

    def train(self, epoch):
        self.models['realnvp'].train()
        self.models['deq'].train()
        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
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
        self.models['deq'].eval()

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
        if logprob_mean.item() >= self.records['best_logprob']:
            is_best = True
            self.records['best_logprob'] = logprob_mean.item()
        output = 'Evaluate (epoch {}) -- '.format(epoch) + 'logp(x) = {:.3f} +/- {:.3f}'.format(logprob_mean, logprob_std)
        print(output)
        if self.logger is not None:
            self.logger.info(output)
        return is_best


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run(load_best=False, retrain=False)
import sys
sys.path.append('..')
import marveltoolbox as mt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import horovod.torch as hvd


class Confs(mt.BaseConfs):
    def __init__(self):
        super().__init__()
    
    def get_dataset(self):
        self.dataset = 'mnist'
        self.nc = 1
        self.nz = 10

    def get_flag(self):
        self.flag = 'demo-{}-clf'.format(self.dataset)

    def get_device(self):
        self.device_ids = [0]
        self.ngpu = len(self.device_ids)
        self.device = torch.device(
            "cuda:{}".format(self.device_ids[0]) if \
            (torch.cuda.is_available() and self.ngpu > 0) else "cpu")


class Trainer(mt.BaseTrainer):
    def __init__(self, confs):
        super().__init__(confs)
        self.models['C'] = mt.nn.dcgan.Enet32(confs.nc, confs.nz).to(self.device)
        self.optims['C'] = torch.optim.Adam(
            self.models['C'].parameters(), lr=1e-4, betas=(0.5, 0.99))
        
        self.train_loader, self.val_loader, self.test_loader, _ = \
            mt.datasets.load_data(confs.dataset, 1.0, 0.8, self.batch_size, 32, None, False)

        self.records['acc'] = 0.0

    def train(self, epoch):
        self.models['C'].train()
        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            scores = self.models['C'](x)
            loss = F.cross_entropy(scores, y)
            self.optims['C'].zero_grad()
            loss.backward()
            self.optims['C'].step()
            if i % 100 == 0:
                self.logs['Train Loss'] = loss.item()
                self.print_logs(epoch, i)

        return loss.item()
                
    def eval(self, epoch):
        self.models['C'].eval()
        correct = 0.0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                N = len(x)
                scores = self.models['C'](x)
                pred_y = torch.argmax(scores, dim=1)
                correct += torch.sum(pred_y == y).item()
        N = len(self.val_loader.dataset)
        acc = correct / N
        is_best = False
        if acc >= self.records['acc']:
            is_best = True
            self.records['acc'] = acc
        print('acc: {}'.format(acc))
        return is_best

class Trainer_hvd(mt.HvdTrainer):
    def __init__(self, confs):
        super().__init__(confs)

        self.models['C'] = mt.nn.dcgan.Enet32(confs.nc, confs.nz).to(self.device)
        self.optims['C'] = torch.optim.Adam(
            self.models['C'].parameters(), lr=1e-4 * self.lr_scaler, betas=(0.5, 0.99))
        
        train_loader, val_loader, test_loader, _ = \
            mt.datasets.load_data(confs.dataset, 1.0, 0.8, self.batch_size, 32, None, False)

        self.datasets['train'] = train_loader.dataset
        self.datasets['val'] = val_loader.dataset
        self.datasets['test'] = test_loader.dataset

        self.hvd_preprocessing(op=hvd.Adasum)
        self.records['acc'] = 0.0

    def train(self, epoch):
        self.logs = {}
        self.models['C'].train()
        for i, (x, y) in enumerate(self.dataloaders['train']):
            x, y = x.to(self.device), y.to(self.device)
            scores = self.models['C'](x)
            loss = F.cross_entropy(scores, y)
            self.optims['C'].zero_grad()
            loss.backward()
            self.optims['C'].step()
            
            if i % 100 == 0:
                self.logs['Train Loss'] = loss.item()
                self.print_logs(epoch, i)

        return self.metric_average(loss.item(), 'train_loss')
                
    def eval(self, epoch):
        eval_dataset = 'val'
        self.logs = {}
        self.models['C'].eval()
        correct = 0.0
        test_loss = 0.0
        with torch.no_grad():
            for x, y in self.dataloaders[eval_dataset]:
                x, y = x.to(self.device), y.to(self.device)
                N = len(x)
                scores = self.models['C'](x)
                test_loss += F.cross_entropy(scores, y, reduction='sum').item()
                pred_y = torch.argmax(scores, dim=1)
                correct += torch.sum(pred_y == y).item()

        acc = correct / len(self.samplers[eval_dataset])
        test_loss = test_loss/ len(self.samplers[eval_dataset])
        is_best = False
        acc = self.metric_average(acc, 'acc')
        test_loss = self.metric_average(test_loss, 'test_loss')
        if hvd.rank() == 0:    
            if acc >= self.records['acc']:
                is_best = True
                self.logs['Test Loss'] = test_loss
                self.logs['acc'] = acc
                self.print_logs(epoch, 0)
        return is_best


if __name__ == '__main__':
    confs = Confs()
    trainer = Trainer_hvd(confs)
    trainer.run(load_best=False, retrain=True)


    


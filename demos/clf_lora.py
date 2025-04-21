import marveltoolbox as mt 
import torch
import torch.nn as nn
import torch.nn.functional as F



class Confs(mt.BaseConfs):
    def __init__(self):
        super().__init__()
    
    def get_dataset(self):
        self.dataset = 'mnist'
        self.nc = 1
        self.nz = 10
        self.batch_size = 128
        self.epochs = 50
        self.seed = 0

    def get_flag(self):
        self.flag = 'demo-{}-clf-lora'.format(self.dataset)

    def get_device(self):
        self.device_ids = [0]
        self.ngpu = len(self.device_ids)
        self.device = torch.device(
            "cuda:{}".format(self.device_ids[0]) if \
            (torch.cuda.is_available() and self.ngpu > 0) else "mps")


class Trainer(mt.BaseTrainer, Confs):
    def __init__(self):
        Confs.__init__(self)
        mt.BaseTrainer.__init__(self, self)
        
        self.models['C'] = mt.utils.inject_lora(mt.nn.dcgan.Enet32(self.nc, self.nz), r=2, alpha=1).to(self.device)
        self.optims['C'] = torch.optim.Adam(
            mt.utils.get_lora_parameters(self.models['C']), lr=1e-4, betas=(0.5, 0.99))
        
        self.train_loader, self.val_loader, self.test_loader, _ = \
            mt.datasets.load_data(self.dataset, 1.0, 0.8, self.batch_size, 32, None, False)

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


if __name__ == '__main__':
    trainer = Trainer()
    trainer.run(load_best=False, retrain=True)


    


# marveltoolbox
A marvelous toolbox for DL research!

## Features
- [x] Complex value matrix computition APIs；
- [x] Convenient base trainer and base experment settings；
- [x] Providing commonly used neural network structures (MLP, CNNs, DCGAN).
- [x] Adversarial attacks (CW, adaptive CW, PGD and adaptive PGD)
- [x] Signal processing tools.


# Quick Installation Instructions
- Clone the git repository
```bash
$ git clone https://github.com/xrj-com/marvaltoolbox.git
```
- Navigate to the top level marveltoolbox directory
- Install marveltoolbox
```bash
$ pip install .
```

# Quick Start Instructions
```python
import marveltoolbox as mt
```
- Setting your experiment configs base on **mt.BaseConfs**:
```python
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
```

- Defining your **Trainer** base on mt.BaseTrainer. Using predefined dicts: *models*, *optims*, *schedulers* eta. to preserve your neural networks and optimization settings:
```python
class Trainer(mt.BaseTrainer, Confs):
    def __init__(self, confs):
        Confs.__init__(self)
        mt.BaseTrainer.__init__(self, self)
        self.models['C'] = mt.nn.dcgan.Enet32(confs.nc, confs.nz).to(self.device)
        self.optims['C'] = torch.optim.Adam(
            self.models['C'].parameters(), lr=1e-4, betas=(0.5, 0.99))
        
        self.train_loader, self.val_loader, self.test_loader, _ = \
            mt.datasets.load_data(confs.dataset, 1.0, 0.8, self.batch_size, 32, None, False)
```
- Predefined methods: *train*, *eval*, *main* need to be implemented according to your own needs. For example, if we want to train a classifier, the Trainer can be defined as follow:
```python
class Trainer(mt.BaseTrainer):
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
```

- Training model via:
```python
my_trainer = Trainer()
my_trainer.run(load_best=True, retrain=False)
```
- The model and the optimizer will automatically be saved as 'checkpoint_[your flag].pth.tar' each epoch.

Full code can be found in 'demos/clf.py' .



<!-- ### Training model

Ordinary training：
```bash
cd <yourpath>/marveltoolbox/demos
python clf.py
```


Distributed training via horovod(only for trainers based on mt.HvdTrainer):
```bash
cd <yourpath>/marveltoolbox/demos
horovodrun -np 4 python clf.py
```
Where '-np' is the number of process.  -->

# Citation
If you found this code useful plase cite our work
```
@Electronic{
  Xie2019a,
  author  = {Xie, Renjie and Xu, Wei},
  title   = {{MarvelToolbox}},
  url     = {https://github.com/xrj-com/marveltoolbox},
  year    = {2020}
}
```

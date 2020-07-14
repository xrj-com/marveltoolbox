import sys
sys.path.append('..')
import marveltoolbox as mt
import torch

if __name__ == '__main__':
    device = torch.device('cuda:0')
    net = mt.nn.dcgan.Enet32(1, 10).to(device)
    _, val_loader, _, _ = mt.datasets.load_mnist(0.1, 0.5, 50, 32)
    x, y = next(iter(val_loader))
    x, y = x.to(device), y.to(device)
    attack = mt.attacks.PGDAttack()

    adv, _ = attack.attack_batch(net, x, y)
    print(adv)

import sys
sys.path.append('..')
import marvaltoolbox as mt
import torch

if __name__ == '__main__':
    test_net = mt.nn.wgan.Dnet32(2)
    print(test_net)
    test_net = mt.nn.wgan.Enet32(2, 2)
    print(test_net)

    test_net = mt.nn.dcgan.Dnet32(2)
    print(test_net)
    test_net = mt.nn.dcgan.Enet32(2, 2)
    print(test_net)

    test_net = mt.nn.NormalizedModel(torch.Tensor([0]), torch.Tensor([1]))
    print(test_net)

    test_net = mt.nn.ArcMarginProduct(2, 4)
    print(test_net)

    test_net = mt.nn.ArcMarginProductPlus(2, 4)
    print(test_net)

    test_net = mt.nn.MobileFaceNet(embedding_size=512)
    print(test_net)


import torch
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import matplotlib.pyplot as plt
from torch.distributions import MultivariateNormal


def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def get_jacobian(fun, x, noutputs):
    x_size = list(x.size())
    x = x.unsqueeze(0).repeat(noutputs, *([1]*(len(x_size))) ).detach().requires_grad_(True)
    y = fun(x)
    y.backward(torch.eye(noutputs))
    return x.grad.view(noutputs,*x_size)

def Hessian_matrix(fun,x):
    #y is a scalar Tensor
    def get_grad(xx):
        y = fun(xx)
        grad, = torch.autograd.grad(y,xx,create_graph=True,grad_outputs=torch.ones_like(y))
        return grad
        
    x_size = x.numel()
    return get_jacobian(get_grad, x, x_size)

def analyze_latent_space(z, y, class_num=2):
    gaussians = []
    for c in range(class_num):
        idx = y.eq(c).nonzero().view(-1)
        n = len(idx)
        z_c = z[idx]
        mu_c = torch.mean(z_c, dim=0, keepdim=True)
        v_c = z_c-mu_c.repeat(n, 1)
        var_c = 1/n * v_c.transpose(0, 1).matmul(v_c)
        gaussian = MultivariateNormal(loc=mu_c.clone().detach(), covariance_matrix=var_c.clone().detach())
        gaussians.append(gaussian)
    return gaussians

def log_pz(z, y, gaussians, device):
    n, c = len(z), len(gaussians)
    V = torch.zeros(n, c, device=device)
    mask = F.one_hot(y, num_classes=c).type_as(z).to(device)
    for i in range(c): 
        V[:, i] = gaussians[i].log_prob(z)
    A = V*mask
    A = torch.sum(A, dim=1)
    return A.view(n, -1)

def sample(n, netD, gaussians):    
    samples = []
    K = len(gaussians)
    for i in range(n):
        idx = np.random.choice(np.arange(K))
        samples.append(gaussians[idx].sample())
    sample_z = torch.cat(samples, dim=0).to(device)
    sample_x = netD(sample_z)
    return sample_x


def one_hot_encode(labels, n_labels):
    """
    Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    """

    assert np.min(labels) >= 0 and np.max(labels) < n_labels

    y = np.zeros([labels.size, n_labels])
    y[range(labels.size), labels] = 1

    return y

def logit(x):
    """
    Elementwise logit (inverse logistic sigmoid).
    :param x: numpy array
    :return: numpy array
    """
    return np.log(x / (1.0 - x))
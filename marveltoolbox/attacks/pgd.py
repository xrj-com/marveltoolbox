import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .base import Attack

class PGDAttack(Attack):
    def __init__(self, params={'eps':0.3, 'is_target':False}):
        super().__init__()
        self.step_size = params.get('step_size', 0.01)
        self.k = params.get('k', 40)
        self.eps = params.get('eps', 0.3)
        self.is_target = params.get('is_target', False)
        self.rand = params.get('random_start', True)
        self.lower_bound = params.get('lower_bound', 0.0)
        self.upper_bound = params.get('upper_bound', 1.0)
       
    def get_normed_randinit(self, inputs):
        rand_init = torch.zeros_like(
                inputs.data, device=inputs.device).uniform_(-self.eps, self.eps)
        return rand_init

    def get_loss(self, net, inputs, labels):
        scores = net(inputs)
        loss = F.cross_entropy(scores, labels)
        return loss

    def get_normed_grad(self, inputs):
        return torch.sign(inputs.grad.data)

    def get_normed_clip(self, diff):
        return torch.clamp(diff, min=-self.eps, max=self.eps)

    def PGD_step(self, net, inputs, labels):
        inputs = inputs.clone().detach().requires_grad_()
        net.zero_grad()
        loss = self.get_loss(net, inputs, labels)
        loss.backward()
        grad_normed = self.get_normed_grad(inputs)
        if not self.is_target:
            inputs.data += self.step_size * grad_normed    
        else:
            inputs.data -= self.step_size * grad_normed  
        return inputs

    def attack_batch(self, net, inputs, labels):
        net.eval()
        adv_inputs = inputs.clone().detach().requires_grad_()
        if self.rand:
            adv_inputs.data += self.get_normed_randinit(adv_inputs)
        for _ in range(self.k):
            # calculate changes to x
            adv_inputs = self.PGD_step(net, adv_inputs, labels)
            # if out of the eps-ball, clipped it back
            diff = adv_inputs.data - inputs.data
            diff = self.get_normed_clip(diff)
            adv_inputs.data = torch.clamp(inputs.data + diff, min=self.lower_bound, max=self.upper_bound)
        return adv_inputs, labels
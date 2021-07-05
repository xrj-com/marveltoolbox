import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .base import Attack

class CWAttack(Attack):
    def __init__(self, params={'is_target':False}):
        super().__init__()
        self.T = params.get('T', 1000)
        self.is_target = params.get('is_target', False)
        self.TOLERANCE = params.get('tolerance', 0.05)
        self.CONFIDENCE = params.get('confidence', 0)
        self.LEARNING_RATE = params.get('lr', 1e-2)
        self.INITIAL_BETA = params.get('beta', 1e-0)
        self.MAX_BINARY = params.get('max_bs', 30)
        self.box_max = params.get('box_max', 1.0)
        self.box_min = params.get('box_min', 0.0)
        self.print = True
        self.logs = {}

    def CW_loss(self, scores, labels):
        N, D = scores.shape
        label_mask = F.one_hot(labels, D).type_as(scores)
        correct_logit = torch.sum(label_mask * scores, dim=1)
        n_label = torch.argmax(1-label_mask, dim=1)
        wrong_logit = scores[range(N), n_label]
        loss = F.relu(correct_logit - wrong_logit + self.CONFIDENCE)
        zeros = torch.zeros_like(loss, device=scores.device).detach()
        ones = torch.ones_like(loss, device=scores.device).detach()
        return loss, torch.where(wrong_logit>correct_logit, ones, zeros)

    def D_loss(self, inputs, adv_inputs):
        x = inputs.view(len(inputs), -1)
        x_adv = adv_inputs.view(len(adv_inputs), -1)
        D = torch.norm(x - x_adv, p=2, dim=1)
        return D, D

    def anti_reparameterize(self, x):
        xx = x.view(len(x), -1)
        a = (self.box_max - self.box_min) / 2.0
        b = (self.box_min + self.box_max) / 2.0
        v = torch.atan((xx - b)/a*0.999999)
        return v.view(x.size())

    def reparameterize(self, v):
        vv = v.view(len(v), -1)
        a = (self.box_max - self.box_min) / 2.0
        b = (self.box_min + self.box_max) / 2.0
        x = torch.tanh(v)*a + b
        return x.view(v.size())

    def attack_batch(self, net, inputs, labels):
        x =  inputs.clone().detach()
        v = self.anti_reparameterize(x).clone().detach().to(x.device)
        delta_v = torch.zeros_like(x, device=x.device, requires_grad=True)
        beta = torch.zeros(len(x)).to(x.device) + self.INITIAL_BETA
        lower_beta, upper_beta = torch.zeros(len(x)).to(x.device), torch.zeros(len(x)).to(x.device)+1e9  
        best_adv_x = torch.zeros_like(x).detach().to(x.device)
        best_adv_norm = torch.zeros(len(x)).detach().to(x.device)+1e4
        mean_norm, prev_mean_norm = 1e5, 1e4
        acc = 0


        while abs(prev_mean_norm - mean_norm) >= self.TOLERANCE or acc <= 0.99:
            # optimize to find x-adv
            optimizer = torch.optim.Adam([{'params': delta_v}], lr=self.LEARNING_RATE)
            for t in range(self.T):
                optimizer.zero_grad()
                adv_x = self.reparameterize(v + delta_v)
                scores = net(adv_x)
                loss1, D = self.D_loss(x, adv_x)
                loss2, success_flag = self.CW_loss(scores, labels)
                loss = torch.mean(loss1 + beta*loss2)
                if self.is_target:
                    loss = -loss
                loss.backward()
                optimizer.step()

            # record the best attack so far
            idx = ((success_flag==1).view(-1)*(D<best_adv_norm).view(-1)).nonzero().view(-1)
            best_adv_x[idx] = adv_x[idx].clone()
            best_adv_norm[idx] = D[idx].clone()
            prev_mean_norm = mean_norm
            mean_norm = torch.mean(best_adv_norm).item()
            acc = success_flag.sum().item()/len(x)
            
            # change beta
            new_beta_when_success = (beta + lower_beta)/2
            new_beta_when_failure = beta*10
            upper_beta = torch.where(success_flag==1, beta, upper_beta)
            lower_beta = torch.where(success_flag==0, beta, lower_beta)
            beta = torch.where(success_flag==1, new_beta_when_success, new_beta_when_failure)

            # print
            if self.print:
                print('step:', ' success_rate=', acc, 'mean distortion=', mean_norm)
        print('batch complete.', ' success_rate=', acc, 'mean distortion=', mean_norm, '\n')
        return best_adv_x, labels
    
    def print_logs(self, step):
        print_str = 'Iter:{:0>4d} '.format(step)
        for key, value in self.logs.items():
            print_str += '{}:{:4f} '.format(key, value)
        print(print_str)
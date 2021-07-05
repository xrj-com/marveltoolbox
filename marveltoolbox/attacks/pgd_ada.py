import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .pgd import PGDAttack

class AdaPGDAttack(PGDAttack):
    def __init__(self, params={'eps':0.8, 'is_target':False, 'is_debug':False}):
        super().__init__(params)
        self.is_debug = params.get('is_debug', False)
        self.step_size = self.eps * 0.4/0.3/self.k

    def get_loss(self, net, inputs, labels):
        criterion = nn.CrossEntropyLoss()
        scores, loss_list, flag_list = net(inputs, is_return_more=True)
        loss = criterion(scores, labels)
        # print(loss)
        N = len(inputs)
        class_num = scores.shape[1]
        label_mask = F.one_hot(labels, class_num).type_as(scores)
        correct_logit = torch.sum(label_mask * scores, dim=1)
        n_label = torch.argmax(1-label_mask, dim=1)
        wrong_logit = scores[range(N), n_label]
        zeros = torch.zeros_like(loss, device=scores.device).detach()
        ones = torch.ones_like(loss, device=scores.device).detach()
        loss_mask = torch.where(wrong_logit<correct_logit, ones, zeros)

        # print('mask:', loss_mask)
        if self.is_debug:
            print('loss:', loss)
        for i in range(len(loss_list)):
            loss = loss * loss_mask - loss_list[i]
            if self.is_debug:
                print('cond{}:'.format(i+1), loss_list[i], flag_list[i])
        if self.is_debug:
            print('all loss:', loss)
            print('is attack scusses:', 1-loss_mask)
        return loss.mean()
    
    def eval_attack_acc(self, defense, dataloader, device):
        defense.eval()
        correct = 0.0
        adv_success = 0.0
        i = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            adv_x, _ = self.attack_batch(defense, x, y) 
            scores, loss, flag_list = defense(adv_x.detach(), y, is_return_more=True)
            pred_y = torch.argmax(scores, dim=1)
            flag = torch.prod(torch.stack(flag_list, dim=0), dim=0)
            correct += torch.sum((pred_y == y)).item()
            adv_success += torch.sum((pred_y != y)*flag).item()
            i += 1
        # print(correct/i/64.0)
        acc = correct/len(dataloader.dataset)
        attack_acc = adv_success/len(dataloader.dataset)
        print('clean acc: {}'.format(acc))
        print('attack success rate: {}'.format(attack_acc))
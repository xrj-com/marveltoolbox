import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .cw import CWAttack

class AdaCWAttack(CWAttack):
    def __init__(self, params={'is_target':False, 'turn_on_cwloss':True}):
        super().__init__(params)
        self.is_cwloss = params.get('turn_on_cwloss', True)
    
    def attack_batch(self, defense, inputs, labels):
        x =  inputs.clone().detach()
        v = self.anti_reparameterize(x).clone().detach().to(x.device)
        delta_v = torch.zeros_like(v, device=x.device, requires_grad=True)
        
        loss_num = defense.cond_num
        cond_names = defense.cond_names

        beta_list = []
        lower_beta_list = []
        upper_beta_list = []
        for i in range(loss_num):
            beta_list.append(torch.zeros(len(x)).to(x.device) + self.INITIAL_BETA)
            lower_beta_list.append(torch.zeros(len(x)).to(x.device))
            upper_beta_list.append(torch.zeros(len(x)).to(x.device)+1e9)

        beta = torch.zeros(len(x)).to(x.device) + self.INITIAL_BETA
        lower_beta, upper_beta = torch.zeros(len(x)).to(x.device), torch.zeros(len(x)).to(x.device)+1e9  
        best_adv_x = torch.zeros_like(x).detach().to(x.device)
        best_adv_norm = torch.zeros(len(x)).detach().to(x.device)+1e4
        mean_norm, prev_mean_norm = 1e5, 1e4
        acc = 0

        binary_idx = 0
        while (abs(prev_mean_norm - mean_norm) >= self.TOLERANCE or acc <= 0.99) and (binary_idx < self.MAX_BINARY):
            # optimize to find x-adv
            optimizer = torch.optim.Adam([{'params': delta_v}], lr=self.LEARNING_RATE)
            for t in range(self.T):
                optimizer.zero_grad()
                adv_x = self.reparameterize(v + delta_v)

                scores, loss_list, success_flag_list  = defense(adv_x, is_return_more=True)

                conf_flag = False
                loss, D = self.D_loss(x, adv_x)
                success_flag = (loss == loss).float().detach()
                if self.is_cwloss:
                    loss2, success_flag_cw = self.CW_loss(scores, labels)
                    loss = loss + beta*loss2
                    success_flag = success_flag * success_flag_cw
                
                if success_flag.mean().item() > 0:
                    conf_flag = True
                else:
                    conf_flag = False

                if conf_flag:
                    for i in range(loss_num):
                        loss = loss + beta_list[i] * loss_list[i]
                        success_flag = success_flag * success_flag_list[i]
                    
                loss = torch.mean(loss)
                
                if self.is_target:
                    loss = -loss
                loss.backward()
                if t % 100==0:
                    self.logs['distort'] = D[0].item()
                    self.logs['cw'] = loss2[0].item()
                    for i in range(len(cond_names)):
                        if defense.turn_on[cond_names[i]]:
                            self.logs[cond_names[i]] = loss_list[i][0].item()
                    self.logs['loss'] = loss.item()
                    self.print_logs(t)

                optimizer.step()

            # record the best attack so far
            idx = ((success_flag==1).view(-1)*(D<best_adv_norm).view(-1)).nonzero().view(-1)
            best_adv_x[idx] = adv_x[idx].clone()
            best_adv_norm[idx] = D[idx].clone()
            prev_mean_norm = mean_norm
            mean_norm = torch.mean(best_adv_norm).item()
            acc = success_flag.sum().item()/len(x)
            
            # change beta
            if self.is_cwloss:
                new_beta_when_success = (beta + lower_beta)/2
                new_beta_when_failure = beta*10
                upper_beta = torch.where(success_flag_cw==1, beta, upper_beta)
                lower_beta = torch.where(success_flag_cw==0, beta, lower_beta)
                beta = torch.where(success_flag_cw==1, new_beta_when_success, new_beta_when_failure)

            if conf_flag:
                for i in range(loss_num):
                    new_beta_when_success = (beta_list[i] + lower_beta_list[i])/2
                    new_beta_when_failure = beta_list[i]*10
                    upper_beta_list[i] = torch.where(success_flag_list[i]==1, beta_list[i], upper_beta_list[i])
                    lower_beta_list[i] = torch.where(success_flag_list[i]==0, beta_list[i], lower_beta_list[i])
                    beta_list[i] = torch.where(success_flag_list[i]==1, new_beta_when_success, new_beta_when_failure)

            # print
            if self.print:
                print('step:', ' success_rate=', acc, 'mean distortion=', mean_norm)
            binary_idx += 1
            
        print('batch complete.', ' success_rate=', acc, 'mean distortion=', mean_norm, '\n')
        return best_adv_x, labels
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .cw import CWAttack

class ReparamCWAttack(CWAttack):
    def __init__(self, params={'p': 0.999, 'confidence': 50, 'threshold': 1e-4}, ):
        super().__init__(params)
        self.p = params.get('p', 0.999)
        self.threshold = params.get('threshold', 9999)
    
    def D_loss(self, inputs, adv_inputs):
        x = inputs.view(len(inputs), -1)
        x_adv = adv_inputs.view(len(adv_inputs), -1)
        D = torch.norm(x - x_adv, p=2, dim=1)
        return D, D   
    
    def detective_loss(self, rec_x, adv_x):
        N = len(rec_x)
        D = torch.norm(rec_x.view(N, -1) - adv_x.view(N, -1), p=2, dim=1)
        loss = torch.where(D<self.threshold, torch.zeros_like(D), D*10)
        flag = torch.where(D>=self.threshold, torch.zeros_like(D), torch.ones_like(D)).detach()
        return loss, flag

    def z_loss(self, classifier, z, labels):
        N = len(z)
        label_mask = F.one_hot(labels, self.class_num).type_as(z)
        n_label = torch.argmax(1-label_mask, dim=1)
        
        scores = classifier(z)
        wrong_logit = scores[range(N), n_label]
        loss = F.relu(self.p-wrong_logit)
        flag = torch.where(wrong_logit<self.p, torch.zeros_like(wrong_logit), torch.ones_like(wrong_logit)).detach()
        return 100*loss, flag
    
    def attack_batch(self, defense, inputs, labels):
        x =  inputs.clone().detach()
        z = defense.encode(x).clone().detach().to(x.device)
        rec_x = defense.decode(z).detach()
        delta_z = torch.zeros_like(z, device=x.device, requires_grad=True)
        beta = torch.zeros(len(x)).to(x.device) + self.INITIAL_BETA
        lower_beta, upper_beta = torch.zeros(len(x)).to(x.device), torch.zeros(len(x)).to(x.device)+1e9  
        best_adv_x = torch.zeros_like(x).detach().to(x.device)
        best_adv_norm = torch.zeros(len(x)).detach().to(x.device)+1e4
        mean_norm, prev_mean_norm = 1e5, 1e4
        acc = 0

        while abs(prev_mean_norm - mean_norm) >= self.TOLERANCE or acc <= 0.99:
            # optimize to find x-adv
            optimizer = torch.optim.Adam([{'params': delta_z}], lr=self.LEARNING_RATE)
            for t in range(self.T):
                optimizer.zero_grad()
                new_z = z + delta_z
                new_z_normed = torch.renorm()
                adv_x = defense.decode(new_z)
                loss1, D = self.D_loss(x, adv_x)
                
                loss2, success_flag = self.CW_loss(defense.encoder, adv_x, labels)
                loss3, success_flag2 = self.detective_loss(rec_x, adv_x)
                loss4, success_flag3 = self.z_loss(defense.encoder.classifier, new_z, labels)
                
                success_flag = success_flag * success_flag2 * success_flag3
                loss = torch.mean(loss1 +  beta*loss2 + loss3 + loss4)
                if self.is_target:
                    loss = -loss
                loss.backward()
                if t % 100==0:
                    print(loss.item())
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
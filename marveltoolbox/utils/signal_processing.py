import numpy as np
import matplotlib.pyplot as plt
import torch

def fft_plot(x, samples=200, filename=None):
    N = len(x)
    if samples > N:
        samples = N
    f = np.arange(N)
    if isinstance(x, np.ndarray):
        fft_x = np.fft.fft(x)
        abs_x=np.abs(fft_x/N)                
        angle_x=np.angle(fft_x/N)          
        
        x_imag = x.imag
        x_real = x.real
    
    elif isinstance(x, torch.Tensor):
        if x.shape[-1] == 2:
            fft_x = torch.fft(x, signal_ndim=1, normalized=False)
            abs_x = TorchComplex.abs(fft_x/N).numpy()
            # angle_x = TorchComplex.phase(fft_x/N).numpy()
            angle_x = TorchComplex.phase_np(fft_x/N)
            x_imag = TorchComplex.imag(x).numpy()
            x_real = TorchComplex.real(x).numpy()
        else:
            x_real = x.numpy()
            x_imag = x.numpy() * 0
            angle_x = x.numpy() * 0
            abs_x = x.numpy() * 0
    else:
        print('Can not suppose this dtype!')

    plt.figure(figsize=(20, 8))
    ax1 = plt.subplot(3,1,1)
    ax1.plot(f[:samples],x_real[:samples], '*-')   
    ax1.set_title('time real')

    ax1 = plt.subplot(3,1,2)
    ax1.plot(f[:samples],x_imag[:samples], '*-')   
    ax1.set_title('time img')

    ax2 = plt.subplot(3,2,5)
    ax2.plot(f,abs_x)   
    ax2.set_title('freq')
    
    ax3 = plt.subplot(3,2,6)
    ax3.plot(f,angle_x)   
    ax3.set_title('phase')
    plt.show()
    if not filename is None:
        plt.savefig(filename)
        plt.close()


class TorchComplex:
    def __init__(self):
        pass

    @staticmethod
    def abs(tensor):
        shape = tensor.shape
        if len(shape)>1:
            tensor = tensor.view(-1, 2)
            tensor_abs = (tensor[:, 0]**2 + tensor[:, 1]**2)**0.5
            return tensor_abs.view(*shape[:-1])
        else:
            return (tensor[0]**2 + tensor[1]**2)**0.5

    @staticmethod
    def phase(tensor):
        shape = tensor.shape
        if len(shape)>1:
            tensor = tensor.view(-1, 2)
            return torch.atan(tensor[:, 1]/tensor[:, 0]).view(*shape[:-1])
        else:
            return torch.atan(tensor[1]/tensor[0])

    @staticmethod
    def phase_np(tensor):
        shape = tensor.shape
        if len(shape)>1:
            tensor = tensor.view(-1, 2)
            return np.angle(tensor[:, 0].numpy()+ tensor[:, 1].numpy()*1j)
        else:
            return np.angle(tensor[0].item()+ tensor[1].item()*1j)


    @staticmethod
    def energy(tensor, keep_batch=True):
        if not keep_batch:
            tensor = tensor.view(-1, 2)
            return torch.sum(tensor[:, 0]**2 + tensor[:, 1]**2)
        else:
            N = len(tensor)
            tensor = tensor.view(N, -1, 2)
            return torch.sum(tensor[:, :, 0]**2 + tensor[:, :, 1]**2, dim=1)

    @staticmethod
    def power(tensor):
        return torch.sqrt(torch.mean(tensor[:, 0]**2 + tensor[:, 1]**2))

    @staticmethod
    def imag(tensor):
        return tensor[:, 1]

    @staticmethod
    def real(tensor):
        return tensor[:, 0]

    @staticmethod
    def array2tensor(array):
        shape = array.shape
        real = torch.FloatTensor(array.real).view(-1, 1)
        imag = torch.FloatTensor(array.imag).view(-1, 1)
        tensor = torch.cat([real, imag], dim=1)
        return tensor.view(*shape, 2)

    @staticmethod
    def complex2tensor(complex_num):
        real = torch.FloatTensor([complex_num.real]).view(-1, 1)
        imag = torch.FloatTensor([complex_num.imag]).view(-1, 1)
        tensor = torch.cat([real, imag], dim=1)
        return tensor

    @staticmethod
    def tensor2array(tensor):
        shape = tensor.shape
        tensor = tensor.flatten().view(-1, 2)
        real = tensor[:, :1].numpy()
        imag = tensor[:, 1:].numpy()
        array = real + imag * 1j
        return np.reshape(array, shape[:-1])

    @staticmethod
    def real_array2tensor(array):
        real = torch.FloatTensor(array).view(-1, 1)
        imag = torch.FloatTensor(array).view(-1, 1) * 0.0
        tensor = torch.cat([real, imag], dim=1)
        return tensor

    @staticmethod
    def array_exp(array):
        real = torch.Tensor(array.real)
        imag = torch.Tensor(array.imag)
        exp_real = torch.exp(real)
        exp_imag_cos = torch.cos(imag) * exp_real
        exp_imag_sin = torch.sin(imag) * exp_real
        return torch.cat([exp_imag_cos, exp_imag_sin], dim=1)

    @staticmethod
    def element_inverse(tensor, eps=1e-12):
        '''
        input: tensor shape(*, 2)
        return: inv_tensor shape(*, 2): inverse of element in tensor
        '''
        shape = tensor.shape
        tensor = tensor.flatten().view(-1, 2)
        real = tensor[:, :1]
        imag = tensor[:, 1:]
        denominator = real**2 + imag**2 + eps
        oreal = real/denominator
        oimag = -imag/denominator
        inv_tensor = torch.cat([oreal, oimag], dim=1)
        return inv_tensor.view(*shape)
    
    @staticmethod
    def batch_diag(tensor):
        '''
        input: tensor shape(B, D, 2)
        return: out_tensor shape(B, D, D, 2): batch diagonal matrix by given diagonal elements
        '''
        B, D, _ = tensor.shape
        # out_tensor = torch.eye(D, device=tensor.device).view(1, D, D, 1).repeat(B, 1, 1, 2)
        out_tensor = torch.zeros(B, D, D, 2, device=tensor.device)
        idx = range(D)
        # out_tensor[:, idx, idx, :] = out_tensor[:, idx, idx, :] * tensor
        out_tensor[:, idx, idx, :] = out_tensor[:, idx, idx, :] + tensor
        return out_tensor


    @staticmethod
    def exp(tensor):
        shape = tensor.shape
        tensor = tensor.flatten().view(-1, 2)
        real = tensor[:, :1]
        imag = tensor[:, 1:]
        exp_real = torch.exp(real)
        exp_imag_cos = torch.cos(imag) * exp_real
        exp_imag_sin = torch.sin(imag) * exp_real
        return torch.cat([exp_imag_cos, exp_imag_sin], dim=1).view(*shape)

    @staticmethod
    def prod(tensor0, tensor1):
        shape = tensor0.shape
        tensor0 = tensor0.flatten().view(-1, 2)
        tensor1 = tensor1.flatten().view(-1, 2)
        result_real = tensor0[:, 0] * tensor1[:, 0] - tensor0[:, 1] * tensor1[:, 1]
        result_imag = tensor0[:, 1] * tensor1[:, 0] + tensor0[:, 0] * tensor1[:, 1]
        result = torch.cat([result_real.view(-1, 1), result_imag.view(-1, 1)], dim=1)
        return result.view(*shape)

    @staticmethod
    def mm(tensor0, tensor1):
        shape0 = tensor0.shape
        shape1 = tensor1.shape
        real0 = tensor0.flatten().view(-1, 2)[:, 0].view(*shape0[:-1])
        imag0 = tensor0.flatten().view(-1, 2)[:, 1].view(*shape0[:-1])
        real1 = tensor1.flatten().view(-1, 2)[:, 0].view(*shape1[:-1])
        imag1 = tensor1.flatten().view(-1, 2)[:, 1].view(*shape1[:-1])
        ac = torch.mm(real0, real1)
        bd = torch.mm(imag0, imag1)
        ad = torch.mm(real0, imag1)
        bc = torch.mm(imag0, real1)
        real = ac - bd
        imag = ad + bc
        result = torch.stack([real, imag], dim=-1)
        return result

    @staticmethod
    def bmm(tensor0, tensor1):
        shape0 = tensor0.shape
        shape1 = tensor1.shape
        real0 = tensor0.flatten().view(-1, 2)[:, 0].view(*shape0[:-1])
        imag0 = tensor0.flatten().view(-1, 2)[:, 1].view(*shape0[:-1])
        real1 = tensor1.flatten().view(-1, 2)[:, 0].view(*shape1[:-1])
        imag1 = tensor1.flatten().view(-1, 2)[:, 1].view(*shape1[:-1])
        ac = torch.bmm(real0, real1)
        bd = torch.bmm(imag0, imag1)
        ad = torch.bmm(real0, imag1)
        bc = torch.bmm(imag0, real1)
        real = ac - bd
        imag = ad + bc
        result = torch.stack([real, imag], dim=-1)
        return result

    @staticmethod
    def conj(tensor):
        shape = tensor.shape
        tensor_temp = tensor.flatten().view(-1, 2)  
        tensor_final = torch.cat([tensor_temp[:, :1],  -1 * tensor_temp[:, 1:]], dim=1)
        return tensor_final.view(*shape)

    @staticmethod
    def SNR(x, x_origin, keep_batch=False, eps=1e-12):
        px = TorchComplex.energy(x, keep_batch)
        pn = TorchComplex.energy(x-x_origin, keep_batch)
        return 10 * torch.log10(px/(pn + eps))

    @staticmethod
    def add_noise(x, noise, SNR):
        px = TorchComplex.energy(x)
        pn = TorchComplex.energy(noise)
        pr = px/(10 ** (SNR/10))
        noise_p = torch.sqrt(pr/pn) * noise
        return x + noise_p

    @staticmethod
    def awgn(x, SNR, keep_batch=True, SNR_x=None):
        '''
        x.shape : (N, D, 2)
        '''
        N, D, _ = x.shape
        px = TorchComplex.energy(x, keep_batch)/D
        if not SNR_x is None:
            rate = 10 ** (SNR_x/10)
            px = px/(1.0/rate+1.0)
        noise = torch.randn_like(x)
        pr = px/(10 ** (SNR/10))
        noise_p = torch.sqrt(pr/2) * noise
        return noise_p

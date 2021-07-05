import numpy as np
import matplotlib.pyplot as plt
import torch
from .torch_complex import TorchComplex

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




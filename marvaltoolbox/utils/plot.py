import os
import torch
import torchvision as tv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_tensor(tensor, nrow=8, normalize=True, figsize=(10,10), filename=None):
    plt.figure(figsize=figsize)
    grid = tv.utils.make_grid(tensor, nrow=nrow, normalize=normalize)
    img = tv.transforms.ToPILImage()(grid.cpu())
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    if not filename is None:
        plt.savefig(filename)

def plot_eps_acc(distort_dict, title='test', sample_num=80, begin=0.0, stop=4.0):
    figure = plt.figure()
    legend_list = []
    colours = ['blue', 'green', 'red', 'black', 'yellow', 'orange']
    xs = np.linspace(begin, stop, sample_num)
    for i, (name, distort_list) in enumerate(distort_dict.items()):
        legend_list.append(name)
        acc_list = []
        if type(distort_list) == list:
            distorts = np.array(distort_list)
        elif type(distort_list) == np.ndarray:
            distorts = distort_list
        else:
            print('the type of distort list should be np.ndarray or list!')
            pass
        for j in xs:
            acc = np.sum(distorts < j)/len(distorts)
            acc_list.append(acc)
        plt.plot(xs, acc_list, color=colours[i])
    
    plt.xlabel('eps')
    plt.ylabel('acc')
    plt.title(title)
    plt.legend(legend_list)

def plot_data_dict(data_dict, title='test', xlabel='x', ylabel='y', filename=None):
    figure = plt.figure()
    legend_list = []
    colours = ['gray', 'purple', 'blue', 'green', 'red', 'black', 'yellow', 'orange']
    line_type = ['-.', '--']
    
    for i, (name, data_list) in enumerate(data_dict.items()):
        legend_list.append(name)
        xs = [i+1 for i in range(len(data_list))]
        plt.plot(xs, data_list, line_type[i], color=colours[i])
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(legend_list)
    if not filename is None:
        plt.savefig(filename)
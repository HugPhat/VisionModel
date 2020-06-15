import numpy as np 


import torch
import torch.nn as nn 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection

class VOCseg(Dataset):
    def __init__(self, num_class, class_name):
        super(VOCseg, self).__init__()
        self.default_labels = ['background', 
                               'aeroplane', 
                               'bicycle', 
                               'bird', 
                               'boat', 
                               'bottle', 
                               'bus', 
                               'car', 
                               'cat', 
                               'chair', 
                               'cow', 
                               'diningtable', 
                               'dog', 
                               'horse', 
                               'motorbike', 
                               'person', 
                               'pottedplant', 
                               'sheep', 
                               'sofa', 
                               'train', 
                               'tvmonitor', 
                               'void']
        
    def color_map(self, N=256, normalized=False):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)
        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                #print(r,g,b, c)
                c = c >> 3
            #print(f'end {i} {np.array([r, g, b])}')
            cmap[i] = np.array([r, g, b])
        cmap = cmap/255 if normalized else cmap
        return cmap
    
    def __getitem__(self, index):
        return 

    
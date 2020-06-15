import numpy as np 
import os
from PIL import Image

import torch
import torch.nn as nn 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection

from SegUtils import *

class VOCseg(Dataset):
    def __init__(self, 
                    jpeg_src,
                    mask_src,
                    num_class = 21, 
                    class_name = None,
                    img_size = (224, 224)
                ):
        '''
        Pascal Voc Format for class segmentation
         + Images: Folder contains [jpg, JPG]
         + Labels: Folder contains [png, PNG]
        '''
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
        self.labels = self.default_labels
        self.mask_color = self.color_map()[:len(self.labels)+1]
        self.list_items = []
        
        for each in os.listdir(jpeg_src):
            t = each.split('.')[-1]
            if t in ['jpg', 'JPG']:
                self.list_items.append(t)
                
        #self.image_src = [os.path.join(jpeg_src, each) for each in list_jpeg]
        #self.image_mask = [os.jpeg_src]
        self.image_src = jpeg_src
        self.mask_src = mask_src
        self.img_size = img_size
        #self.labels.pop(0)
        self.labels.pop(-1)
        self.mask_color.pop(-1)
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
    
    def create_mask(self, mask):
        #lmask = np.all(mask == self.mask_color[0], axis=-1)
        lmask = []
        for each in self.mask_color:
            tmask = np.all(mask == each, axis=-1)
            lmask.append(tmask)
        return lmask
    
    def __getitem__(self, index):
        item = self.list_items[index]
        img = os.path.join(self.image_src, item + '.jpg')
        mask = os.path.join(self.mask_src, item + '.png')
        img = Image.open(img).convert('RGB')
        mask = Image.open(mask).convert('RGB')
        img = np.array(img)
        mask = np.array(mask)
        
        if rand():
            img = SegBlur(img)
        if rand():
            img = SegBrightness(img)
        if rand():
            img = SegHue(img)
        if rand():
            img = SegSaturation(img)
        if rand():
            x = random.random()
            img = SegScale(img, x)
            mask = SegScale(mask, x)
        if rand():
            img = SegFlip(img, 'v')
            mask = SegFlip(mask, 'v')
        if rand():
            img = SegFlip(img, 'h')
            mask = SegFlip(mask, 'h')
        
        
        #mask =  
        return 

    
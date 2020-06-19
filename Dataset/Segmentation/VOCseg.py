import numpy as np 
import os
import sys
from PIL import Image

import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection

sys.path.insert(0, os.path.join(os.path.dirname(__file__),  '..'))
sys.path.append(os.path.dirname(__file__))

from Utils import *

class VOCseg(Dataset):
    def __init__(self, 
                    jpeg_src,
                    mask_src,
                    num_class = 21, 
                    class_name = None,
                    img_size = (224, 224),
                    mode = 'train',
                    ratio = 0.8
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
        self.mask_color = list(self.color_map()[:len(self.labels ) -1])
        #self.mask_color.append(np.array([255.0, 255.0, 255.0]))
        self.list_items = []
        self.labels.pop(-1)
        
        for each in os.listdir(mask_src):
            name = each.split('.')[0]
            t = each.split('.')[-1]
            if t in ['png', 'PNG']:
                self.list_items.append(name) 
        if ratio > 1 or ratio < 0:
            raise ValueError('ratio must in [0,1]')
        if mode == 'val':
            tsplit = int(len(self.list_items)*(1 - ratio))
            self.list_items = self.list_items[tsplit:]
        elif mode == 'train':
            tsplit = int(len(self.list_items)*ratio)
            self.list_items = self.list_items[:tsplit]
        else:
            raise ValueError('mode = val or train')  

        self.image_src = jpeg_src
        self.mask_src = mask_src
        self.img_size = img_size

        
        #print(self.labels)
        self.preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
                            ])
        
        
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
        indice = 0
        #plt.imshow(mask)
        #plt.show()
        _mask = mask.copy()
        _mask[_mask == np.max(mask)] = 0
        for i, each in enumerate(self.mask_color):
            tmask = np.all(mask == each, axis=-1)
            indice += tmask*i
            lmask.append(tmask)
        return np.array(indice)
    
    
    def __getitem__(self, index):
        item = self.list_items[index]
        img = os.path.join(self.image_src, item + '.jpg')
        mask = os.path.join(self.mask_src, item + '.png')
        img = Image.open(img).convert('RGB')
        mask = Image.open(mask).convert('RGB')
        img = np.array(img)
        mask = np.array(mask)
        
        #if rand():
        #    angle = random.randint(-40, 40)
        #    img = Rotate(img, angle)
        #    mask = Rotate(mask, angle)
        if rand():
            img = Blur(img)
        if rand(0.1):
            img = Brightness(img)
        if rand(0.1):
            img = Hue(img)
        if rand(0.1):
            img = Saturation(img)
        if rand(0.6):
            x = random.random()
            if x > 0.8 and x < 1.2:
                img = Scale(img, x)
                mask = Scale(mask, x)
        if rand(0.9):
            img = Flip(img, 'v')
            mask = Flip(mask, 'v')
        if rand(0.7):
            img = Flip(img, 'h')
            mask = Flip(mask, 'h')
        if rand(0.7):    
            ratio = random.random()
            img = Crop(img, ratio)
            mask = Crop(mask, ratio)
        if rand(0.8):
            img = Gray(img)           
        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        
        mask = self.create_mask(mask)
        
        img = self.preprocess(img)
        mask = torch.from_numpy(mask)
 
        return img, mask

    def __len__(self):
        return len(self.list_items)

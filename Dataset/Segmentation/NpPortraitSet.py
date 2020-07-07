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

     
class NpPortraitSet(Dataset):
    def __init__(self, 
                    jpeg_src,
                    mask_src,
                    num_class = 21, 
                    class_name = None,
                    img_size = (224, 224),
                    mode = 'train',
                    ratio = 0.8,
                    center_crop = True
                ):
        '''
        '''
        super(NpPortraitSet, self).__init__()
        
        self.center_crop = center_crop
     
        #self.mask_color.append(np.array([255.0, 255.0, 255.0]))
        
        self.image_src = list(np.load(jpeg_src))
        self.mask_src = list(np.load(mask_src))
        
        self.label = {0: 'bg', 1: 'person'}
  

        if ratio > 1 or ratio < 0:
            raise ValueError('ratio must in [0,1]')
        if mode == 'val':
            tsplit = int(len(self.image_src)*(1 - ratio))
            self.image_src = self.image_src[tsplit:]
            self.mask_src = self.mask_src[tsplit:]
            
        elif mode == 'train':
            tsplit = int(len(self.image_src)*ratio)
            print(f'tsplit {tsplit}')
            self.image_src = self.image_src[:tsplit]
            self.mask_src = self.mask_src[:tsplit]
        else:
            raise ValueError('mode = val or train')  

        self.img_size = img_size
     
        #print(self.labels)
        self.preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
                            ])
        
    
    def create_mask(self, mask):
        #lmask = np.all(mask == self.mask_color[0], axis=-1)
        res = mask / np.max(mask)
        return res
    
    
    def __getitem__(self, index):
        print(index)
        img = self.image_src[index]
        mask = self.mask_src[index].squeeze(-1)
        
        if self.center_crop:
            img = CenterCrop(img, self.img_size)
            mask = CenterCrop(mask, self.img_size)        
        
        if rand():
            angle = random.randint(-40, 40)
            img = Rotate(img, angle)
            mask = Rotate(mask, angle)
 
        if rand():
            img = Blur(img)
        if rand(0.2):
            img = Brightness(img)
        if rand(0.1):
            img = Hue(img)
        if rand(0.1):
            img = Saturation(img)
        if rand(0.2):
            x = random.random()
            if x > 0.8 and x < 1.2:
                img = Scale(img, x)
                mask = Scale(mask, x)
        if rand(0.5):
            img = Flip(img, 'v')
            mask = Flip(mask, 'v')
        if rand(0.5):
            img = Flip(img, 'h')
            mask = Flip(mask, 'h')
        if rand(0.8):    
            ratio = random.random()
            img = Crop(img, ratio)
            mask = Crop(mask, ratio)
        if rand(0.9):
            img = Gray(img) 
                      
        img = cv2.resize(img, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        
        mask = self.create_mask(mask)

        img = self.preprocess((img))
        mask = torch.from_numpy(mask)
 
        return img, mask

    def __len__(self):

        return int(len(self.image_src))

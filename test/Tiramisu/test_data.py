import os 
import sys
import matplotlib.pyplot as plt 
import numpy as np 

import torch
import torch.nn as nn 
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from Dataset.Segmentation.NpPortraitSet import *
from Dataset.Segmentation.VOCseg import *

from Backbone.Dense.Tiramisu import *
'''
imgset = r"D:\Code\Dataset\18689-ph\18698-human-portrait\data\img_uint8.npy"
labelset = r"D:\Code\Dataset\18689-ph\18698-human-portrait\data\msk_uint8.npy"

train = torch.utils.data.DataLoader(
    NpPortraitSet(imgset, labelset, center_crop = False), batch_size=1, 
    shuffle=True, num_workers=0, drop_last = False
)
'''
imgset = r'D:\Code\Dataset\PASCAL-VOOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
labelset = r'D:\Code\Dataset\PASCAL-VOOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationClass'
train = torch.utils.data.DataLoader(
    VOCseg(imgset, labelset, center_crop = False, classes=['all']), batch_size=1, 
    shuffle=True, num_workers=0, drop_last = False
)


inv_normalize = transforms.Compose( [transforms.Normalize(
                            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                            std=[1/0.229, 1/0.224, 1/0.255]),
                            ])
for it, (imgs, targets) in enumerate(train):
    imgs = inv_normalize(imgs.squeeze(0)).permute(1,2,0).numpy()
    targets = targets.squeeze(0).numpy().astype('uint8')
    print(np.unique(targets))
    print(targets.shape)
    f, axarr = plt.subplots(ncols=2 , figsize=(10, 5))
    axarr[0].imshow(imgs)
    axarr[1].imshow(targets)
    plt.show()
    

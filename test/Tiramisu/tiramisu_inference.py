import os 
import sys

import cv2
import numpy as np

import torch
import torch.nn as nn 
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Dataset.Segmentation.VOCseg import VOCseg
from Backbone.Dense.Tiramisu103 import Tiramisu103

imgset = r'D:\Code\Dataset\PASCAL-VOOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
labelset = r'D:\Code\Dataset\PASCAL-VOOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationClass'

train = torch.utils.data.DataLoader(
    VOCseg(imgset, labelset), batch_size=1, 
    shuffle=True, num_workers=0, drop_last = False
)

model = Tiramisu103(init_weight = True, num_classes=21)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
loss = nn.BCEWithLogitsLoss()

Tensor = torch.FloatTensor 

#model = nn.DataParallel(model)
#model_chẹkpoints = torch.load('model/tiramisu.pth', map_location="cuda")
#print('loading Model')
#model.load_state_dict(model_chẹkpoints)

model.train()
for it, (imgs, targets) in enumerate(train):
    img = np.asarray(imgs.squeeze(0).permute(1,2,0).cpu())*255
    #img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2RGB).astype('float32')/255.0
    #plt.imshow(img)
    #plt.show()
    #imgs = Variable(imgs.type(Tensor)).cuda()
    #output = model(imgs)
    #
    #res = list(output.type(Tensor).cpu().squeeze().data)
    #
    #for i in range(21):
    #    print(res[i].shape)
    #    plt.imshow(res[i])
    #    plt.show()
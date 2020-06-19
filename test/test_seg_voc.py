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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Dataset.Segmentation.VOCseg import VOCseg
from Backbone.Dense.Tiramisu import *

imgset = r'D:\Code\Dataset\PASCAL-VOOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
labelset = r'D:\Code\Dataset\PASCAL-VOOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationClass'

train = torch.utils.data.DataLoader(
    VOCseg(imgset, labelset), batch_size=3, 
    shuffle=True, num_workers=0, drop_last = False
)

def dice_loss(preds, targets):
      #preds = torch.flatten(preds)
  #targets = torch.flatten(targets)
  #print((preds*targets))
  #print((preds*targets).sum())
  #print(preds.sum() + targets.sum())
  dice = ((preds*targets).sum()*2 +1e-6) / ((torch.square(preds) + torch.square(targets)).sum() + 1e-6)
  return  1 - torch.mean(dice)

def error(preds, targets):
    preds2label = torch.argmax(preds, dim=1)
    bs, w, h = targets.size()
    delta =  (torch.ne(torch.flatten(targets), torch.flatten(preds2label)))
    delta = delta.type(torch.FloatTensor)
    print(delta.sum()) 

    delta = delta.sum() / (bs*w*h)
    return delta

model = Tiramisu57(init_weight = True, num_classes=21)
model.cuda()
model = nn.DataParallel(model)
#model_chẹkpoints = torch.load('model/tiramisu.pth', map_location="cuda")
print('loading Model')
#qmodel.load_state_dict(model_chẹkpoints)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.995, weight_decay=1e-4)
loss = nn.CrossEntropyLoss()
loss = nn.NLLLoss2d()

Tensor = torch.cuda.FloatTensor 
LTensor = torch.cuda.LongTensor 

model.train()
for it, (imgs, targets) in enumerate(train):
    imgs = Variable(imgs.type(Tensor)).cuda()
    print(targets.size())
    optimizer.zero_grad()
    preds = model(imgs)
    b,c,w,h = preds.size()         

    
    targets = Variable(targets.type(LTensor), requires_grad=False)
    loss_value = loss(preds, targets)
    train_error = error(preds, targets)
    acc = 1 - train_error
    print(f'error = {train_error}')
    print(f'acc = {acc}')
    print(f'loss ==> {loss_value.cpu().item()}')
    f, axarr = plt.subplots(ncols=2 , figsize=(10, 5))
    x = torch.argmax(preds, dim=1)[0]
    print(torch.min(x))
    print(torch.min(targets))
    axarr[0].imshow(np.array(x.cpu().data))
    axarr[1].imshow(np.array(targets[0].cpu().data))
    plt.show()
    del x, train_error, acc
    '''
    for i in range(c):
        f, axarr = plt.subplots(ncols=2 , figsize=(10, 5))
        pred = preds[0,i,:,:]
        print(torch.min(pred))
        print(torch.max(pred))
        print(pred)
        #target = targets[0,i,:,:]
        #print(torch.max(target))
        #print((pred * target).sum())
        #print((torch.square(pred) + torch.square(target)).sum())
        print('--------')
        axarr[0].imshow(np.array(pred.cpu().data))
        axarr[1].imshow(np.array(targets[0].cpu().data))

        #print(sum1)
        #plt.imshow(np.array(preds[1,i,:,:].cpu().data))
        plt.show()
    '''
    
    loss_value.backward()
    optimizer.step()
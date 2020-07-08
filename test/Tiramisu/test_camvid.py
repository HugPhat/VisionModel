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
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from Dataset.Segmentation.CamVid import *

from Backbone.Dense.Tiramisu import *

imgset = r'D:\Code\Dataset\CamVid\train'
labelset = r'D:\Code\Dataset\CamVid\trainannot'
#labelset = r'E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\PASCALVOC\VOCdevkit\VOC2012\SegmentationClass'
#imgset = r'E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\PASCALVOC\VOCdevkit\VOC2012\JPEGImages'

train = torch.utils.data.DataLoader(
    CamVid(imgset, labelset, center_crop=True, ratio=1, mode='train'), batch_size=1, 
    shuffle=True, num_workers=0, drop_last = False
)

def dice_loss(preds, targets):
  dice = ((preds*targets).sum()*2 +1e-6) / ((torch.square(preds) + torch.square(targets)).sum() + 1e-6)
  return  1 - torch.mean(dice)

def error(preds, targets):
    preds2label = torch.argmax(preds.data.cpu(), dim=1)
    bs, w, h = targets.size()
    delta =  (torch.ne(torch.flatten(targets.data.cpu()), torch.flatten(preds2label)).cpu())
    delta = delta.type(torch.FloatTensor)
    print(delta.sum()) 

    delta = delta.sum() / (bs*w*h)
    return delta

inv_normalize = transforms.Compose( [transforms.Normalize(
                            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                            std=[1/0.229, 1/0.224, 1/0.255]),
                            ])

model = TiramisuLite(init_weight = True, num_classes=12)
model = nn.DataParallel(model)
model_chẹkpoints = torch.load(os.path.join(os.path.dirname(__file__),r'..\..\Models\Tiramisu\tiramisuLite.pth'), map_location="cuda")
model.cuda()
print('loading Model')
model.load_state_dict(model_chẹkpoints)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-4)

loss = nn.CrossEntropyLoss(weight=torch.Tensor(label_weight).cuda()).cuda()

Tensor = torch.cuda.FloatTensor 
LTensor = torch.cuda.LongTensor 

model.eval()
for it, (imgs, targets) in enumerate(train):
    imgs = Variable(imgs.type(Tensor)).cuda()

    #optimizer.zero_grad()
    with torch.no_grad():
        preds = model(imgs)
    b,c,w,h = preds.size()         
    targets = Variable(targets.type(LTensor), requires_grad=False)
    loss_value = loss(preds, targets)
    train_error = error(preds, targets)
    acc = 1 - train_error
    print(f'error = {train_error}')
    print(f'acc = {acc}')
    print(f'loss ==> {loss_value.item()}')
    
    tmp = np.array(targets[0].cpu().data)
    
    f, axarr = plt.subplots(ncols=3 , figsize=(15, 5))
    #print(targets)
    print(torch.unique(targets))
    axarr[0].imshow(np.array(inv_normalize(imgs[0]).permute(1,2,0).cpu().data))
    axarr[1].imshow(np.array(torch.argmax(preds.data.cpu(), dim=1)[0].cpu().data))
    axarr[2].imshow((tmp).astype('float32'))
    plt.show()


    #loss_value.backward()
    #optimizer.step()

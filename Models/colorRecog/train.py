import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import random
import numpy as np 
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torch.optim import lr_scheduler

img_size = 96

transform = transforms.Compose([
                                    #transforms.CenterCrop(img_size),
                                    transforms.RandomAffine(4),
                                    transforms.ColorJitter( brightness=(0.75, 1.15
                                                                        ), contrast=(0.75, 1.1)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomPerspective(),
                                    transforms.RandomVerticalFlip(p=0.4),
                                    
                                    transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])

train_path = r"D:\Code\Python\vpl-4-captains-body-tracking\data\train"
val_path = r"D:\Code\Python\vpl-4-captains-body-tracking\data\val" 

BS = 25
Epoch = 30
Lr = 0.0011
Mo = 0.912
wd = 9e-4
iter_report = 4
best_loss = np.inf

TrainLoader = DataLoader(ImageFolder(root=train_path, transform=transform), batch_size=BS, shuffle=True)
ValLoader = DataLoader(ImageFolder(root=val_path, transform=transform), batch_size=BS, shuffle=True)

from model import ColorModel
from tqdm import tqdm


Model = ColorModel(img_size= 96, use_bn=True, use_drop=True)


#print('------', Model(x))


loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(Model.parameters(), lr=Lr, weight_decay=wd)
#optimizer = optim.SGD(Model.parameters(), lr=Lr, momentum=Mo, weight_decay=wd)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30)

def accuracy(yhat, y):
    _yhat = torch.argmax(yhat, dim=1)
    batch_size = yhat.size(0)
    res = y.eq(_yhat)
    #print(y, _yhat)
    correct = res.sum().item()
    acc = correct / batch_size
    return acc, correct

cuda = 1
if cuda:
  FloatTensor = torch.cuda.FloatTensor
  LongTensor = torch.cuda.LongTensor
  Model.cuda()
else:
  FloatTensor = torch.FloatTensor
  LongTensor = torch.LongTensor

for epoch in range(1, Epoch + 1):
    Model.train()
    print('#'*16)
    with tqdm(total = len(TrainLoader)) as epoch_pbar:
        epoch_pbar.set_description(f'[Train] Epoch {epoch}')
        vloss = 0
        acc = 0
        correct = 0
        for batch_index, (inp, tar) in enumerate(TrainLoader):
            #scheduler.step()
            inp = inp.type(FloatTensor)
            
            tar = tar.type(LongTensor)
            # zero
            optimizer.zero_grad()
            # pass to Model
            out = Model(inp)
            #print(out)
            # calculate loss
            l = loss(out, tar)
            # cal acc
            _acc, re = accuracy(out, tar)
            # accumulate
            correct += re
            acc += _acc
            vloss += l.item()
            # backpropagation
            l.backward()
            optimizer.step()
            
            if batch_index % iter_report == 0 and batch_index > 0:
                desc = f'[Train] Epoch {epoch} - loss {vloss/iter_report:.4f} - acc {acc/(iter_report):.4f} - correct {int(correct/ iter_report)}'
                epoch_pbar.set_description(desc)
                epoch_pbar.update(iter_report)
                if vloss/iter_report < best_loss:
                    best_loss = vloss/iter_report
                    torch.save(Model.state_dict(), 'color_3.pth')
                acc = 0
                vloss = 0
                correct = 0
    Model.eval()
    accumulate_batch = 1
    with tqdm(total = len(ValLoader)) as epoch_pbar:
        epoch_pbar.set_description(f'( Val ) Epoch {epoch}')
        vloss = 0
        acc = 0
        correct = 0
        for batch_index, (inp, tar) in enumerate(ValLoader):
        
            inp = inp.type(FloatTensor)
            tar = tar.type(LongTensor)
            
            # pass to Model
            with torch.no_grad():
                out = Model(inp)
                #print(out)
            # calculate loss
            l = loss(out, tar)
            # cal acc
            _acc, re = accuracy(out, tar)
            # accumulate
            accumulate_batch = batch_index + 1
            correct += re
            acc += _acc
            vloss += l.item()
           
            desc = f'( Val ) Epoch {epoch} - loss {vloss/accumulate_batch:.4f} - acc {acc/(accumulate_batch):.4f} - correct {int(correct/ accumulate_batch)}'
            epoch_pbar.set_description(desc)
            epoch_pbar.update(accumulate_batch)

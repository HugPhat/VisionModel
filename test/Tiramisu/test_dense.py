import os 
import sys
import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import time

import torch
import torch.nn as nn 
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from Dataset.Segmentation.NpPortraitSet import *

from Backbone.Dense.Tiramisu import *

imgset = r"D:\Code\Dataset\18689-ph\18698-human-portrait\data\test_xtrain.npy"
labelset = r"D:\Code\Dataset\18689-ph\18698-human-portrait\data\test_ytrain.npy"

train = torch.utils.data.DataLoader(
    NpPortraitSet(imgset, labelset, center_crop = False), batch_size=1, 
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

model = Tiramisu103(init_weight = True, num_classes=2)
#model.cuda()
model = nn.DataParallel(model)
model_chẹkpoints = torch.load(os.path.join(os.path.dirname(__file__),r'..\..\Models\Tiramisu\tiramisu103_portrait.pth'), map_location="cuda")
print('loading Model')
model.load_state_dict(model_chẹkpoints)
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-4)

loss = nn.CrossEntropyLoss().cuda()

Tensor = torch.cuda.FloatTensor 
LTensor = torch.cuda.LongTensor 

model.eval()

INPUT_IMG = r"D:\Files\phat_only\pytorchModels\test\Tiramisu\beauty.jpg"
IMG = np.asarray(Image.open(INPUT_IMG))

tic = time.time()
mask = predict(model, INPUT_IMG, use_cuda=False)
print(f'prediction in {round( time.time() - tic,2)}s')

mask = mask.squeeze(0).numpy().astype('uint8')*255

mask = cv2.resize(mask, (IMG.shape[1], IMG.shape[0]), interpolation=cv2.INTER_LANCZOS4  )
##### dilate
kernel = np.ones((5,5),np.uint8)
dilation_mask = cv2.dilate(mask, kernel,iterations = 10)
dilation_mask = ((dilation_mask - mask) * 0.5).astype('uint8')

#dilation_mask = cv2.GaussianBlur(dilation_mask, (7,7),0)
# Normalize the alpha mask to keep intensity between 0 and 1
#dilation_mask = dilation_mask.astype(float)/255

dilation_mask += mask
dilation_mask = cv2.GaussianBlur(dilation_mask, (7,7),0)
dilation_mask = np.array([dilation_mask, dilation_mask, dilation_mask]).transpose(1, 2, 0) / 255.0


IMG = np.uint8(IMG*dilation_mask)


#mask = mask + dilation_mask

#mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

f, axarr = plt.subplots(ncols=3 , figsize=(15, 5))

cv2.imwrite('trimap.jpg', np.uint8(dilation_mask*255))

axarr[0].imshow(IMG)
axarr[1].imshow(mask)
axarr[2].imshow(dilation_mask)
plt.show()








''' 
    
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
'''
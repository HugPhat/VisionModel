import os
import sys

from PIL import Image
import numpy as np 
import cv2
import torchvision.transforms as transforms

from torch.autograd import Variable

sys.path.insert(0, os.path.dirname(__file__))


from DenseLayer import *


class Tiramisu(nn.Module):
    def __init__(self, 
                    growth_rate=16, 
                    flow=[4, 5, 7, 10, 12, 15],
                    mid = 15,
                    num_classes=21,
                    init_weight = False
                    ):
        super(Tiramisu, self).__init__()
        self.growth_rate = growth_rate
        self.model = nn.ModuleList()
        self.flow = flow
        self.mid = mid
        self.num_classes = num_classes
        self.Create()
        if init_weight:
            for m in self.model:
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.xavier_uniform_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.bias)
                    
        
        
    def FirstBlock(self):
        x = nn.Conv2d(in_channels=3, out_channels=48,
                      kernel_size=3, stride=1,
                      padding=(3-1)//2, bias=True)
        return x

    def Create(self):
        self.model.append(self.FirstBlock())
        inp = 48
        out = 48
        lout = []
        # down
        for i in self.flow[:(len(self.flow)-1)]:
            out = inp + self.growth_rate*i
            lout.append(out)
            #print(out)
            self.model.append(DownStep(
                num=i,
                inp=inp,
                out=out,
                growth_rate=self.growth_rate))
            inp = out
        # mid
        out = inp + self.growth_rate * self.mid
        dense = DenseBlock(inp, out, self.mid,
                           self.growth_rate, for_upsample=True)
        self.model.append(dense)
        # up
        uppath = self.flow[::-1]
        lout = lout[::-1]
        for i, item in enumerate(lout):
          j = i+1
          inp_up = self.growth_rate*(uppath[j-1])
          #    #out = lout[i] + self.growth_rate*(uppath[i] + uppath[i-1])
          inp = self.growth_rate*(uppath[j-1]) + lout[i]
          out = self.growth_rate*(uppath[j])
          self.model.append(UpStep(
              num=uppath[j],
              inp_up=inp_up,
              #out_up=out_up,
              inp=inp,
              out=out,
              growth_rate=self.growth_rate))
              
        self.LastLayer = nn.Sequential()
        self.LastLayer.add_module('point_wise',nn.Conv2d( \
                                in_channels=inp + out, 
                                out_channels=self.num_classes, 
                                kernel_size=1, stride=1, padding=0))
        self.LastLayer.add_module('softmax', nn.LogSoftmax(dim=1))
       

    def forward(self, x):
        skip_connection = []
        x = self.model[0](x)
        i = 0
        for it, step in enumerate(self.model[1: len(self.flow)]):
            i = it
            tmp, x = step(x)
            skip_connection.append(tmp)
        _, upsamp = self.model[i+2](x)
        neg_i = -1
        for it, step in enumerate(self.model[i+3:]):
            x, upsamp = step(upsamp, skip_connection[neg_i])
            neg_i -= 1
        
        x = self.LastLayer(x)  # Conv 1x1 | LogSoftmax
       
        return x
    # ==> for class 
    def load_pretrained_weight(self, path_pretrained = r'..\Models\Tiramisu\tiramisu103.pth', watching = True):
        # 0. load pretrained state dict
        pretrained_dict = torch.load(path_pretrained)
        # 0.1 get state dict
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        tmp_pretrained_dict = {}
        for k, v in list(pretrained_dict.items())[:-2]:
            if k in model_dict:
                if watching:
                    print(f'match==>{k}')
                tmp_pretrained_dict.update({k: v})
            else:
                if watching:
                    print(f'unmatch==>{k}')
        # 2. overwrite entries in the existing state dict
        model_dict.update(tmp_pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)

def preprocess(img, size):
    normalize_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225]),
                            ])
    img = cv2.resize(img, size)
    img = normalize_image(img).unsqueeze(0)
    return img

    
def predict(model, img, size = (224, 224), use_cuda = True):
    if type(img) == str:
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    elif type(img) != np.ndarray:
        raise ValueError("args [img] should be numpy array or path to image")
    
    img = preprocess(img, size)
 
    img = Variable(img.type(torch.cuda.FloatTensor)).cuda() if use_cuda  \
                            else  Variable(img.type(torch.FloatTensor))
    
    model.eval()
    with torch.no_grad():
        preds = model(img)
        

    res = torch.argmax(preds, dim=1).cpu().data   
        
    return res
    
def Tiramisu103(num_classes = 21, init_weight=False):
    return Tiramisu(
                    growth_rate=16, 
                    flow=[4, 5, 7, 10, 12, 15],
                    mid = 15,
                    num_classes=num_classes,
                    init_weight = init_weight
                    )
def Tiramisu57(num_classes = 21, init_weight=False):
    return Tiramisu(
                    growth_rate=16, 
                    flow=[4, 4, 4, 4, 4, 4],
                    mid = 4,
                    num_classes=num_classes,
                    init_weight = init_weight
                    )
def Tiramisu67(num_classes = 21, init_weight=False):
    return Tiramisu(
                    growth_rate=16, 
                    flow=[5, 5, 5, 5, 5, 5],
                    mid = 5,
                    num_classes=num_classes,
                    init_weight = init_weight
                    )
def TiramisuLite(num_classes=2, init_weight = False):
    return Tiramisu(
                    growth_rate=7, 
                    flow=[ 2, 3, 4, 5, 7],
                    mid = 7,
                    num_classes=num_classes,
                    init_weight = init_weight,
                    )
    
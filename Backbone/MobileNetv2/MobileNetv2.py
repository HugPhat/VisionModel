import torch
import torch.nn as nn
import torch.nn.functional as F 
from collections import OrderedDict
from torchvision import transforms as T

class InvertedBlock(nn.Module):
    def __init__(self, input_size, output_size, stride, factor, Id ):
        super(InvertedBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.stride = stride 
        self.factor = factor
        self.Id  = Id
        self.add = False
        
        
        self.module = self.create_block()
        
    def create_block(self):

        DWs = self.input_size * self.factor
        module = nn.Sequential()
        _prefix = "InvertedBlock_{}_".format(self.Id)

        if self.Id:
            module.add_module(_prefix + 'CONV_expand',
                              nn.Conv2d(self.input_size, DWs,
                                        kernel_size=1, bias=False))
            module.add_module(_prefix + 'BN_expand',
                              nn.BatchNorm2d(DWs))
            module.add_module(_prefix + 'RELU6_expand',
                              nn.ReLU6(inplace=True))
        else:
            _prefix = 'EXPANDED_CONV_'
            
        if (self.input_size == self.output_size) and self.stride == 1 :
            self.add = True
        # depthwise 
        pad =  (3-1)//2 #if self.stride == 1 else ()
        module.add_module(_prefix + 'DW_CONV_' + str(self.Id), \
                                nn.Conv2d(DWs, DWs, kernel_size=3, 
                                            stride= (self.stride, self.stride), 
                                            padding= pad, groups=DWs, bias=False))
        module.add_module(_prefix + 'DW_BN_' + str(self.Id), \
                                nn.BatchNorm2d(DWs))
        module.add_module(_prefix + 'DW_RELU6_' + str(self.Id), \
                                nn.ReLU6(inplace=True))
        # projection
        module.add_module(_prefix + 'PRJ_CONV_' + str(self.Id),
                          nn.Conv2d(DWs, self.output_size, kernel_size=1, padding=(1-1)//2, bias=False))
        module.add_module(_prefix + 'PRJ_BN_' + str(self.Id),
                          nn.BatchNorm2d(self.output_size))
        return module
        
    def forward(self, x):
        inp = x
        if self.add:
            return self.module(x) + inp
        else:
            return self.module(x)
        
class MobileNetv2(nn.Module):
    def __init__(self, input_size = 224, n_classes = 10, debug= False):
        super(MobileNetv2, self).__init__()
        self.input_size = input_size
        self.debug = debug
        self.last_channel = 1280
        self.n_classes = n_classes
        self.model_hyperparams = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        self.constructNet()
        self.T = T.Compose([
                            T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225]),
                            ])
    def firstBlock(self, module):
        out_filter = 32
        # 3: RGB image
        module.add_module('FirstBlock_Conv2D', nn.Conv2d(3, out_filter, 
                                                kernel_size= 3, 
                                                stride=2, 
                                                padding= (3-1)//2, 
                                                bias=False))
        module.add_module('FirstBlock_BN', nn.BatchNorm2d(out_filter))
        module.add_module('FirstBlock_RELU6', nn.ReLU6(inplace=True))
        return out_filter
     
    def conv_1x1_bn(self, inp, oup):
        return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
        nn.Dropout2d(0.2)
        )
        
        
    def constructNet(self):
        fmodule = nn.Sequential()
        self.model = nn.ModuleList()
        self.blocks = list()
        prev_filter = self.firstBlock(fmodule) # first output filter 
        self.blocks.append('FirstBlock')
        self.model.append(fmodule)
        
        Id = 0
        for t, c, n, s in self.model_hyperparams:
            for block in range(n):
                stride = s if block == 0 else 1
                if self.debug:
                    print(f'[{n}] with Id {Id}==> block {block} : [in]: {prev_filter}, [out]: {c}')
                self.model.append(InvertedBlock(input_size=prev_filter, 
                                                output_size= c, 
                                                stride= stride, 
                                                factor= t, 
                                                Id= Id))
                self.blocks.append('InvertedBlock_' + str(Id))
                Id+=1
                prev_filter = c
        self.blocks.append('last_conv_1_1')        
        self.model.append(self.conv_1x1_bn(prev_filter, self.last_channel))
        self.classifier=  nn.Linear(self.last_channel, self.n_classes)
        if self.debug:
            print(f'[1] with Id {Id}==> block {block} : [in]: {prev_filter}, [out]: {self.last_channel}')
            print(f'[1] with Id {Id + 1}==> block {block} : [in]: {self.last_channel}, [out]: {self.n_classes}')
    
    def forward(self, x):
        for block_name, module in zip(self.blocks, self.model):
            x = module(x)
            if self.debug:
                print(x.size())    
        x = F.avg_pool2d(x, 7)
        x = x.mean(3).mean(2)
        print(x.size())
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x    

    def predict(self, img, cuda = False):
        inp = self.T(img).unsqueeze(0)
        if cuda: inp = inp.to('cuda')
        out = self(inp)
        clss = torch.argmax(out)
        conf = torch.max(out).item()
        return clss, conf
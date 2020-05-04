import torch
import torch.nn as nn
import torch.functional as F 
from collections import OrderedDict

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
                                        kernel_size=1, bias=False, padding=0))
            module.add_module(_prefix + 'BN_expand',
                              nn.BatchNorm2d(DWs))
            module.add_module(_prefix + 'RELU6_expand',
                              nn.ReLU6(inplace=True))
        else:
            _prefix = 'EXPANDED_CONV_'

        if self.stride == 2:
            self.add = True
            # Bug in padding -> ?
            #module.add(_prefix + 'pad', nn.ZeroPad2d(padding= 0))
        
        # depthwise 
        module.add_module(_prefix + 'DW_CONV_' + str(self.Id), \
                                nn.Conv2d(DWs, DWs, kernel_size=3, padding= (3-1)//2, groups=self.factor, bias=False, stride= self.stride))
        module.add_module(_prefix + 'DW_BN_' + str(self.Id), \
                                nn.BatchNorm2d(DWs))
        module.add_module(_prefix + 'DW_RELU6_' + str(self.Id), \
                                nn.ReLU6(inplace=True))
        # projection
        module.add_module(_prefix + 'DW_CONV_' + str(self.Id),
                          nn.Conv2d(DWs, self.output_size, kernel_size=1, padding=(1-1)//2, bias=False))
        module.add_module(_prefix + 'DW_BN_' + str(self.Id),
                          nn.BatchNorm2d(DWs))
        return module
        
    def forward(self, x):
        if self.add:
            return self.module(x) + x
        else:
            return self.module(x)
        
class MobileNetv2(nn.Module):
    def __init__(self, input_size = 224, debug= False):
        super(MobileNetv2, self).__init__()
        self.input_size = input_size
        self.debug = debug
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
        
    def firstBlock(self, module):
        out_filter = 32
        # 3: RGB image
        module.add_module('FirstBlock_Conv2D', nn.Conv2d(3, out_filter, kernel_size= 3, stride=2, padding= (3-1)//2, bias=False))
        module.add_module('FirstBlock_BN', nn.BatchNorm2d(out_filter))
        module.add_module('FirstBlock_RELU6', nn.ReLU6(inplace=True))
        return out_filter 

    def constructNet(self):
        fmodule = nn.Sequential()
        self.model = nn.ModuleList()
        self.blocks = list()
        prev_filter = self.firstBlock(fmodule) # first output filter 
        self.model.append(fmodule)
        self.blocks.append('FirstBlock')
        Id = 0
        for t, c, n, s in self.model_hyperparams:
            for block in range(n):
                if self.debug:
                    print(f'[{n}] with Id {Id}==> block {block} : [in]: {prev_filter}, [out]: {c}')
                self.model.append(InvertedBlock(input_size=prev_filter, output_size= c, stride= s, factor= t, Id= Id))
                self.blocks.append('InvertedBlock_' + str(Id))
                Id+=1
                prev_filter = c

    def forward(self, x):
        for block_name, module in zip(self.blocks, self.model):
            if self.debug:
                print(f'We reach {block_name}')
            x = module(x)
            
        return x    

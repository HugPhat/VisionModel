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
        _prefix = "InvertedBlock_{}_".format(Id)

        if Id:
            module.add_module(_prefix + 'CONV_expand',
                              nn.Conv2d(self.input_size, DWs,
                                        kernel_size=1, bias=False, padding=0))
            module.add_module(_prefix + 'BN_expand',
                              nn.BatchNorm2d(DWs))
            module.add_module(_prefix + 'RELU6_expand',
                              nn.ReLU6(inplace=True))
        else:
            _prefix = 'EXPANDED_CONV_'

        if stride == 2:
            self.add = True
            # Bug in padding -> ?
            module.add(_prefix + 'pad', nn.ZeroPad2d(padding= 0))
        
        # depthwise 
        module.add_module(_prefix + 'DW_CONV_' + str(self.Id), \
                                nn.Conv2d(DWs, DWs, kernel_size=3, padding= (3-1)//2, groups=self.factor, bias=False, stride= self.stride))
        module.add_module(_prefix + 'DW_BN_' + str(self.Id), \
                                nn.BatchNorm2d(DWs))
        module.add_module(_prefix + 'DW_RELU6_' + str(self.Id), \
                                nn.ReLU6(inplace=True))
        # projection
        module.add_module(_prefix + 'DW_CONV_' + str(self.Id),
                          nn.Conv2d(DWs, self.output_size, kernel_size=1, padding=(1-1)//2, groups=self.factor, bias=False))
        module.add_module(_prefix + 'DW_BN_' + str(self.Id),
                          nn.BatchNorm2d(DWs))
        return module
        
    def forward(self, x):
        if self.add:
            return self.module(x) + x
        else:
            return self.module(x)
class MobileNetv2(nn.Module):
    def __init__(self, input_size = 224):
        super(MobileNetv2, self).__init__()
        self.input_size = input_size
    
    def firstBlock(self):
        module = nn.Sequential()
        module.add_module('ZeroPad_0', nn.ZeroPad2d(32))

    def invertedBlock(self, Input, stride, factor, Id):
        # Calculate parameters
        






    def strideBlock(self, id, stride):
        pass

    def constructNet(self):
        pass

    def forward(self, x):
        pass
    

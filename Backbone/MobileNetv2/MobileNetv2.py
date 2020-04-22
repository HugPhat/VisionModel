import torch
import torch.nn as nn
import torch.functional as F 
from collections import OrderedDict

class MobileNetv2(nn.Module):
    def __init__(self, input_size = 224):
        super(MobileNetv2, self).__init__()
        self.input_size = input_size
    def firstBlock(self):
        module = nn.Sequential()
        module.add_module('ZeroPad_0', nn.ZeroPad2d())
    def invertedBlock(self,*args, **kwargs):
        pass
    
    def strideBlock(self):
        pass

    def constructNet(self):
        pass

    def forward(self, x):
        pass
    
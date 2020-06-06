import torch 
import torch.nn as nn 
import torch.functional as F 

class DenseLayer(nn.Module):
    def __init__(self, n_input, n_output):
        super(DenseLayer, self).__init__()
        self.module = self.init_layer(n_input, n_output)

    def init_layer(self, n_input, n_output):
        module = nn.Sequential()
        module.add_module('BN', nn.BatchNorm2d(n_input))
        module.add_module('ReLU', nn.ReLU(inplace=True))
        module.add_module('CONV', nn.Conv2d(
                          n_input, n_output,
                          kernel_size=3, 
                          padding= (3-1)//2,
                          stride=1
                          ))
        module.add_module('Drop', nn.Dropout2d(0.2))
        return module
        
    def forward(self, x):
        return self.module(x)

class DenseBlock(nn.Module):
    def __init__(self, n_input, n_output, n_layer, growth_rate, debug = False):
        super(DenseBlock, self).__init__()
        self.debug = debug
        self.n_input, self.n_output, self.growth_rate = n_input, n_output, growth_rate
        self.n_layer = n_layer
        self.module = nn.ModuleList()
        self.init_layer()

    def init_layer(self):
        n_input = self.n_input
        n_out = self.growth_rate 
        for _ in range(1, self.n_layer + 1):
            tmp = DenseLayer(n_input, n_out)
            self.module.append(tmp)
            # cat : prev and curr
            n_input = n_input + self.growth_rate
            n_out = self.growth_rate 
            #if self.debug:
            #    print(f'inp-> {n_input}')
            #    print(tmp.module[2].weight.size())
            #    print(f'out-> {n_out}')

    def forward(self, x):
        prev = x
        for i, layer in enumerate(self.module):
            x = layer(x)
            x = torch.cat([prev, x], dim=1)
            if self.debug: 
                print(x.size())
            prev = x
        return x

        
class TransitionDown(nn.Module):
    def __init__(self, n_input, n_out):
        super(TransitionDown,self).__init__()
        self.module = nn.Sequential()
        self.module.add_module('BN', nn.BatchNorm2d(n_input))
        self.module.add_module('ReLu', nn.ReLU(inplace=True))
        self.module.add_module('Conv', nn.Conv2d(
                                in_channels=n_input,
                                out_channels=n_out,
                                kernel_size=(1,1),
                                stride=1,
                                padding= (1-1)//2            
                                ))
        self.module.add_module('Drop', nn.Dropout2d(0.2))
        self.module.add_module('MaxPool', nn.MaxPool2d(kernel_size=(2,2)))
    
    def forward(self, x):
        return self.module(x)

'''
class TransitionUp(nn.Module):
    def __init__(self, n_inp, n_out, stride = 2):
        super(TransitionUp, self).__init__()

        self.TU = nn.ConvTranspose2d(in_channels=n_inp, 
                                     out_channels=n_out,
                                     kernel_size=(3,3),
                                     stride=stride,)

    def forward(self, x):
        x = self.TU(x)
        return x
'''

class TransitionUp(nn.Sequential):
    def __init__(self, n_inp, n_out):
        super(TransitionUp, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(n_inp))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(n_inp, n_out,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

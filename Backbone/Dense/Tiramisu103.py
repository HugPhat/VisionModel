import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from DenseLayer import *


class Tiramisu103(nn.Module):
    def __init__(self, growth_rate=16, flow=[4, 5, 7, 10, 12, 15], num_classes=10):
        super(Tiramisu103, self).__init__()
        self.growth_rate = growth_rate
        self.model = nn.ModuleList()
        self.flow = flow
        self.mid = 15
        self.num_classes = num_classes
        self.Create()

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
        self.LastLayer.add_module('lastconv', nn.Conv2d( \
                                in_channels=inp + out, 
                                out_channels=self.num_classes, 
                                kernel_size=1, stride=1, padding=0))
        self.LastLayer.add_module('softmax', nn.Softmax(dim=1))    
        

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
        
        x = self.LastLayer[0](x)# Conv 1x1
        b,c,w,h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = x.resize(b*h*w, c)
        x = F.softmax(x, 0)
        x = x.resize(b,c,w,h)
        
        return x

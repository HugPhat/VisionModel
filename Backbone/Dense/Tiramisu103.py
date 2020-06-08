import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from DenseLayer import *

class Tiramisu103(nn.Module):
    def __init__(self, growth_rate = 16):
        super(Tiramisu103, self).__init__()
        self.growth_rate = growth_rate
        self.model = nn.ModuleList()
        self.flow = [4, 5, 7, 10, 12]
        self.mid = 15
        self.Create()

    def FirstBlock(self):
        x =  nn.Conv2d(in_channels= 3, out_channels= 48, 
                        kernel_size=3, stride=1, 
                        padding=(3-1)//2, bias=True)
        return x

    def Create(self):
        self.model.append(self.FirstBlock())
        inp = 48
        out = 48
        lout = []
        # down 
        for i in self.flow:
            out = inp + self.growth_rate*i
            lout.append(out)
            print(out)
            self.model.append(DownStep(
                                        num= i, 
                                        inp= inp, 
                                        out= out, 
                                        growth_rate= self.growth_rate))
            inp = out
        # mid
        out = inp + self.growth_rate * self.mid
        dense = DenseBlock(self.mid, inp, out, self.growth_rate)
        self.model.append(dense)
        # up
        uppath = self.flow[::-1]
        lout = lout[::-1]
        for i, item in enumerate(uppath):
            if item == max(uppath):
                out = out + self.growth_rate*item
                print(f'=>out {out}')
            else:
                out = lout[i] + self.growth_rate*(uppath[i] + uppath[i-1]) 
                print(f'uu=>out {out}')
            self.model.append(UpStep(
                                    num=item,
                                    inp=inp,
                                    out=out,
                                    growth_rate=self.growth_rate))
            inp = out

    def forward(self, x):
        
        skip_connection = []

        x = self.model[0](x)
        i = 0
        for it, step in enumerate(self.model[1: len(self.flow)]):
            i = it
            tmp, x = step(x)
            print(f'i=>> {i}')
            print(step)
            skip_connection.append(tmp)
        print(self.model[i+1])
        x = self.model[i+1](x)
        neg_i = -1
        for it, step in enumerate(self.model[i+2:]):
            #print(x.size())
            x = step(x, skip_connection[neg_i])
            neg_i -= 1
        
        return x
            
                

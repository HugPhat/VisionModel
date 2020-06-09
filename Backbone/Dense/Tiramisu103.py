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
        dense = DenseBlock(inp, out, self.mid, self.growth_rate, for_upsample= True)
        #inp = out
        self.model.append(dense)
        # up
        uppath = self.flow[::-1]
        lout = lout[::-1]
        inp_up = out
        for i, item in enumerate(uppath):
            if item == max(uppath):
                out_up = self.growth_rate*(self.mid)
                out = out + self.growth_rate*item
            else:
                out_up = self.growth_rate*(uppath[i-1])
                out = lout[i] + self.growth_rate*(uppath[i] + uppath[i-1]) 
            
            print(f'uu=>inp_up {inp_up}')
            print(f'uu=>out_up {out_up}')
            print(f'uu=>out {out}')
            self.model.append(UpStep(
                                    num=item,
                                    inp_up=inp_up,
                                    out_up=out_up,
                                    #inp=inp,
                                    out=out,
                                    growth_rate=self.growth_rate))
            inp_up = out

    def forward(self, x):
        
        skip_connection = []

        x = self.model[0](x)
        i = 0
        for it, step in enumerate(self.model[1: len(self.flow)+1]):
            i = it
            tmp, x = step(x)
            print(f'i=>> {i}')
            print(x.size())
            skip_connection.append(tmp)
        for_upsample  = self.model[i+2](x)
        #print(self.model[i+2])
        print(for_upsample.size())
        neg_i = -1
        for it, step in enumerate(self.model[i+3:]):

            for_upsample = step(x, for_upsample, skip_connection[neg_i])
            neg_i -= 1
        
        return x
            
                

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                          padding=(3-1)//2,
                          stride=1
                          ))
        module.add_module('Drop', nn.Dropout2d(0.2))
        return module

    def forward(self, x):
        #if isinstance(x, torch.Tensor):
        #    inp = [x]
        #else:
        #    inp = x
        return self.module(x)


class DenseBlock(nn.Module):
    def __init__(self, n_input, n_output, n_layer, growth_rate, debug=False, for_upsample=False):
        super(DenseBlock, self).__init__()
        self.debug = debug
        self.for_upsample = for_upsample
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

    def forward(self, x):
        feature = x
        tmp = []
        for i, layer in enumerate(self.module):
            x = layer(feature)
            if self.for_upsample:
                tmp.append(x)
            feature = torch.cat([feature, x], 1)

        if self.for_upsample:
            return feature, torch.cat(tmp, 1)
        else:
            return feature


class TransitionDown(nn.Module):
    def __init__(self, n_input, n_out):
        super(TransitionDown, self).__init__()
        self.module = nn.Sequential()
        self.module.add_module('BN', nn.BatchNorm2d(n_input))
        self.module.add_module('ReLu', nn.ReLU(inplace=True))
        self.module.add_module('Conv', nn.Conv2d(
            in_channels=n_input,
            out_channels=n_out,
            kernel_size=(1, 1),
            stride=1,
            padding=0
        ))
        self.module.add_module('Drop', nn.Dropout2d(0.2))
        self.module.add_module('MaxPool', nn.MaxPool2d(kernel_size=(2, 2)))

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


class TransitionUp(nn.Module):
    def __init__(self, n_inp, n_out):
        super(TransitionUp, self).__init__()
        self.bn = nn.BatchNorm2d(n_inp)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.ConvTranspose2d(n_inp, n_out,
                                       kernel_size=3, stride=2,
                                       padding=1,
                                       bias=True,
                                       )
        #self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x, output_size):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x, output_size=output_size)
        return x


class DownStep(nn.Module):
    def __init__(self, num, inp, out, growth_rate, ):
        super(DownStep, self).__init__()
        self.dense_block = DenseBlock(
            inp, out, num, growth_rate, for_upsample=False)
        self.down = TransitionDown(out, out)

    def forward(self, x):
        x1 = self.dense_block(x)
        x2 = self.down(x1)
        return x1, x2


class UpStep(nn.Module):
    def __init__(self, num, inp_up, inp, out, growth_rate):
        super(UpStep, self).__init__()
        self.up = TransitionUp(inp_up, inp_up)
        self.dense_block = DenseBlock(
            inp, out, num, growth_rate, for_upsample=True)

    def forward(self, x, skip_connect):
        b, c, h, w = list(x.size())
        # , output_size = [b,c,h*2,w*2]
        x = self.up(x, output_size=[b, c, h*2, w*2])
        t = torch.cat([x, skip_connect], dim=1)
        x, forupsample = self.dense_block(t)
        return x, forupsample

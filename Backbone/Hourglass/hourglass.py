import torch 
import torch.nn as nn 

use_bb_with_dw = False

def basic_block(inp, use_bn = True):
    _basic_block = nn.Sequential()
    if not use_bb_with_dw:
        # Conv1x1
        _basic_block.add_module('Con2d_1x1', nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0))
        if use_bn:
            _basic_block.add_module('BN_1x1', nn.BatchNorm2d(inp, momentum=0.9))
        _basic_block.add_module('Relu_1x1', nn.ReLU(inplace=True))
        # 
        # Bottle neck
        out = inp // 2
        #
        _basic_block.add_module('Con2d_1x1_1', nn.Conv2d(inp, out, kernel_size=1, stride=1, padding=0))
        if use_bn:
            _basic_block.add_module('BN_1x1_1', nn.BatchNorm2d(out, momentum=0.9))
        _basic_block.add_module('Relu_1x1_1', nn.ReLU(inplace=True))
        #
        _basic_block.add_module('Con2d_0', nn.Conv2d(out, out, kernel_size=3, stride=1, padding=1))
        if use_bn:
            _basic_block.add_module('BN_0', nn.BatchNorm2d(out, momentum=0.9))
        _basic_block.add_module('Relu_0', nn.ReLU(inplace=True))
        #
        _basic_block.add_module('Con2d_1', nn.Conv2d(out, inp, kernel_size=1, stride=1, padding=0))
        if use_bn:
            _basic_block.add_module('BN_1', nn.BatchNorm2d(inp, momentum=0.9))
        _basic_block.add_module('Relu_1', nn.ReLU(inplace=True))
    else:
        #print('dw')
        t = 1
        dws = t*inp
        # Conv1x1
        _basic_block.add_module('Con2d_1x1', nn.Conv2d(inp, dws, kernel_size=1, stride=1, padding=0))
        if use_bn:
            _basic_block.add_module('BN_1x1', nn.BatchNorm2d(dws))
        _basic_block.add_module('Relu_1x1', nn.ReLU(inplace=True))
        # 
        # Bottle neck with depth wise operation
        #
        _basic_block.add_module('Con2d_0', nn.Conv2d(dws, dws, kernel_size=3, stride=1, padding=1, groups=dws))
        if use_bn:
            _basic_block.add_module('BN_0', nn.BatchNorm2d(dws))
        _basic_block.add_module('Relu_0', nn.ReLU(inplace=True))
        #
        _basic_block.add_module('Con2d_1', nn.Conv2d(dws, inp, kernel_size=1, stride=1, padding=0))
        if use_bn:
            _basic_block.add_module('BN_1', nn.BatchNorm2d(inp))
        _basic_block.add_module('Relu_1', nn.ReLU(inplace=True))
        
    return _basic_block

def downSampleLayer( k, s):
    return nn.MaxPool2d(kernel_size=k, stride=s)

class down_stream(nn.Module):
    def __init__(self, bb, fil, pool):
        super(down_stream, self).__init__()
        self.bb = bb
        self.pool = pool 
        self.fil = fil
        
    def forward(self, x):
        '''
        
        *return:
            + x1: output from branch
            + x2: output from pool
            
        '''
        x += self.bb(x)
        x1 = self.fil(x)
        x2 = self.pool(x)
        return x1, x2

############ Hourglass Module ##############

class hourglass_module(nn.Module):
    def __init__(self, input_size, basic_block_callback, n_reduce_blocks= 4,n_basis_blocks= 3, n_filter_blocks=3):
        super(hourglass_module, self).__init__()
        self.inp = input_size
        self.n_reduce_blocks = n_reduce_blocks
        self.n_basis_blocks = n_basis_blocks
        self.n_filter_blocks = n_filter_blocks
        self.model = self.construct_module(basic_block_callback)
    
    def construct_module(self, basic_block_callback):
        # pre define input/output_size
        inp = self.inp
        
        model = nn.ModuleDict({
                'down' : nn.ModuleList(),
                'up': nn.ModuleList(),
                'post': basic_block_callback(inp, use_bn=True)
            }
        )
        
        # construct network
        for i in range(self.n_reduce_blocks):
            basic_blocks = nn.Sequential()
            for j in range(self.n_basis_blocks):
                basic_blocks.add_module('bblock_' + str(i) + '_' + str(j), 
                                        basic_block_callback(inp, use_bn=True))
            basic_filter = nn.Sequential()
            for f in range(self.n_filter_blocks):
                basic_filter.add_module('fblock_'+ str(i) + '_' + str(f),
                                        basic_block_callback(inp, use_bn=True)
                                        )
                
            basic_blocks = down_stream(basic_blocks, basic_filter, downSampleLayer((2,2), 2))
            model['down'].append(basic_blocks)   
            model['up'].append(nn.Upsample(scale_factor=2, mode='nearest'))
        
        
        return model
        
    def forward(self, x):
        branch = []
        # down sample
        for block in self.model['down']:
            #print(block)
            x1, x = block(x)
            branch.append(x1)
        # upsample 
        for i, block in enumerate(self.model['up']):
            #print(x.size(), branch[-(i + 1)].size())
            x = branch[-(i + 1)] + block(x)
        
        x = self.model['post'](x)
        
        return x
        
        
###### Hourglass stacked ########
 
class hourglass(nn.Module):
    
    def __init__(self, stacked_ch=192, n_stacked_hg=3):
        super(hourglass, self).__init__()
        self.stacked_ch = stacked_ch 
        

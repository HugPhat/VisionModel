import torch 
import torch.nn as nn 

use_bb_with_dw = False
dws_ratio = 1

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
        dws = dws_ratio*inp
        # Conv1x1
        _basic_block.add_module('Con2d_1x1', nn.Conv2d(inp, dws, kernel_size=1, stride=1, padding=0, groups=inp))
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
    def __init__(self, input_size, basic_block_callback, n_reduce_blocks= 4,n_basis_blocks= 3, n_filter_blocks=3, use_bn= True):
        super(hourglass_module, self).__init__()
        self.inp = input_size
        self.n_reduce_blocks = n_reduce_blocks
        self.n_basis_blocks = n_basis_blocks
        self.n_filter_blocks = n_filter_blocks
        self.use_bn= use_bn
        self.model = self.construct_module(basic_block_callback)
    
    def construct_module(self, basic_block_callback):
        # pre define input/output_size
        inp = self.inp
        
        model = nn.ModuleDict({
                'down' : nn.ModuleList(),
                'up': nn.ModuleList(),
                #'post': basic_block_callback(inp, use_bn=True)
            }
        )
        
        # construct network
        for i in range(self.n_reduce_blocks):
            basic_blocks = nn.Sequential()
            for j in range(self.n_basis_blocks):
                basic_blocks.add_module('bblock_' + str(i) + '_' + str(j), 
                                        basic_block_callback(inp, use_bn=self.use_bn))
            basic_filter = nn.Sequential()
            for f in range(self.n_filter_blocks):
                basic_filter.add_module('fblock_'+ str(i) + '_' + str(f),
                                        basic_block_callback(inp, use_bn=self.use_bn)
                                        )
                
            basic_blocks = down_stream(basic_blocks, basic_filter, downSampleLayer((2,2), 2))
            model['down'].append(basic_blocks) 
              
            up = nn.Sequential()
            up.add_module('upsample_'+ str(i), nn.Upsample(scale_factor=2, mode='nearest'))
            up.add_module('up_bblock'+ str(i), basic_block_callback(inp, use_bn= self.use_bn))
            model['up'].append(up)
        
        
        return model
        
    def forward(self, x):
        branch = []
        # down sample
        for block in self.model['down']:
            #print(block)
            x1, x = block(x)
            branch.append(x1)
        # upsample 
        for i in range(0, len(self.model['up'])):
            #print(x.size(), branch[-(i + 1)].size())
            x = branch[-(i + 1)] + self.model['up'][i](x)
            
        #print('x----', x.size())
        return x
        
####### Intermediate Supervision ###########

class intermediate_supervision(nn.Module):
    def __init__(self, input_size,n_heatmaps, bb):
        super(intermediate_supervision, self).__init__()
        self.bb = bb
        self.inter = nn.Conv2d(input_size, input_size, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sequential()
        self.output.add_module('out_conv', 
                               nn.Conv2d(input_size, n_heatmaps, kernel_size=1, stride=1, padding=0))
        self.output.add_module('out_inter',
                                nn.Conv2d(n_heatmaps, input_size, kernel_size=1, stride=1, padding=0)) 
        
    def forward(self, x):
        y = self.bb(x)
        y1 = self.inter(y)
        y2 = self.output(y)
        
        y = y + y1 + y2
        #print(y.size())
        return y
        
###### Hourglass stacked ########
 
def default_input_block(out):
    prefix = 'default'
    prefix_conv = lambda x: prefix+ '_conv2d_' + str(x)
    prefix_relu = lambda x: prefix+ '_relu_' + str(x)
    prefix_bn  = lambda x: prefix+ '_bn_' + str(x)
    prefix_pool  = lambda x: prefix+ '_pool_' + str(x)
    module = nn.Sequential()
    module.add_module(prefix_conv(0), nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3))
    module.add_module(prefix_bn(0), nn.BatchNorm2d(64))
    module.add_module(prefix_relu(0), nn.ReLU(64))
    module.add_module(prefix_conv('1x1_128'), nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0))
    module.add_module(prefix_conv('bb0'), basic_block(128))
    module.add_module(prefix_pool('maxpool'), nn.MaxPool2d(kernel_size=2, stride=2))
    module.add_module(prefix_conv('1x1_256'), nn.Conv2d(128, out, kernel_size=1, stride=1, padding=0))
    module.add_module(prefix_conv('bb1'), basic_block(out))
    
    return module
    
    
    
    
class hourglass(nn.Module):
    
    def __init__(self, 
                 n_heatmaps,
                 n_stacked_hg=3, 
                 stacked_ch=192, 
                 n_reduce_blocks=4, 
                 n_basis_blocks=3, 
                 n_filter_blocks=3,
                 use_bn=True,
                 input_block = None, 
                 basic_block_callback=None,
                 ):
        super(hourglass, self).__init__()
        
        self.n_heatmaps = n_heatmaps
        #### 
        self.stacked_ch = stacked_ch 
        self.n_stacked_hg = n_stacked_hg
        ####
        self.n_reduce_blocks = n_reduce_blocks
        self.n_basis_blocks    = n_basis_blocks
        self.n_filter_blocks = n_filter_blocks
        self.use_bn = use_bn
        ####
        self.bb_cb = basic_block_callback if basic_block_callback else basic_block
        self.input_block = input_block if input_block else default_input_block
        self.model = self.create_network()

    def create_network(self):
        model = nn.ModuleList()
        model.append(self.input_block(self.stacked_ch))
        for i in range(self.n_stacked_hg):
            block = hourglass_module(self.stacked_ch, self.bb_cb, self.n_reduce_blocks, self.n_basis_blocks, self.n_filter_blocks, self.use_bn)
            model.append(intermediate_supervision(self.stacked_ch, self.n_heatmaps, block))
        return model
    
    def forward(self, x):
        for i, block in enumerate(self.model):
            #print(i)
            x = block(x)
            
        return x
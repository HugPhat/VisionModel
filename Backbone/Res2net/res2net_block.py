import torch
import torch.nn as nn 

class res2net_block(nn.Module):
    def __init__(self, in_channel, out_channel, k_size):
        super(res2net_block, self).__init__()

        
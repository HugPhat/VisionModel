import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np 

class YoloPreHead(nn.Module):
    def __init__(self,
                 batch,
                 CH,
                 Grid1,
                 Grid2,
                 n_anchor,
                 n_class
                 ):
        super(YoloPreHead, self).__init__()
        '''
        The last layer before YoloHead
        k_size: 1x1
        + Input: [Batch, CH,Grid1, Grid2]
        + Output: [Batch, [n_anchor *(4 + 1 + num_class)], Grid1, Grid2]
        '''
        self.conv1x1 = nn.Conv2d(
                                in_channels=CH,
                                out_channels= (n_anchor * (4 + 1 + n_class)),
                                kernel_size=(1,1),
                                padding=0,
                                stride=1
                                )
        
        
    def forward(self, x):
        return self.conv1x1(x)

class YoloHead(nn.Module):
    def __init__(self, 
                    anchor,
                    num_classes,
                    img_size,
                    is_train, 
                    ignore_thresh,
                    lambda_noobj,
                    lambda_obj,
                    lambda_box
                    ):
        super(YoloHead, self).__init__()
        '''
        Head (output) of Yolov3
        + Input: 
        Tensor from YoloPreHead
            => Applying author formular:
                + bx = sig(tx) + cx
                + by = sig(ty) + cy
                + bw = pw * exp(tw)
                + bh = ph * exp(th)
                + conf = sig(conf)
                + class = sig(class)
        + Output: 
            Tensor [n_batch, grid1*grid2*n_anchor, 5+ n_class]
        '''
        self.anchor = anchor
        self.n_anchor = len(anchor)
        self.num_classes = num_classes
        self.cell_len = 5 + num_classes
        self.img_size = img_size
        self.ignore_thresh = ignore_thresh
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box
        
    def forward(self, x):
        
        n_batch = x.size(0)
        n_grid1 = x.size(2)
        n_grid2 = x.size(3)
        
        # [n_batch, n_anchor*( len_cell), grid1, grid2]
        # -> [n_batch, n_anchor, grid1, grid2, len_cell]
        x = x.view(n_batch, self.n_anchor, n_grid1, n_grid2, self.cell_len).contiguous()
        
        grid1_ratio = img_size[0]/ n_grid1
        grid2_ratio = img_size[1]/ n_grid2
        
        FT = torch.cuda.FloatTensor if x.is_cuda() else torch.FloatTensor
        LT = torch.cuda.LongTensor if x.is_cuda() else torch.LongTensor
        BT = torch.cuda.BoolTensor if x.is_cuda() else torch.BoolTensor
        
        ########## Box Coordinate #############
        x = torch.sigmoid(x[..., 0])
        y = torch.sigmoid(x[..., 1])
        w = x[..., 2]
        h = x[..., 3]
        coor_x = torch.arange(n_grid1).repeat(n_grid1, 1).view(
                [1, 1, n_grid1, n_grid2]).type(FT)
        coor_y = torch.arange(n_grid2).repeat(n_grid2, 1).t().view(
                [1, 1, n_grid1, n_grid2]).type(FT)
        scaled_anchors = FT([(aw/ grid1_ratio, ah/ grid2_ratio) for aw, ah in self.anchor])
        anchor_x = scaled_anchors[:, :1].view((1, self.n_anchor, 1, 1))
        anchor_y = scaled_anchors[:, 1:].view((1, self.n_anchor, 1, 1))
        ########### Conf / Class score #########
        Conf = torch.sigmoid(x[..., 4])
        Class = torch.sigmoid(x[..., 5:])
        ########################################
        PredBoxes = FT(x[..., :4].size()) # [n_batch, n_anchor, grid1, grid2, 4]
        # bx, by
        PredBoxes[..., 0] = x.data + coor_x
        PredBoxes[..., 1] = y.data + coor_y
        # w, h
        PredBoxes[..., 2] = torch.exp(w)*anchor_x
        PredBoxes[..., 3] = torch.exp(h)*anchor_y
        
        output = torch.cat(
            [
                PredBoxes.view(n_batch, self.n_anchor,  n_grid1, n_grid2, 4),
                Conf.view(n_batch, self.n_anchor,  n_grid1, n_grid2, 1),
                Class.view(n_batch, self.n_anchor,  n_grid1, n_grid2, self.num_classes)
            ], dim=-1
        )
        
        return output
    
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

class skip_connection(nn.Module):
    def __init__(self):
        super(skip_connection, self).__init__()


class yoloDetection(nn.Module):
    def __init__(self,  anchor, 
                        num_classes, 
                        img_size, 
                        device=None, 
                        ignore_thresh = 0.5,
                        lambda_noobj=0.5,
                        lambda_obj = 1,
                        lambda_class = 1.5,
                        lambda_coord = 2,
                        ):

        super(yoloDetection, self).__init__()
        self.anchor = anchor
        self.num_anchor = len(self.anchor)
        self.num_classes = num_classes
        self.img_size = img_size
        self.grid_info = num_classes + 5
        self.ignore_thres = ignore_thresh
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class
        self.lambda_coord = lambda_coord
        self.mse_loss = nn.MSELoss(size_average=True)
        self.bce_loss = nn.BCELoss(size_average=True)
        self.ce_loss = nn.CrossEntropyLoss()
       

    def forward(self, x, targets=None):

        isTrain = True if targets != None else False
        numBatches = x.size(0)
        numGrids = x.size(2)

        ratio = self.img_size / numGrids

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        BoolTensor = torch.cuda.BoolTensor if x.is_cuda else torch.BoolTensor
        predict = x.view(numBatches, self.num_anchor,
                         self.grid_info, numGrids, numGrids).permute(0, 1, 3, 4, 2).contiguous()  #
        x = torch.sigmoid(predict[..., 0])
        y = torch.sigmoid(predict[..., 1])
        w = predict[..., 2]
        h = predict[..., 3]
        conf = torch.sigmoid(predict[...,  4])
        clss = torch.sigmoid(predict[...,  5:])
        coor_x = torch.arange(numGrids).repeat(numGrids, 1).view(
            [1, 1, numGrids, numGrids]).type(FloatTensor)
        coor_y = torch.arange(numGrids).repeat(numGrids, 1).t().view(
            [1, 1, numGrids, numGrids]).type(FloatTensor)
        scale_anchors = FloatTensor([(aw/ratio, ah/ratio)
                                     for aw, ah in self.anchor])
        anchor_x = scale_anchors[:, 0:1].view((1, self.num_anchor, 1, 1))
        anchor_y = scale_anchors[:, 1:2].view((1, self.num_anchor, 1, 1))
<<<<<<< HEAD
 
 
=======

>>>>>>> 1766a63a8eed803fc611c2d8a485a05215da2953
        pred_boxes = FloatTensor(predict[..., :4].shape)
        pred_boxes[..., 0] = x.data + coor_x

        pred_boxes[..., 1] = y.data + coor_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_x
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_y


        if isTrain:
            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=conf.cpu().data,
                pred_cls=clss.cpu().data,
                target=targets.cpu().data,
                anchors=scale_anchors.cpu().data,
                num_anchors=self.num_anchor,
                num_classes=self.num_classes,
                grid_size=numGrids,
                ignore_thres=self.ignore_thres,
                img_dim=self.img_size,
            )

            nProposals = int((conf > 0.75).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = 0
            if nProposals > 0:
                precision = float(nCorrect / nProposals)

            # Handle masks
            mask = Variable(mask.type(BoolTensor))
            conf_mask = Variable(conf_mask.type(BoolTensor))
            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask  ^ mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = (self.bce_loss(conf[conf_mask_false], tconf[conf_mask_false])*self.lambda_noobj
                         + self.bce_loss(conf[conf_mask_true], tconf[conf_mask_true])*self.lambda_obj )
            loss_cls = self.ce_loss(clss[mask], torch.argmax(tcls[mask], 1))
            loss = (loss_x + loss_y + loss_w + loss_h) *self.lambda_coord \
                                                    + loss_conf \
                                                    + loss_cls*self.lambda_class
            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )
        else:
            output = torch.cat(
                (
                    pred_boxes.view(numBatches, -1, 4)*ratio,
                    conf.view(numBatches, -1, 1),
                    clss.view(numBatches, -1, self.num_classes),
                ), -1,
            )

        return output

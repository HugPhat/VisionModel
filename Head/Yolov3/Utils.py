import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def TargetBuilder(PredBoxes,
                  Conf,
                  Class,
                  targets,
                  anchors,
                  n_class,
                  n_grid1,
                  n_grid2,
                  ignore_thresh = 0.5,
                  iou_thresh = 0.5,
                  class_thresh = 0.5,
                  ):
    NB = PredBoxes.size(0) # num  batches
    
    NA = len(anchors) # num anchors
    NG1 = n_grid1
    NG2 = n_grid2
    
    cell_len = 5 + n_class
    
    #mask of objectness score
    obj_mask = torch.zeros(NB, NA, NG1, NG2)
    # For loss calculation
    tx = torch.zeros(NB, NA, NG1, NG2)
    ty = torch.zeros(NB, NA, NG1, NG2)
    tw = torch.zeros(NB, NA, NG1, NG2)
    th = torch.zeros(NB, NA, NG1, NG2)
    
    tconf  = torch.zeros(NB, NA, NG1, NG2)
    tclass = torch.zeros(NB, NA, NG1, NG2, n_class)
    
    N_GroundTruth = 0
    N_Correct = 0
    
    # Go through every batches
    for batch in range(NB):
        # at this iteration , go through this target
        for targ in range(targets.size(1)):
            N_GroundTruth +=1
            # find the grid cell of (x, y)
            ngx = targets[batch, targ, 1] * NG1
            ngy = targets[batch, targ, 2] * NG2
            ngw = targets[batch, targ, 3] * NG1
            ngh = targets[batch, targ, 4] * NG2
        
            ngi = int(ngx) if ngx < NG1 else NG1-1
            ngj = int(ngy) if ngy < NG2 else NG2-1
  
            gt_box = torch.FloatTensor(np.array([0, 0, ngw, ngh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate(
                (np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            obj_mask[batch, anch_ious > ignore_thresh, ngj, ngi] = 1
            
            # best fit anchor
            best_anchor = np.argmax(anch_ious)
            mask[batch,  ngi, ngj]
            
            # Ground truth box
            gt_box = torch.FloatTensor(np.array([ngx, ngy, ngw, ngh])).unsqueeze(0)
            # Best predicted box
            pred_box = PredBoxes[batch, best_anchor, ngi, ngj]
            # 
            tx[batch, best_anchor, ngi, ngj] = ngx -ngi
            ty[batch, best_anchor, ngi, ngj] = ngy -ngj
            
            tw[batch, best_anchor, ngi, ngj] = math.log( ngw /(anchors[best_anchor][0] + 1e-16))
            th[batch, best_anchor, ngi, ngj] = math.log( ngh /(anchors[best_anchor][1] + 1e-16))
            
            tar_label = int(targets[batch, targ, 0])
            
            tclass[batch, best_anchor, ngi, ngj, tar_label] = 1
            tconf[batch, best_anchor, ngi, ngj] = 1
            
            iou = bbox_iou(gt_box, pred_box,x1y1x2y2=False)
            
            pred_label = torch.argmax(Class[batch, best_anchor, ngi, ngj, :])
            score = Conf[batch, best_anchor, ngi, ngj]
            
            if score > class_thresh and \
                pred_label == targets and \
                    iou > iou_thresh:
                N_Correct += 1
    
    return tx, ty, tw, th, tclass, tconf, obj_mask, N_GroundTruth, N_Correct
                        
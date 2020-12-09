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
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2).clamp(min=0)
    inter_rect_y2 = torch.min(b1_y2, b2_y2).clamp(min=0)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou



def build_targets(
    pred_boxes,
    pred_conf,
    pred_cls,
    target,
    anchors,
    num_anchors,
    num_classes,
    grid_size,
    ignore_thres,
    IoU_thresh=0.5,
    score_thresh=0.5
):
    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if pred_boxes.is_cuda else torch.LongTensor

    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    nG = grid_size
    mask = BoolTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = BoolTensor(nB, nA, nG, nG).fill_(1)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tconf = BoolTensor(nB, nA, nG, nG).fill_(0)
    tcls = BoolTensor(nB, nA, nG, nG, nC).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG
            # Get grid box indices
            gi = gx.floor().type(LongTensor)
            gj = gy.floor().type(LongTensor)
            # Get shape of gt box
            gt_box = FloatTensor([0, 0, gw, gh]).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = FloatTensor(torch.cat(
                (FloatTensor(len(anchors), 2).fill_(0).data, FloatTensor(anchors).data), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than ignoring threshold set mask to zero (ignore)
            #
            noobj_mask[b, anch_ious > ignore_thres, gj, gi] = 0  # $warn
            #
            # Find the best matching anchor box
            best_n = torch.argmax(anch_ious)
            # Get ground truth box
            gt_box = FloatTensor([gx, gy, gw, gh]).unsqueeze(0)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            noobj_mask[b, best_n, gj, gi] = 0
            # Coordinates
            tx[b, best_n, gj, gi] = (gx - gi).clamp(EPSILON, 1 - EPSILON)
            ty[b, best_n, gj, gi] = (gy - gj).clamp(EPSILON, 1 - EPSILON)
            # Width and height
            tw[b, best_n, gj, gi] = torch.log(
                gw / (anchors[best_n][0] + 1e-8)).clamp(min=EPSILON)
            th[b, best_n, gj, gi] = torch.log(
                gh / (anchors[best_n][1] + 1e-8)).clamp(min=EPSILON)
            # One-hot encoding of label
            target_label = int(target[b, t, 0].cpu().data)
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi]).cpu().data
            score = pred_conf[b, best_n, gj, gi]
            if iou > IoU_thresh and pred_label == target_label and score.item() > score_thresh:
                nCorrect += 1
    return nGT, nCorrect, mask, noobj_mask, tx, ty, tw, th, tconf, tcls

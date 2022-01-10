import torch
import torch.nn as nn

from faster_rcnn.rpn.utils import box_iou

class AnchorTargetLayer(nn.Module):
    
    def __init__(self, cfg, device):
        super(AnchorTargetLayer, self).__init__()
        self.img_size = cfg.IMG_SIZE
        self.pos_iou_thresh = cfg.RPN_POSITIVE_OVERLAP
        self.neg_iou_thresh = cfg.RPN_NEGATIVE_OVERLAP


    def forward(self, anchors, targets):
        r"""
        Algorithms:
        -----------
            1. Generate all possible anchors over image (Given `Anchors`).
            2. Exclude anchors which are outside the image.
            3. Find the IOU with respect to ground truth bbox.
            4. Assign the labels and location of objects (with respect to the anchor) to each and every anchor.
                a) assign "+"ve label to anchor which has highest iou with ground truth
                b) assign "+"ve label to anchors that has an IoU overlap higher than `self.pos_iou_thresh` with ground-truth box.
                c) assign "-"ve label to anchors that has an IoU overlap less than `self.neg_iou_thresh` with ground-truth box.
        
        """
        inside_idx = torch.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= self.img_size[0]) &
            (anchors[:, 3] <= self.img_size[1])
        )[0]
        anchors = anchors[inside_idx]

        labels_list, matched_gt_boxes_list = [], []
        for target in targets:
            gt_boxes = target["bbox"]
            
            ious = box_iou(gt_boxes, anchors) # size == (M(gt), N(pred))        
            
            # i) the highest iou for each gt_box and its corresponding anchor box
            # ii) find the anchor_boxes which have this max_ious (gt_max_ious)
            gt_max_ious, gt_argmax_ious = ious.max(axis=1)         
            gt_argmax_ious = torch.where(ious.T == gt_max_ious)[0]

            # iii) the highest iou for each anchor box and its corresponding ground truth box
            max_ious, argmax_ious = ious.max(dim=0) # returns max_values, max_index

            below_thresh = max_ious < self.neg_iou_thresh
            between_thresh = (max_ious >= self.neg_iou_thresh) & (max_ious < self.pos_iou_thresh)
            argmax_ious[below_thresh] = -1
            argmax_ious[between_thresh] = -2
            matched_gt_boxes = gt_boxes[argmax_ious.clamp(min=0)]       

            labels = torch.empty((len(inside_idx), ), dtype=torch.int32).fill_(-1)
            labels[gt_argmax_ious] = 1 # [a]
            labels[max_ious >= self.pos_iou_thresh] = 1 # [b]
            labels[max_ious < self.neg_iou_thresh] = 0 # [c]

            labels_list.append(labels)
            matched_gt_boxes_list.append(matched_gt_boxes)

        return labels_list, matched_gt_boxes_list

        



    
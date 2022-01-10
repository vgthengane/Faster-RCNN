import torch
import torch.nn as nn

from faster_rcnn.rpn.utils import nms


class ProposalLayer(nn.Module):
    def __init__(self, cfg, device):
        super(ProposalLayer, self).__init__()
        self.image_size = cfg.IMG_SIZE
        self.pre_nms_top_n = cfg.RPN_PRE_NMS_TOP_N
        self.post_nms_top_n = cfg.RPN_POST_NMS_TOP_N
        self.min_box_size = cfg.RPN_MIN_SIZE
        self.nms_thresh = cfg.RPN_NMS_THRESH

        self.device = device


    def forward(self, batch_proposals, batch_cls_prob):
        r"""
        Arguments:
        ----------
            anchors (List[torch.tensor]): list all possible anchor on each location of feature maps
            cls_scores (torch.tensor): class probability (bg/fg) of size == (batch_size, num_anchors * 2, H, W) 
            bbox_pred (torch.tensor): bounding box prediction of size == (batch_size, num_anchors * 4, H, W)
        """
        # batch_cls_prob = batch_cls_prob.permute(0, 2, 3, 1).contiguous().view(self.batch_size, -1, 2)
        # batch_bbox_pred = batch_bbox_pred.permute(0, 2, 3, 1).contiguous().view(self.batch_size, -1, 4)
        # anchors = anchors.expand(self.batch_size, *(anchors.shape))

        final_proposals = []
        final_cls_prob = []
        # 1. Generate all anchors and apply bbox_pred to obtain proposals via bbox transformations
        # shift in parent file
        
        batch_cls_prob = batch_cls_prob[:, :, 1]
        for proposals, cls_prob in zip(batch_proposals, batch_cls_prob):
            # 2. Clip proposals to image size and filter boxes
            proposals = self._clip_boxes_to_image(proposals)
            keep = self._remove_small_boxes(proposals)
            proposals, cls_prob = proposals[keep], cls_prob[keep]

            # 3. Select top N boxes before applying nms
            pre_top_n = self._get_pre_nms_top_n(cls_prob) # only consize fg i.e == 1
            proposals, cls_prob = proposals[pre_top_n], cls_prob[pre_top_n]

            # # 4. Remove low scoring boxes
            # keep = torch.where(cls_prob >= self.score_thresh)[0]
            # proposals, cls_prob = proposals[keep], cls_prob[keep]

            # 5. Non-maximum suppression, independently done per level
            keep = nms(proposals, cls_prob, self.nms_thresh)

            # 6. select top N boxes after applying nms
            post_top_n = keep[:self.post_nms_top_n]
            proposals, cls_prob = proposals[post_top_n], cls_prob[post_top_n]

            final_proposals.append(proposals)
            final_cls_prob.append(cls_prob)
        return final_cls_prob, final_proposals        
        

    def _get_pre_nms_top_n(self, cls_prob):
        num_anchors = cls_prob.shape[0]
        pre_nms_top_n = min(self.pre_nms_top_n, num_anchors)
        return cls_prob.topk(pre_nms_top_n)[1]

    def _remove_small_boxes(self, boxes):
        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        keep = (ws >= self.min_box_size) & (hs >= self.min_box_size)
        keep = torch.where(keep)[0]
        return keep

    def _clip_boxes_to_image(self, boxes):
        dim = boxes.dim()
        boxes_x = boxes[..., 0::2]
        boxes_y = boxes[..., 1::2]
        height, width = self.image_size

        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)

        clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
        return clipped_boxes.reshape(boxes.shape)

    

    

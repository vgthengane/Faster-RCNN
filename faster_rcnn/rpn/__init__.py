import torch.nn as nn

from faster_rcnn.rpn.rpn_head import RPNHead
from faster_rcnn.rpn.generate_anchors import AnchorGenerator
from faster_rcnn.rpn.proposal_layer import ProposalLayer
from faster_rcnn.rpn.anchors_target_layer import AnchorTargetLayer
from faster_rcnn.rpn.utils import bbox_transform, bbox_transform_inv


class RPN(nn.Module):
    def __init__(self, cfg, device):
        super(RPN, self).__init__()

        self.anchor_generator = AnchorGenerator(device, cfg.ANCHOR_SIZES, cfg.ASPECT_RATIOS)
        self.rpn_head = RPNHead(cfg.OUT_CHANNELS, self.anchor_generator.n_anchors_per_loc())
        self.proposal_layer = ProposalLayer(cfg, device)
        self.anchors_target_layer = AnchorTargetLayer(cfg, device)

    
    def forward(self, images, features, targets):
        batch_size = features.size(0)
        objectness, bbox_pred_deltas = self.rpn_head(features)
        objectness = objectness.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        bbox_pred_deltas = bbox_pred_deltas.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        
        anchors = self.anchor_generator(images, features)
        proposals = bbox_transform_inv(bbox_pred_deltas.detach(), anchors.expand(batch_size, *(anchors.shape)))
        cls_scores, bbox_pred = self.proposal_layer(proposals, objectness.detach())
                
        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.anchors_target_layer(anchors, targets)
            regression_targets = bbox_transform(matched_gt_boxes, anchors.expand(batch_size, *(anchors.shape)))
            cls_loss, reg_loss = self.compute_loss(objectness, bbox_pred_deltas, labels, regression_targets)

            losses = {
                "cls_loss": cls_loss,
                "reg_loss": reg_loss
            }

        return bbox_pred, losses

    
    def compute_loss(self, ):
        return 0, 0


if __name__ == "__main__":
    rpn = RPN()




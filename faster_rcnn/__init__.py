import torch
import torch.nn as nn

from .backbone import VGG16
from .rpn import RPN


class FasterRCNN(nn.Module):
    r"""TO BE DOCUMENTED.
    """
    def __init__(self, cfg, device) -> None:
        super(FasterRCNN, self).__init__()
        self.backbone = VGG16(bn=False)
        self.rpn = RPN(cfg, device)


    def forward(self, images, targets):
        features = self.backbone(images)
        bbox_pred, losses = self.rpn(images, features, targets)
    

        # return cls_scores, bbox_pred







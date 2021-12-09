import torch
import torch.nn as nn

from .vgg16 import VGG16
from .rpn import RPN
from .anchor_gen import AnchorGenerator


class FasterRCNN(nn.Module):
    r"""TO BE DOCUMENTED.
    """
    def __init__(self, out_channels, n_anchors_per_loc) -> None:
        super(FasterRCNN, self).__init__()
        self.backbone = VGG16(bn=False)
        self.rpn = RPN(out_channels, n_anchors_per_loc)
        # self.anchor_generator = AnchorGenerator()  need to implement

    def forward(self, img):
        features = self.backbone(img)
        cls_scores, bbox_pred = self.rpn(features)


        # return cls_scores, bbox_pred







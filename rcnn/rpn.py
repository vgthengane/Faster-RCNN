import torch
import torch.nn as nn
from .layers import Conv2d


class RPN(nn.Module):
    def __init__(self, in_channels, n_anchors_per_loc):
        super(RPN, self).__init__()

        self.n_anchors_per_loc = n_anchors_per_loc
        self.conv1 = Conv2d(in_channels, in_channels, 3, same_padding=True)
        # self.score_conv = Conv2d(in_channels, n_anchors_per_loc * 2, 1, relu=False, same_padding=False)
        self.score_conv = Conv2d(in_channels, n_anchors_per_loc, 1, relu=False, same_padding=False)
        self.bbox_conv = Conv2d(in_channels, n_anchors_per_loc * 4, 1, relu=False, same_padding=False)

    def forward(self, features):
        rpn_conv1 = self.conv1(features)

        # rpn score
        rpn_cls_score = self.score_conv(rpn_conv1)
        # Here I am not sure about adding softmax layer, most of the implmentation does not 
        # softmax layer (eg. PyTorch official implementation)
        # rpn_cls_prob = F.softmax(self.reshape_layer(rpn_cls_score, 2))
        # rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, self.n_anchors_per_loc * 2)

        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)

        # return rpn_cls_prob_reshape, rpn_bbox_pred
        return rpn_cls_score, rpn_bbox_pred

    @staticmethod
    def reshape_layer(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x


if __name__ == "__main__":
    rpn = RPN(512, 9)
    image = torch.randn((1, 512, 50, 50))
    out = rpn(image)
    print(out.shape)
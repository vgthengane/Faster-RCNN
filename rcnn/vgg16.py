import torch
import torch.nn as nn
import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self, bn=False) -> None:
        super(VGG16, self).__init__()
        if bn:
            model = models.vgg16_bn(pretrained=True)
        else:
            model = models.vgg16(pretrained=True)
        feat_layers = list(model.features)
        self.layers = nn.Sequential(*feat_layers[:-1])

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    test_img = torch.randn(1, 3, 800, 800)
    model = VGG16()
    feature_map = model(test_img)
    print(feature_map.shape)
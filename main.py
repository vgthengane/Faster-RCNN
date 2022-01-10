import torch


from faster_rcnn import FasterRCNN
from faster_rcnn.config import cfg


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
model = FasterRCNN(cfg.TRAIN, device)

inputs = torch.randn(2, 3, 800, 800)
targets = [
    {
        "bbox": torch.tensor([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=torch.float32),
        "labels": torch.tensor([6, 8], dtype=torch.int8)
    },

    {
        "bbox": torch.tensor([[40, 60, 600, 500], [450, 280, 670, 750]], dtype=torch.float32),
        "labels": torch.tensor([5, 4], dtype=torch.int8)
    },
]

model(inputs, targets)

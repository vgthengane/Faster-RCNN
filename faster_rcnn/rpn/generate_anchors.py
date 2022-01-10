import torch
import torch.nn as nn
from typing import List


class AnchorGenerator(nn.Module):
    def __init__(self, device, anchor_sizes=[128, 256, 512], aspect_ratios=[0.5, 1.0, 2.0]) -> None:
        super(AnchorGenerator, self).__init__()
        self.anchor_sizes = anchor_sizes
        self.aspect_ratios = aspect_ratios

        self.cell_anchors = self.generate_anchors(anchor_sizes, aspect_ratios, device=device)

    def forward(self, images, features):
        grid_sizes = features.shape[-2:]
        image_size = images.shape[-2:]
        dtype, device = features.dtype, features.device
        strides = [
                torch.tensor(image_size[0] // grid_sizes[0], dtype=torch.int64, device=device),
                torch.tensor(image_size[1] // grid_sizes[1], dtype=torch.int64, device=device),
            ]
        self.set_cell_anchors(dtype, device)
        return self.grid_anchors(grid_sizes, strides)


    def grid_anchors(self, grid_sizes, strides):
        cell_anchors = self.cell_anchors

        grid_height, grid_width = grid_sizes
        stride_height, stride_width = strides
        device = cell_anchors.device

        # For output anchor, compute [x_center, y_center, x_center, y_center]
        shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * stride_width
        shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * stride_height
        # shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        # For every (base anchor, output anchor) pair,
        # offset each zero-centered base anchor by the center of the output anchor.

        return (shifts.view(-1, 1, 4) + cell_anchors.view(1, -1, 4)).reshape(-1, 4)

    def set_cell_anchors(self, dtype, device):
        self.cell_anchors = self.cell_anchors.to(dtype=dtype, device=device)

    def n_anchors_per_loc(self) -> int:
        return len(self.anchor_sizes) * len(self.aspect_ratios)
      
    def generate_anchors(
        self,
        scales: List[int],
        aspect_ratios: List[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()



import torch





# def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
#     dets = np.hstack((pred_boxes,
#                       scores[:, np.newaxis])).astype(np.float32)
#     keep = nms(dets, nms_thresh)
#     if inds is None:
#         return pred_boxes[keep], scores[keep]
#     return pred_boxes[keep], scores[keep], inds[keep]




# from typing import List, Tuple

# import torch
# from torch import Tensor


# class ImageList:
#     """
#     Structure that holds a list of images (of possibly
#     varying sizes) as a single tensor.
#     This works by padding the images to the same size,
#     and storing in a field the original sizes of each image
#     Args:
#         tensors (tensor): Tensor containing images.
#         image_sizes (list[tuple[int, int]]): List of Tuples each containing size of images.
#     """

#     def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]) -> None:
#         self.tensors = tensors
#         self.image_sizes = image_sizes

#     def to(self, device: torch.device) -> "ImageList":
#         cast_tensor = self.tensors.to(device)
#         return ImageList(cast_tensor, self.image_sizes)
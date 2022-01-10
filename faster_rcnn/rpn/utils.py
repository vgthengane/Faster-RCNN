import torch

def nms(dets, scores, nms_thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = torch.sort(scores, descending=True)[1]

    keep = []
    while order.size(0) > 0:
        i = order[0]
        keep.append(i)
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = torch.maximum(torch.tensor(0.0), xx2 - xx1 + 1)
        h = torch.maximum(torch.tensor(0.0), yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = torch.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def box_iou(boxes1, boxes2):
    r"""Adopeted from https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    Arguments:
    ----------
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
    Returns :
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2    
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter

    return inter / union


def bbox_transform_inv(bbox_pred, boxes):
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = bbox_pred[:, :, 0::4]
    dy = bbox_pred[:, :, 1::4]
    dw = bbox_pred[:, :, 2::4]
    dh = bbox_pred[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = bbox_pred.clone()
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def bbox_transform(reference_boxes, proposals):
    boxes_per_image = [len(b) for b in reference_boxes]
    reference_boxes = torch.cat(reference_boxes, dim=0).view(2, -1, 4)
    print(reference_boxes.shape, proposals.shape)

    ex_widths = proposals[:, :, 2] - proposals[:, :, 0] + 1.0
    ex_heights = proposals[:,:, 3] - proposals[:,:, 1] + 1.0
    ex_ctr_x = proposals[:, :, 0] + 0.5 * ex_widths
    ex_ctr_y = proposals[:, :, 1] + 0.5 * ex_heights

    gt_widths = reference_boxes[:, :, 2] - reference_boxes[:, :, 0] + 1.0
    gt_heights = reference_boxes[:, :, 3] - reference_boxes[:, :, 1] + 1.0
    gt_ctr_x = reference_boxes[:, :, 0] + 0.5 * gt_widths
    gt_ctr_y = reference_boxes[:, :, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)


    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh), dim=2)
    print(targets.shape)
    return targets.split(boxes_per_image, 0)



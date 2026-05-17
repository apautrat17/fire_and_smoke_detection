import torch.nn.functional as F

from src.utils.helpers import build_targets

def detection_loss(preds, targets, num_classes=2):

    B, C, H, W = preds.shape

    preds = preds.permute(0, 2, 3, 1)

    pred_box = preds[..., :4]
    pred_obj = preds[..., 4]
    pred_cls = preds[..., 5:]

    obj_t, box_t, cls_t = build_targets(targets, B, H, num_classes, preds.device)

    # objectness (is there an object in this box ?)
    loss_obj = F.binary_cross_entropy_with_logits(pred_obj, obj_t)

    # bbox regression
    loss_box = F.l1_loss(pred_box, box_t.permute(0, 2, 3, 1))

    # classification
    loss_cls = F.binary_cross_entropy_with_logits(pred_cls, cls_t.permute(0, 2, 3, 1))

    return loss_obj + loss_box + loss_cls
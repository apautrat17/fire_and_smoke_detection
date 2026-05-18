import torch
import torch.nn.functional as F

from src.utils.helpers import build_targets


def detection_loss(
    preds, targets, num_classes=2, lambda_obj=2.0, lambda_box=5.0, lambda_cls=2.0
):
    B, C, H, W = preds.shape
    preds = preds.permute(0, 2, 3, 1)

    pred_box = preds[..., :4]
    pred_obj = preds[..., 4]
    pred_cls = preds[..., 5:]

    obj_t, box_t, cls_t = build_targets(targets, B, H, num_classes, preds.device)

    pos_mask = obj_t.bool()
    neg_mask = ~pos_mask

    loss_obj_pos = (
        F.binary_cross_entropy_with_logits(pred_obj[pos_mask], obj_t[pos_mask])
        if pos_mask.any()
        else torch.tensor(0.0, device=preds.device)
    )

    loss_obj_neg = (
        F.binary_cross_entropy_with_logits(pred_obj[neg_mask], obj_t[neg_mask])
        if neg_mask.any()
        else torch.tensor(0.0, device=preds.device)
    )

    loss_obj = 2.0 * loss_obj_pos + 1.0 * loss_obj_neg

    if pos_mask.any():
        loss_box = F.l1_loss(
            torch.sigmoid(pred_box[pos_mask]), box_t.permute(0, 2, 3, 1)[pos_mask]
        )
        loss_cls = F.binary_cross_entropy_with_logits(
            pred_cls[pos_mask], cls_t.permute(0, 2, 3, 1)[pos_mask]
        )
    else:
        loss_box = torch.tensor(0.0, device=preds.device)
        loss_cls = torch.tensor(0.0, device=preds.device)

    return lambda_obj * loss_obj + lambda_box * loss_box + lambda_cls * loss_cls

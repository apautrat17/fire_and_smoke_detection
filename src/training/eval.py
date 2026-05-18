import torch
from tqdm import tqdm
from src.training.losses import detection_loss
from src.training.metrics import compute_metrics
from src.utils.helpers import decode_predictions, decode_targets


def evaluate(model, loader, device):

    model.eval()

    total_loss = 0

    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0
    epoch_ap = 0

    validation_bar = tqdm(loader, desc="Evaluating", leave=False)

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)

            loss = detection_loss(preds, targets)

            total_loss += loss.item()

            pred_boxes = decode_predictions(preds)
            gt_boxes = decode_targets(targets)

            precision, recall, f1, ap = compute_metrics(
                pred_boxes, gt_boxes, iou_thresh=0.3
            )

            epoch_precision += precision
            epoch_recall += recall
            epoch_f1 += f1
            epoch_ap += ap

            validation_bar.update(1)
            validation_bar.set_postfix(
                {
                    "loss": total_loss / (validation_bar.n + 1),
                    "precision": epoch_precision / (validation_bar.n + 1),
                    "recall": epoch_recall / (validation_bar.n + 1),
                    "f1": epoch_f1 / (validation_bar.n + 1),
                    "mAP": epoch_ap / (validation_bar.n + 1),
                    "pred_boxes": len(pred_boxes),
                }
            )

    validation_bar.close()

    n = len(loader)

    return {
        "loss": total_loss / n,
        "precision": epoch_precision / n,
        "recall": epoch_recall / n,
        "f1": epoch_f1 / n,
        "mAP": epoch_ap / n,
    }

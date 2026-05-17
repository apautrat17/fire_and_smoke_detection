import torch
from src.training.losses import detection_loss
from src.training.metrics import compute_metrics
from src.utils.helpers import decode_predictions, decode_targets

def evaluate(model, loader, device):

    model.eval()

    total_loss = 0

    epoch_precision = 0
    epoch_recall = 0
    epoch_f1 = 0

    with torch.no_grad():

        for images, targets in loader:

            images = images.to(device)
            targets = targets.to(device)

            preds = model(images)

            loss = detection_loss(preds, targets)

            total_loss += loss.item()

            pred_boxes = decode_predictions(preds)
            gt_boxes = decode_targets(targets)

            precision, recall, f1 = compute_metrics(
                pred_boxes,
                gt_boxes
            )

            epoch_precision += precision
            epoch_recall += recall
            epoch_f1 += f1

    n = len(loader)

    return {
        "loss": total_loss / n,
        "precision": epoch_precision / n,
        "recall": epoch_recall / n,
        "f1": epoch_f1 / n
    }
import time
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

from src.data.dataloader import create_dataloader
from src.models.base_model import FireSmokeModel
from src.main import logger, config
from src.utils.helpers import decode_predictions, decode_targets, save_model
from src.training.losses import detection_loss
from src.training.metrics import compute_metrics
from src.training.eval import evaluate


def train(EPOCHS, BATCH_SIZE, LR, SHUFFLE, NUM_WORKERS, DEVICE):

    # Dataloaders
    train_loader = create_dataloader(
        images_dir=config.processed_train_images_path,
        labels_dir=config.processed_train_labels_path,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS,
    )
    val_loader = create_dataloader(
        images_dir=config.processed_test_images_path,
        labels_dir=config.processed_test_labels_path,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    yolo_model = YOLO("yolov8m.pt")

    yolo_nn = yolo_model.model

    feature_maps = {}

    def save_features(module, inputs, output):
        feature_maps["x"] = output

    hook_layer = yolo_nn.model[9]
    hook_layer.register_forward_hook(save_features)

    class YOLOFeatureBackbone(torch.nn.Module):
        def __init__(self, yolo_model, hook_key="x"):
            super().__init__()
            self.yolo_model = yolo_model
            self.hook_key = hook_key

        def forward(self, x):
            _ = self.yolo_model(x)
            return feature_maps[self.hook_key]

    backbone = YOLOFeatureBackbone(yolo_nn)
    model = FireSmokeModel(backbone, in_channels=576).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    run_name = f"yolov8m_finetune_{time.strftime('%Y%m%d-%H%M%S')}"

    writer = SummaryWriter(log_dir=f"./runs/training/{run_name}")

    training_pbar = tqdm(total=EPOCHS, desc="Training epochs", unit="epoch")

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        epoch_precision = 0
        epoch_recall = 0
        epoch_f1 = 0
        epoch_ap = 0

        batch_pbar = tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{EPOCHS}",
            unit="batch",
            leave=False,
        )

        for batch_idx, (images, targets) in enumerate(train_loader):
            global_step = epoch * len(train_loader) + batch_idx

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(images)

            loss = detection_loss(preds, targets, num_classes=2)

            with torch.no_grad():

                pred_boxes = decode_predictions(preds)
                gt_boxes = decode_targets(targets)

                precision, recall, f1, ap = compute_metrics(
                    pred_boxes, gt_boxes, iou_thresh=0.5
                )

                epoch_precision += precision
                epoch_recall += recall
                epoch_f1 += f1
                epoch_ap += ap

                writer.add_scalar(
                    "Train/loss (compute per batch)", loss.item(), global_step
                )
                writer.add_scalar("Train/precision", precision, global_step)
                writer.add_scalar("Train/recall", recall, global_step)
                writer.add_scalar("Train/f1", f1, global_step)
                writer.add_scalar("Train/mAP", ap, global_step)
                writer.add_scalar("Train/num_pred_boxes", len(pred_boxes), global_step)

                if batch_idx % 50 == 0:
                    writer.flush()  # Ensure logs are written to disk

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            batch_pbar.update(1)
            batch_pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "prec": f"{precision:.4f}",
                    "rec": f"{recall:.4f}",
                    "f1": f"{f1:.4f}",
                    "ap": f"{ap:.4f}",
                }
            )

        batch_pbar.close()

        # Compute average loss and log metrics
        n_batches = len(train_loader)

        avg_loss = total_loss / n_batches
        avg_precision = epoch_precision / n_batches
        avg_recall = epoch_recall / n_batches
        avg_f1 = epoch_f1 / n_batches
        avg_ap = epoch_ap / n_batches

        writer.add_scalar("Train/avg_loss (compute per epoch)", avg_loss, epoch)

        val_metrics = evaluate(model, val_loader, DEVICE)

        save_dir = "./runs/checkpoints"
        # save checkpoint at each epoch with val_f1 in filename
        save_model(
            model,
            epoch,
            save_dir,
            optimizer,
            val_metrics["f1"],
            f"epoch_{epoch}_f1_{val_metrics['f1']:.4f}",
        )

        training_pbar.update(1)

        writer.add_scalar("Val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("Val/precision", val_metrics["precision"], epoch)
        writer.add_scalar("Val/recall", val_metrics["recall"], epoch)
        writer.add_scalar("Val/f1", val_metrics["f1"], epoch)
        writer.add_scalar("Val/mAP", val_metrics["mAP"], epoch)

        training_pbar.set_postfix(
            {
                "loss": f"{avg_loss:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
                "prec": f"{avg_precision:.4f}",
                "rec": f"{avg_recall:.4f}",
                "f1": f"{avg_f1:.4f}",
                "val_f1": f"{val_metrics['f1']:.4f}",
                "map": f"{avg_ap:.4f}",
            }
        )

        logger.info(
            f"Epoch {epoch}: Loss={avg_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}"
        )
    training_pbar.close()
    writer.close()

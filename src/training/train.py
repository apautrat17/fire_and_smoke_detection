import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO

from src.data.dataloader import create_dataloader
from src.models.base_model import FireSmokeModel
from src.main import logger, config
from src.utils.helpers import decode_predictions, decode_targets
from src.training.losses import detection_loss
from src.training.metrics import compute_metrics

EPOCHS = config.epochs
BATCH_SIZE = config.batch_size
LR = config.learning_rate
SHUFFLE = config.shuffle_data
NUM_WORKERS = config.num_workers
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train(EPOCHS, BATCH_SIZE, LR, SHUFFLE, NUM_WORKERS, DEVICE):
    train_loader = create_dataloader(
        images_dir=config.processed_train_images_path,
        labels_dir=config.processed_train_labels_path,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS
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

        batch_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch", leave=False)

        for batch_idx, (images, targets) in enumerate(train_loader):
            global_step = epoch * len(train_loader) + batch_idx

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(images)

            loss = detection_loss(preds, targets, num_classes = 2)

            with torch.no_grad():

                pred_boxes = decode_predictions(preds)

                gt_boxes = decode_targets(targets)

                precision, recall, f1 = compute_metrics(
                    pred_boxes,
                    gt_boxes,
                    iou_thresh=0.5
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            batch_pbar.update(1)
            batch_pbar.set_postfix({
                "loss": loss.item(),
                "prec": precision,
                "rec": recall,
                "f1": f1
            })

            writer.add_scalar("Metrics/precision", precision, global_step)
            writer.add_scalar("Metrics/recall", recall, global_step)
            writer.add_scalar("Metrics/f1", f1, global_step)
            writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], global_step)

        batch_pbar.close()

        # Compute average loss and log metrics
        avg_loss = total_loss / len(train_loader)
        
        # Log metrics to TensorBoard
        writer.add_scalar("Loss/train", avg_loss, epoch)
        
        training_pbar.update(1)
        training_pbar.set_postfix({
            "loss": avg_loss,
            "prec": precision,
            "rec": recall,
            "f1": f1
        })
        logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}")
    training_pbar.close()
    writer.close()
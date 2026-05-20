import time
from torch.utils.tensorboard import SummaryWriter
from src.main import config, logger


class Callbacks:

    def __init__(self):
        self.run_name = f"yolov8m_finetune_{time.strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(
            log_dir=f"./{config.tensorboard_log_dir}/{self.run_name}"
        )

        self.state = {
            "batch_count": 0,
            "epoch_loss": 0.0,
        }

    # On batches: total loss, lr, box/cls/dfl losses if available
    def on_train_batch_end(self, trainer):
        epoch = trainer.epoch
        nb = len(trainer.train_loader)
        self.state["batch_count"] += 1
        global_step = epoch * nb + self.state["batch_count"]

        # trainer.loss = scalar 0-dim, batch total loss
        loss_val = trainer.loss.item()
        self.state["epoch_loss"] += loss_val

        self.writer.add_scalar("Train/loss", loss_val, global_step)
        self.writer.add_scalar(
            "Train/learning_rate",
            trainer.optimizer.param_groups[0]["lr"],
            global_step,
        )

        # [box, cls, dfl] with tloss (running mean epoch)
        if (
            hasattr(trainer, "tloss")
            and trainer.tloss is not None
            and trainer.tloss.numel() == 3
        ):
            self.writer.add_scalar(
                "Train/box_loss", float(trainer.tloss[0]), global_step
            )
            self.writer.add_scalar(
                "Train/cls_loss", float(trainer.tloss[1]), global_step
            )
            self.writer.add_scalar(
                "Train/dfl_loss", float(trainer.tloss[2]), global_step
            )

        if self.state["batch_count"] % 50 == 0:
            self.writer.flush()

    # Epoch train end: avg loss, lr

    def on_train_epoch_start(self, trainer):
        self.state["batch_count"] = 0
        self.state["epoch_loss"] = 0.0

    def on_train_epoch_end(self, trainer):
        epoch = trainer.epoch
        n = max(self.state["batch_count"], 1)
        avg_loss = self.state["epoch_loss"] / n

        self.writer.add_scalar("Train/avg_loss", avg_loss, epoch)
        logger.info(
            f"Epoch {epoch}: avg_loss={avg_loss:.4f}, "
            f"learning_rate={trainer.optimizer.param_groups[0]['lr']:.6f}"
        )

    # Epoch val end: precision, recall, mAP50, mAP50-95
    def on_fit_epoch_end(self, trainer):
        """
        Déclenché après la validation. trainer.metrics est un dict avec les clés :
            'metrics/precision(B)', 'metrics/recall(B)',
            'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
        """
        epoch = trainer.epoch
        metrics = trainer.metrics

        try:
            p = float(metrics.get("metrics/precision(B)", 0))
            r = float(metrics.get("metrics/recall(B)", 0))
            map50 = float(metrics.get("metrics/mAP50(B)", 0))
            map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0))
            f1 = 2 * p * r / (p + r + 1e-8)

            self.writer.add_scalar("Val/precision", p, epoch)
            self.writer.add_scalar("Val/recall", r, epoch)
            self.writer.add_scalar("Val/f1", f1, epoch)
            self.writer.add_scalar("Val/mAP50", map50, epoch)
            self.writer.add_scalar("Val/mAP50-95", map50_95, epoch)

            logger.info(
                f"Epoch {epoch} val: P={p:.4f}, R={r:.4f}, "
                f"F1={f1:.4f}, mAP50={map50:.4f}, mAP50-95={map50_95:.4f}"
            )
        except Exception as e:
            logger.warning(f"[TensorBoard on_fit_epoch_end] {e}")

        self.writer.flush()

    def on_train_end(self, trainer):
        self.writer.close()
        logger.info(
            f"Training finished. TensorBoard → {config.tensorboard_log_dir}/{self.run_name}"
        )

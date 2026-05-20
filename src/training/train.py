from ultralytics import YOLO
from src.utils.callbacks import Callbacks
from src.main import config


def train():

    # Callbacks
    callbacks = Callbacks()
    model = YOLO("yolov8m.pt")
    model.add_callback("on_train_epoch_start", callbacks.on_train_epoch_start)
    model.add_callback("on_train_batch_end", callbacks.on_train_batch_end)
    model.add_callback("on_train_epoch_end", callbacks.on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", callbacks.on_fit_epoch_end)
    model.add_callback("on_train_end", callbacks.on_train_end)

    # Training
    model.train(
        data=config.data_yaml_path,
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch_size,
        optimizer=config.optimizer,
        lr0=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        cos_lr=config.cos_lr,
        mosaic=config.mosaic_prob,
        close_mosaic=config.close_mosaic_epoch,
        mixup=config.mixup_prob,
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        degrees=config.degrees,
        translate=config.translate,
        scale=config.scale,
        fliplr=config.fliplr,
        workers=config.num_workers,
        device=config.device,
        project=config.tensorboard_log_dir,
        name=callbacks.run_name,
        save=True,
        verbose=False,
    )

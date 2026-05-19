import torch
from src.main import logger, config
from src.data.pipeline import ProcessPipeline
from src.training.train import train

if __name__ == "__main__":

    logger.info("Starting project...")

    if config.process_data:
        process_pipeline = ProcessPipeline(
            resize=config.resize, resize_size=config.resize_size
        )
        process_pipeline.run(
            raw_images_path=config.raw_train_images_path,
            raw_labels_path=config.raw_train_labels_path,
            processed_images_path=config.processed_train_images_path,
            processed_labels_path=config.processed_train_labels_path,
        )
        process_pipeline.run(
            raw_images_path=config.raw_test_images_path,
            raw_labels_path=config.raw_test_labels_path,
            processed_images_path=config.processed_test_images_path,
            processed_labels_path=config.processed_test_labels_path,
        )

    EPOCHS = config.epochs
    BATCH_SIZE = config.batch_size
    LR = config.learning_rate
    SHUFFLE = config.shuffle_data
    NUM_WORKERS = config.num_workers
    DEVICE = config.device  # "cuda" if torch.cuda.is_available() else "cpu"
    WEIGHT_DECAY = config.weight_decay
    WARMUP_EPOCHS = config.warmup_epochs
    COS_LR = config.cos_lr
    CLOSE_MOSAIC = config.close_mosaic

    if config.train == True:

        train(
            EPOCHS,
            BATCH_SIZE,
            LR,
            SHUFFLE,
            NUM_WORKERS,
            DEVICE,
            WEIGHT_DECAY,
            WARMUP_EPOCHS,
            COS_LR,
            CLOSE_MOSAIC,
        )

    if config.inference_image == True:
        from src.models.base_model import create_fire_smoke_model
        from src.visualization.visualize import visualize_predictions
        from src.inference.predict_image import (
            load_model_from_checkpoint,
            predict_image,
        )
        from src.utils.helpers import load_image

        model = create_fire_smoke_model()
        infer_model = load_model_from_checkpoint(
            model, "runs/checkpoints/epoch_81_f1_0.0128.pt", DEVICE
        )

        image = load_image("data/dataset/raw/test/images/WEB10751.jpg")

        pred_boxes = predict_image(infer_model, image, config.resize_size, DEVICE)

        visualize_predictions(image, pred_boxes)

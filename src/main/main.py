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

    if config.train == True:

        EPOCHS = config.epochs
        BATCH_SIZE = config.batch_size
        LR = config.learning_rate
        SHUFFLE = config.shuffle_data
        NUM_WORKERS = config.num_workers
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        train(EPOCHS, BATCH_SIZE, LR, SHUFFLE, NUM_WORKERS, DEVICE)

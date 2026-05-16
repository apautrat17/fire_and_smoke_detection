from src.main import logger, config
from src.data.pipeline import ProcessPipeline

if __name__ == "__main__":

    logger.info("Starting project...")

    if config.process_data:
        process_pipeline = ProcessPipeline(resize=config.resize, resize_size=config.resize_size)
        process_pipeline.run(
            raw_images_path=config.raw_train_images_path,
            raw_labels_path=config.raw_train_labels_path,
            processed_images_path=config.processed_train_images_path,
            processed_labels_path=config.processed_train_labels_path
        )
        process_pipeline.run(
            raw_images_path=config.raw_test_images_path,
            raw_labels_path=config.raw_test_labels_path,
            processed_images_path=config.processed_test_images_path,
            processed_labels_path=config.processed_test_labels_path
        )
from ultralytics import YOLO
from src.main import logger, config
from src.training.train import train
from src.inference.predict_image import predict_image
from src.inference.predict_video import predict_video

if __name__ == "__main__":

    logger.info("Starting project...")

    if config.do_train == True:
        train()

    if config.do_inference_image == True:

        infer_model = YOLO(config.model_path)

        predict_image(
            infer_model,
            image_path=config.image_name,
            folder_path=config.image_folder_path,
            conf_threshold=config.conf_threshold,
            random=config.random_image,
            stop_after_one=config.stop_after_one,
        )

    if config.do_inference_video == True:

        infer_model = YOLO(
            "runs/detect/runs/training/yolov8m_finetune_20260520-113028/weights/best.pt"
        )

        predict_video(
            infer_model,
            video_path=config.video_path,
            conf_threshold=config.conf_threshold,
            show=config.show_video,
            save=config.save_video,
        )

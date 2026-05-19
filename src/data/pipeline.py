import os
import cv2
import numpy as np
from tqdm import tqdm
from src.main import logger
from src.utils.helpers import load_image, load_label_as_array
from src.data.preprocess import letterbox, unletterbox


class ProcessPipeline:

    def __init__(self, resize: bool = False, resize_size: int = 640):
        self.resize = resize
        self.resize_size = resize_size

    def process_image_and_label(
        self, image_path: str, label_path: str
    ) -> tuple[np.ndarray, np.ndarray]:

        image = load_image(image_path)
        labels = load_label_as_array(label_path)

        if self.resize:
            image, labels = letterbox(image, labels, resize_size=self.resize_size)

        return image, labels

    def unprocess_label(
        self, image: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        if self.resize:
            labels = unletterbox(image, labels)

        return labels

    def run(
        self,
        raw_images_path: str = None,
        raw_labels_path: str = None,
        processed_images_path: str = None,
        processed_labels_path: str = None,
    ) -> None:

        # Check that all images have a corresponding label
        image_files = [f for f in os.listdir(raw_images_path)]
        label_files = [f for f in os.listdir(raw_labels_path)]

        nb_images = len(image_files)
        nb_labels = len(label_files)

        for image_file in image_files:
            if image_file.replace(".jpg", ".txt") not in label_files:
                logger.critical(
                    f"Mismatch between number of images ({nb_images}) and labels ({nb_labels})"
                )
                raise ValueError("Mismatch between number of images and labels")

        logger.info(
            f"Processing {nb_images} images and labels from {raw_images_path} and {raw_labels_path} and saving to {processed_images_path} and {processed_labels_path}..."
        )

        # Progress bar for processing images and labels
        pbar = tqdm(total=nb_images, desc="Processing images and labels", unit="image")

        for image_file in image_files:
            # image = load_image(f"{raw_images_path}/{image_file}")
            # labels = load_label_as_array(
            #     f"{raw_labels_path}/{image_file.replace('.jpg', '.txt')}"
            # )

            # if self.resize:
            #     image, labels = letterbox(image, labels, resize_size=self.resize_size)

            image, labels = self.process_image_and_label(
                f"{raw_images_path}/{image_file}",
                f"{raw_labels_path}/{image_file.replace('.jpg', '.txt')}",
            )

            # Save processed image and labels
            output_image_path = f"{processed_images_path}/{image_file}"
            output_label_path = (
                f"{processed_labels_path}/{image_file.replace('.jpg', '.txt')}"
            )
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
            cv2.imwrite(f"{processed_images_path}/{image_file}", image)
            np.savetxt(
                f"{processed_labels_path}/{image_file.replace('.jpg', '.txt')}",
                labels,
                fmt="%f",
            )
            pbar.update(1)
        logger.info(
            f"Processed {nb_images} images and labels from {raw_images_path} and {raw_labels_path} and saved to {processed_images_path} and {processed_labels_path}"
        )

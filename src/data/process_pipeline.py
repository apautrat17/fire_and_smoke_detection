import os
import cv2
import numpy as np
from src.main import logger
from src.utils.helpers import load_image, load_label_as_array

class ProcessPipeline:

    def __init__(self, resize: bool = False, resize_size: int = 640):
        self.resize = resize
        self.resize_size = resize_size

    def resize_image(self, image: np.ndarray, new_w: int, new_h: int, new_size: int, pad_x: int, pad_y: int) -> np.ndarray:
        # resize image
        image_resized = cv2.resize(image, (new_w, new_h))

        image_padded = cv2.copyMakeBorder(
            image_resized,
            pad_y, new_size - new_h - pad_y,
            pad_x, new_size - new_w - pad_x,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        return image_padded

    def resize_labels(self, labels, w: int, h: int, new_size: int, pad_x: int, pad_y: int, scale: float) -> np.ndarray:
        new_labels = []

        for cls, x_c, y_c, bw, bh in labels:

            # YOLO normalized -> pixel
            x = x_c * w
            y = y_c * h
            bw_px = bw * w
            bh_px = bh * h

            # scale
            x *= scale
            y *= scale
            bw_px *= scale
            bh_px *= scale

            # add padding
            x += pad_x
            y += pad_y

            # back to normalized (new image size)
            x_c_new = x / new_size
            y_c_new = y / new_size
            bw_new = bw_px / new_size
            bh_new = bh_px / new_size

            new_labels.append([cls, x_c_new, y_c_new, bw_new, bh_new])

        return np.array(new_labels, dtype=np.float32)

    def letterbox(self, image, labels):
        """
        Resize + padding YOLO-style (letterbox)
        labels must be in YOLO format: [class, x_center, y_center, w, h]
        all normalized [0,1]
        """

        h, w = image.shape[:2]
        new_size = self.resize_size

        # scale to fit new size while keeping aspect ratio
        scale = min(new_size / w, new_size / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        # padding
        pad_x = (new_size - new_w) // 2
        pad_y = (new_size - new_h) // 2

        image_padded = self.resize_image(image, new_w, new_h, new_size, pad_x, pad_y)
        labels_resized = self.resize_labels(labels, w, h, new_size, pad_x, pad_y, scale)        

        return image_padded, labels_resized

    def normalize_pixels(self, image):
        return image.astype('float32') / 255.0

    def run(self, raw_images_path: str = None, raw_labels_path: str = None, processed_images_path: str = None, processed_labels_path: str = None) -> None:
        # Check dataset length
        image_files = [f for f in os.listdir(raw_images_path)]
        label_files = [f for f in os.listdir(raw_labels_path)]

        nb_images = len(image_files)
        nb_labels = len(label_files)

        if nb_images != nb_labels:
            logger.critical(f"Mismatch between number of images ({nb_images}) and labels ({nb_labels})")
            raise ValueError("Mismatch between number of images and labels")
        
        logger.info(f"Processing {nb_images} images and labels from {raw_images_path} and {raw_labels_path} and saving to {processed_images_path} and {processed_labels_path}...")
        
        for image_file in image_files:
            image = load_image(f"{raw_images_path}/{image_file}")
            labels = load_label_as_array(f"{raw_labels_path}/{image_file.replace('.jpg', '.txt')}")

            if self.resize:
                image, labels = self.letterbox(image, labels)

            # Save processed image and labels
            output_image_path = f"{processed_images_path}/{image_file}"
            output_label_path = f"{processed_labels_path}/{image_file.replace('.jpg', '.txt')}"
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
            cv2.imwrite(f"{processed_images_path}/{image_file}", image)
            np.savetxt(f"{processed_labels_path}/{image_file.replace('.jpg', '.txt')}", labels, fmt="%f")
        
        logger.info(f"Processed {nb_images} images and labels from {raw_images_path} and {raw_labels_path} and saved to {processed_images_path} and {processed_labels_path}")
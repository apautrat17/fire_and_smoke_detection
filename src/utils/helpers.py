import cv2
import numpy as np
from src.main import logger

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
            logger.critical(f"Cannot load image: {image_path}")
            raise ValueError(f"Cannot load image: {image_path}")
    return img

def load_label_as_array(label_path):
    with open(label_path, 'r') as f:
        label_lines = f.readlines()
    
    labels = []

    for line in label_lines:
        class_id, x_center, y_center, width, height = map(float, line.split())
        labels.append((class_id, x_center, y_center, width, height))
    return np.array(labels)
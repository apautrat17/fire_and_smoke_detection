import cv2
import yaml
import numpy as np
from src.config.config_parser import Config

def load_image(image_path):
    return cv2.imread(image_path)

def load_label_as_array(label_path):
    with open(label_path, 'r') as f:
        label_lines = f.readlines()
    
    labels = []

    for line in label_lines:
        class_id, x_center, y_center, width, height = map(float, line.split())
        labels.append((class_id, x_center, y_center, width, height))
    return np.array(labels)

def load_config(config_path: str) -> Config:
    """
    Load configuration parameters from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Config: An instance of the Config class initialized with the loaded parameters.
    """
    with open(config_path, "r") as f:
        configuration = yaml.safe_load(f)

    return Config(**configuration)


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("true", "1", "yes")


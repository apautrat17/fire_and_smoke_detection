import cv2
import torch


def basic_transforms(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return image

import cv2
import numpy as np
import torch
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

def build_targets(targets, batch_size, grid_size, num_classes, device):

    B = batch_size

    obj_target = torch.zeros((B, grid_size, grid_size), device=device)
    box_target = torch.zeros((B, 4, grid_size, grid_size), device=device)
    cls_target = torch.zeros((B, num_classes, grid_size, grid_size), device=device)

    if targets.numel() == 0:
        return obj_target, box_target, cls_target

    for t in targets:

        b, cls, x, y, w, h = t
        i = int(x * grid_size)
        j = int(y * grid_size)

        obj_target[int(b), j, i] = 1
        box_target[int(b), :, j, i] = torch.tensor([x, y, w, h], device=device)
        cls_target[int(b), int(cls), j, i] = 1

    return obj_target, box_target, cls_target

def decode_predictions(preds):
    """
    preds: [B, 5+C, H, W]
    return list of bbox per image
    """

    B = preds.shape[0]

    boxes = []

    preds = preds.permute(0, 2, 3, 1)

    for b in range(B):

        pred = preds[b]

        obj = torch.sigmoid(pred[..., 4])

        mask = obj > 0.5

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):

                if mask[i, j]:

                    x, y, w, h = pred[i, j, :4]
                    cls = torch.argmax(pred[i, j, 5:])

                    boxes.append([b, cls.item(), x.item(), y.item(), w.item(), h.item()])

    return boxes

def decode_targets(targets):

    gt_boxes = []

    for t in targets:

        b, cls, x, y, w, h = t.tolist()

        gt_boxes.append([int(b), int(cls), x, y, w, h])

    return gt_boxes
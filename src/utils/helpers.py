import os

import cv2
import numpy as np
import torch
from src.training.metrics import nms
from src.main import logger


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        logger.critical(f"Cannot load image: {image_path}")
        raise ValueError(f"Cannot load image: {image_path}")
    return img


def load_label_as_array(label_path):
    with open(label_path, "r") as f:
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
        i = min(int(x * grid_size), grid_size - 1)
        j = min(int(y * grid_size), grid_size - 1)

        obj_target[int(b), j, i] = 1
        box_target[int(b), :, j, i] = torch.tensor([x, y, w, h], device=device)
        cls_target[int(b), int(cls), j, i] = 1

    return obj_target, box_target, cls_target


def decode_predictions(preds, conf_thresh=0.2, nms_thresh=0.5):

    B = preds.shape[0]
    preds = preds.permute(0, 2, 3, 1)

    all_boxes = []

    for b in range(B):
        pred = preds[b]
        boxes = []
        scores = []
        classes = []

        H, W, _ = pred.shape

        for i in range(H):
            for j in range(W):

                obj_score = torch.sigmoid(pred[i, j, 4]).item()

                if obj_score < conf_thresh:
                    continue

                cls_logits = pred[i, j, 5:]
                cls = torch.argmax(cls_logits).item()

                x, y, w, h = torch.sigmoid(pred[i, j, :4]).tolist()

                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                boxes.append([x, y, w, h])
                scores.append(obj_score)
                classes.append(cls)

        keep = nms(boxes, scores, nms_thresh)

        for idx in keep:

            x, y, w, h = boxes[idx]
            cls = classes[idx]

            all_boxes.append([b, cls, scores[idx], x, y, w, h])

    return all_boxes


def decode_targets(targets):

    gt_boxes = []

    for t in targets:

        b, cls, x, y, w, h = t.tolist()

        gt_boxes.append([int(b), int(cls), x, y, w, h])

    return gt_boxes


def save_model(model, epoch, save_dir, optimizer, val_f1, checkpoint_name):

    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1": val_f1,
        },
        f"{save_dir}/{checkpoint_name}.pt",
    )

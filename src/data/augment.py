import cv2
import torch
import random
import numpy as np
import albumentations as A
from src.utils.helpers import yolo_to_pascal, pascal_to_yolo


def basic_transforms(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return image


# def random_translate(image, boxes, translate=0.1):

#     h, w = image.shape[:2]

#     tx = random.uniform(-translate, translate) * w
#     ty = random.uniform(-translate, translate) * h

#     M = np.array([
#         [1, 0, tx],
#         [0, 1, ty]
#     ], dtype=np.float32)

#     image = cv2.warpAffine(image, M, (w, h))

#     boxes[:, [0, 2]] += tx
#     boxes[:, [1, 3]] += ty

#     return image, boxes


def mixup(image1, boxes1, image2, boxes2, alpha):

    mixed = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)

    boxes = np.concatenate([boxes1, boxes2], axis=0).astype(np.float32)

    return mixed, boxes


def mosaic(images, boxes_list, img_size=832):

    yc = img_size // 2
    xc = img_size // 2

    mosaic_img = np.full((img_size, img_size, 3), 114, dtype=np.uint8)

    positions = [(0, 0), (0, xc), (yc, 0), (yc, xc)]

    final_boxes = []

    for i, (img, boxes) in enumerate(zip(images, boxes_list)):

        img = cv2.resize(img, (xc, yc), interpolation=cv2.INTER_LINEAR)

        y, x = positions[i]

        mosaic_img[y : y + yc, x : x + xc] = img

        boxes = boxes.copy()

        boxes[:, [0, 2]] *= 0.5
        boxes[:, [1, 3]] *= 0.5

        boxes[:, [0, 2]] += x
        boxes[:, [1, 3]] += y

        final_boxes.append(boxes)

    if len(final_boxes) == 0:
        return mosaic_img, np.zeros((0, 5), dtype=np.float32)

    final_boxes = np.concatenate(final_boxes, axis=0)

    return mosaic_img, final_boxes


def basic_augmentations(image, boxes, **kwargs):
    """Apply basic augmentations to the image and boxes using Albumentations.
    Augmentations applied:
    - Horizontal flip
    - ShiftScaleRotate (shift, scale, rotate)
    - RandomBrightnessContrast
    - HueSaturationValue
    - GaussianBlur
    """

    h, w = image.shape[:2]

    pascal_boxes = yolo_to_pascal(boxes, w, h)

    transform = A.Compose(
        [
            A.HorizontalFlip(p=kwargs.get("horizontal_flip_prob", 0.5)),
            A.ShiftScaleRotate(
                shift_limit=kwargs.get("shift_limit", 0.1),
                scale_limit=kwargs.get("scale_limit", 0.5),
                rotate_limit=kwargs.get("rotate_limit", 10),
                border_mode=kwargs.get("border_mode", 0),
                p=kwargs.get("shift_scale_rotate_prob", 0.5),
            ),
            A.RandomBrightnessContrast(
                brightness_limit=kwargs.get("brightness_limit", 0.2),
                contrast_limit=kwargs.get("contrast_limit", 0.2),
                p=kwargs.get("random_brightness_contrast_prob", 0.3),
            ),
            A.HueSaturationValue(
                hue_shift_limit=kwargs.get("hsv_h_shift_limit", 5),
                sat_shift_limit=kwargs.get("hsv_s_shift_limit", 20),
                val_shift_limit=kwargs.get("hsv_v_shift_limit", 20),
                p=kwargs.get("hsv_prob", 0.3),
            ),
            A.GaussianBlur(
                blur_limit=tuple(kwargs.get("gaussian_blur_limit", (3, 5))),
                p=kwargs.get("gaussian_blur_prob", 0.2),
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=kwargs.get("bbox_min_visibility", 0.2),
        ),
    )

    # Apply transforms on image and labels
    transformed = transform(
        image=image,
        bboxes=pascal_boxes[:, :4],
        class_labels=pascal_boxes[:, 4],
    )

    image = transformed["image"]

    boxes = pascal_to_yolo(
        transformed["bboxes"],
        w,
        h,
        transformed["class_labels"],
    )

    return image, boxes

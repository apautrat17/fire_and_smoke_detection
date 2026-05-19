import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.helpers import load_image
from src.data.collate import collate_fn
from src.data.augment import basic_transforms, basic_augmentations, mosaic, mixup
from src.main import aumgentation_parameters, config


class FireSmokeDataset(Dataset):

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        current_epoch: int = 0,
        close_mosaic: int = None,
        use_augmentations: bool = False,
    ):

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.current_epoch = current_epoch
        self.close_mosaic = close_mosaic
        self.use_augmentations = use_augmentations

        self.image_files = sorted(
            [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
        )

    def __len__(self):
        return len(self.image_files)

    def transform(self, image, label):
        image = basic_transforms(image)
        image, label = (
            basic_augmentations(image, label, **aumgentation_parameters)
            if self.use_augmentations
            else (image, label)
        )
        return image, label

    def load_single(self, idx):

        image_name = self.image_files[idx]

        image_path = os.path.join(self.images_dir, image_name)

        label_path = os.path.join(
            self.labels_dir, image_name.rsplit(".", 1)[0] + ".txt"
        )

        # Load image
        image = load_image(image_path)

        # Load labels
        if os.path.getsize(label_path) > 0:
            labels = np.loadtxt(label_path, ndmin=2)
        else:
            # In case the label file is empty, return an empty array with the correct shape
            labels = np.zeros((0, 5), dtype=np.float32)

        return image, labels

    def apply_mosaic(self, idx):

        indices = [idx] + list(np.random.choice(len(self.image_files), 3))

        images = []
        boxes_list = []

        for i in indices:
            img, boxes = self.load_single(i)
            images.append(img)
            boxes_list.append(boxes)

        image, boxes = mosaic(images, boxes_list, img_size=config.resize_size)

        return image, boxes

    def apply_mixup(self, idx):

        idx2 = np.random.randint(0, len(self.image_files))

        img1, boxes1 = self.load_single(idx)
        img2, boxes2 = self.load_single(idx2)

        alpha = np.random.uniform(
            tuple(aumgentation_parameters.get("mixup_limits", [0.4, 0.6]))
        )

        image, boxes = mixup(img1, boxes1, img2, boxes2, alpha)

        return image, boxes

    def __getitem__(self, idx):

        image, labels = self.load_single(idx)

        disable_augmentation = True if not self.use_augmentations else False

        # Check if we should disable mosaic
        disable_mosaic = (
            self.close_mosaic is not None and self.current_epoch >= self.close_mosaic
        )

        p_mosaic = (
            0.0
            if (disable_mosaic or disable_augmentation)
            else aumgentation_parameters.get("mosaic_prob", 1.0)
        )
        p_mixup = (
            0.0
            if disable_augmentation
            else aumgentation_parameters.get("mixup_prob", 0.1)
        )

        r = np.random.random()

        # Mosaic
        if r < p_mosaic:
            image, labels = self.apply_mosaic(idx)

        # Mixup
        elif r < p_mosaic + p_mixup:
            image, labels = self.apply_mixup(idx)

        # Basic transforms and augmentations
        else:
            image, labels = self.transform(image, labels)

        return image, labels


def create_dataloader(
    images_dir: str,
    labels_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    current_epoch: int = 0,
    close_mosaic: int = None,
    use_augmentations: bool = False,
):

    dataset = FireSmokeDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        current_epoch=current_epoch,
        close_mosaic=close_mosaic,
        use_augmentations=use_augmentations,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return loader

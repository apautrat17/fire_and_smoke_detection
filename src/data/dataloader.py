import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.helpers import load_image
from src.data.collate import collate_fn
from src.data.augment import basic_transforms


class FireSmokeDataset(Dataset):

    def __init__(self, images_dir: str, labels_dir: str):

        self.images_dir = images_dir
        self.labels_dir = labels_dir

        self.image_files = sorted(
            [f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
        )

    def __len__(self):
        return len(self.image_files)

    def transform(self, image):
        return basic_transforms(image)

    def __getitem__(self, idx):

        image_name = self.image_files[idx]

        image_path = os.path.join(self.images_dir, image_name)

        label_path = os.path.join(
            self.labels_dir, image_name.rsplit(".", 1)[0] + ".txt"
        )

        # Load image
        image = load_image(image_path)
        image = self.transform(image)

        # Load labels
        if os.path.getsize(label_path) > 0:
            labels = np.loadtxt(label_path, ndmin=2)
        else:
            # In case the label file is empty, return an empty array with the correct shape
            labels = np.zeros((0, 5), dtype=np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        return image, labels


def create_dataloader(
    images_dir: str,
    labels_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
):

    dataset = FireSmokeDataset(images_dir=images_dir, labels_dir=labels_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return loader

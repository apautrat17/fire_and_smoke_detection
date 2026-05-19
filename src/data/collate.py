import torch
import numpy as np


def collate_fn(batch):
    """
    Custom collate function to handle batches of images and labels. (Each image has a different number of labels)

    Args:
        batch: A list of tuples (image, labels) where:
            - image is a tensor of shape (C, H, W)
            - labels is a tensor of shape (num_labels, 5) with [class, x_center, y_center, width, height]

    Returns:
        images: A tensor of shape (batch_size, C, H, W)
        targets: A tensor of shape (total_labels, 6) with [batch_idx, class, x_center, y_center, width, height]
    """

    images = []
    targets = []

    for i, (image, labels) in enumerate(batch):

        # Ensure image is a torch tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        elif not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        images.append(image)

        # Convert labels (numpy) to tensor
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).float()
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)

        # If no labels, skip
        if labels.numel() == 0:
            continue

        # add batch index (float to match other targets)
        batch_idx = torch.full((labels.shape[0], 1), i, dtype=torch.float32)
        labels = torch.cat([batch_idx, labels], dim=1)

        targets.append(labels)

    images = torch.stack(images)

    if len(targets) == 0:
        targets = torch.zeros((0, 6), dtype=torch.float32)
    else:
        targets = torch.cat(targets, dim=0)

    return images, targets

import torch


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

        images.append(image)

        if labels.numel() == 0:
            continue

        # add batch index
        batch_idx = torch.full((labels.shape[0], 1), i)

        labels = torch.cat([batch_idx, labels], dim=1)

        targets.append(labels)

    images = torch.stack(images)
    targets = torch.cat(targets, dim=0)

    return images, targets

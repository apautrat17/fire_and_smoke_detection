import cv2
import numpy as np

def resize_image(image: np.ndarray, new_w: int, new_h: int, new_size: int, pad_x: int, pad_y: int) -> np.ndarray:
    # resize image
    image_resized = cv2.resize(image, (new_w, new_h))

    image_padded = cv2.copyMakeBorder(
        image_resized,
        pad_y, new_size - new_h - pad_y,
        pad_x, new_size - new_w - pad_x,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    return image_padded

def resize_labels(labels: np.ndarray, w: int, h: int, new_size: int, pad_x: int, pad_y: int, scale: float) -> np.ndarray:
    new_labels = []

    for cls, x_c, y_c, bw, bh in labels:

        # YOLO normalized -> pixel
        x = x_c * w
        y = y_c * h
        bw_px = bw * w
        bh_px = bh * h

        # scale
        x *= scale
        y *= scale
        bw_px *= scale
        bh_px *= scale

        # add padding
        x += pad_x
        y += pad_y

        # back to normalized (new image size)
        x_c_new = x / new_size
        y_c_new = y / new_size
        bw_new = bw_px / new_size
        bh_new = bh_px / new_size

        new_labels.append([cls, x_c_new, y_c_new, bw_new, bh_new])

    return np.array(new_labels, dtype=np.float32)

def letterbox(image, labels, resize_size=640):
    """
    Resize + padding YOLO-style (letterbox)
    labels must be in YOLO format: [class, x_center, y_center, w, h]
    all normalized [0,1]
    """

    h, w = image.shape[:2]
    new_size = resize_size

    # scale to fit new size while keeping aspect ratio
    scale = min(new_size / w, new_size / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    # padding
    pad_x = (new_size - new_w) // 2
    pad_y = (new_size - new_h) // 2

    image_padded = resize_image(image, new_w, new_h, new_size, pad_x, pad_y)
    labels_resized = resize_labels(labels, w, h, new_size, pad_x, pad_y, scale)        

    return image_padded, labels_resized

def normalize_pixels(image):
    return image.astype('float32') / 255.0
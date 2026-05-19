from pydantic import BaseModel


class Config(BaseModel):
    """
    Configuration class: this class will be used to load all configuration parameters from a YAML file.
    """

    # Data paths
    raw_test_images_path: str
    raw_test_labels_path: str
    raw_train_images_path: str
    raw_train_labels_path: str
    processed_test_images_path: str
    processed_test_labels_path: str
    processed_train_images_path: str
    processed_train_labels_path: str

    # Data processing / augmentation
    resize: bool
    resize_size: int

    horizontal_flip_prob: float

    shift_limit: float
    scale_limit: float
    rotate_limit: int
    border_mode: int
    shift_scale_rotate_prob: float

    brightness_limit: float
    contrast_limit: float
    random_brightness_contrast_prob: float

    hsv_h_shift_limit: int
    hsv_s_shift_limit: int
    hsv_v_shift_limit: int
    hsv_prob: float

    gaussian_blur_limit: list
    gaussian_blur_prob: float

    bbox_min_visibility: float

    mixup_prob: float
    mixup_limits: list

    mosaic_prob: float

    # Model
    nb_in_channels: int
    num_classes: int
    conf_threshold: float
    nms_threshold: float
    iou_threshold: float

    # Training
    learning_rate: float
    epochs: int
    batch_size: int
    shuffle_data: bool
    num_workers: int
    weight_decay: float
    warmup_epochs: int
    cos_lr: bool
    close_mosaic: int

    device: str

    # Main pipeline
    process_data: bool
    train: bool
    inference_image: bool
    inference_video: bool

    # Logger
    log_folder_path: str
    log_level: str

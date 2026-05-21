from pydantic import BaseModel


class Config(BaseModel):
    """
    Configuration class: this class will be used to load all configuration parameters from a YAML file.
    """

    # Data processing / augmentation
    data_yaml_path: str

    imgsz: int

    mosaic_prob: float
    close_mosaic_epoch: int

    mixup_prob: float

    hsv_h: float
    hsv_s: float
    hsv_v: float

    degrees: int
    translate: float
    scale: float
    fliplr: float

    # Training
    learning_rate: float
    epochs: int
    batch_size: int
    optimizer: str
    shuffle_data: bool
    num_workers: int
    weight_decay: float
    warmup_epochs: int
    cos_lr: bool

    device: str

    # Logger
    log_folder_path: str
    log_level: str

    # Tensorboard
    tensorboard_log_dir: str

    # Inference
    model_path: str
    conf_threshold: float

    # Image inference
    image_folder_path: str
    image_name: str
    test_image_path: str
    random_image: bool
    stop_after_one: bool

    # Video inference
    video_path: str
    show_video: bool
    save_video: bool

    # Main pipeline
    do_train: bool
    do_inference_image: bool
    do_inference_video: bool

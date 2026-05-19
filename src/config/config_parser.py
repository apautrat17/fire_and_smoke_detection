from pydantic import BaseModel


class Config(BaseModel):
    """
    Configuration class: this class will be used to load all configuration parameters from a YAML file.
    """

    # Data processing
    raw_test_images_path: str
    raw_test_labels_path: str
    raw_train_images_path: str
    raw_train_labels_path: str
    processed_test_images_path: str
    processed_test_labels_path: str
    processed_train_images_path: str
    processed_train_labels_path: str
    resize: bool
    resize_size: int

    # Model
    nb_in_channels: int
    num_classes: int

    # Training
    learning_rate: float
    epochs: int
    batch_size: int
    shuffle_data: bool
    num_workers: int

    device: str

    # Main pipeline
    process_data: bool
    train: bool
    inference_image: bool
    inference_video: bool

    # Logger
    log_folder_path: str
    log_level: str

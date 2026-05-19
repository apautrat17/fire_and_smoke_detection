import time
from src.utils.logger import get_config, Logger

config = get_config()

LOG_FILE_PATH = (
    f"{config.log_folder_path}/logs_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
)
logger = Logger(level=config.log_level, log_file=LOG_FILE_PATH).get()

aumgentation_parameters = {
    "horizontal_flip_prob": config.horizontal_flip_prob,
    "shift_limit": config.shift_limit,
    "scale_limit": config.scale_limit,
    "rotate_limit": config.rotate_limit,
    "border_mode": config.border_mode,
    "shift_scale_rotate_prob": config.shift_scale_rotate_prob,
    "brightness_limit": config.brightness_limit,
    "contrast_limit": config.contrast_limit,
    "random_brightness_contrast_prob": config.random_brightness_contrast_prob,
    "hsv_h_shift_limit": config.hsv_h_shift_limit,
    "hsv_s_shift_limit": config.hsv_s_shift_limit,
    "hsv_v_shift_limit": config.hsv_v_shift_limit,
    "hsv_prob": config.hsv_prob,
    "gaussian_blur_limit": config.gaussian_blur_limit,
    "gaussian_blur_prob": config.gaussian_blur_prob,
    "bbox_min_visibility": config.bbox_min_visibility,
    "mixup_prob": config.mixup_prob,
    "mixup_limits": config.mixup_limits,
    "mosaic_prob": config.mosaic_prob,
}

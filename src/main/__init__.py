import time
from src.utils.logger import get_config, Logger

config = get_config()

LOG_FILE_PATH = (
    f"{config.log_folder_path}/logs_{time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
)
logger = Logger(level=config.log_level, log_file=LOG_FILE_PATH).get()

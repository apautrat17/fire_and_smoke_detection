import os
import yaml
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from src.config.config_parser import Config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="src/config/config.yaml")

    # overrides optionnels
    parser.add_argument("--log-level", dest="log_level", type=str)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_inference_image", action="store_true")
    parser.add_argument("--do_inference_video", action="store_true")
    parser.add_argument("--image_path", dest="test_image_path", type=str)
    parser.add_argument("--video_path", dest="video_path", type=str)

    return parser.parse_args()


def load_config(config_path: str) -> Config:
    """
    Load configuration parameters from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Config: An instance of the Config class initialized with the loaded parameters.
    """
    with open(config_path, "r") as f:
        configuration = yaml.safe_load(f)

    return Config(**configuration)


def override_config(config_dict: dict, args) -> dict:
    for key, value in vars(args).items():
        if key == "config":
            continue
        if isinstance(value, bool) and value is False:
            continue
        if value is not None:
            config_dict[key] = value
    return config_dict


def get_config() -> Config:
    args = parse_args()
    if sum([args.do_train, args.do_inference_image, args.do_inference_video]) > 1:
        raise ValueError("Choose only one mode among train / image / video")
    config_dict = load_config(args.config).model_dump()
    config_dict = override_config(config_dict, args)
    return Config(**config_dict)


class Logger:
    """Logger class to set up and retrieve a configured logger instance."""

    class ConsoleFormatter(logging.Formatter):
        # ANSI escape codes for colors
        COLORS = {
            logging.DEBUG: "\033[90m",  # gris
            logging.INFO: "\033[97m",  # blanc
            logging.WARNING: "\033[93m",  # jaune
            logging.ERROR: "\033[38;5;208m",  # orange (256 colors mode)
            logging.CRITICAL: "\033[91m",  # rouge
        }
        RESET = "\033[0m"

        def format(self, record):
            record.relativepath = os.path.relpath(record.pathname)

            color = self.COLORS.get(record.levelno, self.RESET)
            message = super().format(record)

            return f"{color}{message}{self.RESET}"

    class LogFileFormatter(logging.Formatter):

        def format(self, record):
            record.relativepath = os.path.relpath(record.pathname)

            return super().format(record)

    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
            except Exception:
                self.handleError(record)

    def __init__(self, level="INFO", name="Logger", log_file=None):
        """
        Initialize the Logger with the specified logging level, name, and optional log file.

        Args:
            level (str): The logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'NOTSET').
            name (str): The name of the logger.
            log_file (str | None): Path to a log file. If provided, logs also go to that file.
        """

        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # Avoid duplicate handlers if Logger is constructed multiple times
        if not self.logger.handlers:

            base_log_format = "[%(asctime)s - %(levelname)s] - %(message)s (%(relativepath)s:%(lineno)d)"

            console_formatter = self.ConsoleFormatter(
                base_log_format,
                datefmt="%H:%M:%S",
            )

            # Console handler
            sh = (
                self.TqdmLoggingHandler()
            )  # Safe if tqdm pbar is used, otherwise it behaves like a normal StreamHandler
            sh.setFormatter(console_formatter)
            self.logger.addHandler(sh)

            # File handler
            if log_file:
                # Ensure parent directory exists (best-effort)
                try:
                    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                file_formatter = self.LogFileFormatter(
                    base_log_format,
                    datefmt="%H:%M:%S",
                )
                fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
                fh.setFormatter(file_formatter)
                self.logger.addHandler(fh)

    def get(self):
        """Retrieve the configured logger instance."""
        return self.logger

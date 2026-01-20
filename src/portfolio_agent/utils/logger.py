import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os

from portfolio_agent.config.settings import get_settings


def get_logger(name: str = "portfolio_agent") -> logging.Logger:
    """
    Create or retrieve a logger with consistent formatting.
    Usage: logger = get_logger(__name__)
    """

    logger = logging.getLogger(name)

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    settings = get_settings()

    # ---- Define log file path ----
    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "portfolio_agent.log"

    # ---- File handler ----
    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=5
    )
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_format)

    # ---- Console handler ----
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    )

    # ---- Attach handlers ----
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

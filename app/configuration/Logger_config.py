import logging
import os
from datetime import datetime
import colorlog

def setup_logger(name='MarketPredictor', log_level='INFO', log_file=None):
    """
    Setup comprehensive logging system with colored console output.
    Called once (e.g., in main).
    """
    logger = logging.getLogger(name)

    # Se il logger ha già handler, non ricrearlo
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper()))
    logger.propagate = False

    # Formatter per file
    file_formatter = logging.Formatter(
        '%(asctime)s | %(filename)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Formatter colorato per console
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s | %(filename)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    # Console handler
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    # File handler (se richiesto)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

import logging
import os
from datetime import datetime
import colorlog


def setup_logger(name='MarketPredictor', log_level='INFO', log_file=None):
    """
    Setup comprehensive logging system with colored console output
    """
    # Crea logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Rimuovi eventuali handler esistenti
    logger.handlers = []

    # Formatter per file (senza colori)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Formatter colorato per la console
    color_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s | %(name)s | %(levelname)s | %(message)s',
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

    # File handler (solo se richiesto)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# Logger di default
logger = setup_logger()

if __name__ == "__main__":
    # Esempio di log
    logger.debug("Debug di prova")
    logger.info("Operazione completata")
    logger.warning("Attenzione: file mancante")
    logger.error("Errore di connessione")
    logger.critical("Errore critico di sistema")

import logging
from pathlib import Path
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
Path("logs").mkdir(exist_ok=True)

def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    fh = logging.FileHandler(log_file)
    fh.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(fh)

    return logger


train_logger = setup_logger('train_logger', 'logs/train.log')
val_logger = setup_logger('val_logger', 'logs/val.log')


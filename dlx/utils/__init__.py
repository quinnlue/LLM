"""
Utility functions and helpers for DLX.

This module contains utility functions including backend management,
logging, and learning rate scheduling.
"""

from .backend import xp
from .logger import setup_logger, train_logger, val_logger
from .lr_scheduler import LRScheduler

__all__ = [
    "xp",
    "setup_logger",
    "train_logger", 
    "val_logger",
    "LRScheduler"
]

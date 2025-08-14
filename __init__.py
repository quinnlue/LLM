"""
GPT-1 Training Project

A personal project for training GPT-1 style models.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main components for easy access
from .training.model import Model
from .preprocess.dataloader import DataLoader, Dataset
from .tokenizer.tokenizer import tokenizer

__all__ = [
    "Model",
    "DataLoader", 
    "Dataset",
    "tokenizer"
]

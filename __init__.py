"""
LLM Training Project

A personal project for training LLM models.
"""

__version__ = "0.1.0"
__author__ = "Quinn Lue"

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

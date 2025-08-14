"""
DLX - Deep Learning eXperiments

A deep learning framework with custom tensor operations, neural network modules,
and transformer implementations.
"""

__version__ = "0.1.0"
__author__ = "DLX Team"

# Import main components for easy access
from .nn.tensor import Tensor
from .nn.module import Module
from .nn.losses import MeanSquaredError, BinaryCrossEntropyWithLogits, CrossEntropyWithLogits
from .nn.optim import AdamW, SGD
from .modules.linear import Linear
from .modules.layernorm import LayerNorm
from .modules.transformer import Transformer
from .utils.backend import xp

__all__ = [
    "Tensor",
    "Module", 
    "MeanSquaredError",
    "BinaryCrossEntropyWithLogits", 
    "CrossEntropyWithLogits",
    "AdamW",
    "SGD",
    "Linear",
    "LayerNorm",
    "Transformer",
    "xp"
]

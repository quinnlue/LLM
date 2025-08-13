import torch
import random
try:
    import cupy as xp
    if not xp.cuda.is_available():
        print("CUDA not available")
        raise ImportError("CUDA not available")
except (ImportError, ModuleNotFoundError):
    import numpy as xp



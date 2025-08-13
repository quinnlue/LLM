import torch
import random
try:
    raise ImportError("CUDA not available")
    import cupy as xp
    if not xp.cuda.is_available():
        raise ImportError("CUDA not available")
except (ImportError, ModuleNotFoundError):
    import numpy as xp

def set_seed(seed=42):
    xp.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)


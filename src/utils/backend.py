try:
    import cupy as xp
    if not xp.cuda.is_available():
        raise ImportError("CUDA not available")
except (ImportError, ModuleNotFoundError):
    import numpy as xp



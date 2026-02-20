"""
Normalize audio to RMS level
"""

import numpy as np
from pathlib import Path

from src.utils.general_utils import _load_config

CONFIG_PATH = (Path(__file__).parent / "config.yaml").resolve()

_CONFIG = _load_config(CONFIG_PATH)
if _CONFIG:
    print(f"normalize: Loaded config from {CONFIG_PATH}")
else:
    print(f"normalize: Warning: Could not load config from {CONFIG_PATH}")
_DEFAULT_TARGET_DBFS = _CONFIG.get("target_dbfs", -14.0)

def rms_normalize(x, target_dbfs=_DEFAULT_TARGET_DBFS, eps=1e-12):
    """
    Normalize audio to RMS level

    x: np.ndarray, shape (n,) or (n, ch), float in [-1, 1] recommended
    target_dbfs: target RMS level in dBFS (0 dBFS == full-scale sine RMS=1.0)
    """
    x = np.asarray(x, dtype=np.float32)

    # RMS over all samples (and channels if present)
    rms = np.sqrt(np.mean(x**2) + eps)

    target_rms = 10 ** (target_dbfs / 20.0)  # dBFS -> linear RMS
    gain = target_rms / max(rms, eps)

    y = x * gain
    return y, gain

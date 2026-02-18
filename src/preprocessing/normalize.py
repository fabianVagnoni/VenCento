"""
Normalize audio to RMS level
"""

import numpy as np

def rms_normalize(x, target_dbfs=-14.0, eps=1e-12):
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

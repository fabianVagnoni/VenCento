"""
Utils for audio processing and security checks
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

def ensure_mono_float32(audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Ensures mono float32 numpy array in [-1, 1] (recommended).
    Accepts:
      - (n,) mono
      - (n, ch) or (ch, n) -> will average channels
    """
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = np.asarray(audio)

    audio_np = audio_np.astype(np.float32, copy=False)

    # Handle channel shapes
    if audio_np.ndim == 1:
        mono = audio_np
    elif audio_np.ndim == 2:
        # If shape is (ch, n), transpose to (n, ch)
        if audio_np.shape[0] <= 8 and audio_np.shape[1] > audio_np.shape[0]:
            audio_np = audio_np.T
        mono = audio_np.mean(axis=1)
    else:
        raise ValueError(f"Audio must be 1D or 2D array, got shape={audio_np.shape}")

    return mono


def numpy_to_torch_1d(audio_mono: np.ndarray) -> torch.Tensor:
    """
    Converts mono numpy float32 array to 1D torch tensor.
    """
    return torch.from_numpy(audio_mono).float().contiguous()


def slice_audio(
    audio_mono: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    """
    Returns audio segment [start_sec, end_sec] from mono numpy audio.
    """
    start_i = max(0, int(round(start_sec * sr)))
    end_i = min(len(audio_mono), int(round(end_sec * sr)))
    return audio_mono[start_i:end_i]
"""IO helpers for raw PCM s16le"""

from pathlib import Path

import numpy as np

from src.utils.audio_helpers import ensure_mono_float32


def load_pcm_s16le(pcm_path: str, *, channels: int = 1) -> np.ndarray:
    """
    Load raw PCM (s16le) into float32 in [-1, 1].
    Returns mono float32 shape (n,).
    """
    pcm_path = str(pcm_path)
    data_i16 = np.fromfile(pcm_path, dtype=np.int16)
    if data_i16.size == 0:
        raise ValueError(f"PCM file is empty: {pcm_path}")

    # If channels > 1, interpret as interleaved (n_frames * channels)
    if channels > 1:
        if data_i16.size % channels != 0:
            raise ValueError(
                f"PCM length {data_i16.size} not divisible by channels={channels}: {pcm_path}"
            )
        data_i16 = data_i16.reshape(-1, channels)

    # int16 -> float32 [-1,1]
    audio = (data_i16.astype(np.float32) / 32768.0)
    audio = ensure_mono_float32(audio)  # ensures mono float32 (n,)
    return audio


def save_pcm_s16le(audio_mono_f32: np.ndarray, out_path: str) -> None:
    """
    Save mono float32 [-1,1] as raw PCM s16le.
    """
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(audio_mono_f32, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -1.0, 1.0)

    # float32 [-1,1] -> int16
    x_i16 = (x * 32767.0).astype(np.int16)
    x_i16.tofile(out_path)

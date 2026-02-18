"""
Convert PCM to WAV
"""

import numpy as np
import soundfile as sf

def pcm16_bytes_to_wav(pcm_bytes, out_wav, sr=16000, channels=1):
    """
    Convert PCM to WAV

    Args:
        pcm_bytes: PCM data as bytes
        out_wav: Output WAV file path
        sr: Sample rate
        channels: Number of channels

    Returns:
        None
    """
    x = np.frombuffer(pcm_bytes, dtype=np.int16)

    # reshape if interleaved multi-channel
    if channels > 1:
        x = x.reshape(-1, channels)

    # convert int16 -> float32 in [-1, 1]
    y = (x.astype(np.float32) / 32768.0)

    sf.write(out_wav, y, sr, subtype="PCM_16")


def normalized_float_to_wav(y, out_wav, sr=16000, subtype="PCM_16", peak=0.999):
    """
    Writes a WAV you can listen to that matches what the model gets.

    Args:
        y: normalized audio (float), shape (n,) or (n, ch)
        out_wav: Output WAV file path
        sr: Sample rate
        subtype: WAV subtype
        peak: Peak level

    Returns:
        None
    """
    y = np.asarray(y, dtype=np.float32)

    # Optional safety: prevent clipping in the file you write
    m = float(np.max(np.abs(y))) if y.size else 0.0
    if m > 1.0:
        y = y * (peak / m)

    sf.write(out_wav, y, sr, subtype=subtype)

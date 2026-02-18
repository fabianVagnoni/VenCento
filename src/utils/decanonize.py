"""
Convert PCM to WAV
"""

import numpy as np
import soundfile as sf

def pcm16_bytes_to_wav(pcm_bytes, out_wav, sr=16000, channels=1):
    """
    Convert PCM to WAV
    """
    x = np.frombuffer(pcm_bytes, dtype=np.int16)

    # reshape if interleaved multi-channel
    if channels > 1:
        x = x.reshape(-1, channels)

    # convert int16 -> float32 in [-1, 1]
    y = (x.astype(np.float32) / 32768.0)

    sf.write(out_wav, y, sr, subtype="PCM_16")

if __name__ == "__main__":
    pcm16_bytes_to_wav(b"", "out.wav", sr=16000, channels=1)
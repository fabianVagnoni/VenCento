from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.preprocessing.canonize import canonize
from src.preprocessing.normalize import rms_normalize
from src.preprocessing.vad_cleaning import VadParams, get_vad_model, vad_clean_audio

from src.utils.io_helpers import load_pcm_s16le, save_pcm_s16le


# -------------------------
# Params
# -------------------------

@dataclass(frozen=True)
class PreprocessParams:
    raw_data_path: str               # root folder for raw audio files
    processed_data_path: str         # root folder for outputs (pcms + chunks)
    sr: int = 16000
    channels: int = 1
    vad_params: VadParams = VadParams()
    sample_fmt: str = "s16le"        # must match canonize(... sample_fmt=...)
    target_dbfs: float = -14.0       # passed to rms_normalize


# -------------------------
# Main orchestrator function
# -------------------------

def preprocess_audio(
    audio_path: str,
    params: PreprocessParams,
    *,
    vad_model=None,
) -> List[str]:
    """
    Pipeline:
      1) canonize input -> canonical PCM on disk
      2) load PCM -> float32
      3) RMS normalize to target_dbfs
      4) VAD clean -> get (start,end) segments
      5) slice segments from normalized audio and save each chunk to disk as PCM

    Returns:
      list of written chunk paths
    """
    in_path = Path(audio_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # ---- (1) Canonize to raw PCM on disk (s16le, sr, channels)
    # canonize(...) signature comes from your file :contentReference[oaicite:4]{index=4}
    processed_root = Path(params.processed_data_path)
    canonical_pcm_path = processed_root / "canonical" / f"{in_path.stem}.pcm"

    canonize(
        wav_path=str(in_path),
        pcm_path=str(canonical_pcm_path),
        sr=params.sr,
        channels=params.channels,
        sample_fmt=params.sample_fmt,
    )

    # ---- (2) Load canonical PCM
    audio = load_pcm_s16le(str(canonical_pcm_path), channels=params.channels)

    # ---- (3) Safety checks: correct type, finite, non-zero
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio after ensure_mono_float32, got shape={audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Audio contains NaN/Inf: {canonical_pcm_path}")
    if np.max(np.abs(audio)) <= 0.0:
        raise ValueError(f"Audio is all zeros: {canonical_pcm_path}")

    # ---- (4) Normalize RMS to target dBFS using your normalize module :contentReference[oaicite:5]{index=5}
    audio_norm, _gain = rms_normalize(audio, target_dbfs=params.target_dbfs)

    # ---- (5) VAD cleaning: returns list of (start_sec,end_sec) :contentReference[oaicite:6]{index=6}
    model = vad_model or get_vad_model(cache=True)  # :contentReference[oaicite:7]{index=7}
    clip_ranges: List[Tuple[float, float]] = vad_clean_audio(
        audio_norm,
        params=params.vad_params,
        vad_model=model,
        return_audio_clips=False,
    )

    # ---- (6) Convert segments to PCM chunks + save
    out_dir = processed_root / "chunks" / in_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths: List[str] = []
    for idx, (start_s, end_s) in enumerate(clip_ranges):
        # slice in samples
        s0 = int(round(start_s * params.sr))
        s1 = int(round(end_s * params.sr))
        s0 = max(0, min(s0, len(audio_norm)))
        s1 = max(0, min(s1, len(audio_norm)))

        if s1 <= s0:
            continue

        chunk = audio_norm[s0:s1]

        # Name must be same as original audio path with index of chunk
        # e.g. foo.wav -> foo_000.pcm, foo_001.pcm, ...
        chunk_path = out_dir / f"{in_path.stem}_{idx:03d}.pcm"
        save_pcm_s16le(chunk, str(chunk_path))
        chunk_paths.append(str(chunk_path))

    return chunk_paths


# # -------------------------
# # Example usage (optional)
# # -------------------------

# def preprocess_one_file_example():
#     params = PreprocessParams(
#         raw_data_path="data/raw",
#         processed_data_path="data/processed",
#         sr=16000,
#         channels=1,
#         vad_params=VadParams(sampling_rate=16000),
#         sample_fmt="s16le",
#         target_dbfs=-14.0,
#     )

#     wav = r"data\raw\youtube__los-tipos-de-acento-maracucho-segun-nandatayo__CxUupicxwhY__seg000__60000ms-180000ms.wav"
#     chunk_paths = preprocess_audio(wav, params)
#     print("Wrote chunks:")
#     for p in chunk_paths:
#         print("  ", p)
 

# if __name__ == "__main__":
#     preprocess_one_file_example()

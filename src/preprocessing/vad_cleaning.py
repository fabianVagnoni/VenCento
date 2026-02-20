"""
vad_cleaning.py

Silero VAD + segment packing into 2–6s clips.

Purpose:
- This module is ONLY meant to be called by a preprocessing orchestrator.
- It provides modular functions and a single high-level function:
    vad_clean_audio(...)

Expected orchestrator responsibilities:
- Decide input/output paths
- Save audio clips if needed
- Update / write JSONL manifests
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from pathlib import Path

from silero_vad import load_silero_vad, get_speech_timestamps

from src.utils.audio_helpers import ensure_mono_float32, numpy_to_torch_1d, slice_audio
from src.utils.general_utils import _load_config

CONFIG_PATH = (Path(__file__).parent / "config.yaml").resolve()

_CONFIG = _load_config(CONFIG_PATH)
if _CONFIG:
    print(f"vad_cleaning: Loaded config from {CONFIG_PATH}")
    if "max_sec" in _CONFIG:
         print(f"vad_cleaning: Found max_sec={_CONFIG['max_sec']} in config")
else:
    print(f"vad_cleaning: Warning: Could not load config from {CONFIG_PATH}")

# -------------------------
# Types
# -------------------------

@dataclass(frozen=True)
class VadParams:
    sampling_rate: int = _CONFIG.get("sampling_rate", 16000)               # Silero supports 8000 or 16000
    threshold: float = _CONFIG.get("threshold", 0.5)                   # 0..1
    min_speech_sec: float = _CONFIG.get("min_speech_sec", 0.25)             # drop tiny bursts
    min_sec: float = _CONFIG.get("min_sec", 2.0)                     # final clip min duration
    max_sec: float = _CONFIG.get("max_sec", 6.0)                     # final clip max duration
    max_merge_gap_sec: float = _CONFIG.get("max_merge_gap_sec", 0.10)          # merge segments separated by <= this
    return_seconds: bool = _CONFIG.get("return_seconds", True)              # internal convenience
    padding: float = _CONFIG.get("padding", 0.15)


# -------------------------
# Model loading
# -------------------------

_VAD_MODEL: Optional[torch.nn.Module] = None

def get_vad_model(cache: bool = True) -> torch.nn.Module:
    """
    Loads Silero VAD model (CPU). Optionally caches it globally.
    """
    global _VAD_MODEL
    if cache and _VAD_MODEL is not None:
        return _VAD_MODEL

    model = load_silero_vad()  # CPU model by default
    if cache:
        _VAD_MODEL = model
    return model



# -------------------------
# VAD + chunking logic
# -------------------------

def run_silero_vad(
    audio_torch_1d: torch.Tensor,
    model: torch.nn.Module,
    sr: int,
    threshold: float,
    min_speech_sec: float,
) -> List[Dict[str, int]]:
    """
    Runs Silero VAD and returns raw speech timestamps in SAMPLES:
      [{"start": sample_idx, "end": sample_idx}, ...]
    """
    # NOTE: get_speech_timestamps expects 1D tensor
    speech_ts = get_speech_timestamps(
        audio_torch_1d,
        model,
        sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=int(min_speech_sec * 1000),
        return_seconds=False,
    )
    return speech_ts


def timestamps_samples_to_seconds(
    speech_ts_samples: Sequence[Dict[str, int]],
    sr: int,
) -> List[Dict[str, float]]:
    return [{"start": t["start"] / sr, "end": t["end"] / sr} for t in speech_ts_samples]


def merge_and_pack_segments(
    segments_s: List[Dict[str, float]],
    min_sec: float,
    max_sec: float,
    max_merge_gap_sec: float,
    padding: float,
    audio_len: int,
) -> List[Dict[str, float]]:
    """
    Take raw VAD segments (start/end in seconds) and:
    1) merge segments separated by <= max_merge_gap_sec
    2) split long segments into <= max_sec windows
    3) pack adjacent segments to reach at least min_sec (<= max_sec)
       If a packed segment can't reach min_sec, it's dropped.
    """
    if not segments_s:
        return []

    # Add padding to each segment
    for seg in segments_s:
        seg["start"] = max(0, seg["start"] - padding)
        seg["end"] = min(audio_len, seg["end"] + padding)

    # 1) Merge close segments
    merged: List[Dict[str, float]] = []
    cur = dict(segments_s[0])
    for seg in segments_s[1:]:
        if seg["start"] <= cur["end"] + max_merge_gap_sec:
            cur["end"] = max(cur["end"], seg["end"])
        else:
            merged.append(cur)
            cur = dict(seg)
    merged.append(cur)

    # 2) Split long segments into <= max_sec
    split: List[Dict[str, float]] = []
    for seg in merged:
        start, end = seg["start"], seg["end"]
        if (end - start) <= max_sec:
            split.append({"start": start, "end": end})
        else:
            s = start
            while s < end:
                e = min(s + max_sec, end)
                split.append({"start": s, "end": e})
                s = e

    # 3) Pack adjacent segments to reach min_sec (not exceeding max_sec)
    packed: List[Dict[str, float]] = []
    i = 0
    while i < len(split):
        clip_start = split[i]["start"]
        clip_end = split[i]["end"]

        while (clip_end - clip_start) < min_sec and (i + 1) < len(split):
            nxt = split[i + 1]
            proposed_end = nxt["end"]
            if (proposed_end - clip_start) <= max_sec:
                i += 1
                clip_end = proposed_end
            else:
                break

        if (clip_end - clip_start) >= min_sec:
            packed.append({"start": clip_start, "end": clip_end})

        i += 1

    return packed


# -------------------------
# Orchestrator-facing API
# -------------------------

def vad_clean_audio(
    audio: Union[np.ndarray, torch.Tensor],
    params: VadParams = VadParams(),
    vad_model: Optional[torch.nn.Module] = None,
    *,
    return_audio_clips: bool = False,
) -> Union[
    List[Tuple[float, float]],
    Tuple[List[Tuple[float, float]], List[np.ndarray]],
]:
    """
    Main entrypoint for the orchestrator.

    Inputs:
      audio: raw waveform already at params.sampling_rate (recommended).
             - mono shape (n,) OR stereo (n, ch)/(ch, n)
      params: VAD + chunking configuration
      vad_model: optionally pass a preloaded model (recommended in orchestrator)
      return_audio_clips: if True, also return list of numpy audio clips

    Returns:
      If return_audio_clips is False:
        [(start_sec, end_sec), ...]
      Else:
        ([(start_sec, end_sec), ...], [clip_audio_np, ...])

    Notes:
      - This function does NOT save files and does NOT touch JSON.
      - The orchestrator should save clips and update manifests.
    """
    if params.sampling_rate not in (8000, 16000):
        raise ValueError("Silero VAD supports sampling_rate=8000 or 16000.")

    audio_mono = ensure_mono_float32(audio)
    audio_torch = numpy_to_torch_1d(audio_mono)
    audio_len = len(audio_mono)/params.sampling_rate

    model = vad_model or get_vad_model(cache=True)

    speech_ts_samples = run_silero_vad(
        audio_torch_1d=audio_torch,
        model=model,
        sr=params.sampling_rate,
        threshold=params.threshold,
        min_speech_sec=params.min_speech_sec,
    )

    segments_s = timestamps_samples_to_seconds(speech_ts_samples, params.sampling_rate)

    packed = merge_and_pack_segments(
        segments_s=segments_s,
        min_sec=params.min_sec,
        max_sec=params.max_sec,
        max_merge_gap_sec=params.max_merge_gap_sec,
        padding=params.padding,
        audio_len=audio_len
    )

    clip_ranges = [(seg["start"], seg["end"]) for seg in packed]

    if not return_audio_clips:
        return clip_ranges

    clips_audio = [
        slice_audio(audio_mono, params.sampling_rate, start, end)
        for (start, end) in clip_ranges
    ]
    return clip_ranges, clips_audio


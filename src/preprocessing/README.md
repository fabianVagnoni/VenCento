# Preprocessing Pipeline (Audio)

This directory implements an **offline preprocessing pipeline** that converts raw audio files into **canonical PCM**, applies **RMS normalization**, runs **Silero VAD** to find speech regions, and writes **cleaned speech chunks** back to disk as PCM files. :contentReference[oaicite:0]{index=0}

---

## Contents

- `preprocessing.py` — main orchestrator: canonize → load → normalize → VAD-clean → save chunks :contentReference[oaicite:1]{index=1}  
- `vad_cleaning.py` — Silero VAD + segment merging/splitting/packing into ~2–6s clips :contentReference[oaicite:2]{index=2}  
- `normalize.py` — RMS normalization to a target dBFS (returns normalized audio + gain) :contentReference[oaicite:3]{index=3}  
- `canonize.py` — converts input audio to canonical raw PCM (default: 16kHz, mono, s16le) using ffmpeg :contentReference[oaicite:4]{index=4}  
- `config.yaml` — configuration shared by modules via `_load_config(...)` (sampling rate, channels, target dBFS, VAD params, etc.) :contentReference[oaicite:5]{index=5}  

---

## `preprocessing.py` (Main orchestrator)

### `PreprocessParams` (dataclass)

Holds end-to-end preprocessing settings, including:
- `raw_manifest_path`: JSONL manifest path (used to locate metadata for the given `audio_path`)
- `raw_data_path`: root folder for raw audio files
- `processed_data_path`: root folder for outputs
- `sr`, `channels`: defaults from config
- `vad_params`: `VadParams()` (defaults from config)
- `sample_fmt`: `"s16le"` (must match `canonize`)
- `target_dbfs`: passed into RMS normalization :contentReference[oaicite:6]{index=6}  

### `preprocess_audio(audio_path, params, vad_model=None) -> List[str]`

Pipeline steps (as implemented):

1) **Validate file exists** :contentReference[oaicite:7]{index=7}  
2) **Look up entry in manifest** (expects exactly one match) and build a stable PCM filename based on `id`, `speaker_id`, and optional `segment_index` :contentReference[oaicite:8]{index=8}  
3) **Canonize**: convert input audio to canonical raw PCM on disk (`s16le`, `sr`, `channels`) :contentReference[oaicite:9]{index=9}  
4) **Load PCM**: reads canonical PCM into numpy audio (mono expected) :contentReference[oaicite:10]{index=10}  
5) **Safety checks**: mono shape, finite values, non-zero signal :contentReference[oaicite:11]{index=11}  
6) **RMS normalize** to `target_dbfs` using `rms_normalize` :contentReference[oaicite:12]{index=12}  
7) **VAD cleaning**: runs Silero VAD + packs segments (start/end in seconds) :contentReference[oaicite:13]{index=13}  
8) **Write chunks**: slices normalized audio for each segment and saves as PCM (`.pcm`) under:
   `processed_data_path/chunks/<audio_path>/<audio_path>_<idx>.pcm` :contentReference[oaicite:14]{index=14}  

Returns: list of written chunk paths. :contentReference[oaicite:15]{index=15}  

### Example entrypoint

`preprocess_one_file_example()` shows how to instantiate `PreprocessParams` and run `preprocess_audio(...)`. :contentReference[oaicite:16]{index=16}  

---

## `vad_cleaning.py` (VAD + chunk packing)

Purpose: **Silero VAD + segment packing into 2–6s clips**, designed to be called by an external orchestrator (this directory’s `preprocessing.py`). :contentReference[oaicite:17]{index=17}  

### `VadParams` (dataclass)

Configurable parameters (defaults read from `config.yaml`), including:
- `sampling_rate` (Silero supports **8000 or 16000**)
- `threshold`
- `min_speech_sec` (drop tiny bursts)
- `min_sec`, `max_sec` (final clip duration bounds)
- `max_merge_gap_sec` (merge nearby segments)
- `padding` (expand segments slightly) :contentReference[oaicite:18]{index=18}  

### Key functions

- `get_vad_model(cache=True)` loads Silero VAD and optionally caches globally. :contentReference[oaicite:19]{index=19}  
- `run_silero_vad(...)` returns raw speech timestamps in **samples**. :contentReference[oaicite:20]{index=20}  
- `merge_and_pack_segments(...)`:
  1) adds padding  
  2) merges nearby segments  
  3) splits long segments to `<= max_sec`  
  4) packs adjacent segments until `>= min_sec` without exceeding `max_sec` :contentReference[oaicite:21]{index=21}  
- `vad_clean_audio(...)` is the orchestrator-facing API:
  - accepts numpy or torch audio
  - ensures mono float32
  - returns `[(start_sec, end_sec), ...]` or (optionally) also returns the sliced clips :contentReference[oaicite:22]{index=22}  

---

## `normalize.py` (RMS normalization)

Defines `rms_normalize(x, target_dbfs, eps)`:
- computes RMS over all samples (and channels if present)
- converts `target_dbfs` to linear RMS
- returns `(y, gain)` where `y = x * gain` :contentReference[oaicite:23]{index=23}  

Target dBFS default is loaded from config (fallback `-14.0`). :contentReference[oaicite:24]{index=24}  

---

## `canonize.py` (canonical PCM conversion)

Converts audio (e.g., WAV) to raw PCM using **ffmpeg**, defaulting to:
- `sampling_rate` from config (fallback 16000)
- `channels` from config (fallback 1)
- sample format `s16le` :contentReference[oaicite:25]{index=25}  

Key elements:
- `canonize(wav_path, pcm_path, sr, channels, sample_fmt)` builds and runs an ffmpeg command producing raw PCM. :contentReference[oaicite:26]{index=26}  
- Includes a CLI interface (`py -m src.preprocessing.canonize ...`). :contentReference[oaicite:27]{index=27}  

⚠️ Note: This implementation points to a **Windows-specific ffmpeg.exe path** and raises if it doesn’t exist. :contentReference[oaicite:28]{index=28}  

---

## End-to-end output

Given an input `audio_path` in the manifest, `preprocess_audio(...)` will write:
- canonical PCM under: `<raw_data_path>/canonical/<audio_id...>.pcm` :contentReference[oaicite:29]{index=29}  
- chunk PCMs under: `<processed_data_path>/chunks/<audio_path>/<audio_path>_<idx>.pcm` :contentReference[oaicite:30]{index=30}  

---

## Quick usage sketch

```python
from src.preprocessing.preprocessing import PreprocessParams, preprocess_audio

params = PreprocessParams(
    raw_manifest_path="data/raw/manifest.jsonl",
    raw_data_path="data/raw",
    processed_data_path="data/processed",
)

chunk_paths = preprocess_audio("path/to/input.wav", params)
# Utils Directory

This directory groups small, reusable helpers that are shared across the codebase: **JSONL utilities**, **audio shape/safety helpers**, **raw PCM I/O**, **PCM→WAV “decanonization” for listening/debug**, and a **generic config loader**.

---

## Contents (what each file covers)

- `jsonl_helpers.py` — streaming JSONL reading + robust path-based record lookup :contentReference[oaicite:0]{index=0}  
- `audio_helpers.py` — audio “safety” utilities: ensure mono float32, torch↔numpy conversion, slicing by seconds :contentReference[oaicite:1]{index=1}  
- `io_helpers.py` — load/save raw **PCM s16le** to/from normalized float32 audio :contentReference[oaicite:2]{index=2}  
- `decanonize.py` — convert PCM bytes / normalized float audio to listenable WAV (debugging) :contentReference[oaicite:3]{index=3}  
- `general_utils.py` — minimal YAML config loader `_load_config(config_path)` :contentReference[oaicite:4]{index=4}  

---

## `jsonl_helpers.py` — JSONL streaming + path matching

### Functionality
- **Stream JSON objects from a `.jsonl` file** safely (line-by-line, with line numbers on errors). :contentReference[oaicite:5]{index=5}  
- **Normalize paths** for robust matching across Windows/POSIX:
  - expands `~`
  - resolves/absolutizes best-effort
  - normalizes slashes
  - lowercases on Windows :contentReference[oaicite:6]{index=6}  

### Key utilities
- `iter_jsonl(jsonl_path)` → generator of dicts; raises `ValueError` with line number if JSON is invalid. :contentReference[oaicite:7]{index=7}  
- `find_records_by_path(jsonl_path, target_path, keys="file_path", mode="exact|contains|basename", first_only=False)` → returns a list of `Match(obj, line_no, matched_key)` results. :contentReference[oaicite:8]{index=8}  
- `find_first_record_by_path(...)` → convenience wrapper returning one match or `None`. :contentReference[oaicite:9]{index=9}  

**Typical use:** locate the manifest row corresponding to an audio file path (or match by basename when absolute paths differ between machines).

---

## `audio_helpers.py` — audio shape + “safety” helpers

### Functionality
- Standardizes inputs to **mono float32 numpy** (recommended range `[-1, 1]`), handling both numpy arrays and torch tensors. :contentReference[oaicite:10]{index=10}  
- Provides small helpers for:
  - numpy→torch conversion
  - slicing a waveform by time in seconds :contentReference[oaicite:11]{index=11}  

### Key utilities
- `ensure_mono_float32(audio)`:
  - accepts `(n,)` mono
  - accepts `(n, ch)` or `(ch, n)` and averages channels
  - raises if ndim > 2 :contentReference[oaicite:12]{index=12}  
- `numpy_to_torch_1d(audio_mono)` → `torch.Tensor` (contiguous float32). :contentReference[oaicite:13]{index=13}  
- `slice_audio(audio_mono, sr, start_sec, end_sec)` → returns `[start, end]` slice in samples. :contentReference[oaicite:14]{index=14}  

**Typical use:** enforce consistent audio format before VAD/augmentation, and slice speech segments cleanly.

---

## `io_helpers.py` — raw PCM s16le I/O

### Functionality
- Load and save **raw PCM** files in **s16le** format (the canonical format used by your preprocessing pipeline).
- Converts between:
  - **int16 PCM** on disk
  - **float32 waveform** in `[-1, 1]` in memory :contentReference[oaicite:15]{index=15}  

### Key utilities
- `load_pcm_s16le(pcm_path, channels=1)`:
  - reads `int16` from file
  - reshapes interleaved data if `channels > 1`
  - converts to float32 `[-1,1]`
  - calls `ensure_mono_float32` to return mono `(n,)` :contentReference[oaicite:16]{index=16}  
- `save_pcm_s16le(audio_mono_f32, out_path)`:
  - creates parent directories
  - replaces NaNs/Infs with 0
  - clips to `[-1, 1]`
  - writes int16 PCM via `tofile()` :contentReference[oaicite:17]{index=17}  

**Typical use:** store normalized/cleaned audio as compact PCM for fast downstream loading.

---

## `decanonize.py` — “make it listenable” WAV helpers

### Functionality
Utilities to convert internal PCM/normalized signals to **WAV files** for listening/debugging.

### Key utilities
- `pcm16_bytes_to_wav(pcm_bytes, out_wav, sr=16000, channels=1)`:
  - `bytes` → `int16` → float32 `[-1,1]`
  - writes WAV via `soundfile` :contentReference[oaicite:18]{index=18}  
- `normalized_float_to_wav(y, out_wav, sr=16000, subtype="PCM_16", peak=0.999)`:
  - writes float audio to WAV
  - optional safety: if peak exceeds 1.0, scales down to avoid clipping :contentReference[oaicite:19]{index=19}  

**Typical use:** after preprocessing/augmentation, export examples you can audition (and verify no clipping).

---

## `general_utils.py` — config loader

### Functionality
- `_load_config(config_path)` loads YAML into a dict using `yaml.safe_load`.
- On error, prints a message and returns `{}`. :contentReference[oaicite:20]{index=20}  

**Typical use:** lightweight shared config loader used by other modules (preprocessing/augmentation).

---

## How these utils fit together

A common flow in your codebase looks like:

1. Use `general_utils._load_config(...)` to read SR/channels/params. :contentReference[oaicite:21]{index=21}  
2. Use `io_helpers.load_pcm_s16le(...)` to load canonical PCM into float32 mono. :contentReference[oaicite:22]{index=22}  
3. Use `audio_helpers.slice_audio(...)` to cut segments in seconds (e.g., from VAD timestamps). :contentReference[oaicite:23]{index=23}  
4. Use `io_helpers.save_pcm_s16le(...)` to write processed chunks. :contentReference[oaicite:24]{index=24}  
5. Optionally use `decanonize.normalized_float_to_wav(...)` to export a WAV for listening/debug. :contentReference[oaicite:25]{index=25}  
6. If using manifests, use `jsonl_helpers.find_*_by_path(...)` to map a file path back to its JSONL record. :contentReference[oaicite:26]{index=26}  
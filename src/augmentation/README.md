# Audio Augmentation Pipeline (On-the-fly)

This directory contains a **PyTorch / torchaudio** augmentation pipeline for speech/audio training. It applies **waveform-level augmentations** (crop, gain, speed perturbation), optional **Opus codec simulation**, and optional **feature extraction + SpecAugment**. :contentReference[oaicite:0]{index=0}

---

## Contents

- `main_augmentation.py` — main augmentation module (config loading + augmenter) :contentReference[oaicite:1]{index=1}  
- `basic_operations.py` — waveform augmentation primitives (crop, gain, speed perturb) :contentReference[oaicite:2]{index=2}  
- `codec_opus.py` — Opus encode→decode simulation for robustness :contentReference[oaicite:3]{index=3}  
- `config.yaml` — YAML configuration (loaded by `_load_config`) :contentReference[oaicite:4]{index=4}  

---

## `main_augmentation.py`

### Key components

- `AugmentConfig` (dataclass): holds all augmentation parameters, including:
  - `crop_len` (samples)
  - gain params (`p_gain`, `gain_db_min/max`)
  - speed perturb params (`p_speed`, `speed_min/max`)
  - Opus params (`p_opus`, `opus_impl`)
  - SpecAugment params (`use_specaugment`, `p_specaugment`, mask sizes)
  - Optional feature extraction params (`return_features`, `n_mels`, `n_fft`, `hop_length`) :contentReference[oaicite:5]{index=5}

- `load_augment_config(...)`: loads `config.yaml`, maps `pipeline` ops to `AugmentConfig` and reads `feature_aug` settings. :contentReference[oaicite:6]{index=6}

- `create_augmenter(...)`: creates an `OnTheFlyAugmenter` from YAML config. :contentReference[oaicite:7]{index=7}

- `OnTheFlyAugmenter(nn.Module)`: core augmenter callable as `aug(x, sample_rate)`. :contentReference[oaicite:8]{index=8}

### Augmentation flow (forward)

Inside `OnTheFlyAugmenter.forward`:

1. Shape check/standardization: only accepts `(T,)` or `(B, T)` by default. :contentReference[oaicite:9]{index=9}  
2. Initial crop (bounds compute). :contentReference[oaicite:10]{index=10}  
3. Random gain (probabilistic). :contentReference[oaicite:11]{index=11}  
4. Speed perturbation (probabilistic; can change length). :contentReference[oaicite:12]{index=12}  
5. Opus codec simulation (probabilistic; can change length slightly). :contentReference[oaicite:13]{index=13}  
6. Final crop to ensure fixed output length. :contentReference[oaicite:14]{index=14}  
7. Optional: compute mel features and apply SpecAugment; optionally returns features instead of waveform. :contentReference[oaicite:15]{index=15}  

### Feature extraction / SpecAugment

If `return_features` or `use_specaugment` is enabled, it uses:

- `torchaudio.transforms.MelSpectrogram`
- `torchaudio.transforms.AmplitudeToDB`
- `FrequencyMasking` and `TimeMasking` for SpecAugment :contentReference[oaicite:16]{index=16}  

The helper `_wav_to_mel` recreates the MelSpectrogram transform if the runtime `sample_rate` differs from the cached one. :contentReference[oaicite:17]{index=17}  

---

## `basic_operations.py`

Waveform operations used by the augmenter:

- `random_crop_audio(x, crop_len)`
  - Supports `(T,)` or `(B, T)`
  - Pads with zeros if shorter than `crop_len`
  - Randomly selects a crop window :contentReference[oaicite:18]{index=18}  

- `random_gain_db(x, gain_db_min, gain_db_max)`
  - Samples a single gain in dB uniformly and applies it to the entire tensor
  - Works with many shapes (broadcast-friendly) :contentReference[oaicite:19]{index=19}  

- `speed_perturb(x, factor_min, factor_max)`
  - Chooses a random resampling factor
  - Normalizes shape to interpolate in `(B, C, T)` form internally
  - Output length changes unless cropped/padded afterward :contentReference[oaicite:20]{index=20}  

---

## `codec_opus.py`

Provides `codec_opus(...)`: a **WhatsApp/VoIP-like** robustness augmentation by simulating **Opus encode → decode**. :contentReference[oaicite:21]{index=21}  

Highlights:

- Supports shapes `(T,)`, `(B,T)`, `(C,T)`, `(B,C,T)` and returns **mono** audio. :contentReference[oaicite:22]{index=22}  
- Random bitrate selection, including optional “hard mode” low-bitrate bucket. :contentReference[oaicite:23]{index=23}  
- Typical Opus toolchains operate at **48 kHz**; the code resamples to 48 kHz, runs codec, then resamples back. :contentReference[oaicite:24]{index=24}  
- Two implementations:
  - `ffmpeg_libopus`: uses `ffmpeg` subprocess calls with `libopus` :contentReference[oaicite:25]{index=25}  
  - `torchaudio`: uses torchaudio save/load through FFmpeg-backed support :contentReference[oaicite:26]{index=26}  
- After processing batch items, it pads/crops to a common length before stacking. :contentReference[oaicite:27]{index=27}  

---

## Configuration (`config.yaml`)

`load_augment_config()` expects a section (default: `"accent_encoder"`) containing:

- `sampling_rate`
- `pipeline`: list of ops with `name`, optional `p`, and `params`
  - `random_crop` -> `seconds` mapped into `crop_len = int(sr * seconds)` :contentReference[oaicite:28]{index=28}  
  - `gain` -> `p`, `gain_db_min`, `gain_db_max` :contentReference[oaicite:29]{index=29}  
  - `speed_perturb` -> `p`, `factor_min`, `factor_max` :contentReference[oaicite:30]{index=30}  
  - `codec_opus` -> `p`, `implementation` :contentReference[oaicite:31]{index=31}  
- `feature_aug`: supports `specaugment` via `enabled`, `type`, `p`, and mask params :contentReference[oaicite:32]{index=32}  

---

## Example usage

```python
from pathlib import Path
from main_augmentation import create_augmenter

aug = create_augmenter(config_path=Path("config.yaml"), section="accent_encoder")

# x: (T,) or (B, T)
y = aug(x, sample_rate=16000)
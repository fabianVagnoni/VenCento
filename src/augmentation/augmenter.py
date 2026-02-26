# main_augmentation.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Callable, List
import torch
import torch.nn as nn

import torchaudio

from basic_operations import random_crop_audio, random_gain_db, speed_perturb
from codec_opus import codec_opus
from src.utils.general_utils import _load_config


def _maybe(p: float) -> bool:
    return torch.rand(()) < p


def _ensure_1d_or_2d(x: torch.Tensor) -> torch.Tensor:
    # We will standardize to (T,) or (B,T). Most of your ops support these well.
    if x.dim() == 1:
        return x
    if x.dim() == 2:
        return x
    # If you have (C,T) or (B,C,T), you can decide your convention here.
    # For training speech models, many pipelines store mono anyway.
    raise ValueError(f"Unsupported waveform shape: {tuple(x.shape)}")


@dataclass
class AugmentConfig:
    # Length control
    crop_len: int  # in samples, e.g. int(sr * 4.0) for 4s

    # Waveform augments
    p_gain: float = 0.5
    gain_db_min: float = -6.0
    gain_db_max: float = 6.0

    p_speed: float = 0.3
    speed_min: float = 0.95
    speed_max: float = 1.05

    p_opus: float = 0.15
    opus_impl: str = "ffmpeg_libopus"  # or "torchaudio"

    # SpecAugment (only if you compute features here)
    use_specaugment: bool = False
    p_specaugment: float = 0.5
    freq_mask_param: int = 15
    time_mask_param: int = 35

    # Feature extraction (optional: only if use_specaugment or you want model-ready features)
    return_features: bool = False
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160


def _find_pipeline_op(pipeline: List[dict], name: str) -> Optional[dict]:
    """Find a pipeline op by name."""
    for op in pipeline or []:
        if op.get("name") == name:
            return op
    return None


def load_augment_config(
    config_path: Optional[Path] = None,
    section: str = "accent_encoder",
    sample_rate: Optional[int] = None,
) -> AugmentConfig:
    """
    Load AugmentConfig from config.yaml.
    Maps pipeline ops (random_crop, gain, speed_perturb, codec_opus) and feature_aug to AugmentConfig.
    """
    if config_path is None:
        config_path = (Path(__file__).parent / "config.yaml").resolve()

    raw = _load_config(str(config_path))
    if not raw:
        return AugmentConfig(crop_len=16000 * 4)  # fallback: 4s at 16kHz

    section_cfg = raw.get(section) or {}
    sr = sample_rate or section_cfg.get("sampling_rate", 16000)
    pipeline = section_cfg.get("pipeline") or []
    feature_aug = section_cfg.get("feature_aug") or {}

    # random_crop -> crop_len (samples)
    crop_op = _find_pipeline_op(pipeline, "random_crop")
    crop_seconds = 4.0
    if crop_op and crop_op.get("params"):
        crop_seconds = crop_op["params"].get("seconds", crop_seconds)
    crop_len = int(sr * crop_seconds)

    # gain
    gain_op = _find_pipeline_op(pipeline, "gain")
    p_gain = 0.5
    gain_db_min, gain_db_max = -6.0, 6.0
    if gain_op:
        p_gain = gain_op.get("p", p_gain)
        if gain_op.get("params"):
            gain_db_min = gain_op["params"].get("gain_db_min", gain_db_min)
            gain_db_max = gain_op["params"].get("gain_db_max", gain_db_max)

    # speed_perturb
    speed_op = _find_pipeline_op(pipeline, "speed_perturb")
    p_speed = 0.3
    speed_min, speed_max = 0.95, 1.05
    if speed_op:
        p_speed = speed_op.get("p", p_speed)
        if speed_op.get("params"):
            speed_min = speed_op["params"].get("factor_min", speed_min)
            speed_max = speed_op["params"].get("factor_max", speed_max)

    # codec_opus
    opus_op = _find_pipeline_op(pipeline, "codec_opus")
    p_opus = 0.15
    opus_impl = "ffmpeg_libopus"
    if opus_op:
        p_opus = opus_op.get("p", p_opus)
        if opus_op.get("params"):
            opus_impl = opus_op["params"].get("implementation", opus_impl)

    # feature_aug (specaugment)
    use_specaugment = feature_aug.get("enabled", False) and feature_aug.get("type") == "specaugment"
    p_specaugment = feature_aug.get("p", 0.5)
    fa_params = feature_aug.get("params") or {}
    freq_mask_param = fa_params.get("freq_mask_max_bins", 15)
    time_mask_param = fa_params.get("time_mask_max_frames", 35)

    return AugmentConfig(
        crop_len=crop_len,
        p_gain=p_gain,
        gain_db_min=gain_db_min,
        gain_db_max=gain_db_max,
        p_speed=p_speed,
        speed_min=speed_min,
        speed_max=speed_max,
        p_opus=p_opus,
        opus_impl=opus_impl,
        use_specaugment=use_specaugment,
        p_specaugment=p_specaugment,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
    )


def create_augmenter(
    config_path: Optional[Path] = None,
    section: str = "accent_encoder",
) -> OnTheFlyAugmenter:
    """Create OnTheFlyAugmenter from config.yaml."""
    cfg = load_augment_config(config_path=config_path, section=section)
    return OnTheFlyAugmenter(cfg)


class OnTheFlyAugmenter(nn.Module):
    """
    Callable on the fly:
        y = aug(x, sample_rate)
    Optionally:
        feats = aug(x, sample_rate) if return_features=True
    """

    def __init__(self, cfg: AugmentConfig):
        super().__init__()
        self.cfg = cfg

        # If you want to do feature-space augmentation here:
        if cfg.use_specaugment or cfg.return_features:
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,  # will override at call-time if needed
                n_fft=cfg.n_fft,
                hop_length=cfg.hop_length,
                n_mels=cfg.n_mels,
                center=True,
                power=2.0,
            )
            self.amptodb = torchaudio.transforms.AmplitudeToDB(stype="power")

        if cfg.use_specaugment:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=cfg.freq_mask_param)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=cfg.time_mask_param)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, sample_rate: int) -> torch.Tensor:
        x = _ensure_1d_or_2d(x)

        # 1) Ensure a fixed length early or late.
        # Typically: do speed/codec first (they may change length), then crop last.
        # But you CAN also do an initial crop to bound compute.
        # Here: bound compute first, then final crop again after length-changing ops.
        x = random_crop_audio(x, self.cfg.crop_len)

        # 2) Gain
        if self.cfg.p_gain > 0 and _maybe(self.cfg.p_gain):
            x = random_gain_db(x, self.cfg.gain_db_min, self.cfg.gain_db_max)

        # 3) Speed perturb (changes length)
        if self.cfg.p_speed > 0 and _maybe(self.cfg.p_speed):
            x = speed_perturb(x, self.cfg.speed_min, self.cfg.speed_max)

        # 4) Opus codec simulation (can change length slightly)
        if self.cfg.p_opus > 0 and _maybe(self.cfg.p_opus):
            x = codec_opus(
                x=x,
                sample_rate=sample_rate,
                implementation=self.cfg.opus_impl,
            )

        # 5) Final length fix (important!)
        x = random_crop_audio(x, self.cfg.crop_len)

        # 6) Optional: compute mel + specaugment + return features
        if self.cfg.return_features or self.cfg.use_specaugment:
            # torchaudio expects (channels, time); standardize
            if x.dim() == 1:
                wav = x.unsqueeze(0)
            else:
                # (B,T) -> treat as batch; convert per-item (simple but not fastest)
                # If you need speed: vectorize with reshape and grouped ops.
                feats = []
                for i in range(x.shape[0]):
                    feats.append(self._wav_to_mel(x[i].unsqueeze(0), sample_rate))
                feats = torch.stack(feats, dim=0)  # (B, n_mels, frames)
                return feats

            mel = self._wav_to_mel(wav, sample_rate)  # (n_mels, frames)
            return mel

        return x

    def _wav_to_mel(self, wav_ch_t: torch.Tensor, sample_rate: int) -> torch.Tensor:
        # recreate melspec if SR differs (clean + explicit)
        if getattr(self.melspec, "sample_rate", None) != sample_rate:
            self.melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                n_mels=self.cfg.n_mels,
                center=True,
                power=2.0,
            )
        mel = self.melspec(wav_ch_t)            # (ch, n_mels, frames)
        mel_db = self.amptodb(mel)              # (ch, n_mels, frames)
        mel_db = mel_db.squeeze(0)              # (n_mels, frames)

        if self.cfg.use_specaugment and _maybe(self.cfg.p_specaugment):
            # SpecAugment expects (freq, time) or (batch, freq, time) depending on version;
            # this works for typical torchaudio usage.
            mel_db = self.freq_mask(mel_db)
            mel_db = self.time_mask(mel_db)

        return mel_db
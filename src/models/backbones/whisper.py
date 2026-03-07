"""
Thin HuggingFace wrapper for Whisper encoder backbones.

Returns frame-level encoder hidden states only — no pooling, no classification head.
This mirrors the interface of the project's wav2vec2 / WavLM backbones so the
accent detector can swap backbones via config.

Important difference vs wav2vec2/WavLM:
- Whisper's encoder takes log-Mel spectrogram features (80 x frames), not raw waveforms.
  This module therefore runs the WhisperFeatureExtractor inside forward().

Tradeoffs:
- HF WhisperFeatureExtractor uses NumPy under the hood, so gradients do NOT flow
  through the feature extraction stage. Gradients *do* flow through the Whisper encoder
  itself (from input_features onward), which is typically what you want anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel


@dataclass(frozen=True)
class WhisperBackboneOutput:
    """Optional typed output if you later want to return both features + lengths."""
    features: torch.Tensor  # (B, L, D)
    lengths: torch.Tensor   # (B,) in frames (encoder time steps)


class WhisperBackbone(nn.Module):
    """
    Frozen-or-trainable wrapper around a Whisper encoder.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. "openai/whisper-small", "openai/whisper-large-v3".
    freeze:
        If True, all backbone parameters are frozen after loading.
    sampling_rate:
        Expected waveform sampling rate. Whisper is trained for 16k.
    return_output:
        If True, forward() returns WhisperBackboneOutput(features, lengths).
        If False (default), forward() returns features only, matching other backbones.

    Attributes
    ----------
    feature_dim:
        Encoder hidden size D.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        freeze: bool = False,
        sampling_rate: int = 16000,
        return_output: bool = False,
    ) -> None:
        super().__init__()

        config: WhisperConfig = WhisperConfig.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name)
        self.encoder = self.model.encoder  # explicit for clarity

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.sampling_rate = sampling_rate

        # WhisperConfig uses d_model for hidden size
        self.feature_dim: int = int(getattr(config, "d_model", config.hidden_size))

        self.return_output = return_output

        if freeze:
            self.freeze_backbone()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """Freeze all encoder parameters in-place."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _to_feature_extractor_inputs(
        self,
        waveforms: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[np.ndarray]:
        """
        Convert a batch tensor (B, T) into a list of NumPy arrays, optionally
        trimming to true lengths from attention_mask.
        """
        if waveforms.dim() != 2:
            raise ValueError(f"Expected waveforms shape (B, T), got {tuple(waveforms.shape)}")

        waveforms_cpu = waveforms.detach().to("cpu")

        if attention_mask is None:
            # Whole tensor is real; convert each row
            return [waveforms_cpu[b].numpy() for b in range(waveforms_cpu.size(0))]

        if attention_mask.shape != waveforms.shape:
            raise ValueError(
                f"attention_mask must match waveforms shape. "
                f"Got waveforms={tuple(waveforms.shape)}, mask={tuple(attention_mask.shape)}"
            )

        mask_cpu = attention_mask.detach().to("cpu")
        lengths = mask_cpu.long().sum(dim=1).tolist()
        outs: List[np.ndarray] = []
        for b, n in enumerate(lengths):
            n = max(int(n), 1)
            outs.append(waveforms_cpu[b, :n].numpy())
        return outs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        waveforms: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | WhisperBackboneOutput:
        """
        Encode raw waveforms into Whisper encoder frame-level hidden states.

        Parameters
        ----------
        waveforms:
            Float32 tensor of shape (B, T) in [-1, 1] at 16kHz.
        attention_mask:
            Optional tensor of shape (B, T): 1 for real samples, 0 for padding.
            Used only to trim waveforms before feature extraction.

        Returns
        -------
        If return_output=False:
            Float32 tensor (B, L, D)
        If return_output=True:
            WhisperBackboneOutput(features=(B, L, D), lengths=(B,))
        """
        # Ensure float32 (HF feature extractor expects float)
        if waveforms.dtype != torch.float32:
            waveforms = waveforms.float()

        # Convert to list of np arrays (variable length safe)
        wav_list = self._to_feature_extractor_inputs(waveforms, attention_mask)

        # HF Whisper feature extraction -> log-Mel features
        # Whisper expects fixed 30s features (3000 frames), so force pad/truncate to max length.
        fe = self.feature_extractor(
            wav_list,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.feature_extractor.n_samples,  # 30s @ 16k (in samples)
            truncation=True,
        )
        input_features: torch.Tensor = fe.input_features  # (B, 80, 3000)

        # Move features to the same device as encoder
        device = next(self.encoder.parameters()).device
        input_features = input_features.to(device)

        # Whisper encoder forward
        enc_out = self.encoder(input_features=input_features)
        features = enc_out.last_hidden_state  # (B, L, D)

        if not self.return_output:
            return features

        # Approximate frame lengths: based on padding in input_features.
        # WhisperFeatureExtractor pads in the time dimension of log-mel frames.
        # We infer lengths by counting non-zero frames (safe because padded frames are zeros).
        # Shape: (B, 80, frames) -> sum energy over mel bins
        frame_energy = (input_features.abs().sum(dim=1) > 0).long()  # (B, frames)
        lengths = frame_energy.sum(dim=1)  # (B,)

        # Encoder may downsample or keep similar L; to avoid mismatch, clamp to L
        lengths = torch.clamp(lengths, max=features.size(1)).to(features.device)

        return WhisperBackboneOutput(features=features, lengths=lengths)
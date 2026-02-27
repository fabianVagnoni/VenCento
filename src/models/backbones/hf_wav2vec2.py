"""
Thin HuggingFace wrapper for wav2vec2 / HuBERT-style encoders.

Returns frame-level features only — no pooling, no classification head.
The model is loaded via ``AutoModel.from_pretrained`` so it works with any
architecture in the wav2vec2 family (wav2vec2-base, wav2vec2-large, HuBERT,
mms-*, etc.) as long as the model outputs ``last_hidden_state``.

Integration map
---------------
- :mod:`transformers` — ``AutoModel``, ``AutoConfig`` (HuggingFace)
- Consumed by ``src.models.accent_detector`` (Sprint 3+), which wraps this
  backbone with a pooling head and classifier.

Public API
----------
    Wav2Vec2Backbone   – ``nn.Module`` that produces frame-level features
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class Wav2Vec2Backbone(nn.Module):
    """
    Frozen-or-trainable wrapper around a wav2vec2 / HuBERT encoder.

    Loads any HuggingFace model in the wav2vec2 family via
    ``AutoModel.from_pretrained`` and exposes its frame-level hidden states.
    No pooling or classification logic is included; those belong in the head.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g.
        ``"facebook/wav2vec2-base-960h"`` or ``"facebook/hubert-base-ls960"``.
    freeze:
        If ``True``, all backbone parameters are frozen immediately after
        loading (``requires_grad = False``).  Set to ``True`` for Sprint 3
        (frozen encoder + linear head) and ``False`` for fine-tuning later.

    Attributes
    ----------
    feature_dim:
        Hidden size of the encoder (i.e. ``D`` in the output shape).  Derived
        from the model config; downstream heads should read this attribute
        rather than hard-coding a dimension.

    Example
    -------
    >>> backbone = Wav2Vec2Backbone("facebook/wav2vec2-base-960h", freeze=True)
    >>> waveforms = torch.zeros(2, 16000)           # (B, T)
    >>> features  = backbone(waveforms)             # (B, L, D)
    >>> features.shape
    torch.Size([2, 49, 768])
    """

    def __init__(self, model_name: str, freeze: bool = False) -> None:
        super().__init__()

        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.feature_dim: int = config.hidden_size

        if freeze:
            self.freeze_backbone()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def freeze_backbone(self) -> None:
        """
        Freeze all encoder parameters in-place.

        Calling this after ``__init__`` is equivalent to passing
        ``freeze=True``.  Useful when you want to unfreeze selectively later.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        waveforms: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode raw waveforms into frame-level hidden states.

        Parameters
        ----------
        waveforms:
            Float32 tensor of shape ``(B, T)`` in ``[-1, 1]``, where ``T`` is
            the number of raw audio samples at 16 000 Hz.  This matches the
            ``waveforms`` field produced by :class:`src.data.collate.CollateFn`.
        attention_mask:
            Optional int/bool tensor of shape ``(B, T)``.  ``1`` at real
            samples, ``0`` at zero-padded positions.  Pass
            ``batch.attention_mask`` directly from
            :class:`src.data.collate.Batch`.  When ``None`` the model treats
            every position as real (safe for variable-length inference).

        Returns
        -------
        torch.Tensor:
            Float32 tensor of shape ``(B, L, D)`` where:

            - ``B`` – batch size
            - ``L`` – downsampled time frames (roughly ``T / 320`` for
              wav2vec2-base, e.g. 299 frames for 3 s audio)
            - ``D`` – :attr:`feature_dim` (e.g. 768 for *-base models)
        """
        outputs = self.encoder(
            input_values=waveforms,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state  # (B, L, D)

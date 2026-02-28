"""
Thin HuggingFace wrapper for WavLM-style encoders.

Returns frame-level features only — no pooling, no classification head.
The interface is identical to :mod:`src.models.backbones.hf_wav2vec2` so that
``accent_detector.py`` can swap backbones by changing a single config key.

WavLM is kept in a separate module from wav2vec2 because:

- Config field names and best default checkpoints differ (e.g. WavLM uses a
  different ``num_hidden_layers`` convention and exposes weighted-layer-sum
  outputs that may be exploited in later sprints).
- Future versions may use ``output_hidden_states=True`` and a learnable
  per-layer weighting over all transformer layers, which is specific to WavLM.

Integration map
---------------
- :mod:`transformers` — ``AutoModel``, ``AutoConfig`` (HuggingFace)
- Consumed by ``src.models.accent_detector`` (Sprint 3+), swappable with
  :class:`src.models.backbones.hf_wav2vec2.Wav2Vec2Backbone` via config.

Public API
----------
    WavLMBackbone   – ``nn.Module`` that produces frame-level features
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class WavLMBackbone(nn.Module):
    """
    Frozen-or-trainable wrapper around a WavLM encoder.

    Loads any WavLM checkpoint via ``AutoModel.from_pretrained`` (HuggingFace
    supports ``microsoft/wavlm-base``, ``microsoft/wavlm-base-plus``,
    ``microsoft/wavlm-large``, etc.) and exposes its frame-level hidden states.
    No pooling or classification logic is included.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier, e.g. ``"microsoft/wavlm-base-plus"``.
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
    >>> backbone = WavLMBackbone("microsoft/wavlm-base-plus", freeze=True)
    >>> waveforms = torch.zeros(2, 16000)          # (B, T)
    >>> features  = backbone(waveforms)            # (B, L, D)
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
        ``freeze=True``.  Useful when you want to unfreeze selectively later
        (e.g. fine-tune only the top transformer layers in Sprint 4).
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
              WavLM-base variants, e.g. 299 frames for 3 s audio)
            - ``D`` – :attr:`feature_dim` (e.g. 768 for *-base models,
              1024 for *-large)
        """
        outputs = self.encoder(
            input_values=waveforms,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state  # (B, L, D)

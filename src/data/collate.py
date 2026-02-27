"""
Collate function for audio batches fed to the DataLoader.

Integration map
---------------
- :mod:`src.utils.jsonl_helpers` – ``load_jsonl`` to read split files
- :mod:`src.data.splits`         – produces the per-split record lists
- :mod:`src.data.manifest`       – ``encode_labels`` / ``load_label_to_id``
- :mod:`src.utils.io_helpers`    – ``load_pcm_s16le`` (float32 PCM loader)

Sample format (what the Dataset must produce)
---------------------------------------------
Each item passed to the DataLoader must be a plain dict::

    {
        "waveform":   torch.Tensor,  # shape (n_samples,), float32 in [-1, 1]
        "label":      int,           # integer class id from label_to_id mapping
        "chunk_id":   str,           # e.g. "5364f2fddff08cc5_000" (optional)
        "speaker_id": str,           # e.g. "spk_008"              (optional)
    }

Use :func:`load_sample` to convert a manifest record dict to this format.

Collate output
--------------
:class:`Batch` – a frozen dataclass::

    waveforms:      torch.Tensor  # (B, T)  float32
    attention_mask: torch.Tensor  # (B, T)  int64  – 1 = real, 0 = padding
    labels:         torch.Tensor  # (B,)    int64
    chunk_ids:      list[str]
    speaker_ids:    list[str]

Attention mask
--------------
When a waveform is shorter than ``target_samples`` it is right-padded with
zeros.  The corresponding ``attention_mask`` positions are set to ``0`` so
that the model ignores them.  For waveforms that were cropped (longer than
``target_samples``) every position is ``1``.

This convention matches the HuggingFace ``attention_mask`` expected by
``Wav2Vec2ForSequenceClassification``, ``HubertForSequenceClassification``,
and the ``WhisperFeatureExtractor`` pipeline::

    outputs = model(
        input_values   = batch.waveforms,
        attention_mask = batch.attention_mask,
        labels         = batch.labels,
    )

Audio lengths
-------------
Preprocessing (config.yaml) produces chunks between ``min_sec=2.0`` and
``max_sec=10.0`` seconds at ``sr=16 000`` Hz.  The collate function fixes
every waveform to ``target_samples`` by:

- **padding** (right, zeros) when the waveform is shorter
- **cropping** (from the start) when the waveform is longer

Default target: **6 s × 16 000 = 96 000 samples**.

Public API
----------
    DEFAULT_SR             – 16 000  (matches config.yaml)
    DEFAULT_TARGET_SEC     – 6.0 s   (upper bound for most clips)
    DEFAULT_TARGET_SAMPLES – 96 000  (= DEFAULT_TARGET_SEC × DEFAULT_SR)
    load_sample(record, label_to_id)  – manifest record → sample dict
    CollateFn(target_samples)         – callable collate for DataLoader
    Batch                             – output dataclass
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.io_helpers import load_pcm_s16le
from src.utils.jsonl_helpers import JSONDict

# ---------------------------------------------------------------------------
# Constants  (mirror src/preprocessing/config.yaml)
# ---------------------------------------------------------------------------

DEFAULT_SR: int = 16_000          # sample rate in Hz
DEFAULT_TARGET_SEC: float = 6.0   # fixed clip length in seconds
DEFAULT_TARGET_SAMPLES: int = int(DEFAULT_TARGET_SEC * DEFAULT_SR)  # 96 000


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Batch:
    """
    Output of :class:`CollateFn`.

    Attributes
    ----------
    waveforms:
        Float32 tensor of shape ``(B, T)`` in ``[-1, 1]``.
    attention_mask:
        Int64 tensor of shape ``(B, T)``.  ``1`` at every real sample,
        ``0`` at every zero-padded position.  Pass directly to HuggingFace
        model ``attention_mask`` argument.
    labels:
        Int64 tensor of shape ``(B,)``.
    chunk_ids:
        Passthrough string identifiers, one per item.
    speaker_ids:
        Passthrough speaker tags, one per item.
    """

    waveforms:      torch.Tensor  # (B, T)
    attention_mask: torch.Tensor  # (B, T)
    labels:         torch.Tensor  # (B,)
    chunk_ids:      List[str]
    speaker_ids:    List[str]


# ---------------------------------------------------------------------------
# Core primitive: pad / crop a single waveform
# ---------------------------------------------------------------------------


def _pad_or_crop(waveform: torch.Tensor, target: int) -> torch.Tensor:
    """
    Return a 1-D waveform of exactly ``target`` samples.

    - Shorter → right-pad with zeros (silence).
    - Longer  → keep the first ``target`` samples (deterministic crop).
    - Equal   → return as-is (no copy).

    Parameters
    ----------
    waveform:
        1-D float32 tensor, shape ``(n,)``.
    target:
        Desired number of samples.

    Returns
    -------
    torch.Tensor:
        Shape ``(target,)``.
    """
    n = waveform.shape[0]
    if n < target:
        return F.pad(waveform, (0, target - n))
    if n > target:
        return waveform[:target]
    return waveform


# ---------------------------------------------------------------------------
# Bridge: manifest record → sample dict
# ---------------------------------------------------------------------------


def load_sample(
    record: JSONDict,
    label_to_id: Dict[str, int],
) -> Dict[str, object]:
    """
    Convert one manifest record to the sample dict expected by :class:`CollateFn`.

    This is the single place where a raw record (dict from a ``.jsonl`` file)
    becomes a typed Python dict ready for the DataLoader.  Call it from
    ``Dataset.__getitem__``.

    Parameters
    ----------
    record:
        A manifest record as returned by
        :func:`src.utils.jsonl_helpers.load_jsonl`.  Must contain at minimum
        ``chunk_path``, ``zuliano``, ``speaker_id``, and ``chunk_id``.
    label_to_id:
        String-label → integer mapping produced by
        :func:`src.data.manifest.build_label_to_id` (or loaded via
        :func:`src.data.manifest.load_label_to_id`).

    Returns
    -------
    dict:
        ``{"waveform": Tensor(n,), "label": int,
           "chunk_id": str, "speaker_id": str}``

    Raises
    ------
    KeyError:
        If the record's label is absent from *label_to_id*.
    FileNotFoundError / ValueError:
        Propagated from :func:`src.utils.io_helpers.load_pcm_s16le`.
    """
    # PCM → float32 numpy (n,) in [-1, 1]
    audio_np: np.ndarray = load_pcm_s16le(str(record["chunk_path"]))

    # numpy → torch tensor
    waveform = torch.from_numpy(audio_np)

    # String label → integer via the shared mapping
    label_str = "zuliano" if record.get("zuliano") else "non_zuliano"
    label: int = label_to_id[label_str]

    return {
        "waveform":   waveform,
        "label":      label,
        "chunk_id":   str(record.get("chunk_id",   "")),
        "speaker_id": str(record.get("speaker_id", "")),
    }


# ---------------------------------------------------------------------------
# Collate callable
# ---------------------------------------------------------------------------


class CollateFn:
    """
    Collate a list of sample dicts into a :class:`Batch`.

    Pass an instance directly to ``torch.utils.data.DataLoader``::

        collate = CollateFn(target_samples=96_000)
        loader  = DataLoader(dataset, batch_size=32, collate_fn=collate)

    Each sample dict must contain:

    - ``"waveform"``   – :class:`torch.Tensor` shape ``(n,)``, float32
    - ``"label"``      – ``int``
    - ``"chunk_id"``   – ``str`` *(optional, defaults to "")*
    - ``"speaker_id"`` – ``str`` *(optional, defaults to "")*

    Parameters
    ----------
    target_samples:
        Every waveform is padded / cropped to this length before stacking.
        Defaults to :data:`DEFAULT_TARGET_SAMPLES` (96 000 = 6 s @ 16 kHz).
    """

    def __init__(self, target_samples: int = DEFAULT_TARGET_SAMPLES) -> None:
        self.target_samples = target_samples

    def __call__(self, samples: List[Dict[str, object]]) -> Batch:
        """
        Stack a list of sample dicts into a single :class:`Batch`.

        Parameters
        ----------
        samples:
            List of dicts as returned by :func:`load_sample` (or an equivalent
            ``Dataset.__getitem__``).

        Returns
        -------
        Batch:
            ``waveforms`` and ``attention_mask`` shape ``(B, T)``,
            ``labels`` shape ``(B,)``.
        """
        # Real length of each clip, capped at target (used to build the mask).
        # Clips longer than target_samples are cropped, so every sample is real.
        lengths = torch.tensor(
            [min(s["waveform"].shape[0], self.target_samples) for s in samples],
            dtype=torch.int64,
        )                                                           # (B,)

        waveforms = torch.stack(
            [_pad_or_crop(s["waveform"], self.target_samples) for s in samples]
        )                                                           # (B, T)

        # Vectorised mask construction: arange (T,) broadcast vs lengths (B,1)
        attention_mask = (
            torch.arange(self.target_samples, device=waveforms.device).unsqueeze(0) < lengths.unsqueeze(1)
        ).to(torch.int64)                                           # (B, T)

        labels = torch.tensor(
            [s["label"] for s in samples], dtype=torch.int64
        )                                                           # (B,)

        chunk_ids   = [str(s.get("chunk_id",   "")) for s in samples]
        speaker_ids = [str(s.get("speaker_id", "")) for s in samples]

        return Batch(
            waveforms=waveforms,
            attention_mask=attention_mask,
            labels=labels,
            chunk_ids=chunk_ids,
            speaker_ids=speaker_ids,
        )

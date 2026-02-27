"""
PyTorch Dataset for audio chunks produced by the preprocessing pipeline.

Design for parallelism
-----------------------
``torch.utils.data.DataLoader`` parallelises loading by spawning *worker
processes*, each of which receives a **copy** of the dataset object via
``pickle``.  To keep this safe and fast:

- ``__init__`` stores only plain Python objects and (optionally) the
  augmenter ``nn.Module``, which is picklable.
- All I/O (reading the ``.pcm`` file from disk) happens inside
  ``__getitem__``, which runs inside the worker process.
- The manifest is loaded once in the main process and cheaply pickled
  to each worker.

Augmentation
------------
Pass an :class:`~src.augmentation.augmenter.OnTheFlyAugmenter` as the
``augmenter`` argument **only for the training dataset**.  It is called as::

    waveform = augmenter(waveform, sample_rate)   # → (crop_len,) Tensor

The augmenter applies random gain, speed perturbation, Opus codec simulation
and — critically — its own **random crop** to ``cfg.crop_len`` samples.
Because the augmenter already fixes the waveform length, the collate
function's ``_pad_or_crop`` step becomes a no-op for training items when
``CollateFn.target_samples`` is aligned with ``augmenter.cfg.crop_len``::

    aug      = create_augmenter()                     # crop_len = 3 s × 16 000
    train_ds = AudioChunkDataset.from_jsonl(
        "data/splits/train.jsonl", augmenter=aug
    )
    collate  = CollateFn(target_samples=aug.cfg.crop_len)   # ← match!
    loader   = DataLoader(train_ds, batch_size=32,
                          collate_fn=collate, num_workers=4,
                          shuffle=True, pin_memory=True,
                          persistent_workers=True)

Val / test datasets receive no augmenter and are simply padded / cropped by
the collate to its ``target_samples``::

    val_ds = AudioChunkDataset.from_jsonl(
        "data/splits/val.jsonl",
        label_to_id = train_ds.label_to_id,  # same mapping as train
    )

Integration map
---------------
- :mod:`src.utils.jsonl_helpers`       – ``load_jsonl``
- :mod:`src.data.manifest`             – ``validate_manifest``, ``build_label_to_id``
- :mod:`src.data.collate`              – ``load_sample``, ``DEFAULT_SR``
- :mod:`src.augmentation.augmenter`   – ``OnTheFlyAugmenter`` (train only)

Public API
----------
    AudioChunkDataset(records, label_to_id, augmenter, transform, sample_rate)
        Core Dataset class.
    AudioChunkDataset.from_jsonl(path, label_to_id, augmenter, transform,
                                  validate, sample_rate)
        Convenience constructor from a ``.jsonl`` split file.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from src.data.collate import DEFAULT_SR, load_sample
from src.data.manifest import build_label_to_id, validate_manifest
from src.utils.jsonl_helpers import JSONDict, load_jsonl


class AudioChunkDataset(Dataset):
    """
    Map-style dataset over a list of manifest chunk records.

    Each call to ``__getitem__`` reads one ``.pcm`` file from disk and
    returns the sample dict expected by :class:`~src.data.collate.CollateFn`::

        {
            "waveform":   torch.Tensor,  # (n_samples,) float32 in [-1, 1]
            "label":      int,           # integer class id
            "chunk_id":   str,
            "speaker_id": str,
        }

    Parameters
    ----------
    records:
        List of manifest dicts.  Stored in-memory; only the ``chunk_path``
        field triggers I/O, and only inside ``__getitem__``.
    label_to_id:
        String-label → integer mapping.  Build once from the training set
        with :func:`~src.data.manifest.build_label_to_id` and share the
        same object across all splits so class indices are consistent.
    augmenter:
        An :class:`~src.augmentation.augmenter.OnTheFlyAugmenter` instance.
        When provided it is called as ``augmenter(waveform, sample_rate)``
        inside ``__getitem__`` (worker process).  **Pass only for the
        training dataset**; leave ``None`` for val / test.
    transform:
        Optional callable ``(Tensor) -> Tensor`` applied *after* the
        augmenter.  Intended for lightweight, deterministic operations
        such as normalisation or feature extraction.
    sample_rate:
        Sample rate in Hz forwarded to the augmenter.
        Defaults to :data:`~src.data.collate.DEFAULT_SR` (16 000 Hz).
    """

    def __init__(
        self,
        records: List[JSONDict],
        label_to_id: Dict[str, int],
        augmenter: Optional[torch.nn.Module] = None,
        transform: Optional[Callable] = None,
        sample_rate: int = DEFAULT_SR,
    ) -> None:
        self.records     = records
        self.label_to_id = label_to_id
        self.augmenter   = augmenter
        self.transform   = transform
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """
        Load one chunk from disk, augment (training only), and return.

        Execution order inside the DataLoader worker:

        1. ``load_sample`` – PCM → float32 Tensor + label resolution
        2. ``augmenter``   – random crop / gain / speed / opus  *(train only)*
        3. ``transform``   – any deterministic post-augment operation

        Parameters
        ----------
        idx:
            Index into ``self.records``.

        Returns
        -------
        dict:
            ``{"waveform": Tensor(n,), "label": int,
               "chunk_id": str, "speaker_id": str}``
        """
        sample = load_sample(self.records[idx], self.label_to_id)

        # Augmentation runs only when an augmenter was provided (train split).
        # The augmenter crops the waveform to its own cfg.crop_len, so after
        # this step the length is fixed and the collate pad/crop is a no-op.
        if self.augmenter is not None:
            sample["waveform"] = self.augmenter(
                sample["waveform"], self.sample_rate
            )

        if self.transform is not None:
            sample["waveform"] = self.transform(sample["waveform"])

        return sample

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        label_to_id: Optional[Dict[str, int]] = None,
        augmenter: Optional[torch.nn.Module] = None,
        transform: Optional[Callable] = None,
        validate: bool = False,
        sample_rate: int = DEFAULT_SR,
    ) -> "AudioChunkDataset":
        """
        Build a dataset directly from a ``.jsonl`` split file.

        Typical usage — build the label map once from the training set and
        reuse it for val / test so class indices are guaranteed consistent::

            from src.augmentation.augmenter import create_augmenter
            from src.data.collate import CollateFn

            aug      = create_augmenter()   # loads src/augmentation/config.yaml
            train_ds = AudioChunkDataset.from_jsonl(
                "data/splits/train.jsonl",
                augmenter = aug,
            )
            val_ds   = AudioChunkDataset.from_jsonl(
                "data/splits/val.jsonl",
                label_to_id = train_ds.label_to_id,
            )
            # Align collate target with the augmenter's crop length
            collate  = CollateFn(target_samples=aug.cfg.crop_len)

        Parameters
        ----------
        path:
            Path to a ``.jsonl`` split file produced by
            :func:`~src.data.splits.split_manifest`.
        label_to_id:
            Pre-built label mapping.  When *None*, derived from *path*
            via :func:`~src.data.manifest.build_label_to_id`.
        augmenter:
            On-the-fly augmenter for the training split.  Leave ``None``
            for val / test.
        transform:
            Optional post-augment waveform transform.
        validate:
            Run :func:`~src.data.manifest.validate_manifest` on load.
            Useful during development; skip in production.
        sample_rate:
            Forwarded to the augmenter (default 16 000 Hz).

        Returns
        -------
        AudioChunkDataset
        """
        records = load_jsonl(path)

        if validate:
            validate_manifest(records)

        if label_to_id is None:
            label_to_id = build_label_to_id(records)

        return cls(records, label_to_id, augmenter, transform, sample_rate)

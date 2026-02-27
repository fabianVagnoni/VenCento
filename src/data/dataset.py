"""
PyTorch Dataset for audio chunks produced by the preprocessing pipeline.

Design for parallelism
-----------------------
``torch.utils.data.DataLoader`` parallelises loading by spawning *worker
processes*, each of which receives a **copy** of the dataset object via
``pickle``.  To keep this safe and fast:

- ``__init__`` stores only plain Python objects (list of dicts, a dict).
  No open file handles, no unpicklable state.
- All I/O (reading the ``.pcm`` file from disk) happens inside
  ``__getitem__``, which runs inside the worker process.
- The manifest is loaded once in the main process and then cheaply pickled
  to each worker.

Recommended DataLoader setup::

    dataset = AudioChunkDataset.from_jsonl("data/splits/train.jsonl")
    loader  = DataLoader(
        dataset,
        batch_size      = 32,
        shuffle         = True,        # only for training
        num_workers     = 4,           # I/O parallelism
        pin_memory      = True,        # faster host→GPU transfer
        collate_fn      = CollateFn(),
        persistent_workers = True,     # avoid fork overhead between epochs
    )

Integration map
---------------
- :mod:`src.utils.jsonl_helpers`  – ``load_jsonl``  to read split files
- :mod:`src.data.manifest`        – ``validate_manifest``, ``build_label_to_id``
- :mod:`src.data.collate`         – ``load_sample``  (record → sample dict)

Public API
----------
    AudioChunkDataset(records, label_to_id, transform)
        Core Dataset class.
    AudioChunkDataset.from_jsonl(path, label_to_id, transform, validate)
        Convenience constructor: load records from a ``.jsonl`` split file.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from torch.utils.data import Dataset

from src.data.collate import load_sample
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
        List of manifest dicts (e.g. as returned by
        :func:`~src.utils.jsonl_helpers.load_jsonl`).  Stored in-memory;
        only the ``chunk_path`` field triggers I/O, and only inside
        ``__getitem__``.
    label_to_id:
        String-label → integer mapping.  Build once with
        :func:`~src.data.manifest.build_label_to_id` and share the same
        object across train / val / test datasets so class indices are
        consistent.
    transform:
        Optional callable applied to the raw waveform tensor *before* it
        is returned.  Runs inside the DataLoader worker, so it is the right
        place for data augmentation (e.g. ``torchaudio`` transforms).
        Signature: ``(waveform: Tensor) -> Tensor``.
    """

    def __init__(
        self,
        records: List[JSONDict],
        label_to_id: Dict[str, int],
        transform: Optional[Callable] = None,
    ) -> None:
        self.records     = records
        self.label_to_id = label_to_id
        self.transform   = transform

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """
        Load one chunk from disk and return its sample dict.

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
        transform: Optional[Callable] = None,
        validate: bool = False,
    ) -> "AudioChunkDataset":
        """
        Build a dataset directly from a ``.jsonl`` split file.

        This is the recommended entry point.  Typical usage::

            # Build label map once from the training set, reuse for val/test.
            train_ds = AudioChunkDataset.from_jsonl("data/splits/train.jsonl")
            val_ds   = AudioChunkDataset.from_jsonl(
                "data/splits/val.jsonl",
                label_to_id = train_ds.label_to_id,  # same mapping
            )

        Parameters
        ----------
        path:
            Path to a ``.jsonl`` split file produced by
            :func:`~src.data.splits.split_manifest`.
        label_to_id:
            Pre-built label mapping.  When *None*, it is derived from
            the records in *path* via
            :func:`~src.data.manifest.build_label_to_id`.  Always pass
            the training-set mapping to val/test datasets so that class
            indices are identical across splits.
        transform:
            Optional waveform transform (see class docstring).
        validate:
            When ``True``, run :func:`~src.data.manifest.validate_manifest`
            on the loaded records before constructing the dataset.  Useful
            during development; skip in production for speed.

        Returns
        -------
        AudioChunkDataset
        """
        records = load_jsonl(path)

        if validate:
            validate_manifest(records)

        if label_to_id is None:
            label_to_id = build_label_to_id(records)

        return cls(records, label_to_id, transform)

"""
Speaker-wise train / val / test splitting for the processed chunk manifest.

Design decisions
----------------
- Splitting is done **independently within each speaker group** (zuliano vs
  non-zuliano), so target percentages are honoured for both groups separately.
- A speaker is never split across sets: every chunk of a given speaker lands
  in exactly one split.
- All chunks are treated as equally-weighted (chunk count = data size proxy).
- The greedy assignment sorts speakers largest-first and, for each speaker,
  assigns them to whichever non-train split still has the largest unfilled
  quota. Once both quotas are satisfied, remaining speakers go to train.
- The algorithm gives a close approximation of the requested percentages but
  cannot guarantee exactness when speaker sizes are coarse-grained.

Public API
----------
    split_manifest(manifest_path, output_dir, val_frac, test_frac)
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Tuple

from src.utils.jsonl_helpers import JSONDict, load_jsonl, write_jsonl

# ---------------------------------------------------------------------------
# Domain type aliases (specific to splitting; not re-exported from helpers)
# ---------------------------------------------------------------------------

# speaker_id -> list of chunk records belonging to that speaker
SpeakerMap = Dict[str, List[JSONDict]]

# split name ("train" | "val" | "test") -> list of chunk records
SplitMap = Dict[str, List[JSONDict]]


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_by_speaker(records: List[JSONDict]) -> Tuple[SpeakerMap, SpeakerMap]:
    """
    Partition chunk records into two speaker maps: zuliano and non-zuliano.

    Parameters
    ----------
    records:
        Flat list of chunk records (e.g. as returned by :func:`load_jsonl`).

    Returns
    -------
    (zuliano_map, non_zuliano_map):
        Each map has shape ``{speaker_id: [record, ...]}``.
    """
    zuliano: SpeakerMap = defaultdict(list)
    non_zuliano: SpeakerMap = defaultdict(list)

    for rec in records:
        target = zuliano if rec.get("zuliano") else non_zuliano
        target[str(rec["speaker_id"])].append(rec)

    return dict(zuliano), dict(non_zuliano)


# ---------------------------------------------------------------------------
# Greedy speaker assignment
# ---------------------------------------------------------------------------

def assign_speakers(
    speaker_map: SpeakerMap,
    val_frac: float,
    test_frac: float,
) -> Dict[str, str]:
    """
    Greedily assign each speaker to ``"train"``, ``"val"``, or ``"test"``.

    Algorithm
    ---------
    1. Count chunks per speaker.
    2. Sort speakers largest-first so greedy fill is more accurate.
    3. For each speaker, assign them to whichever of ``val`` / ``test`` has
       the largest remaining unfilled quota. Fall back to ``train`` once both
       quotas are satisfied.

    Parameters
    ----------
    speaker_map:
        ``{speaker_id: [chunk_record, ...]}``.
    val_frac:
        Target fraction of chunks for validation (e.g. ``0.10``).
    test_frac:
        Target fraction of chunks for testing (e.g. ``0.20``).

    Returns
    -------
    Dict[str, str]:
        ``{speaker_id: split_name}``.
    """
    if not speaker_map:
        return {}

    counts: Dict[str, int] = {spk: len(recs) for spk, recs in speaker_map.items()}
    total = sum(counts.values())

    target = {"val": val_frac * total, "test": test_frac * total}
    filled = {"val": 0.0, "test": 0.0}
    assignment: Dict[str, str] = {}

    for spk in sorted(counts, key=counts.__getitem__, reverse=True):
        deficit_val  = target["val"]  - filled["val"]
        deficit_test = target["test"] - filled["test"]

        if deficit_test >= deficit_val and deficit_test > 0:
            split = "test"
        elif deficit_val > 0:
            split = "val"
        else:
            split = "train"

        assignment[spk] = split
        if split in filled:
            filled[split] += counts[spk]

    return assignment


# ---------------------------------------------------------------------------
# Applying the assignment
# ---------------------------------------------------------------------------

def apply_assignment(
    speaker_map: SpeakerMap,
    assignment: Dict[str, str],
) -> SplitMap:
    """
    Bucket every chunk record according to its speaker's assigned split.

    Parameters
    ----------
    speaker_map:
        ``{speaker_id: [chunk_record, ...]}``.
    assignment:
        ``{speaker_id: split_name}`` as returned by :func:`assign_speakers`.

    Returns
    -------
    SplitMap:
        ``{"train": [...], "val": [...], "test": [...]}``.
    """
    result: SplitMap = {"train": [], "val": [], "test": []}
    for spk, recs in speaker_map.items():
        result[assignment[spk]].extend(recs)
    return result


# ---------------------------------------------------------------------------
# Merging groups
# ---------------------------------------------------------------------------

def merge_splits(*split_maps: SplitMap) -> SplitMap:
    """
    Concatenate several :class:`SplitMap` dicts into one.

    Used to combine the zuliano and non-zuliano split results before writing.

    Parameters
    ----------
    *split_maps:
        Any number of ``{"train": [...], "val": [...], "test": [...]}`` dicts.

    Returns
    -------
    SplitMap:
        Single merged dict with the same three keys.
    """
    merged: SplitMap = {"train": [], "val": [], "test": []}
    for sm in split_maps:
        for split, records in sm.items():
            merged[split].extend(records)
    return merged


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def split_stats(split_map: SplitMap) -> Dict[str, Dict[str, object]]:
    """
    Return per-split chunk counts and fractions for quick inspection.

    Parameters
    ----------
    split_map:
        ``{"train": [...], "val": [...], "test": [...]}`` dict of records.

    Returns
    -------
    Dict[str, Dict]:
        ``{split: {"n_chunks": int, "fraction": float}}``.
    """
    totals = {split: len(recs) for split, recs in split_map.items()}
    grand_total = sum(totals.values())
    return {
        split: {
            "n_chunks": n,
            "fraction": round(n / grand_total, 4) if grand_total else 0.0,
            "n_speakers": len(set([j["speaker_id"] for j in split_map[split]])),
            "n_zuliano_speakers": sum([1 for j in split_map[split] if j["zuliano"]])
        }
        for split, n in totals.items()
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def split_manifest(
    manifest_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
) -> Dict[str, Dict[str, object]]:
    """
    End-to-end pipeline: load manifest → split by speaker → write outputs.

    The split runs **independently** for zuliano and non-zuliano speakers so
    that both groups honour the requested percentages.

    Parameters
    ----------
    manifest_path:
        Path to the processed ``manifest.jsonl``.
    output_dir:
        Directory where ``train.jsonl``, ``val.jsonl``, ``test.jsonl`` are
        written (created if absent).
    val_frac:
        Target fraction of chunks for validation  (default ``0.10``).
    test_frac:
        Target fraction of chunks for testing     (default ``0.20``).

    Returns
    -------
    Dict[str, Dict]:
        Split statistics from :func:`split_stats` for quick inspection.

    Raises
    ------
    ValueError:
        If ``val_frac + test_frac >= 1.0`` (no room for a training set).
    """
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_frac ({val_frac}) + test_frac ({test_frac}) must be < 1.0"
        )

    records = load_jsonl(manifest_path)

    zuliano_map, non_zuliano_map = group_by_speaker(records)

    zuliano_splits     = apply_assignment(zuliano_map,     assign_speakers(zuliano_map,     val_frac, test_frac))
    non_zuliano_splits = apply_assignment(non_zuliano_map, assign_speakers(non_zuliano_map, val_frac, test_frac))

    final = merge_splits(zuliano_splits, non_zuliano_splits)

    for split_name, split_records in final.items():
        write_jsonl(split_records, f"{output_dir}/{split_name}.jsonl")

    return split_stats(final)

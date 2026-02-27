"""
Manifest schema, validation, label utilities, and query helpers.

A *manifest* is a flat list of chunk records, typically loaded from a .jsonl
file via :func:`src.utils.jsonl_helpers.load_jsonl`.  Each record describes
one audio chunk that will be consumed by the model.

Current record shape (all fields are required unless noted)::

    {
        "audio_id":   "5364f2fddff08cc5",          # hex id of the source file
        "chunk_id":   "5364f2fddff08cc5_000",       # unique per chunk
        "zuliano":    true,                          # dialect label (bool)
        "speaker_id": "spk_008",                    # anonymised speaker tag
        "chunk_path": "data/processed/chunks/...",  # path to .pcm file
        # optional ---
        "sr":         16000,                        # sample rate in Hz
        "duration":   1.92,                         # duration in seconds
    }

Public API
----------
Schema
    ManifestRecord              – TypedDict for one chunk record
    REQUIRED_FIELDS             – list of fields every record must contain
    LABEL_TRUE / LABEL_FALSE    – canonical string label names

Validation  (fail early, explicit errors)
    validate_manifest(...)      – structural check: fields present, label valid
    assert_paths_exist(...)     – every chunk_path must exist on disk
    validate_audio_metadata(...)– sr / speaker_id / label value checks

Summary
    summarize_manifest(...)     – counts, speakers, label & duration stats

Label utilities  (consistent across all scripts)
    build_label_to_id(...)      – derive label→int mapping from records
    encode_labels(...)          – map records to integer labels
    save_label_to_id(...)       – persist mapping to JSON
    load_label_to_id(...)       – reload mapping from JSON

Query helpers
    group_by_speaker(...)       – {speaker_id: [record, …]}
    filter_records(...)         – sub-select by label and/or speaker_id
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Union
from typing import TypedDict

from src.utils.jsonl_helpers import JSONDict

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class _ManifestRecordRequired(TypedDict):
    """Required fields – every manifest record must have all five."""

    audio_id:   str   # hex identifier of the source audio file
    chunk_id:   str   # "<audio_id>_<NNN>" – unique per chunk
    zuliano:    bool  # True = zuliano dialect, False = non-zuliano
    speaker_id: str   # anonymised speaker tag, e.g. "spk_008"
    chunk_path: str   # relative / absolute path to the .pcm file


class ManifestRecord(_ManifestRecordRequired, total=False):
    """
    Typed schema for one manifest row.

    Inherits the five required fields from ``_ManifestRecordRequired``.
    The optional fields below (``sr``, ``duration``) may or may not be
    present; validation functions skip them when absent.
    """

    sr:       int    # sample rate in Hz, e.g. 16000
    duration: float  # chunk duration in seconds


# Fields that every record must have.  Import this constant instead of
# hard-coding field names in other modules.
REQUIRED_FIELDS: List[str] = [
    "audio_id", "chunk_id", "zuliano", "speaker_id", "chunk_path"
]

# Canonical string labels derived from the boolean ``zuliano`` flag.
LABEL_TRUE  = "zuliano"
LABEL_FALSE = "non_zuliano"


def _label_of(rec: JSONDict) -> str:
    """Return the canonical string label for a record."""
    return LABEL_TRUE if rec.get("zuliano") else LABEL_FALSE


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_manifest(
    records: List[JSONDict],
    required_fields: Optional[List[str]] = None,
    allowed_labels: Optional[List[str]] = None,
) -> None:
    """
    Structural validation: every record must contain all required fields and
    (optionally) only use labels from the allowed set.

    Checks are intentionally strict – the function raises on the *first*
    problem found so the error is as specific as possible.

    Parameters
    ----------
    records:
        List of chunk records to validate.
    required_fields:
        Fields that must be present in every record.
        Defaults to :data:`REQUIRED_FIELDS`.
    allowed_labels:
        Set of permitted string labels (e.g. ``["zuliano", "non_zuliano"]``).
        When *None* no label constraint is applied.

    Raises
    ------
    ValueError:
        On the first record that fails any check, with a descriptive message
        that includes the record index and offending field/value.
    """
    if required_fields is None:
        required_fields = REQUIRED_FIELDS

    for idx, rec in enumerate(records):
        # --- field presence ---
        for field in required_fields:
            if field not in rec:
                raise ValueError(
                    f"Record {idx} is missing required field '{field}': {rec}"
                )

        # --- label allow-list ---
        if allowed_labels is not None:
            label = _label_of(rec)
            if label not in allowed_labels:
                raise ValueError(
                    f"Record {idx} has label '{label}' which is not in "
                    f"allowed_labels={allowed_labels}: {rec}"
                )


def assert_paths_exist(
    records: List[JSONDict],
    key: str = "chunk_path",
) -> None:
    """
    Assert that the path stored in *key* exists on disk for every record.

    This is intentionally a separate function from :func:`validate_manifest`
    because path checks are expensive and may be skipped in CI environments
    where the audio files are not mounted.

    Parameters
    ----------
    records:
        List of chunk records.
    key:
        The field holding the file path (default: ``"chunk_path"``).

    Raises
    ------
    FileNotFoundError:
        On the first record whose path does not exist, with the record index
        and the missing path.
    """
    for idx, rec in enumerate(records):
        path = Path(str(rec[key]))
        if not path.exists():
            raise FileNotFoundError(
                f"Record {idx}: path does not exist – {path}"
            )


def validate_audio_metadata(
    records: List[JSONDict],
    expected_sr: Optional[int] = 16_000,
    allowed_labels: Optional[List[str]] = None,
) -> None:
    """
    Value-level sanity checks on audio metadata fields.

    Catches "silent bugs" such as all zuliano clips having ``sr=48000``.
    Fields that are absent in a record are skipped (future records may not
    carry all optional fields).

    Parameters
    ----------
    records:
        List of chunk records.
    expected_sr:
        Expected sample rate in Hz.  Pass ``None`` to skip sr validation.
    allowed_labels:
        Permitted label strings.  Defaults to ``[LABEL_TRUE, LABEL_FALSE]``.

    Raises
    ------
    ValueError:
        On the first record that violates any check.
    """
    if allowed_labels is None:
        allowed_labels = [LABEL_TRUE, LABEL_FALSE]

    for idx, rec in enumerate(records):
        # speaker_id must be a non-empty string
        spk = str(rec.get("speaker_id", "")).strip()
        if not spk:
            raise ValueError(f"Record {idx} has empty speaker_id: {rec}")

        # label must be in the allowed set
        label = _label_of(rec)
        if label not in allowed_labels:
            raise ValueError(
                f"Record {idx} has label '{label}' not in {allowed_labels}: {rec}"
            )

        # sample rate check (only when field is present)
        if expected_sr is not None and "sr" in rec:
            sr = rec["sr"]
            if sr != expected_sr:
                raise ValueError(
                    f"Record {idx} has sr={sr}, expected {expected_sr}: {rec}"
                )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize_manifest(records: List[JSONDict]) -> Dict[str, object]:
    """
    Return a concise statistical summary of a manifest.

    Parameters
    ----------
    records:
        List of chunk records.

    Returns
    -------
    dict with keys:

    ``n_chunks``
        Total number of chunk records.
    ``n_speakers``
        Number of unique speaker IDs.
    ``label_distribution``
        ``{label_string: count}`` over all records.
    ``duration_stats``
        ``{min, mean, max}`` in seconds, or ``None`` if no record has a
        ``duration`` field.
    """
    n_chunks = len(records)
    speakers = {str(r["speaker_id"]) for r in records if "speaker_id" in r}

    label_dist: Dict[str, int] = {}
    for rec in records:
        lbl = _label_of(rec)
        label_dist[lbl] = label_dist.get(lbl, 0) + 1

    durations = [float(r["duration"]) for r in records if "duration" in r]
    if durations:
        duration_stats: Optional[Dict[str, float]] = {
            "min":  min(durations),
            "mean": statistics.mean(durations),
            "max":  max(durations),
        }
    else:
        duration_stats = None

    return {
        "n_chunks":           n_chunks,
        "n_speakers":         len(speakers),
        "label_distribution": label_dist,
        "duration_stats":     duration_stats,
    }


# ---------------------------------------------------------------------------
# Label mapping utilities
# ---------------------------------------------------------------------------


def build_label_to_id(records: List[JSONDict]) -> Dict[str, int]:
    """
    Derive a deterministic ``label → int`` mapping from the records.

    Labels are sorted alphabetically so the mapping is stable regardless of
    the order records appear in the manifest.

    Parameters
    ----------
    records:
        List of chunk records.

    Returns
    -------
    Dict[str, int]:
        E.g. ``{"non_zuliano": 0, "zuliano": 1}``.
    """
    labels = sorted({_label_of(r) for r in records})
    return {lbl: idx for idx, lbl in enumerate(labels)}


def encode_labels(
    records: List[JSONDict],
    label_to_id: Dict[str, int],
) -> List[int]:
    """
    Map each record to its integer label.

    Parameters
    ----------
    records:
        List of chunk records.
    label_to_id:
        Mapping from string label to integer, as returned by
        :func:`build_label_to_id`.

    Returns
    -------
    List[int]:
        One integer per record, in the same order.

    Raises
    ------
    KeyError:
        If a record's label is not present in *label_to_id*.
    """
    return [label_to_id[_label_of(r)] for r in records]


def save_label_to_id(
    path: Union[str, os.PathLike],
    mapping: Dict[str, int],
) -> None:
    """
    Persist a label→int mapping to a JSON file.

    Parent directories are created automatically.  Use this alongside model
    checkpoints so the class ordering is never ambiguous.

    Parameters
    ----------
    path:
        Destination .json file.
    mapping:
        Dict returned by :func:`build_label_to_id`.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2, ensure_ascii=False)


def load_label_to_id(path: Union[str, os.PathLike]) -> Dict[str, int]:
    """
    Reload a label→int mapping that was saved by :func:`save_label_to_id`.

    Parameters
    ----------
    path:
        Path to the .json file.

    Returns
    -------
    Dict[str, int]:
        The restored mapping.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def group_by_speaker(records: List[JSONDict]) -> Dict[str, List[JSONDict]]:
    """
    Partition records into per-speaker buckets.

    Unlike the two-map variant in :mod:`src.data.splits` (which separates
    zuliano vs non-zuliano groups), this function returns a single flat dict
    keyed by ``speaker_id``.  Use it for leakage tests, per-speaker stats,
    and any code that does not need the dialect split.

    Parameters
    ----------
    records:
        Flat list of chunk records.

    Returns
    -------
    Dict[str, List[JSONDict]]:
        ``{speaker_id: [record, …]}``.
    """
    result: Dict[str, List[JSONDict]] = {}
    for rec in records:
        spk = str(rec["speaker_id"])
        result.setdefault(spk, []).append(rec)
    return result


def filter_records(
    records: List[JSONDict],
    *,
    label: Optional[str] = None,
    speaker_id: Optional[str] = None,
) -> List[JSONDict]:
    """
    Return records matching all specified filters.

    Each keyword argument acts as an independent AND filter; omitting it means
    "no constraint on that dimension".

    Parameters
    ----------
    records:
        List of chunk records to filter.
    label:
        If given, keep only records whose string label equals this value
        (``"zuliano"`` or ``"non_zuliano"``).
    speaker_id:
        If given, keep only records from this speaker.

    Returns
    -------
    List[JSONDict]:
        Filtered subset of *records* (original dicts, not copies).

    Examples
    --------
    >>> zuliano_spk8 = filter_records(records, label="zuliano", speaker_id="spk_008")
    """
    out = records
    if label is not None:
        out = [r for r in out if _label_of(r) == label]
    if speaker_id is not None:
        out = [r for r in out if str(r.get("speaker_id", "")) == speaker_id]
    return out

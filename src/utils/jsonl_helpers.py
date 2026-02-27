"""
Generic utilities for reading, writing, and querying JSON Lines (.jsonl) files.

This module is intentionally domain-agnostic: it knows nothing about speakers,
splits, or manifests. Any file in the project that needs to touch a .jsonl file
should import from here rather than re-implementing I/O.

Public API
----------
    iter_jsonl(path)                   – stream records lazily (generator)
    load_jsonl(path)                   – load all records into a list
    write_jsonl(records, path)         – write records to a .jsonl file
    jsonl_to_csv(jsonl_path, csv_path) – convert a .jsonl to .csv
    find_records_by_path(...)          – look up records by a path field value
    find_first_record_by_path(...)     – convenience single-match wrapper
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

# ---------------------------------------------------------------------------
# Shared type alias – imported by any module that handles JSONL records
# ---------------------------------------------------------------------------

JSONDict = Dict[str, object]


# ---------------------------------------------------------------------------
# I/O primitives
# ---------------------------------------------------------------------------

def iter_jsonl(path: Union[str, os.PathLike]) -> Iterable[JSONDict]:
    """
    Lazily stream JSON objects from a .jsonl file (one per non-blank line).

    Prefer this over :func:`load_jsonl` when the file is large and you do not
    need all records in memory at once.

    Parameters
    ----------
    path:
        Path to the .jsonl file.

    Yields
    ------
    JSONDict:
        Parsed JSON object for each non-blank line.

    Raises
    ------
    ValueError:
        On malformed JSON, with the offending line number.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc


def load_jsonl(path: Union[str, os.PathLike]) -> List[JSONDict]:
    """
    Load all records from a .jsonl file into a list.

    Thin wrapper around :func:`iter_jsonl` for callers that need random access
    or need to know the total record count upfront.

    Parameters
    ----------
    path:
        Path to the .jsonl file.

    Returns
    -------
    List[JSONDict]:
        All records in file order.
    """
    return list(iter_jsonl(path))


def write_jsonl(
    records: Iterable[JSONDict],
    path: Union[str, os.PathLike],
) -> None:
    """
    Write an iterable of dicts to a .jsonl file (one JSON object per line).

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    records:
        Iterable of dicts to serialise.
    path:
        Destination .jsonl file path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CSV conversion
# ---------------------------------------------------------------------------

def jsonl_to_csv(
    jsonl_path: Union[str, os.PathLike],
    csv_path: Union[str, os.PathLike],
    *,
    fieldnames: Optional[List[str]] = None,
) -> List[str]:
    """
    Convert a .jsonl file to .csv.

    Parameters
    ----------
    jsonl_path:
        Path to the input .jsonl file.
    csv_path:
        Path to the output .csv file (parent dirs created if absent).
    fieldnames:
        Explicit column order. If omitted, columns are inferred from all
        records in first-seen key order.

    Returns
    -------
    List[str]:
        The column names written to the CSV header.
    """
    records = load_jsonl(jsonl_path)

    if fieldnames is None:
        seen: set = set()
        fieldnames = []
        for obj in records:
            for key in obj:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for obj in records:
            writer.writerow(obj)

    return fieldnames


# ---------------------------------------------------------------------------
# Path-based record lookup
# ---------------------------------------------------------------------------

def _norm_path(p: Union[str, os.PathLike]) -> str:
    """
    Normalise a path for cross-platform comparison.

    - Expands ``~``
    - Resolves to absolute (best-effort; path need not exist)
    - Converts backslashes to forward slashes
    - Lowercases on Windows to avoid case-sensitivity mismatches
    """
    s = os.path.expanduser(os.fspath(p))
    try:
        s = str(Path(s).resolve(strict=False))
    except Exception:
        s = os.path.abspath(s)
    s = s.replace("\\", "/")
    if os.name == "nt":
        s = s.lower()
    return s


@dataclass(frozen=True)
class Match:
    """A record found by :func:`find_records_by_path`, with source context."""

    obj: JSONDict
    line_no: int
    matched_key: str


def find_records_by_path(
    jsonl_path: Union[str, os.PathLike],
    target_path: Union[str, os.PathLike],
    *,
    keys: Union[str, List[str]] = "file_path",
    mode: str = "exact",
    first_only: bool = False,
) -> List[Match]:
    """
    Find records in a .jsonl file whose path field matches *target_path*.

    Parameters
    ----------
    jsonl_path:
        Path to the .jsonl file to search.
    target_path:
        The path value to look for.
    keys:
        JSON key(s) whose value is treated as a file path.
    mode:
        ``"exact"``    – normalised equality,
        ``"contains"`` – substring match between normalised paths,
        ``"basename"`` – filename-only match.
    first_only:
        Return at most one result (short-circuits the search).

    Returns
    -------
    List[Match]:
        All matching records with their line number and matched key.
    """
    if isinstance(keys, str):
        keys = [keys]

    if mode not in {"exact", "contains", "basename"}:
        raise ValueError(f"Unknown mode '{mode}'. Use 'exact', 'contains', or 'basename'.")

    t_norm = _norm_path(target_path)
    t_base = Path(os.fspath(target_path)).name
    hits: List[Match] = []

    for line_no, raw in enumerate(
        open(jsonl_path, "r", encoding="utf-8"), start=1
    ):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj: JSONDict = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

        for key in keys:
            val = obj.get(key)
            if not val:
                continue
            rec_path = os.fspath(val)
            rec_norm = _norm_path(rec_path)
            rec_base = Path(rec_path).name

            matched = (
                rec_norm == t_norm                              if mode == "exact"    else
                (t_norm in rec_norm or rec_norm in t_norm)     if mode == "contains" else
                rec_base == t_base
            )
            if matched:
                hits.append(Match(obj=obj, line_no=line_no, matched_key=key))
                if first_only:
                    return hits

    return hits


def find_first_record_by_path(
    jsonl_path: Union[str, os.PathLike],
    target_path: Union[str, os.PathLike],
    *,
    keys: Union[str, List[str]] = "file_path",
    mode: str = "exact",
) -> Optional[Match]:
    """
    Return the first record matching *target_path*, or ``None``.

    Convenience wrapper around :func:`find_records_by_path`.
    """
    hits = find_records_by_path(
        jsonl_path, target_path, keys=keys, mode=mode, first_only=True
    )
    return hits[0] if hits else None

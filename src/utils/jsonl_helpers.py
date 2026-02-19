"""a"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


JSONDict = Dict[str, Any]


def _norm_path(p: Union[str, os.PathLike]) -> str:
    """
    Normalize a path for robust matching across Windows/POSIX:
    - expands ~
    - resolves/absolutizes (best-effort; file need not exist)
    - converts backslashes to forward slashes
    - lowercases on Windows to avoid case-mismatch surprises
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


def iter_jsonl(jsonl_path: Union[str, os.PathLike]) -> Iterable[JSONDict]:
    """
    Stream JSON objects from a JSON Lines (.jsonl) file.
    Raises ValueError with line number on invalid JSON.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e


@dataclass(frozen=True)
class Match:
    """A match result with context."""
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
    Find JSON objects in a JSONL file whose path field matches `target_path`.

    Parameters
    ----------
    jsonl_path:
        Path to the .jsonl file.
    target_path:
        The path you want to match (string or PathLike).
    keys:
        JSON key(s) to check. Default "file_path".
        Example: ["file_path", "source_file_path"]
    mode:
        - "exact": normalized path equality
        - "contains": substring match between normalized paths
        - "basename": match only by filename (basename)
    first_only:
        If True, return at most one Match.

    Returns
    -------
    List[Match]:
        Each Match contains (obj, line_no, matched_key).
    """
    if isinstance(keys, str):
        keys = [keys]

    if mode not in {"exact", "contains", "basename"}:
        raise ValueError(f"Unknown mode '{mode}'. Use 'exact', 'contains', or 'basename'.")

    t_norm = _norm_path(target_path)
    t_base = Path(os.fspath(target_path)).name

    hits: List[Match] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj: JSONDict = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e

            for key in keys:
                val = obj.get(key)
                if not val:
                    continue

                rec_path = os.fspath(val)
                rec_norm = _norm_path(rec_path)
                rec_base = Path(rec_path).name

                ok = False
                if mode == "exact":
                    ok = (rec_norm == t_norm)
                elif mode == "contains":
                    ok = (t_norm in rec_norm) or (rec_norm in t_norm)
                elif mode == "basename":
                    ok = (rec_base == t_base)

                if ok:
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
    Convenience wrapper returning only the first match (or None).
    """
    matches = find_records_by_path(
        jsonl_path,
        target_path,
        keys=keys,
        mode=mode,
        first_only=True,
    )
    return matches[0] if matches else None

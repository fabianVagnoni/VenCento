#!/usr/bin/env python3
"""
Import a sample of Common Voice clips into data/raw and register them in manifest.jsonl.

The script:
  1. Accepts a list of clip filenames/paths (from the cv-corpus-24 dataset).
  2. Copies them from the CV clips directory into data/raw/.
  3. Looks up each clip in the TSV to retrieve metadata (client_id, sentence, etc.).
  4. Appends a manifest entry for each clip so preprocessing.py can process them normally.

Usage examples
--------------
# Pass clip names directly:
  python scripts/cv_importer.py common_voice_es_18390729.mp3 common_voice_es_18390730.mp3

# Pass a file with one path/filename per line:
  python scripts/cv_importer.py --paths-file my_selection.txt

# Override defaults:
  python scripts/cv_importer.py --paths-file selection.txt \\
      --clips-dir "data/raw/cv-corpus-24.0-2025-12-05-es/cv-corpus-24.0-2025-12-05/es/clips" \\
      --tsv "data/raw/cv-corpus-24.0-2025-12-05-es/cv-corpus-24.0-2025-12-05/es/train.tsv" \\
      --out data/raw --manifest data/raw/manifest.jsonl

Manifest fields written per clip
---------------------------------
  id          : deterministic hex ID derived from the clip filename
  platform    : "cv-corpus-24"
  url         : null
  media       : null
  zuliano     : false
  speaker_id  : "spk_cv_<id>"
  segments    : null
  file_path   : relative path inside data/raw after copying
  kind        : "full"
  client_id   : original CV client_id (from TSV, if found)
  sentence    : transcription text (from TSV, if found)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CLIPS_DIR = (
    "data/raw/cv-corpus-24.0-2025-12-05-es/cv-corpus-24.0-2025-12-05/es/clips"
)
_DEFAULT_TSV = (
    "data/raw/cv-corpus-24.0-2025-12-05-es/cv-corpus-24.0-2025-12-05/es/train.tsv"
)
_DEFAULT_OUT = "data/raw"
_DEFAULT_MANIFEST = "data/raw/manifest.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_cv_id(filename: str, length: int = 16) -> str:
    """Deterministic ID based on the original clip filename."""
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:length]


def load_tsv_index(tsv_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load the Common Voice TSV and return a dict keyed by the `path` column
    (the clip filename, e.g. 'common_voice_es_18390729.mp3').
    Returns an empty dict if the file doesn't exist.
    """
    if not tsv_path.exists():
        print(f"WARNING: TSV not found at {tsv_path} — metadata fields will be omitted.")
        return {}

    try:
        import csv

        index: Dict[str, Dict[str, Any]] = {}
        with tsv_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                key = row.get("path", "").strip()
                if key:
                    index[key] = dict(row)
        print(f"Loaded {len(index):,} entries from TSV.")
        return index
    except Exception as exc:
        print(f"WARNING: Could not parse TSV ({exc}) — metadata fields will be omitted.")
        return {}


def load_existing_file_paths(manifest_path: Path) -> set[str]:
    """Return the set of file_path values already in the manifest (to skip duplicates)."""
    existing: set[str] = set()
    if not manifest_path.exists():
        return existing
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fp = rec.get("file_path")
                if fp:
                    existing.add(fp)
            except json.JSONDecodeError:
                pass
    return existing


def append_jsonl(jsonl_path: Path, record: Dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_clip_path(name_or_path: str, clips_dir: Path) -> Optional[Path]:
    """
    Accept either:
      - a bare filename:  common_voice_es_12345.mp3
      - an absolute/relative path pointing directly to the file
    Returns the resolved Path if it exists, else None.
    """
    p = Path(name_or_path)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to CWD first
    if p.exists():
        return p.resolve()
    # Treat as a bare filename inside clips_dir
    candidate = clips_dir / p.name
    if candidate.exists():
        return candidate
    return None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def import_clips(
    names: List[str],
    clips_dir: Path,
    tsv_path: Path,
    out_dir: Path,
    manifest_path: Path,
    dry_run: bool = False,
    overwrite: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    tsv_index = load_tsv_index(tsv_path)
    existing_paths = load_existing_file_paths(manifest_path)

    successes: List[str] = []
    skipped: List[str] = []
    failures: List[str] = []

    for name in names:
        name = name.strip()
        if not name:
            continue

        src = resolve_clip_path(name, clips_dir)
        if src is None:
            print(f"MISSING: {name!r} — not found in {clips_dir}. Skipping.")
            failures.append(name)
            continue

        filename = src.name
        dst = out_dir / filename

        # Build the manifest file_path (always forward-slash, relative)
        rel_dst = dst.relative_to(Path(".").resolve()) if dst.is_absolute() else dst
        # Use forward slashes for cross-platform consistency
        file_path_str = str(rel_dst).replace("\\", "/")

        if file_path_str in existing_paths and not overwrite:
            print(f"SKIP (already in manifest): {filename}")
            skipped.append(name)
            continue

        # Generate deterministic ID
        clip_id = make_cv_id(filename)
        speaker_id = f"spk_cv_{clip_id}"

        # Look up TSV metadata
        meta = tsv_index.get(filename, {})
        client_id: Optional[str] = meta.get("client_id") or None
        sentence: Optional[str] = meta.get("sentence") or None
        accents: Optional[str] = meta.get("accents") or None

        # Build the manifest record
        record: Dict[str, Any] = {
            "id": clip_id,
            "platform": "cv-corpus-24",
            "url": None,
            "media": None,
            "zuliano": False,
            "speaker_id": speaker_id,
            "segments": None,
            "file_path": file_path_str,
            "kind": "full",
        }
        if client_id is not None:
            record["client_id"] = client_id
        if sentence is not None:
            record["sentence"] = sentence
        if accents is not None:
            record["accents"] = accents

        if dry_run:
            print(f"[DRY RUN] Would copy: {src} -> {dst}")
            print(f"[DRY RUN] Would append to manifest: {json.dumps(record, ensure_ascii=False)}")
            successes.append(name)
            continue

        # Copy file
        if dst.exists() and not overwrite:
            print(f"NOTE: {dst} already exists, skipping copy (use --overwrite to replace).")
        else:
            shutil.copy2(src, dst)
            print(f"Copied: {src.name} -> {dst}")

        # Append to manifest
        append_jsonl(manifest_path, record)
        print(f"Manifest +1: {filename} (id={clip_id})")
        existing_paths.add(file_path_str)
        successes.append(name)

    # Summary
    print(f"\nDone.")
    print(f"  Imported : {len(successes)}")
    print(f"  Skipped  : {len(skipped)}")
    print(f"  Failed   : {len(failures)}")
    print(f"  Manifest : {manifest_path}")
    if failures:
        print("\nMissing files:")
        for f in failures:
            print(f"  - {f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Import Common Voice clips into data/raw and register in manifest.jsonl."
    )
    ap.add_argument(
        "clips",
        nargs="*",
        metavar="CLIP",
        help="One or more clip filenames or paths (e.g. common_voice_es_12345.mp3).",
    )
    ap.add_argument(
        "--paths-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Text file with one clip filename/path per line (can be combined with positional args).",
    )
    ap.add_argument(
        "--clips-dir",
        type=Path,
        default=Path(_DEFAULT_CLIPS_DIR),
        metavar="DIR",
        help=f"Directory containing the CV clips. Default: {_DEFAULT_CLIPS_DIR}",
    )
    ap.add_argument(
        "--tsv",
        type=Path,
        default=Path(_DEFAULT_TSV),
        metavar="FILE",
        help=f"Common Voice TSV file for metadata lookup. Default: {_DEFAULT_TSV}",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(_DEFAULT_OUT),
        metavar="DIR",
        help=f"Destination directory for audio files. Default: {_DEFAULT_OUT}",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path(_DEFAULT_MANIFEST),
        metavar="FILE",
        help=f"Path to the manifest JSONL. Default: {_DEFAULT_MANIFEST}",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-copy and re-register clips that are already in the manifest.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without copying files or modifying the manifest.",
    )
    args = ap.parse_args()

    # Collect clip names from positional args and/or paths file
    names: List[str] = list(args.clips)
    if args.paths_file is not None:
        if not args.paths_file.exists():
            ap.error(f"--paths-file not found: {args.paths_file}")
        lines = args.paths_file.read_text(encoding="utf-8").splitlines()
        names.extend(line.strip() for line in lines if line.strip())

    if not names:
        ap.error("No clip paths provided. Pass clip filenames as arguments or use --paths-file.")

    import_clips(
        names=names,
        clips_dir=args.clips_dir,
        tsv_path=args.tsv,
        out_dir=args.out,
        manifest_path=args.manifest,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

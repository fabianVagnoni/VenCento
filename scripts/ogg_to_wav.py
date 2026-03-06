#!/usr/bin/env python3
"""Convert all .ogg files in data/raw to .wav, save in the same dir, and delete the .ogg."""

from pathlib import Path

import soundfile as sf

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    raw_dir = RAW_DIR
    if not raw_dir.exists():
        print(f"Directory not found: {raw_dir}")
        return

    ogg_files = list(raw_dir.glob("*.ogg"))
    if not ogg_files:
        print(f"No .ogg files found in {raw_dir}")
        return

    for ogg_path in ogg_files:
        wav_path = ogg_path.with_suffix(".wav")
        try:
            data, samplerate = sf.read(ogg_path)
            sf.write(wav_path, data, samplerate)
            ogg_path.unlink()
            print(f"Converted: {ogg_path.name} -> {wav_path.name} (removed .ogg)")
        except Exception as e:
            print(f"Failed {ogg_path.name}: {e}")


if __name__ == "__main__":
    main()

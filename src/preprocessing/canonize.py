"""
Canonize audio files to raw PCM (default: 16 kHz, mono, s16le).

CLI:
  py -m src.preprocessing.canonize path/to/input.wav
  py -m src.preprocessing.canonize path/to/input.wav --output data/processed/custom_name.pcm
"""

import argparse
import subprocess
from pathlib import Path

FFMPEG_BIN = Path(
    r"C:/Users/fabia/AppData/Local/Microsoft/WinGet/Packages/"
    r"Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin"
)
FFMPEG_EXE = FFMPEG_BIN / "ffmpeg.exe"


def default_output_path(input_path: str) -> str:
    """
    Default output: data/processed/<input_filename>.pcm
    Example: data/raw/foo.wav -> data/processed/foo.pcm
    """
    in_path = Path(input_path)
    return str(Path("data") / "processed" / (in_path.stem + ".pcm"))


def canonize(
    wav_path: str,
    pcm_path: str,
    sr: int = 16000,
    channels: int = 1,
    sample_fmt: str = "s16le",
) -> None:
    """
    Canonizes audio files to PCM format (default 16kHz, 1 channel, s16le).

    Args:
        wav_path: Path to input WAV file
        pcm_path: Path to output PCM file
        sr: Sample rate
        channels: Number of channels
        sample_fmt: Sample format

    Returns:
        None
    """
    wav_path = str(Path(wav_path))
    pcm_path = str(Path(pcm_path))

    if not FFMPEG_EXE.exists():
        raise FileNotFoundError(f"ffmpeg.exe not found at: {FFMPEG_EXE}")

    # Ensure output directory exists
    Path(pcm_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(FFMPEG_EXE),
        "-y",
        "-i",
        wav_path,
        "-f",
        sample_fmt,
        "-acodec",
        f"pcm_{sample_fmt}",
        "-ar",
        str(sr),
        "-ac",
        str(channels),
        pcm_path,
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert audio to raw PCM using ffmpeg.")
    parser.add_argument("input", help="Input audio file (e.g., .wav)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output raw PCM file (default: data/processed/<input_name>.pcm)",
    )
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (default: 16000)")
    parser.add_argument("--channels", type=int, default=1, help="Channels (default: 1)")
    parser.add_argument("--sample-fmt", default="s16le", help="Sample format (default: s16le)")
    args = parser.parse_args()

    out_path = args.output or default_output_path(args.input)
    canonize(args.input, out_path, sr=args.sr, channels=args.channels, sample_fmt=args.sample_fmt)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()


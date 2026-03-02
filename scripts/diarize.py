import sys
from pathlib import Path
import warnings

# ---- silence torchcodec noise (optional) ----
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly*")
warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated*")
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0*")

# ---- CRITICAL: patch torchaudio BEFORE importing speechbrain/pyannote ----
import torchaudio
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]  # any non-empty list is fine

import soundfile as sf
import torch

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


def load_audio(path: Path) -> dict:
    waveform, sample_rate = sf.read(str(path), dtype="float32")
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    else:
        waveform = waveform.T
    return {"waveform": torch.from_numpy(waveform), "sample_rate": sample_rate}


def seconds_to_hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main(audio_path: str, hf_token: str, num_speakers: int = 2, out_path: str = "segments.txt"):
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(audio_file)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )

    # Optional: speed up if CUDA exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    print("Using device:", device)

    audio_dict = load_audio(audio_file)

    with ProgressHook() as hook:
        diarization = pipeline(audio_dict, num_speakers=num_speakers, hook=hook)

    lines = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        lines.append(f"{seconds_to_hhmmss(turn.start)} - {seconds_to_hhmmss(turn.end)}  {speaker}")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(lines)} segments to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: py -m scripts.diarize <audio_file> <HF_TOKEN> [num_speakers] [out_file]")
        sys.exit(1)

    audio = sys.argv[1]
    token = sys.argv[2]
    nspk = int(sys.argv[3]) if len(sys.argv) >= 4 else 2
    out = sys.argv[4] if len(sys.argv) >= 5 else "segments.txt"
    main(audio, token, nspk, out)
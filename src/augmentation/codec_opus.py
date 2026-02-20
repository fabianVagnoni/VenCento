import io
import math
import random
import subprocess
import tempfile
import wave
from typing import Iterable, Optional, Sequence, Tuple

import torch

try:
    import torchaudio
except Exception:
    torchaudio = None


def _to_mono_1d(x: torch.Tensor) -> torch.Tensor:
    # Accept (T,), (C,T), (B,T), (B,C,T) -> return (B,T) mono (B may be 1)
    if x.dim() == 1:          # (T,)
        return x[None, :]     # (1,T)
    if x.dim() == 2:          # (C,T) or (B,T) ambiguous; assume (T,) batch is more common => treat as (B,T)
        return x              # (B,T)
    if x.dim() == 3:          # (B,C,T)
        return x.mean(dim=1)  # downmix to mono
    raise ValueError(f"Unsupported shape {tuple(x.shape)}")


def _write_wav_int16(path: str, y: torch.Tensor, sample_rate: int) -> None:
    """y: (T,) float in [-1,1] (roughly)."""
    y = torch.clamp(y, -1.0, 1.0)
    pcm = (y * 32767.0).round().to(torch.int16).cpu().numpy().tobytes()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def _read_wav_float32(path: str) -> Tuple[torch.Tensor, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        if n_channels != 1 or sampwidth != 2:
            raise RuntimeError("Expected mono int16 WAV from decode step.")
        n_frames = wf.getnframes()
        pcm = wf.readframes(n_frames)
    y = torch.frombuffer(pcm, dtype=torch.int16).to(torch.float32) / 32767.0
    return y.clone(), sr  # clone to detach from underlying buffer


def codec_opus(
    x: torch.Tensor,
    sample_rate: int,
    implementation: str = "ffmpeg_libopus",  # or "torchaudio"
    bitrate_kbps_choices: Sequence[int] = (10, 12, 16, 20, 24),
    hard_mode_enabled: bool = True,
    hard_mode_p: float = 0.10,
    hard_mode_bitrate_kbps_choices: Sequence[int] = (8, 10),
) -> torch.Tensor:
    """
    WhatsApp/VoIP-like robustness: simulate Opus encode->decode.

    - implementation="ffmpeg_libopus": uses `ffmpeg` CLI + libopus (must be installed).
    - implementation="torchaudio": uses torchaudio's FFmpeg integration to write/read Opus (also requires FFmpeg).
      Note: torchaudio streaming encode/decode APIs are deprecated in torchaudio 2.8 and planned for removal in 2.9. :contentReference[oaicite:0]{index=0}

    Shapes:
      - (T,), (B,T), (C,T), (B,C,T)
    Output:
      - returns mono audio, same batch shape as input where possible.
    """
    device = x.device
    dtype = x.dtype

    # choose bitrate (hard mode bucket)
    if hard_mode_enabled and random.random() < hard_mode_p:
        bitrate_kbps = int(random.choice(list(hard_mode_bitrate_kbps_choices)))
    else:
        bitrate_kbps = int(random.choice(list(bitrate_kbps_choices)))

    xb = _to_mono_1d(x)  # (B,T) mono
    B, T = xb.shape

    # Opus is typically encoded/decoded at 48kHz in common toolchains; FFmpeg decoding uses 48kHz. :contentReference[oaicite:1]{index=1}
    target_sr = 48_000

    def process_one(wav_1d: torch.Tensor) -> torch.Tensor:
        # Always do codec I/O on CPU float32
        wav_1d = wav_1d.detach()
        wav_cpu = wav_1d.to("cpu", torch.float32)

        # resample to 48k for Opus friendliness
        if sample_rate != target_sr:
            if torchaudio is None:
                raise RuntimeError("torchaudio is required for resampling when sample_rate != 48000.")
            wav_cpu = torchaudio.functional.resample(wav_cpu, sample_rate, target_sr)

        if implementation == "ffmpeg_libopus":
            with tempfile.TemporaryDirectory() as td:
                in_wav = f"{td}/in.wav"
                out_ogg = f"{td}/encoded.ogg"
                out_wav = f"{td}/decoded.wav"

                _write_wav_int16(in_wav, wav_cpu, target_sr)

                # Encode Opus in Ogg container, then decode back to WAV.
                # (libopus encoder via ffmpeg; bitrate in kbps)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        in_wav,
                        "-c:a",
                        "libopus",
                        "-b:a",
                        f"{bitrate_kbps}k",
                        out_ogg,
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        out_ogg,
                        "-ac",
                        "1",
                        "-ar",
                        str(target_sr),
                        out_wav,
                    ],
                    check=True,
                )
                y, sr2 = _read_wav_float32(out_wav)
                if sr2 != target_sr:
                    raise RuntimeError(f"Unexpected decoded sample rate {sr2}")

        elif implementation == "torchaudio":
            if torchaudio is None:
                raise RuntimeError("torchaudio is not available.")
            # Keep it simple: still write/read through torchaudio to temp files.
            # (Uses FFmpeg libs under the hood when available.) :contentReference[oaicite:2]{index=2}
            with tempfile.TemporaryDirectory() as td:
                out_ogg = f"{td}/encoded.ogg"
                out_wav = f"{td}/decoded.wav"

                # torchaudio.save expects (channels, time)
                torchaudio.save(
                    out_ogg,
                    wav_cpu[None, :],
                    sample_rate=target_sr,
                    format="ogg",
                    encoding="OPUS",
                    bits_per_sample=None,
                    compression=bitrate_kbps,  # this maps to ffmpeg/sox compression settings depending on backend
                )
                y2, sr2 = torchaudio.load(out_ogg)
                y = y2.mean(dim=0).to(torch.float32)  # ensure mono
                if sr2 != target_sr:
                    # many backends decode Opus at 48k; if not, we’ll treat sr2 as truth and continue.
                    target_sr_local = sr2
                else:
                    target_sr_local = target_sr
                # normalize sample rate variable for later resample
                if target_sr_local != target_sr:
                    # bring to target_sr so the next step is consistent
                    y = torchaudio.functional.resample(y, target_sr_local, target_sr)
        else:
            raise ValueError('implementation must be "ffmpeg_libopus" or "torchaudio".')

        # resample back to original sample_rate
        if sample_rate != target_sr:
            y = torchaudio.functional.resample(y, target_sr, sample_rate)

        return y

    ys = [process_one(xb[i]) for i in range(B)]
    # pad/crop to a common length (codec + resample can change length slightly)
    max_len = max(y.numel() for y in ys)
    yb = torch.stack(
        [torch.nn.functional.pad(y, (0, max_len - y.numel())) for y in ys],
        dim=0,
    )  # (B, max_len)

    # Return shape similar to input (but mono):
    if x.dim() == 1:
        out = yb[0]
    elif x.dim() == 2:
        out = yb
    elif x.dim() == 3:
        out = yb[:, None, :]  # (B,1,T)
    else:
        out = yb

    return out.to(device=device, dtype=dtype)
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import AudioChunkDataset
from src.utils.jsonl_helpers import write_jsonl
from src.utils.io_helpers import save_pcm_s16le


# -----------------------------
# Helpers
# -----------------------------

def _write_pcm(tmp_path: Path, name: str, audio_f32: np.ndarray) -> str:
    """
    Write float32 mono audio in [-1,1] as raw PCM s16le and return path as str.
    """
    pcm_path = tmp_path / name
    save_pcm_s16le(audio_f32, str(pcm_path))
    return str(pcm_path)


def _rec(*, audio_id: str, chunk_id: str, zuliano: bool, speaker_id: str, chunk_path: str, sr: int = 16000):
    """
    Construct a minimal manifest-like record compatible with Dataset/load_sample.
    """
    return {
        "audio_id": audio_id,
        "chunk_id": chunk_id,
        "zuliano": zuliano,
        "speaker_id": speaker_id,
        "chunk_path": chunk_path,
        "sr": sr,
    }


# -----------------------------
# Core Dataset interface tests
# -----------------------------

def test_len_matches_records_count(tmp_path):
    pcm = _write_pcm(tmp_path, "a.pcm", np.zeros(100, dtype=np.float32))
    records = [
        _rec(audio_id="a", chunk_id="a_000", zuliano=True, speaker_id="spk1", chunk_path=pcm),
        _rec(audio_id="b", chunk_id="b_000", zuliano=False, speaker_id="spk2", chunk_path=pcm),
    ]
    label_to_id = {"non_zuliano": 0, "zuliano": 1}

    ds = AudioChunkDataset(records, label_to_id)
    assert len(ds) == 2


def test_getitem_returns_expected_schema_and_types(tmp_path):
    audio = np.linspace(-0.5, 0.5, 32).astype(np.float32)
    pcm = _write_pcm(tmp_path, "x.pcm", audio)

    records = [
        _rec(audio_id="x", chunk_id="x_000", zuliano=True, speaker_id="spkX", chunk_path=pcm),
    ]
    label_to_id = {"non_zuliano": 0, "zuliano": 1}

    ds = AudioChunkDataset(records, label_to_id)

    sample = ds[0]
    assert isinstance(sample, dict)

    # Required keys produced by load_sample / Dataset.__getitem__
    assert set(sample.keys()) == {"waveform", "label", "chunk_id", "speaker_id"}

    assert isinstance(sample["waveform"], torch.Tensor)
    assert sample["waveform"].dtype == torch.float32
    assert sample["waveform"].ndim == 1
    assert sample["waveform"].shape[0] > 0

    assert isinstance(sample["label"], int)
    assert sample["label"] == 1  # zuliano -> "zuliano" -> id 1

    assert sample["chunk_id"] == "x_000"
    assert sample["speaker_id"] == "spkX"

    # PCM roundtrip sanity: values should be close to original (quantized by int16)
    assert torch.max(torch.abs(sample["waveform"])) <= 1.0 + 1e-6
    assert torch.mean(sample["waveform"]).item() == pytest.approx(np.mean(audio), abs=1e-3)


def test_label_mapping_respected_for_both_classes(tmp_path):
    pcm_t = _write_pcm(tmp_path, "t.pcm", np.ones(16, dtype=np.float32) * 0.25)
    pcm_f = _write_pcm(tmp_path, "f.pcm", np.ones(16, dtype=np.float32) * -0.25)

    records = [
        _rec(audio_id="t", chunk_id="t_000", zuliano=True, speaker_id="spk1", chunk_path=pcm_t),
        _rec(audio_id="f", chunk_id="f_000", zuliano=False, speaker_id="spk2", chunk_path=pcm_f),
    ]

    # Intentionally non-default ordering to ensure dataset uses the provided mapping
    label_to_id = {"zuliano": 7, "non_zuliano": 3}

    ds = AudioChunkDataset(records, label_to_id)

    assert ds[0]["label"] == 7
    assert ds[1]["label"] == 3


def test_negative_indexing_uses_python_list_semantics(tmp_path):
    pcm = _write_pcm(tmp_path, "a.pcm", np.zeros(8, dtype=np.float32))
    records = [
        _rec(audio_id="a", chunk_id="a_000", zuliano=True, speaker_id="s1", chunk_path=pcm),
        _rec(audio_id="b", chunk_id="b_000", zuliano=False, speaker_id="s2", chunk_path=pcm),
    ]
    ds = AudioChunkDataset(records, {"non_zuliano": 0, "zuliano": 1})

    # Since dataset indexes directly into list, negative index should work like list[-1]
    assert ds[-1]["chunk_id"] == "b_000"


def test_transform_is_applied_to_waveform_only(tmp_path):
    audio = np.ones(10, dtype=np.float32) * 0.1
    pcm = _write_pcm(tmp_path, "x.pcm", audio)

    records = [_rec(audio_id="x", chunk_id="x_000", zuliano=False, speaker_id="spk", chunk_path=pcm)]
    label_to_id = {"non_zuliano": 0, "zuliano": 1}

    def transform(w: torch.Tensor) -> torch.Tensor:
        assert isinstance(w, torch.Tensor)
        return w * 2.0

    ds = AudioChunkDataset(records, label_to_id, transform=transform)
    sample = ds[0]

    assert sample["label"] == 0
    assert sample["chunk_id"] == "x_000"
    assert sample["speaker_id"] == "spk"

    # Waveform should be transformed
    assert torch.allclose(sample["waveform"], torch.from_numpy(audio) * 2.0, atol=1e-3)


def test_dataset_is_picklable_and_still_loads(tmp_path):
    pcm = _write_pcm(tmp_path, "a.pcm", np.random.uniform(-0.2, 0.2, 64).astype(np.float32))
    records = [
        _rec(audio_id="a", chunk_id="a_000", zuliano=True, speaker_id="s1", chunk_path=pcm),
        _rec(audio_id="b", chunk_id="b_000", zuliano=False, speaker_id="s2", chunk_path=pcm),
    ]
    ds = AudioChunkDataset(records, {"non_zuliano": 0, "zuliano": 1})

    blob = pickle.dumps(ds)
    ds2 = pickle.loads(blob)

    assert len(ds2) == 2
    s = ds2[0]
    assert set(s.keys()) == {"waveform", "label", "chunk_id", "speaker_id"}
    assert s["chunk_id"] == "a_000"

# -----------------------------
# Error / edge-case tests
# -----------------------------

def test_getitem_raises_file_not_found_for_missing_pcm(tmp_path):
    missing = str(tmp_path / "does_not_exist.pcm")
    records = [_rec(audio_id="x", chunk_id="x_000", zuliano=True, speaker_id="spk", chunk_path=missing)]
    ds = AudioChunkDataset(records, {"non_zuliano": 0, "zuliano": 1})

    # load_pcm_s16le uses np.fromfile; missing file usually raises FileNotFoundError/OSError
    with pytest.raises((FileNotFoundError, OSError)):
        _ = ds[0]


def test_getitem_raises_value_error_for_empty_pcm(tmp_path):
    empty_path = tmp_path / "empty.pcm"
    empty_path.write_bytes(b"")  # empty file

    records = [_rec(audio_id="e", chunk_id="e_000", zuliano=False, speaker_id="spk", chunk_path=str(empty_path))]
    ds = AudioChunkDataset(records, {"non_zuliano": 0, "zuliano": 1})

    # io_helpers.load_pcm_s16le raises ValueError on empty PCM
    with pytest.raises(ValueError):
        _ = ds[0]


def test_getitem_raises_key_error_if_label_missing_in_mapping(tmp_path):
    pcm = _write_pcm(tmp_path, "a.pcm", np.zeros(10, dtype=np.float32))
    records = [_rec(audio_id="a", chunk_id="a_000", zuliano=True, speaker_id="s1", chunk_path=pcm)]

    # Missing "zuliano"
    ds = AudioChunkDataset(records, {"non_zuliano": 0})

    with pytest.raises(KeyError):
        _ = ds[0]

# -----------------------------
# from_jsonl constructor tests
# -----------------------------

def test_from_jsonl_builds_label_to_id_when_none(tmp_path):
    pcm1 = _write_pcm(tmp_path, "z.pcm", np.zeros(10, dtype=np.float32))
    pcm2 = _write_pcm(tmp_path, "n.pcm", np.zeros(10, dtype=np.float32))

    records = [
        _rec(audio_id="a", chunk_id="a_000", zuliano=True,  speaker_id="s1", chunk_path=pcm1),
        _rec(audio_id="b", chunk_id="b_000", zuliano=False, speaker_id="s2", chunk_path=pcm2),
    ]

    jsonl_path = tmp_path / "split.jsonl"
    write_jsonl(records, jsonl_path)

    ds = AudioChunkDataset.from_jsonl(str(jsonl_path), label_to_id=None, validate=False)

    # build_label_to_id sorts labels alphabetically -> non_zuliano then zuliano
    assert ds.label_to_id == {"non_zuliano": 0, "zuliano": 1}
    assert len(ds) == 2
    assert ds[0]["label"] in (0, 1)


def test_from_jsonl_uses_provided_label_to_id_without_rebuilding(tmp_path):
    pcm = _write_pcm(tmp_path, "a.pcm", np.zeros(10, dtype=np.float32))

    records = [
        _rec(audio_id="a", chunk_id="a_000", zuliano=True, speaker_id="s1", chunk_path=pcm),
    ]
    jsonl_path = tmp_path / "split.jsonl"
    write_jsonl(records, jsonl_path)

    custom_map = {"zuliano": 99, "non_zuliano": 12}
    ds = AudioChunkDataset.from_jsonl(str(jsonl_path), label_to_id=custom_map, validate=False)

    # Must keep the exact mapping object (important when sharing across splits)
    assert ds.label_to_id is custom_map
    assert ds[0]["label"] == 99


def test_from_jsonl_validate_true_rejects_missing_required_fields(tmp_path):
    # Missing chunk_path should be caught by validate_manifest
    bad_records = [{
        "audio_id": "a",
        "chunk_id": "a_000",
        "zuliano": True,
        "speaker_id": "s1",
        # "chunk_path": "..."  # missing
    }]

    jsonl_path = tmp_path / "bad.jsonl"
    write_jsonl(bad_records, jsonl_path)

    with pytest.raises(ValueError):
        _ = AudioChunkDataset.from_jsonl(str(jsonl_path), validate=True)


def test_from_jsonl_validate_false_allows_bad_schema_but_getitem_fails_later(tmp_path):
    # Demonstrates that validate=False skips schema checks.
    bad_records = [{
        "audio_id": "a",
        "chunk_id": "a_000",
        "zuliano": True,
        "speaker_id": "s1",
        # chunk_path missing -> will break inside __getitem__ / load_sample
    }]

    jsonl_path = tmp_path / "bad.jsonl"
    write_jsonl(bad_records, jsonl_path)

    ds = AudioChunkDataset.from_jsonl(str(jsonl_path), validate=False)

    # __len__ works, but access fails because required field missing
    assert len(ds) == 1
    with pytest.raises(KeyError):
        _ = ds[0]
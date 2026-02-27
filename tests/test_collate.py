import torch
from src.data.collate import CollateFn


def test_collate_shapes_and_mask():
    T = 10
    collate = CollateFn(target_samples=T)

    samples = [
        {"waveform": torch.ones(6),  "label": 0, "chunk_id":"a", "speaker_id":"s1"},
        {"waveform": torch.ones(10), "label": 1, "chunk_id":"b", "speaker_id":"s2"},
        {"waveform": torch.ones(14), "label": 0, "chunk_id":"c", "speaker_id":"s3"},
    ]

    batch = collate(samples)

    assert batch.waveforms.shape == (3, T)
    assert batch.attention_mask.shape == (3, T)
    assert batch.labels.shape == (3,)
    assert batch.labels.dtype == torch.int64

    # lengths should be [6, 10, 10]
    expected_mask = torch.tensor([
        [1,1,1,1,1,1,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
    ], dtype=torch.int64)

    assert torch.equal(batch.attention_mask, expected_mask)

    # padded region should be zero for the short sample
    assert torch.all(batch.waveforms[0, 6:] == 0)

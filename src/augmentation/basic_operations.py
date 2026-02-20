import torch

def random_crop_audio(x: torch.Tensor, crop_len: int) -> torch.Tensor:
    """
    Randomly crop audio to `crop_len` samples.
    Supports shapes: (T,) or (B, T).
    If shorter than crop_len, it pads with zeros at the end.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)  # (1, T)

    B, T = x.shape
    if T < crop_len:
        pad = crop_len - T
        x = torch.nn.functional.pad(x, (0, pad))  # pad last dim
        T = crop_len

    start = torch.randint(0, T - crop_len + 1, (1,), device=x.device).item()
    out = x[:, start:start + crop_len]

    return out.squeeze(0) if out.shape[0] == 1 else out
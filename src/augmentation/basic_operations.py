import torch
import torch.nn.functional as F

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


def random_gain_db(x: torch.Tensor, gain_db_min: float = -6.0, gain_db_max: float = 6.0) -> torch.Tensor:
    """
    Apply a random gain (in dB) uniformly sampled from [gain_db_min, gain_db_max].
    Works for any shape (e.g., (T,), (B,T), (C,T), (B,C,T), ...).
    """
    # sample one gain value (shared across the whole tensor)
    gain_db = (gain_db_max - gain_db_min) * torch.rand(1, device=x.device, dtype=x.dtype) + gain_db_min
    gain_lin = 10.0 ** (gain_db / 20.0)  # dB -> linear amplitude

    return x * gain_lin


def speed_perturb(
    x: torch.Tensor,
    factor_min: float = 0.95,
    factor_max: float = 1.05,
) -> torch.Tensor:
    """
    Speed perturbation via resampling (linear interpolation).
    Pick factor ~ Uniform[factor_min, factor_max] and resample.

    Shapes supported:
      - (T,)
      - (B, T)
      - (C, T)
      - (B, C, T)

    Notes:
      - factor > 1.0 -> faster -> shorter output
      - factor < 1.0 -> slower -> longer output
      - This changes the number of samples unless you crop/pad afterward.
    """

    factor = (factor_max - factor_min) * torch.rand(1).item() + factor_min

    # Normalize to (B, C, T) for interpolate
    orig_dim = x.dim()
    if orig_dim == 1:        # (T,)
        x_ = x[None, None, :]
    elif orig_dim == 2:      # (B,T) or (C,T) (ambiguous). We'll treat as (B,T).
        x_ = x[:, None, :]
    elif orig_dim == 3:      # (B,C,T)
        x_ = x
    else:
        raise ValueError(f"Unsupported shape {tuple(x.shape)}")

    B, C, T = x_.shape
    new_T = max(1, int(round(T / factor)))  # faster => smaller new_T

    # linear interpolation expects float
    x_in = x_.to(torch.float32)
    y = F.interpolate(x_in, size=new_T, mode="linear", align_corners=False)

    # cast back
    y = y.to(dtype=x.dtype)

    # Restore original shape
    if orig_dim == 1:
        return y[0, 0, :]
    elif orig_dim == 2:
        return y[:, 0, :]
    else:  # orig_dim == 3
        return y
"""Torch-only pitch helpers extracted from upstream utils.py.

Keep legacy pitch-bin arithmetic for existing checkpoint compatibility. Improving
pitch representation is a separate experiment, not an implicit migration.
"""
import math
import torch


def normalize_f0(f0, x_mask, uv, random_scale=True):
    uv_sum = torch.sum(uv, dim=1, keepdim=True)
    uv_sum[uv_sum == 0] = 9999
    means = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / uv_sum
    if random_scale:
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        factor = torch.ones(f0.shape[0], 1).to(f0.device)
    f0_norm = (f0 - means.unsqueeze(-1)) * factor.unsqueeze(-1)
    if torch.isnan(f0_norm).any():
        raise ValueError('F0 normalization produced NaN; inspect pitch/voicing features')
    return f0_norm * x_mask


def f0_to_coarse(f0):
    f0_bin = 256
    f0_mel_min = 1127 * math.log(1 + 50.0 / 700)
    f0_mel_max = 1127 * math.log(1 + 1100.0 / 700)
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
    return f0_coarse

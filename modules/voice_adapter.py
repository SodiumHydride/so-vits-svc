"""Identity-initialized, frame-local adapters for SVC latent representations.

This is a bottleneck residual adapter, not LoRA and not a zero-shot voice encoder.
The same pointwise map is used for cropped training and full-length inference.
"""
import torch
from torch import nn
from torch.nn import functional as F


class VoiceAdapter(nn.Module):
    def __init__(self, channels: int, speaker_channels: int, rank: int):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (channels, speaker_channels, rank)):
            raise ValueError('Adapter dimensions must be positive integers')
        self.down = nn.Conv1d(channels, rank, 1)
        self.speaker = nn.Conv1d(speaker_channels, rank, 1)
        self.pitch = nn.Conv1d(3, rank, 1)
        self.up = nn.Conv1d(rank, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x, f0, g, uv=None, volume=None, mask=None):
        if x.ndim != 3 or f0.shape != (x.shape[0], x.shape[-1]):
            raise ValueError('Expected x [B,C,T] and aligned f0 [B,T]')
        if g.ndim != 3 or g.shape[0] != x.shape[0] or g.shape[-1] not in (1, x.shape[-1]):
            raise ValueError('Expected speaker conditioning [B,G,1] or [B,G,T]')
        if uv is not None and uv.shape != f0.shape:
            raise ValueError('Voicing must match f0')
        if volume is not None and volume.shape != f0.shape:
            raise ValueError('Volume must match f0')
        voiced = (f0 > 0).float() if uv is None else uv.float()
        log_pitch = torch.log2(f0.float().clamp_min(1.) / 220.).clamp(-8., 8.) * voiced
        energy = torch.zeros_like(log_pitch) if volume is None else torch.log1p(volume.float().clamp_min(0.))
        condition = torch.stack((log_pitch, voiced, energy), dim=1).to(x.dtype)
        hidden = self.down(x) + self.speaker(g.to(x.dtype)) + self.pitch(condition)
        residual = self.up(F.silu(hidden)).to(x.dtype)
        if mask is not None:
            if mask.shape != (x.shape[0], 1, x.shape[-1]):
                raise ValueError('Mask must have shape [B,1,T]')
            residual = residual * mask
        return x + residual


def configure_trainable(model, mode='full'):
    """Select generator parameters before optimizer/DDP construction.

    adapters: only adapters; adapters+speaker: adapters and the existing embedding
    table. The latter requires zero weight decay for that table to avoid modifying
    unused rows. It does not add new speaker IDs or expand the checkpoint vocabulary.
    """
    if mode not in ('full', 'adapters', 'adapters+speaker'):
        raise ValueError('finetune_mode must be full, adapters, or adapters+speaker')
    if getattr(model, 'inference_only', False):
        raise ValueError('Inference-only models cannot be trained')
    if mode != 'full' and not getattr(model, 'adapter_rank', 0):
        raise ValueError('Adapter training requires model.adapter_rank > 0')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(mode == 'full' or name.startswith(('prior_adapter.', 'decoder_adapter.'))
                                 or (mode == 'adapters+speaker' and name.startswith('emb_g.')))
        parameter.grad = None
    model.finetune_mode = mode
    model.train(model.training)
    total = sum(p.numel() for p in model.parameters())
    trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'mode': mode, 'generator_parameters': total, 'trainable_generator_parameters': trained}


def optimizer_groups(model, weight_decay=0.01):
    regular, speaker = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (speaker if name.startswith('emb_g.') and model.finetune_mode != 'full' else regular).append(parameter)
    groups = []
    if regular:
        groups.append({'params': regular, 'weight_decay': weight_decay})
    if speaker:
        groups.append({'params': speaker, 'weight_decay': 0.0})
    if not groups:
        raise ValueError('No trainable generator parameters')
    return groups

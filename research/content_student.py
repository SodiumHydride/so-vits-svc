"""Experimental waveform encoder; NOT a pretrained/drop-in ContentVec replacement.

16 kHz mono -> valid 20 ms frames -> separate content/performance latents.
No ContentVec, ASR, text LLM, or teacher is invoked in forward(). The first
implementation uses ordinary convolution and attention, not Gated DeltaNet/JEPA.
SPDX-License-Identifier: AGPL-3.0-only
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

FORMAT = "svc-content-student-v1"
KERNELS = (10, 3, 3, 3, 3, 2, 2)
STRIDES = (5, 2, 2, 2, 2, 2, 2)


@dataclass(frozen=True)
class StudentConfig:
    sample_rate: int = 16000
    width: int = 192
    content_dim: int = 192
    performance_dim: int = 32
    blocks: int = 6
    heads: int = 4
    attention_every: int = 3

    def __post_init__(self):
        if self.sample_rate != 16000:
            raise ValueError("Resample explicitly to 16000 Hz; the frame grid is fixed")
        for name in ("width", "content_dim", "performance_dim", "blocks", "heads", "attention_every"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(name + " must be a positive integer")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


def frame_lengths(lengths):
    result = lengths.clone()
    for kernel, stride in zip(KERNELS, STRIDES):
        result = torch.div(result - kernel, stride, rounding_mode="floor") + 1
    return result.clamp_min(0)


def frame_times(count, *, device=None):
    # Seven VALID convolutions: receptive field 400 samples, stride 320.
    # These are indexing centers, not a measured acoustic/phoneme alignment.
    return (torch.arange(count, device=device, dtype=torch.float64) * 320 + 199.5) / 16000


def _mask(lengths, size):
    return torch.arange(size, device=lengths.device)[None] < lengths[:, None]


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation, heads, attention):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 7, padding=3 * dilation,
                                   dilation=dilation, groups=width)
        self.expand = nn.Linear(width, 2 * width)
        self.project = nn.Linear(width, width)
        self.attn_norm = nn.LayerNorm(width) if attention else None
        self.attn = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True) if attention else None

    def forward(self, x, mask):
        valid = mask[..., None].to(x.dtype)
        # Mask AFTER affine normalization too: its bias must not enter valid frames.
        y = self.norm(x) * valid
        y = self.depthwise(y.transpose(1, 2)).transpose(1, 2)
        x = (x + self.project(F.glu(self.expand(y), dim=-1))) * valid
        if self.attn is not None:
            y = self.attn_norm(x) * valid
            y = self.attn(y, y, y, key_padding_mask=~mask, need_weights=False)[0]
            x = (x + y) * valid
        return x


class ContentStudent(nn.Module):
    """Offline encoder. Future frames are used; this is NOT a streaming model."""
    def __init__(self, config=StudentConfig()):
        super().__init__()
        self.config = config
        channels = (32, 48, 64, 64, 96, 128, config.width)
        previous = 1
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for channel, kernel, stride in zip(channels, KERNELS, STRIDES):
            self.convs.append(nn.Conv1d(previous, channel, kernel, stride=stride))
            self.norms.append(nn.LayerNorm(channel))
            previous = channel
        self.blocks = nn.ModuleList([
            TemporalBlock(config.width, 2 ** (i % 3), config.heads,
                          (i + 1) % config.attention_every == 0)
            for i in range(config.blocks)])
        self.content = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, config.content_dim))
        # A shallow, separately supervised branch; shared frontend != disentanglement.
        self.performance = nn.Sequential(nn.LayerNorm(config.width),
            nn.Linear(config.width, config.performance_dim), nn.Tanh())

    def forward(self, waveform, lengths=None, *, sample_rate=16000):
        if sample_rate != self.config.sample_rate:
            raise ValueError("Incorrect sample rate; no implicit resampling")
        if waveform.ndim != 2 or not waveform.is_floating_point() or waveform.shape[0] == 0:
            raise ValueError("waveform must be a nonempty floating [batch, samples] tensor")
        if not bool(torch.isfinite(waveform).all()):
            raise ValueError("waveform contains non-finite values")
        if lengths is None:
            lengths = torch.full((waveform.shape[0],), waveform.shape[1], dtype=torch.long, device=waveform.device)
        if lengths.dtype != torch.long or lengths.device != waveform.device or lengths.shape != (waveform.shape[0],):
            raise ValueError("lengths must be an int64 [batch] tensor on waveform.device")
        if bool(((lengths < 400) | (lengths > waveform.shape[1])).any()):
            raise ValueError("Each clip must contain at least 400 real samples and fit its padding")
        current = lengths.clone()
        x = (waveform * _mask(lengths, waveform.shape[1])).unsqueeze(1)
        for conv, norm, kernel, stride in zip(self.convs, self.norms, KERNELS, STRIDES):
            x = conv(x)
            current = torch.div(current - kernel, stride, rounding_mode="floor") + 1
            mask = _mask(current, x.shape[-1])
            x = F.gelu(norm(x.transpose(1, 2))).transpose(1, 2) * mask[:, None]
        x = x.transpose(1, 2)
        performance = self.performance(x) * mask[..., None]
        for block in self.blocks:
            x = block(x, mask)
        content = self.content(x) * mask[..., None]
        return {"content": content, "performance": performance, "mask": mask,
                "lengths": current, "times": frame_times(x.shape[1], device=x.device)}


def align_teacher(features, hop_seconds, offset_seconds, target_times):
    """Interpolate on physical frame times. Never stretch an utterance to fit.

    Caller must supply actual teacher frame-center metadata. Out-of-support
    queries are masked out, NOT treated as valid copies of endpoint frames.
    """
    if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("Teacher features must be nonempty [time, channels]")
    if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
        raise ValueError("Teacher features must be finite floating values")
    if not math.isfinite(hop_seconds) or hop_seconds <= 0 or not math.isfinite(offset_seconds):
        raise ValueError("Invalid teacher frame grid")
    if target_times.ndim != 1 or not bool(torch.isfinite(target_times).all()):
        raise ValueError("target_times must be a finite vector")
    positions = (target_times.to(features.device, dtype=torch.float64) - offset_seconds) / hop_seconds
    valid = (positions >= -1e-6) & (positions <= features.shape[0] - 1 + 1e-6)
    positions = positions.clamp(0, features.shape[0] - 1)
    left = positions.floor().long()
    right = (left + 1).clamp_max(features.shape[0] - 1)
    weight = (positions - left).to(features.dtype)[:, None]
    aligned = features[left] * (1 - weight) + features[right] * weight
    return aligned, valid


def pool_phoneme_spans(features, span_ids):
    """Pool within annotated contiguous SPANS, never across repeated phoneme IDs.

    IDs must be nondecreasing span indices. -1 means unavailable annotation and
    is left unchanged. This is a training target transform, not a forced aligner.
    """
    if features.ndim != 2 or span_ids.shape != (features.shape[0],) or span_ids.dtype != torch.long:
        raise ValueError("Expected [time, channels] and int64 [time] span IDs")
    if span_ids.device != features.device or bool((span_ids < -1).any()):
        raise ValueError("Invalid span IDs or device")
    known = span_ids[span_ids >= 0]
    if known.numel() > 1 and bool((known[1:] < known[:-1]).any()):
        raise ValueError("Use monotone occurrence IDs, not reusable phoneme-class IDs")
    result = features.clone()
    for value in torch.unique(known):
        indices = torch.where(span_ids == value)[0]
        if indices[-1] - indices[0] + 1 != indices.numel():
            raise ValueError("Each span must be contiguous")
        result[indices] = features[indices].mean(0)
    return result


def export_student(model, destination, *, training_steps, provenance):
    if type(training_steps) is not int or training_steps < 1:
        raise ValueError("Refuse to export an untrained model as a deployment artifact")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("Nonempty provenance is required (including synthetic runs)")
    payload = {"format": FORMAT, "config": asdict(model.config), "training_steps": training_steps,
               "provenance": provenance, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    # Exclusive create: never overwrite the only trained model.
    with Path(destination).open("xb") as stream:
        torch.save(payload, stream)


def load_student(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("Not a content-student artifact; do not load as a So-VITS checkpoint")
    if type(payload.get("training_steps")) is not int or payload["training_steps"] < 1:
        raise ValueError("Missing training step provenance")
    model = ContentStudent(StudentConfig(**payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload["provenance"]

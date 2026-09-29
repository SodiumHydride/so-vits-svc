"""Offline multi-teacher objectives for content_student; no teacher model loading.

Teacher embeddings/pseudo-labels are supervision, not ground truth. No objective
here proves speaker disentanglement or downstream singing quality.
SPDX-License-Identifier: AGPL-3.0-only
"""
from dataclasses import dataclass
import math
import re

import torch
from torch import nn
from torch.nn import functional as F

from research.content_student import ContentStudent, StudentConfig, align_teacher, pool_phoneme_spans


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    dim: int
    checkpoint_sha256: str
    hop_seconds: float
    offset_seconds: float
    kind: str = "features"
    weight: float = 1.0

    def __post_init__(self):
        if not self.name.isidentifier() or type(self.dim) is not int or self.dim < 2:
            raise ValueError("Teacher name/dimension is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", self.checkpoint_sha256):
            raise ValueError("Provide the exact teacher checkpoint SHA-256")
        if self.kind not in ("features", "posterior"):
            raise ValueError("Teacher kind must be features or posterior")
        if not math.isfinite(self.hop_seconds) or self.hop_seconds <= 0 or not math.isfinite(self.offset_seconds):
            raise ValueError("Invalid teacher frame grid")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("Teacher weight must be positive")


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale * gradient, None


def reverse_gradient(x, scale=1.0):
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("Gradient reversal scale must be finite and nonnegative")
    return _ReverseGradient.apply(x, scale)


class StudentTask(nn.Module):
    def __init__(self, config, teachers, *, speaker_count=0):
        super().__init__()
        if not teachers or len({t.name for t in teachers}) != len(teachers):
            raise ValueError("Require at least one teacher and unique teacher names")
        if type(speaker_count) is not int or speaker_count < 0 or speaker_count == 1:
            raise ValueError("speaker_count must be zero (disabled) or at least two")
        self.student = ContentStudent(config)
        self.teachers = list(teachers)
        self.readouts = nn.ModuleDict({t.name: nn.Linear(config.content_dim, t.dim) for t in teachers})
        self.prosody = nn.Linear(config.performance_dim, 3)
        self.speaker = nn.Linear(config.content_dim, speaker_count) if speaker_count else None

    def loss(self, record, *, pair_weight=0.1, prosody_weight=0.1, adversarial_scale=0.0):
        for value in (pair_weight, prosody_weight, adversarial_scale):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Objective weights must be finite and nonnegative")
        if record.get("sample_rate") != 16000:
            raise ValueError("Record must be explicitly resampled to 16000 Hz")
        waveform = record["waveform"]
        if waveform.ndim != 1:
            raise ValueError("A training record contains one mono waveform")
        out = self.student(waveform[None], sample_rate=record["sample_rate"])
        z = out["content"][0]
        losses = {}
        teacher_sum = z.new_zeros(())
        cache = record.get("teachers", {})
        if set(cache) != {t.name for t in self.teachers}:
            raise ValueError("Record teachers differ from experiment configuration")
        for spec in self.teachers:
            entry = cache[spec.name]
            for field in ("checkpoint_sha256", "hop_seconds", "offset_seconds", "kind"):
                if entry.get(field) != getattr(spec, field):
                    raise ValueError("Teacher provenance mismatch: " + spec.name + "/" + field)
            target = entry["features"].detach()
            if target.ndim != 2 or target.shape[-1] != spec.dim:
                raise ValueError("Teacher width mismatch: " + spec.name)
            if spec.kind == "posterior":
                if bool((target < 0).any()) or not torch.allclose(target.sum(-1), torch.ones_like(target[:, 0]), atol=1e-4):
                    raise ValueError("Posterior teacher must contain probability rows, not logits")
            if "span_ids" in entry:
                target = pool_phoneme_spans(target, entry["span_ids"])
            aligned, valid = align_teacher(target, spec.hop_seconds, spec.offset_seconds, out["times"])
            if not bool(valid.any()):
                raise ValueError("No overlapping frames for teacher " + spec.name)
            predicted = self.readouts[spec.name](z)[valid].float()
            expected = aligned[valid].float()
            if spec.kind == "features":
                # A documented target transform; NOT coordinate identity to ContentVec.
                expected = F.layer_norm(expected, (spec.dim,))
                term = F.smooth_l1_loss(predicted, expected)
            else:
                term = F.kl_div(F.log_softmax(predicted, -1), expected, reduction="batchmean")
            losses["teacher/" + spec.name] = term
            teacher_sum = teacher_sum + spec.weight * term
        losses["teachers"] = teacher_sum / sum(t.weight for t in self.teachers)
        total = losses["teachers"]

        if "paired_waveform" in record and pair_weight:
            pair = record["paired_waveform"]
            if record.get("pair_kind") != "verified_timbre_only_same_timing" or pair.shape != waveform.shape:
                raise ValueError("Pair must be externally verified as content/timing preserving")
            other = self.student(pair[None])["content"][0]
            # ONLY the content branch is constrained; never erase pitch by applying
            # this loss to the performance branch. Teacher targets anchor collapse.
            losses["paired_content"] = 0.5 * (F.smooth_l1_loss(z, other.detach()) + F.smooth_l1_loss(other, z.detach()))
            total = total + pair_weight * losses["paired_content"]
        if "prosody" in record and prosody_weight:
            truth = record["prosody"].detach()
            if truth.shape != (z.shape[0], 3) or not bool(torch.isfinite(truth).all()):
                raise ValueError("Prosody must be [student_frames,3]: f0_Hz,voicing,rms")
            f0, uv, rms = truth.unbind(-1)
            if bool(((f0 < 0) | (rms < 0) | ((uv != 0) & (uv != 1)) | ((uv == 1) & (f0 <= 0))).any()):
                raise ValueError("Invalid f0/voicing/rms targets")
            pred = self.prosody(out["performance"][0]).float()
            voiced = uv == 1
            pitch = F.smooth_l1_loss(pred[voiced, 0], torch.log2(f0[voiced] / 220)) if bool(voiced.any()) else pred[:, 0].sum() * 0
            energy = F.smooth_l1_loss(pred[:, 2], torch.log(rms.clamp_min(1e-5)))
            losses["prosody"] = pitch + energy + F.binary_cross_entropy_with_logits(pred[:, 1], uv.float())
            total = total + prosody_weight * losses["prosody"]
        if adversarial_scale:
            if self.speaker is None or type(record.get("speaker_id")) is not int:
                raise ValueError("Adversarial training requires configured labeled speakers")
            speaker_id = record["speaker_id"]
            if not 0 <= speaker_id < self.speaker.out_features:
                raise ValueError("Speaker ID out of range")
            summary = reverse_gradient(z.mean(0, keepdim=True), adversarial_scale)
            logits = self.speaker(summary)
            losses["speaker_adversary"] = F.cross_entropy(logits, torch.tensor([speaker_id], device=z.device))
            total = total + losses["speaker_adversary"]
        losses["total"] = total
        return losses

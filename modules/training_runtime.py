"""Small, independently testable helpers for the experimental trainer."""
from contextlib import contextmanager
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


@contextmanager
def frozen_parameters(module):
    """Freeze parameter gradients, NOT input gradients; restore flags on failure.

    Keep the backward pass inside this context. Do not replace it with no_grad:
    generator training still needs gradients through the discriminator's input.
    This does not change train/eval mode or normalization-buffer updates.
    """
    parameters = list(module.parameters())
    flags = [p.requires_grad for p in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield module
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


def seed_worker(worker_id):
    # DataLoader already seeds torch; align Python/NumPy with that worker seed.
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def resolve_precision(train, device):
    """Return (autocast_enabled, dtype); reject unsupported requested BF16."""
    if not getattr(train, "fp16_run", False):
        return False, torch.float32
    name = getattr(train, "half_type", "fp16")
    if name not in ("fp16", "bf16"):
        raise ValueError("train.half_type must be fp16 or bf16")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("The audio trainer's mixed precision path requires CUDA")
    if name == "bf16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("BF16 is unsupported on this GPU; choose fp16 or disable fp16_run")
    return True, torch.float16 if name == "fp16" else torch.bfloat16


def make_grad_scaler(enabled, dtype):
    # BF16 has a wider exponent range; do not enable FP16 loss scaling for it.
    scale = enabled and dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=scale)
    return torch.cuda.amp.GradScaler(enabled=scale)


def cuda_autocast(enabled, dtype):
    # Preserve an explicit disabled region for numerically sensitive losses.
    effective_dtype = dtype if enabled else torch.float16
    if hasattr(torch, "autocast"):
        return torch.autocast("cuda", enabled=enabled, dtype=effective_dtype)
    return torch.cuda.amp.autocast(enabled=enabled, dtype=effective_dtype)

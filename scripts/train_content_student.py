"""Train the experimental waveform student from trusted, timestamped teacher caches.

Example:
  python scripts/train_content_student.py --config experiment.json \
      --train train.jsonl --valid valid.jsonl --output runs/content-001 --device cuda

Each manifest line: {"path":"relative/record.pt", "source_group":"recording-id"}.
Records are documented in research/CVEC_RESEARCH.md. A fresh output directory is
required. This script is not the So-VITS GAN trainer and does not convert songs.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.content_student import StudentConfig, export_student
from research.content_training import StudentTask, TeacherSpec


def load_manifest(path):
    path = Path(path).resolve()
    result = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry.get("source_group"), str) or not entry["source_group"]:
            raise ValueError("Every record requires its original recording/song group")
        relative = Path(entry["path"])
        resolved = (path.parent / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(path.parent):
            raise ValueError("Record paths must remain inside the manifest directory")
        if not resolved.is_file() or resolved in seen:
            raise ValueError("Missing or duplicate record at manifest line " + str(line_number))
        seen.add(resolved)
        result.append((resolved, entry["source_group"]))
    if not result:
        raise ValueError("Manifest is empty")
    return result


def validate_splits(train, valid):
    if {p for p, _ in train} & {p for p, _ in valid}:
        raise ValueError("Training and validation share a cache file")
    if {g for _, g in train} & {g for _, g in valid}:
        raise ValueError("Split leakage: an original source group crosses train/validation")


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


def load_record(path, expected_group, device):
    record = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(record, dict) or record.get("source_group") != expected_group:
        raise ValueError("Cache source_group disagrees with manifest")
    return to_device(record, device)


def run(args):
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    student_cfg = StudentConfig(**cfg.get("student", {}))
    teacher_specs = [TeacherSpec(**item) for item in cfg["teachers"]]
    if len(teacher_specs) == 1:
        print("NOTICE: single-teacher run; this is a compression baseline, not evidence of surpassing that teacher.")
    train, valid = load_manifest(args.train), load_manifest(args.valid)
    validate_splits(train, valid)
    for key, default in (("epochs", 10), ("seed", 1234), ("max_samples", 160000), ("accumulation", 1)):
        cfg.setdefault(key, default)
        if type(cfg[key]) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(key + " must be a valid integer")
    lr = cfg.get("learning_rate", 3e-4)
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if cfg["max_samples"] < 400:
        raise ValueError("max_samples must cover the frontend receptive field")
    objective = cfg.get("objective", {})
    if set(objective) - {"pair_weight", "prosody_weight", "adversarial_scale"}:
        raise ValueError("Unknown objective option")
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Choose an available CPU/CUDA device; MPS is not validated")
    task = StudentTask(student_cfg, teacher_specs, speaker_count=cfg.get("speaker_count", 0)).to(device)
    optimizer = torch.optim.AdamW(task.parameters(), lr=lr)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    # Hash manifests rather than embedding private absolute paths into artifacts.
    manifest_hashes = {k: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                       for k, p in (("train", args.train), ("valid", args.valid))}
    (output / "experiment.json").write_text(json.dumps(cfg, indent=2) + "\n")
    updates = 0
    for epoch in range(cfg["epochs"]):
        task.train()
        order = list(train)
        random.shuffle(order)
        train_loss = 0.
        accumulation = cfg["accumulation"]
        optimizer.zero_grad(set_to_none=True)
        for index, (path, group) in enumerate(order):
            record = load_record(path, group, device)
            if record["waveform"].numel() > cfg["max_samples"]:
                raise ValueError("Pre-segment long clips AND recalculate frame metadata; no silent crop")
            # Each update averages this exact microbatch group, including the tail.
            start = (index // accumulation) * accumulation
            count = min(accumulation, len(order) - start)
            losses = task.loss(record, **objective)
            if not bool(torch.isfinite(losses["total"])):
                raise FloatingPointError("Non-finite training objective")
            (losses["total"] / count).backward()
            train_loss += float(losses["total"].detach())
            if (index + 1) % accumulation == 0 or index + 1 == len(order):
                torch.nn.utils.clip_grad_norm_(task.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            del record, losses
        task.eval()
        val_loss = 0.
        with torch.no_grad():
            for path, group in valid:
                record = load_record(path, group, device)
                if record["waveform"].numel() > cfg["max_samples"]:
                    raise ValueError("Validation clip exceeds max_samples")
                value = task.loss(record, **objective)["total"]
                if not bool(torch.isfinite(value)):
                    raise FloatingPointError("Non-finite validation objective")
                val_loss += float(value)
        report = {"epoch": epoch + 1, "optimizer_updates": updates,
                  "train_objective": train_loss / len(train), "valid_objective": val_loss / len(valid),
                  "audio_quality_measured": False}
        if device.type == "cuda":
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report) + "\n")
        print(json.dumps(report))
        provenance = {"teachers": [asdict(t) for t in teacher_specs], "manifest_sha256": manifest_hashes,
                      "run_label": cfg.get("run_label", "unrated_research"), "audio_quality_measured": False}
        export_student(task.student, output / ("student_epoch_%03d.pt" % (epoch + 1)),
                       training_steps=updates, provenance=provenance)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "train", "valid", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError, FloatingPointError) as exc:
        parser.exit(2, "Content student experiment failed: " + str(exc) + "\n")


if __name__ == "__main__":
    main()

# Architecture experiments: deployment graph and lightweight adaptation

Downstream development on `SodiumHydride/so-vits-svc`, branch
`exp/low-vram-core`, based on first-pass commit
`1110ad9311c8b60c34b4d2c4387d694c5eea42b8`. The `4.1-Stable` branch remains the
unmodified baseline. These are opt-in experiments, not a new trained voice model.
This document supersedes the first-pass-only architecture status in EXPERIMENTS.md.

## Implemented

**Deployment graph.** `SynthesizerInfer` never constructs the training-only
posterior encoder. Runtime export also omits the auxiliary automatic-F0 predictor
by default; `--keep-f0` preserves it. This does NOT remove the externally extracted
F0, voicing inputs or NSF pitch conditioning. Requesting automatic F0 when it was
omitted raises an error. Training with an inference-only model is rejected.

**Voice adapters.** Two optional, identity-initialized residual bottlenecks modify
the prior input and decoder latent input. They receive current latent features,
speaker conditioning, continuous pitch, voicing and optional volume. They are
frame-local, so the mapping does not depend on whether a training segment or a
full clip is supplied. The decoder adapter's training F0/voicing/volume are cropped
with the exact same indices as its latent segment. This is not LoRA, zero-shot
cloning, or an assertion that these features are perfectly disentangled.

**Trainable subsets.** `train.finetune_mode` supports `full`, `adapters`, and
`adapters+speaker`. `train.py` initializes and freezes the generator before creating
its optimizer or DDP wrapper. Adapter modes require `train.init_generator`; they
refuse to silently freeze random weights. Frozen backbone dropout is disabled,
without disabling required input gradients. The speaker table gets zero weight
decay in adapter+speaker mode. Speaker IDs and table size are not automatically
expanded. Use a fresh optimizer/output directory when changing modes or targets.
Adapters remain conditioned on the existing speaker representation; a shared
adapter can affect multiple IDs. Multi-voice preservation is not yet established.

**Strict artifacts.** Runtime exports contain only runtime tensors and configuration.
Adapter-only exports verify that every supposedly frozen backbone tensor is
unchanged, save only adapted tensors, and bind the delta to the exact base file's
SHA-256. Wrong bases, partially missing adapters, bad shapes and mixed DDP prefixes
are rejected. Output creation refuses overwrites. Artifact loading uses
`weights_only=True`; still use trusted files. Optimizer/RNG/scaler state is not
part of these deployment artifacts. They are not training-resume checkpoints.

**Compatibility and dependency boundaries.** With adapters disabled, legacy tensor
names and dimensions are preserved. Legacy `n_layers_q` remains ignored because
old templates wrote 3 while actual checkpoints used 16 layers. New explicit
`model.posterior_layers` controls experimental depth; its default remains 16.
Changing it breaks shape compatibility and needs separately trained weights.
The neural core no longer imports the top-level utility module's optional FAISS,
librosa, fairseq or sklearn stack. Plotting is imported only when requested in
the standard NSF decoder utilities. Other legacy entry points still have their
original dependencies. No encoder, UI, export directory, LICENSE or attribution
was deleted as cosmetic cleanup.

## Actual parameter counts, not GPU benchmarks

Counted by constructing the default model structure from
`configs_template/config_template.json` on PyTorch 2.10.0+cpu, with randomly
initialized weights. The default 200-row, 768-dimensional speaker table is included.

| Generator configuration | Parameters |
|---|---:|
| Original full training generator | 52,402,957 |
| Posterior encoder alone | 12,067,200 |
| Auxiliary F0 predictor alone | 6,467,521 |
| Inference generator without posterior or auxiliary F0 | 33,868,236 |
| Two rank-16 adapters only (trainable subset) | 37,440 |
| Rank-16 adapters plus speaker table (trainable subset) | 191,040 |

The compact graph contains approximately 35.37% fewer GENERATOR parameters.
FP32 generator tensor state decreases from 209,611,828 to 135,472,944 bytes.
This is not a 35.37% reduction in total system VRAM, FLOPs or training time.
ContentVec/other content encoders, external F0 extractors, optimizer states,
activations, discriminator and optional diffusion are outside that comparison.
The existing GAN discriminator still trains normally in adapter mode. Gradients
may still traverse the frozen decoder; fewer trainable parameters do not imply
proportional compute savings. No GPU training or inference benchmark was run.

## Using the experiment

Start with your real preprocessed dataset config and a compatible, authorized
base generator checkpoint. The generator must already have useful singing
capability; an adapter does not supply a missing general singing model.

```bash
python scripts/make_adapter_config.py configs/config.json configs/adapter.json \
  --base /path/to/base_G.pth --rank 16 --mode adapters
```

This writes a NEW configuration, preserving source config, sampling rate, model
widths and speaker map. Default behavior disables auxiliary automatic F0 prediction
for pitch-preserving conversion. Use `--keep-f0` only for experiments that need it.
Train with the existing `train.py` CLI using this config and a separate output
directory. Do not reuse a full-finetuning optimizer checkpoint in an adapter run.
No learning-rate, batch-size or dataset-duration optimum is claimed.

Export a trained generator for deployment:

```bash
python scripts/svc_artifacts.py runtime --checkpoint /path/to/G_trained.pth \
  --config configs/adapter.json --output /path/to/voice.runtime.pth
```

Load it through the new API:

```python
from modules.model_io import load_runtime
model, config = load_runtime('/path/to/voice.runtime.pth', device='cuda')
# c, f0, uv and speaker_ids must already be correctly extracted/aligned and on
# the same device. Optional vol must match the training configuration.
waveform, used_f0 = model.infer(c, f0, uv, g=speaker_ids)
```

Or export only the learned difference from an unchanged base:

```bash
python scripts/svc_artifacts.py adapter --checkpoint /path/to/G_trained.pth \
  --config configs/adapter.json --base /path/to/base_G.pth \
  --mode adapters --output /path/to/voice.adapter.pth
```

Use `modules.model_io.load_adapter(base_path, adapter_path)` to reconstruct it.
Keep the exact base file: even repacking equivalent tensors changes the file hash.
The legacy WebUI has NOT been migrated to these versioned artifacts. Its existing
full-checkpoint path remains available; do not pass a runtime or adapter artifact
to the old checkpoint loader. The new `.infer` keeps legacy random-seeding behavior.

## Test evidence and limits

41 NEW CPU tests passed locally: 36 in `test_architecture.py`, 5 in
`test_adapter_training.py`. The prior 32 low-VRAM tests remain in the repository;
this pass does not claim they were all rerun. The new tests exercise actual small
prior/flow/posterior/NSF networks, rather than replacing those networks with stubs.
They use synthetic conditioning, random weights and mechanical losses, not real
singing. There is no claim of perceptual quality, speaker similarity or real-data
convergence. Full CUDA/NCCL training, pretrained weights, MPS, full GUI integration
and ONNX exports have NOT been validated.

Checks include bit-exact synthetic full/compact waveform equality, exact zero-
adapter identity, learned adapter artifact round trips, unchanged frozen weights,
nonzero gradients in both adapters, crop/voicing/volume alignment, strict mismatch
rejection, optional-dependency isolation, and two real CPU/Gloo ranks training the
small adapted generator for three steps with synchronized parameters. This DDP
case uses mechanical waveform/prior losses, not the complete GAN objective.

```bash
python -m unittest discover -s tests -p 'test_architecture.py' -v
python -m unittest discover -s tests -p 'test_adapter_training.py' -v
```

The original-model comparison uses `git show 730930d...:models.py` and verifies its
Git blob hash; with a shallow checkout lacking that commit, that one check skips.
An exact baseline source can instead be supplied with `SVC_BASELINE_MODELS`.
No pretrained checkpoint, voice recordings, private data or credentials were
committed. License and data-permission requirements remain those described in
LICENSE and the upstream documentation.

Before treating adapter mode as an improvement, compare it to full fine-tuning
on the SAME authorized data/base/splits. Measure peak allocated and reserved VRAM,
training throughput, inference real-time factor and held-out listening quality.
Check high notes, consonants, breathiness, vibrato, pitch transitions and unused
speaker preservation. Retain a full-finetuning fallback if capacity is inadequate.

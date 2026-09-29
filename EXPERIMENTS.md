# Low-VRAM SVC experiments

Updated: 2026-09-28 (America/New_York). This is an experimental downstream branch,
not an upstream release and not a claim of improved audio quality.

## Branch and provenance

- Repository: `SodiumHydride/so-vits-svc`
- Baseline: `4.1-Stable`, commit `730930d337d171479eadf305f96cbed4bb393e77`
- Experiment branch: `exp/low-vram-core`
- Upstream `4.1-Latest` was inspected: head `578d9389b66d1e42d2dd502b6ba7d0657d60ec30`,
  dated 2023-10-09. It is 32 commits ahead of Stable, touching 14 files, including
  GUI/inference/vocoder changes. It is not a new 2026 architecture. Useful patches
  can be reviewed separately; the branch was not merged wholesale.
- The baseline branch, network definitions, model tensor dimensions, sample rate,
  upstream README files, third-party attribution and LICENSE have not been changed.
  No pull request was opened against the upstream repository.

## Implemented in this first pass

1. `data_utils.py` now honors `train.max_speclen`. Previously it was read but unused:
   clips longer than 800 frames were cropped to a hard-coded 790 frames. Waveform,
   content, pitch, voicing, spectrum and optional volume share the same crop.
2. Validation uses deterministic center crops, a separate loader RNG and a saved/
   restored Torch RNG context. Validation audio is transferred to CPU immediately;
   an optional `max_eval_batches` bounds CPU logging too. This diagnostic validation
   is not a substitute for a fixed, full-song evaluation suite.
3. Single-GPU training no longer initializes DDP or spawns a subprocess. Multi-GPU
   training uses a `DistributedSampler` and calls `set_epoch`. Data is sharded;
   a sampler may still repeat tail examples when the dataset size is not divisible
   by the number of ranks.
4. During the generator update, discriminator parameters are temporarily frozen,
   while gradients through the generated waveform remain enabled. The frozen pass
   bypasses the discriminator's DDP wrapper. Parameters' original flags are restored
   even on exceptions. Normalization buffers and train/eval mode are not frozen.
5. Old gradients are released before the next forward pass; discriminator gradients
   are released before the generator backward pass. Validation no longer retains
   every output waveform on GPU. Peak allocated/reserved VRAM is logged in MiB.
6. Volume conditioning is explicitly transferred to the model's device. Batches with
   inconsistent volume availability or unaligned conditioning fail clearly.
7. AMP selection is explicit. FP16 uses gradient scaling; BF16 does not. Unsupported
   requested BF16 fails with a useful error. Generated-audio STFT/reconstruction
   losses use FP32. AMP still requires real-GPU numerical/audio validation.
8. Tensor feature caches are loaded on CPU with `weights_only=True`. Legacy F0 NumPy
   object arrays still require trusted local data; do not load untrusted caches.
9. Duplicate assignments, unused dataset fields, dead commented-out training code
   and the old reference-loss log were removed from the edited training path.
   Optional frontends/encoders/exporters were NOT deleted by filename alone.

## Configuration

Start with a valid config produced for YOUR dataset; the upstream example speaker
map is not your actual speaker map. Do not overwrite the baseline config.

```bash
python scripts/make_low_vram_config.py configs/config.json configs/experiment_low_vram.json
```

The script exclusively creates a new file and refuses to overwrite an existing
one. Defaults are batch size 2, maximum 256 feature frames, FP16, two loader workers,
no full RAM cache and four validation batches. It preserves `model`, `data`, `spk`
and the generator's `segment_size`. Options: `--batch-size`, `--frames`, and
`--precision fp32|fp16|bf16`. Use the existing training CLI with the derived config
and a separate experiment output directory; do not train into your only checkpoint
backup. GPU selection remains available via `CUDA_VISIBLE_DEVICES`.

These are experiment settings, NOT an 8-GB compatibility guarantee. Reducing batch
size changes the effective batch; gradient accumulation has NOT been added yet.
Shorter windows change training context. Compare 256/384/512 frames for convergence
and held-out singing quality. Old checkpoints' tensor shapes remain compatible by
design, but checkpoint loading and end-to-end training have not been verified with
actual weights. Compressed inference-only checkpoints should not be treated as
training checkpoints. Exact scaler/optimizer/RNG resumption is not claimed.

## Tests and limits of evidence

```bash
python -m unittest discover -s tests -v
```

32 tests passed locally on CPU (PyTorch 2.10.0+cpu). They cover aligned crops, cache
loading, padding, optional volume, config immutability, precision selection,
freeze/unfreeze gradient correctness, isolated trainer steps and validation logic.
One test runs two actual CPU/Gloo DDP processes for three tiny GAN steps and checks
that generator parameters remain synchronized.

The data/trainer tests deliberately isolate definitions from the legacy dependency
stack; mel extraction, plotting and full audio networks are stubbed in control-flow
tests. This is NOT an end-to-end import/environment test, a CUDA/NCCL test, a
pretrained-weight compatibility run or an audio-quality benchmark. No training data
or model weights were downloaded. No GPU training was performed. No measured VRAM
saving, throughput improvement, MOS or speaker-similarity improvement is claimed.
No GitHub Actions run was launched for these tests; this is the local test result.

The first real benchmark should use the same hardware, weights, dataset split,
sample rate and validation songs. Separate an unchanged-batch engineering ablation
from the shorter-window/smaller-batch preset. Warm up before timing, synchronize
CUDA around the measured interval, report peak allocated AND reserved memory, and
keep preprocessing/validation/checkpoint I/O separate from training throughput.
Compare equal audio exposure or effective optimizer updates, not just equal epochs
or one superficially faster training step. Listen to high notes, consonants,
breathy passages, transitions, long vowels and out-of-training-range pitch.

## Research directions — NOT implemented

The intended research order is:

- Make core/preprocessing/retrieval/UI/export dependencies optional and auditable.
  Remove obsolete components only after inspecting imports, CLI entry points and
  configuration references. Old files do not consume VRAM merely by existing.
- Add carefully tested GAN gradient accumulation and selective activation
  checkpointing. They can lower peak memory; they do not guarantee lower total
  compute or identical convergence. Preserve phase/RNG behavior when recomputing.
- Evaluate a shared multi-speaker generator with small pitch/energy-conditioned
  speaker adapters. Do not assume that freezing a single-speaker model supplies
  missing singing capabilities. Compare embedding-only, adapter, partial-unfreeze
  and full-finetuning learning curves.
- Explore teacher-only training assistance and a compact inference student. Reuse
  frozen/precomputed content features initially (the baseline already does this).
  Distill the content encoder only after measuring content, pitch and speaker
  leakage on singing; fewer parameters alone do not prove better representations.
- Explore a cheap harmonic/noise main path with a small, conditionally active
  expressive-residual path. Any gating must be temporally smooth and evaluated for
  consonant loss, breath suppression, phase discontinuities and timbre leakage.
  This is an architectural hypothesis, not a feature or demonstrated speedup.

## License and data

This remains a derivative of the upstream AGPL-3.0 repository. Preserve LICENSE,
copyright notices and upstream attribution, and identify downstream modifications.
Not submitting changes upstream does not remove license obligations. Source
requirements for distribution and AGPL section 13 network use still matter.
The upstream README's additional statements remain intact; this file does not
resolve any licensing ambiguity. Dataset and pretrained-weight permissions are
separate from the code license. Do not commit voice recordings, private datasets,
checkpoints or credentials to this public experiment branch.

## Primary references

- Upstream comparison: https://github.com/svc-develop-team/so-vits-svc/compare/4.1-Stable...4.1-Latest
- PyTorch AMP: https://docs.pytorch.org/docs/2.10/notes/amp_examples.html
- PyTorch DDP: https://docs.pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html
- PyTorch tuning: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- Code license: [LICENSE](LICENSE)

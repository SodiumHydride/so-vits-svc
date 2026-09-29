# Beyond a fixed ContentVec bottleneck: research and executable prototype

Date: 2026-09-29. Repository: `SodiumHydride/so-vits-svc`, branch
`exp/low-vram-core`. Starting commit: `4e960cc41559598a53b2433b32b5cf6bc8635d10`.
This pass is additive. It does not change `4.1-Stable`, production encoder names,
existing checkpoints, or the So-VITS training/inference pipeline.

**Status: functioning research code, NOT trained singing weights, a drop-in
ContentVec replacement, a reproduction of the cited papers, or demonstrated
improvement in conversion quality/GPU usage.**

## 1. What the evidence supports

ContentVec is a HuBERT-derived self-supervised speech representation model, not a
text LLM, F0 extractor, or waveform synthesizer. The current So-VITS wrapper takes
continuous 768-dimensional hidden features from layer 12. The original paper
addresses speaker information leaking into content features, while warning that
removing speaker information can also damage content. Its training uses modified
teacher targets, speaker-related augmentation/contrastive learning, and speaker
conditioning of the prediction network [1]. Caching its output removes the
encoder from per-voice backpropagation, but not its influence on what downstream
models can learn from those features.

A direct 2026 example is Hu et al.'s boundary-aware singing STYLE conversion study
[2]. Its ablation used the same 36 songs. Reported technique-transfer success and
technique leakage were respectively 72.2%/13.9% for ContentVec, 58.3%/25.0% for
unpooled Whisper at lambda=0.1, and 83.3%/8.3% for pooled Whisper at lambda=0.1.
"Converted" includes successful AND slightly successful human ratings. Leakage
means source singing-technique leakage, not a universal timbre-leakage metric.
Their table's MOS column is estimated with SingMOS, not the challenge's human MOS.
The system uses explicit symbolic/technique conditions and phoneme alignments
(MFA at inference). Thus this is evidence that representation choice AND its
injection can limit controllability; it is not proof that raw Whisper alone beats
ContentVec, nor a universal ceiling for all SVC models. We have not reproduced it.

Other relevant primary work:

- DC-Spin [3], an Interspeech 2025 speech-LM tokenizer, combines speaker-invariant
  learning, double codebooks and phonetic/ASR supervision. A practical content
  teacher candidate, not a verified drop-in singing encoder. Larger/finer acoustic
  codebooks and language-model-friendly units serve partly different objectives.
- S-JEPA [4], June 2026, uses soft clustering distributions and adaptive/online
  targets for speech SSL. This motivates support for posterior targets here;
  our code does NOT implement its GMM updates or JEPA pretraining procedure.
- SITA [5], January 2026 preprint (v1 inspected), separates speaker invariance from
  tonal information. Do not declare all pitch differences irrelevant to content,
  particularly for tonal languages. Its evaluation is ASR, not singing conversion.
- Gated DeltaNet-2 [6], May 2026, separates erase/write mechanisms in linear
  attention. Relevant to student computation, not a demonstrated cure for speaker
  leakage. We do NOT implement/copy that architecture or its GPU kernels here.
- Qwen3-TTS [7], January 2026, illustrates the distinction between semantic and
  acoustic speech tokenization. A reconstruction codec need not remove identity;
  reducing its frame rate is not automatically an SVC improvement.

A text/audio LLM can provide supervision, phonetic hypotheses, or context, but
plausibly guessing a missing consonant is not faithful conversion. All such labels
need time alignment and evaluation. If a postprocessor only sees C(x), it cannot
guarantee recovery of distinctions C already collapsed. A student that sees raw x
is a different case: teacher targets do not impose a mathematical ceiling on its
input information. Pure imitation is a compression baseline; independent phonetic,
prosodic and downstream evidence is needed to justify a claim of improved content.

## 2. Implemented hypothesis

```
16 kHz mono waveform
  -> valid convolutional frontend (320-sample stride, 400-sample receptive field)
       -> shallow performance head (32 channels)
       -> temporal convolution + occasional full attention -> content (192 channels)

Training only:
  cached ContentVec / other continuous teacher features -> projection objectives
  cached phonetic/token posterior probabilities -> KL objective
  optional verified phoneme spans -> target pooling before time alignment
  optional verified timbre-only waveform pairs -> content consistency
  optional F0/voicing/RMS -> performance supervision
  optional labeled speaker adversary -> reversed content gradients
```

The waveform student never invokes ContentVec, fairseq, an ASR model, or an LLM
in forward. Teacher projection/classification heads are excluded from its exported
artifact. Teacher names are generic: a multi-teacher interface does not mean those
teachers have been downloaded, trained, or validated. One ContentVec teacher is
allowed as the compression control experiment.

The default student has 1,140,208 parameters (4,560,832 FP32 parameter bytes).
This was counted on PyTorch 2.10.0+cpu with random initialization. It is NOT evidence
that this capacity matches ContentVec's accuracy. Optimizers, activations, teacher
caches, the SVC generator, and the external F0 model are outside this count.
Two of six temporal blocks use ordinary full attention; this prototype is neither
linear-time throughout nor streaming/causal. The performance head shares the
frontend and is not guaranteed identity-free. Do not pass it to a voice converter
without checking source-identity leakage and whether it contains useful detail.

A one-second input produces 49 valid frames, not 50 padded frames. Frame centers
are `(320*i + 199.5)/16000` seconds. These are indexing centers, not measured
phoneme boundaries. Teachers can use different grids: their actual hop and offset
are required, linear interpolation respects physical times, and out-of-support
frames are excluded. Never stretch a short teacher sequence to an unrelated length.

## 3. Cache and experiment contract

Requires Python 3.10+ and PyTorch; CPU 2.10.0 was tested. Resampling, authorized
teacher weights, audio collection and annotation are deliberately external to this
small experiment. Resampling the content input to 16 kHz does not change the
separate So-VITS waveform output sample rate.

Every trusted `.pt` cache is a dictionary containing only tensors/simple values:

```python
record = {
    "sample_rate": 16000,
    "waveform": mono_float_tensor,           # [samples], at least 400 samples
    "source_group": "original-song-id",     # BEFORE slicing/augmentation
    "teachers": {
        "teacher_name": {
            "features": teacher_tensor,     # [T,D]; no batch dimension
            "checkpoint_sha256": "...",    # actual 64-char lowercase SHA-256
            "hop_seconds": actual_hop,
            "offset_seconds": actual_first_frame_center,
            "kind": "features",            # or "posterior" probability rows
            # optional: "span_ids": int64_tensor_of_length_T
        }
    },
    # optional: "prosody": tensor[N,3] of f0_Hz, binary voicing, RMS
    # optional: "paired_waveform": same-length waveform
    # optional: "pair_kind": "verified_timbre_only_same_timing"
    # optional: "speaker_id": integer for a configured speaker classifier
}
```

Prosody N uses the STUDENT grid; it is not silently interpolated. Voiced frames
require positive F0. Span IDs are monotone, contiguous occurrence indices, NOT
reusable phoneme-class IDs; -1 leaves an unannotated teacher frame unchanged.
Posterior rows must be nonnegative and sum to one; logits are rejected. Continuous
teacher targets are layer-normalized per frame after alignment; their absolute
feature scale is intentionally not distilled. Hash/grid/kind/width mismatches fail.

Only set the pair flag after checking the augmentation actually preserves lyrics,
timing and the designated content. A flag is not proof of validity. Content-only
consistency still affects the shared frontend; it does not prove performance is
preserved. Do not apply blanket pitch-invariance constraints to lexical tones.
The optional speaker adversary defaults to zero and can erase useful phonetic or
language information when the data is confounded. A failed speaker probe does not
prove that all identity information was removed.

A JSONL manifest line is:

```json
{"path":"caches/example.pt","source_group":"original-song-id"}
```

Paths must be unique, exist, and stay inside the manifest directory. Train and
validation cannot share paths or source groups. Accurate grouping is the caller's
responsibility; unseen-singer evaluation also needs disjoint singer partitions.

The experiment JSON contains `student` (optional StudentConfig fields), `teachers`
(a list of TeacherSpec dictionaries using the same name/hash/grid/kind), `epochs`,
`learning_rate`, `accumulation`, `max_samples`, `seed`, `run_label`, and optionally
`objective` and `speaker_count`. TeacherSpec additionally requires `dim` and accepts
positive `weight`. Objective keys are `pair_weight` (default 0.1), `prosody_weight`
(default 0.1), and `adversarial_scale` (default 0). Optional terms need matching
records to have any effect. No data-duration or hyperparameter optimum is claimed.

```bash
python scripts/train_content_student.py --config experiment.json \
  --train train.jsonl --valid valid.jsonl --output runs/content-001 --device cuda
```

The output directory must be new. Training uses one clip per microbatch with
configurable gradient accumulation (including a correctly scaled final partial
group). Clips longer than max_samples are rejected: slice and regenerate metadata
upstream instead of silently misaligning cached targets. No AMP, distributed
training, teacher execution, or automatic audio download is implemented here.
It writes metrics, experiment configuration and per-epoch student artifacts.
These artifacts omit task heads and optimizer/RNG state; resuming training from
them is not supported. The reported validation value is a representation objective,
NOT a voice quality score. CUDA memory logging includes the research task and is
not a controlled comparison against the original conversion pipeline.

```python
from research.content_student import load_student
encoder, provenance = load_student("runs/content-001/student_epoch_001.pt", "cpu")
output = encoder(waveform_batch, lengths, sample_rate=16000)
```

Strict artifact loading uses weights_only=True; use trusted files. Export requires
positive training-step metadata and provenance, but this is not a cryptographic
attestation of successful training. Synthetic test exports remain labeled synthetic.

## 4. Integration and experimental decision gates

This 192-dimensional content space is NOT the old 768-dimensional ContentVec
space. Even a projection to 768 does not establish semantic/coordinate equivalence.
A real SVC experiment must align frames and train/validate the matching downstream
projection/converter, or train a teacher-space output head as an explicit separate
baseline. The performance representation requires an additional controlled input
and training objective. Existing adapter exports do not magically perform either
migration. No new production encoder name is registered in this pass.

The first comparison should separate these experiments:

1. Original ContentVec and existing converter; fixed held-out authorized songs.
2. Replacement encoder/feature-pooling baseline with matched downstream adaptation.
3. Waveform student with one ContentVec teacher (compression control).
4. Same student plus independent phonetic targets, then prosody/pair constraints
   one at a time. Pooling is task-dependent: preserving source singing expression
   and intentionally replacing that expression are different evaluation tasks.
5. Only after quality is established, compare convolution/full-attention versus
   efficient state/linear-attention backbones and deployment quantization.

Measure phonetic errors/ABX, pitch/voicing and timing, target similarity, source
identity leakage, and blinded listening on consonants, breaths, high notes,
long vowels and transitions. Probe scores alone are not ground truth. Evaluate
unseen recordings and singers. Keep source/target identity versus source/target
technique goals explicit. Report actual preprocessing, per-voice training and
inference time/memory separately at matched downstream data and compute budgets.
Do not call a faster step, lower training loss, or smaller file a quality win.

## 5. Tests and limits of evidence

```bash
python -m unittest discover -s tests -p 'test_content_student.py' -v
```

34 new CPU tests passed on PyTorch 2.10.0+cpu. They exercise the actual small
student/objective networks, exact frame counts, padding invariance, time alignment,
span validation, target provenance checks, feature/posterior gradients, pair and
prosody branches, gradient-reversal sign, split checks and artifact round trips.
A synthetic end-to-end runner test completes an epoch with two optimizer updates,
exports and reloads a student. An overfit test reduces a synthetic objective from
0.409595 to 0.062239; that number is solely an optimization/mechanics check.

No pretrained teacher was downloaded, no real singing recording was used, no
So-VITS waveform was synthesized, and no CUDA/MPS benchmark was run. Earlier
repository suites were not rerun during this isolated pass. The tests do not show
ContentVec was surpassed, stable speaker disentanglement, sufficient capacity for
singing, or lower measured total GPU requirements. No private audio or credentials
are committed. All new code is explicitly labeled experimental and AGPL-3.0-only.

## Primary sources

[1] Qian et al., ContentVec, ICML 2022.
https://proceedings.mlr.press/v162/qian22b.html
https://github.com/auspicious3000/contentvec

[2] Hu et al., Controllable Singing Style Conversion with Boundary-Aware
Information Bottleneck, arXiv v1, 2026-04-07. Table I and Section IV-C inspected.
https://arxiv.org/abs/2604.05526
https://arxiv.org/html/2604.05526v1

[3] Chang et al., DC-Spin, Interspeech 2025 (preprint 2024).
https://www.isca-archive.org/interspeech_2025/chang25_interspeech.html
https://arxiv.org/html/2410.24177v1
https://github.com/vectominist/spin

[4] S-JEPA, 2026-06-17.
https://arxiv.org/abs/2606.19398
https://github.com/gioannides/s-jepa

[5] SITA, January 2026; method discussion here uses the inspected v1.
https://arxiv.org/html/2601.09050v1

[6] Gated DeltaNet-2, 2026-05-21.
https://arxiv.org/abs/2605.22791
https://github.com/NVlabs/GatedDeltaNet-2

[7] Qwen3-TTS, 2026-01-22.
https://arxiv.org/abs/2601.15621
https://github.com/QwenLM/Qwen3-TTS

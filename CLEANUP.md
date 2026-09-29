# Utility cleanup — 2026-09-29

Baseline for this pass: `25845b010010ca38e257f9aa4255bd3c0cc0c35b`.
Changes are confined to `exp/low-vram-core`; `4.1-Stable` is unchanged.

## Completed

- Consolidated `utils.f0_to_coarse` and `utils.normalize_f0` as re-exports of
  `modules.model_utils`, removing duplicate implementations while retaining the
  historical public names. Finite-input behavior is regression-tested; NaN
  normalization now raises the model-core ValueError instead of exiting with a
  success status.
- Replaced repeated factory branches with allowlisted lazy imports, preserving all
  13 content-encoder names and all 6 F0-predictor names and their constructor kwargs.
- Deferred FAISS, sklearn, librosa and SciPy imports to the functions that need them.
  Importing configuration/tensor utilities no longer requires retrieval or audio
  analysis packages. Missing backends still fail when their feature is requested.
- Deferred KMeans import until cluster loading.
- Consolidated plotting setup and RGB extraction across the three public plotting
  functions. Pixel arrays are copied from the canvas before closing figures.
- Removed dead comments and duplicate utility plumbing; replaced mutable summary
  defaults and side-effect list comprehensions. `utils.py` is 125 lines shorter.

## Not removed or changed

No neural layers, ContentVec weights, feature dimensions, F0 bin arithmetic,
encoders, WebUI, ONNX exporters, diffusion backend, attribution or license were
removed. A code-search miss is not proof that a standalone script or public API is
unused. `diffusion/infer_gt_mel.py` was inspected but retained pending a complete
branch-specific call-site audit. The legacy checkpoint loader's permissive policy
and checkpoint retention selection were not redesigned in this cleanup.

These changes do not establish a reduction in GPU memory, faster training, better
voice conversion, or a complete minimal-dependency environment. Some legacy entry
points still load their own optional dependencies eagerly.

## Validation

33 cleanup regressions passed on PyTorch 2.10.0+cpu, including a fresh rerun on
2026-09-29. Tests use fresh subprocesses with optional imports blocked, compare
finite pitch calculations against legacy arithmetic, check factory dispatch and
kwargs, and test configuration, tensor helpers, synthetic WAV I/O, small linear
checkpoint round trips and real plotting. Backend factories, FAISS and RMS analysis
are mocked where explicitly stated in the tests; no model downloads are performed.

```bash
python -m unittest discover -s tests -p 'test_cleanup.py' -v
```

The earlier architecture and low-VRAM suites were not rerun during this pass. No
CUDA training, real voice data evaluation or end-to-end WebUI conversion was run.
ContentVec representation alternatives are a separate research task and are not
silently substituted by this cleanup.

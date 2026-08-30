# MOSAIC implementation and experiment runbook

Date: 2026-08-30

Branch: `mosaic-ordinal-proof`

The mathematical rationale and preregistered research sequence are in
[`mosaic_plan.md`](mosaic_plan.md). This document describes what the current
code actually does and the next executable experiment.

## Implemented now

- `models/local_efficientnet.py`: one full-canvas, spatially bounded
  EfficientNetV2-S prefix with RF-small/medium/large taps, frozen BatchNorm,
  no permitted post-tap spatial mixing, a fixed canonical support mask, and
  explicit receptive-field metadata.
- `models/mosaic.py`: nested ordinal local states; FP32 simplex- and
  log-tail-stabilized truncated Poisson--binomial counting; a learned boundary
  cardinality law; deterministic minimum dual-proof projection; direct
  continuation and stop probabilities; and fixed-proof pivotality.
- `models/mosaic_model.py`: the image model, with no global/pooled classifier
  path around the proof.
- `losses/mosaic.py`: fold-level at-risk imbalance weights, projected and dense
  continuation likelihoods, and an optional witness-stability primitive.
- `Datasets/mosaic_data.py`: full-canvas APTOS/EyePACS loading, disjoint folds,
  tight-field cropping, direct canonical-square resizing, and the same
  image-independent centered ellipse for every sample.
- `training/mosaic_trainer.py` and `train_mosaic.py`: dense warm-up, proof
  tolerance ramp, validation-QWK checkpointing, complete resume state, atomic
  checkpoints, source-content implementation signatures, held-out test
  evaluation, proof diagnostics, and bounded AMP overflow recovery. The image
  encoder uses autocast, while the lattice-wide local-state reduction and
  exact proof circuit remain FP32.
- `inference/mosaic_certificate.py` and
  `export_mosaic_certificates.py`: SHA-256-protected JSON certificates,
  hardware-tolerant numerical replay, minimum-prefix verification, fixed-proof
  intervention verification, source/checkpoint provenance, and per-grade
  certificate sampling.
- `tools/benchmark_mosaic_core.py`: full-lattice proof runtime plus a calibrated
  mixed-label learning-gradient preflight. Saturated random fields are timed
  separately as a worst-case proof stress.
- `tools/cache_spatial_features.py`: a source-metadata-, preprocessing-, and
  encoder-state-hashed raw local-map cache writer for later head-only studies.
- `tools/audit_mosaic_shortcuts.py`: a validation-only source-dimension and
  canonical-mask shortcut audit; outer-test image headers are not read unless
  explicitly requested.
- `submit_mosaic_aptos_pilot.sh`: the first end-to-end viability job.

The stability loss cannot yet be enabled because the first data path returns a
single photometric view. Adaptive proof replacement, substitute-proof
enumeration, the complete anti-cheating experiment suite, and the formal
matched-baseline cache reader/driver are later experiments, not silently
claimed as completed features.

## Verified locally

Run:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_mosaic.py \
  tests/test_mosaic_loss.py \
  tests/test_local_efficientnet.py \
  tests/test_mosaic_certificate.py \
  tests/test_mosaic_integration.py
```

The local result is **91 passed, 1 CUDA-only skipped**. The skipped
full-lattice certificate test runs automatically in the cluster script; it
checks a CUDA forward followed by canonical CPU replay.

The latest local CPU Gate-0 run at `P=12,544`, batch 4, and `R=32` reported:

- serial/tree agreement on an independent 127-event check: `3.55e-15` max error;
- calibrated full-lattice forward: approximately `0.55 s`;
- calibrated full-lattice backward: approximately `0.61 s`;
- calibrated mean proof size: `1,187` cells;
- minimum local-logit gradient norm: `0.03697`;
- minimum cardinality-alpha gradient norm: `0.49433`; and
- saturated random-logit stress proof: approximately `12,259` cells.

CPU timings are machine-specific. The proof core is practical locally, but
the CUDA preflight remains the authoritative runtime check.

## First experiment: APTOS fold-0 viability pilot

This pilot is intentionally **not formal Gate 1**. Formal Gate 1 in the plan
is a three-seed, cached-feature comparison against matched baseline heads. The
pilot asks a narrower question first: can the complete image-to-proof model
learn a useful APTOS validation signal without collapsing or selecting the
whole retina?

Before submission:

1. synchronize this branch and every untracked MOSAIC file to
   `/dpc/kuin0170/ESPACOL-New`;
2. use the established environment `G`, which is also used by the working
   OPTIC launchers, and install its test-only dependency once with
   `python -m pip install -r requirements-dev.txt`;
3. ensure its PyTorch build supports CUDA 12.6 and `torch.amp`;
4. ensure the EfficientNetV2-S ImageNet weights are cached or downloadable;
5. verify APTOS has `train.csv` and all 3,662 `train_images`; and
6. use a new run directory; set `MOSAIC_RESUME=1` only to continue a compatible
   `last.pth`. Checkpoints reject preprocessing or source-content signature
   drift.

Do not reinstall `requirements.txt` into an already working cluster
environment merely to obtain pytest: it pins a different PyTorch build and its
current `packaging` entry is a machine-local Conda build path. The pilot checks
the active PyTorch, torchvision, CUDA, pretrained weights, and pytest versions
before doing any experiment work.

Submit:

```bash
sbatch submit_mosaic_aptos_pilot.sh
```

To select a different fresh run directory:

```bash
MOSAIC_RUN_DIR=runs/mosaic_aptos_pilot_v2 \
  sbatch submit_mosaic_aptos_pilot.sh
```

To resume the default run deliberately after pre-emption:

```bash
MOSAIC_RESUME=1 sbatch submit_mosaic_aptos_pilot.sh
```

Monitor the job file written in the cluster repository root:

```bash
tail -f mosaic_aptos_pilot_<JOB_ID>.out
```

The job records the git/worktree and software/GPU provenance, checks the
pretrained weights and all 3,662 files, writes a validation-only shortcut
audit, runs the structural and cross-device certificate tests, benchmarks Gate
0 on CUDA, performs configured-batch 896-by-896 dense and projected GradScaler
backward/memory smoke tests, and then trains one seed of APTOS fold 0 with:

- one 896-by-896 full fundus canvas;
- RF-medium (stride 8, theoretical RF 95 pixels);
- a 112-by-112 evidence lattice;
- `R=32` count truncation;
- four dense-circuit warm-up epochs;
- a four-epoch projection-tolerance ramp from 0 to 0.02;
- complement-suppression fraction 0.5;
- effective-number balanced at-risk continuation likelihood; and
- no CTOT, GPA, CGPM, BiomedCLIP, CORAL, or global classifier bypass.

The AMP scale starts at 8,192 rather than PyTorch's default 65,536 because the
local-state bias aggregates roughly ten thousand evidence cells. A detected
AMP overflow skips the optimizer update and reduces the scale, as GradScaler
intends; repeated overflows or any non-AMP non-finite gradient remain fatal.

The permitted EfficientNet trunk is fine-tuned from epoch 1 at the lower
encoder learning rate. This is an intentional viability-pilot deviation from
the later cached head-only Gate-1 protocol.

## Decision files and rule

The important outputs are:

- `runs/mosaic_aptos_pilot/gate0.json`;
- `runs/mosaic_aptos_pilot/shortcut_audit.json`;
- `runs/mosaic_aptos_pilot/fold0/history.csv`;
- `runs/mosaic_aptos_pilot/fold0/best.pth`;
- `runs/mosaic_aptos_pilot/fold0/best_validation_metrics.json`.

The script passes `--skip_test`, so no outer-test predictions or
`test_metrics.json` are produced during architecture development. Architecture
decisions use only `best_validation_metrics.json`; the outer fold remains
untouched until the architecture and hyperparameters are frozen.
Predicted grades use `round(E[Y])`, matching the existing project evaluation;
class argmax is stored only as a diagnostic.

The pilot passes only if:

1. validation QWK clearly exceeds a collapsed/majority solution and validation
   accuracy is useful;
2. grade-3/4 validation recall does not collapse;
3. mean/median/p90 proof fractions are materially below 1 after the ramp;
4. maximum recorded sufficiency and complement-constraint violations remain
   at numerical zero;
5. local and cardinality gradients remain finite and nonzero; and
6. proof fractions do not trend back toward the entire valid retina.

Do not compare the pilot to a published APTOS number with a different split.
The formal comparison must rerun the corrected baseline on this exact fold.

## Certificate smoke test after a passing pilot

Export five deterministic examples per grade (up to 25 total) and independently
replay them:

```bash
python export_mosaic_certificates.py \
  --checkpoint runs/mosaic_aptos_pilot/fold0/best.pth \
  --dataset aptos \
  --data_root Datasets/aptos2019-blindness-detection \
  --fold 0 \
  --split validation \
  --per_grade_limit 5 \
  --output_dir runs/mosaic_aptos_pilot/fold0/validation_certificates
```

Every manifest row must have `replay_ok=True`. Twenty-five certificates are a
replay smoke test, not evidence-quality evaluation.

## Optional overnight exploratory runs

Two validation-only launchers allow longer experiments to run without opening
either dataset's outer test fold:

```bash
sbatch submit_mosaic_aptos_fold1_100.sh
sbatch submit_mosaic_dr_fold0_75.sh
```

The first starts a fresh APTOS fold-1 run through epoch 100 in
`runs/mosaic_aptos_fold1_e100`. The second starts patient-disjoint EyePACS fold
0 through epoch 75 in `runs/mosaic_dr_fold0_e75`. The EyePACS exploration uses
batch size 16 to turn the much larger corpus into a practical two-day job; the
APTOS fold remains at batch size 4 so it is directly comparable to the first
pilot. Both use nonblocking writer locks and refuse to overwrite existing
artifacts. If a wall-time interruption occurs after at least one completed
epoch, resume the identical run with:

```bash
MOSAIC_RESUME=1 sbatch submit_mosaic_aptos_fold1_100.sh
MOSAIC_RESUME=1 sbatch submit_mosaic_dr_fold0_75.sh
```

These are exploratory robustness runs, not substitutes for the matched
APTOS Gate-1 comparison or the preregistered two-fold EyePACS promotion rule.
Their launchers set early-stopping patience equal to the epoch budget so that
the requested learning curves are collected in full. Checkpoint selection
still uses inner-validation QWK, and `--skip_test` keeps the outer folds
untouched.

## Only after the pilot passes

Implement/run formal Gate 1: a frozen cached-map, three-seed matched comparison
between MOSAIC and the closest corrected baselines on the identical validation
split. The preregistered promotion rule is mean validation QWK no more than
0.01 below the corrected baseline, with accuracy then MAE as tie-breakers, plus
nontrivial proof compression. Next run the RF-small/medium/large locality
frontier and anti-cheating tests. EyePACS is last, after architecture and
hyperparameters are frozen.

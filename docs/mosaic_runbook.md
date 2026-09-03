# MOSAIC implementation and experiment runbook

Date: 2026-09-03

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
- `models/mosaic_decoder.py`: a proof-only decision layer whose prospectively
  locked operating rule is raw `posterior_median`. It also exposes five fixed
  alternatives for validation diagnostics, including analytic outcome-weight
  corrections; none has learned or validation-fitted parameters, and none
  accepts image or feature input.
- `training/mosaic_trainer.py` and `train_mosaic.py`: dense warm-up, proof
  tolerance ramp, validation-accuracy checkpointing, complete resume state, atomic
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
- `tools/audit_mosaic_decoders.py`: a fixed-checkpoint, inner-validation-only
  audit of six pre-specified proof decisions, plus boundary-wise empty-proof
  diagnostics. It never constructs an outer-test data loader.
- `submit_mosaic_decoder_audit.sh`: two-argument GPU launcher for that audit;
  it accepts `aptos|dr` and a checkpoint path.
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
The operating rule for all new folds is raw `posterior_median`: selected-proof
continuation scores define the cumulative ordinal cascade and the final grade
is

```text
sum_k 1[P(Y>k) >= 0.5].
```

The at-risk loss uses different stop/advance weights, so the audit also tests
an analytic inverse candidate. For boundary weights ordered as
`[w_stop, w_advance]`, that diagnostic computes

```text
p_continue = w_stop*c / (w_stop*c + w_advance*s)
```

in stable log space before rebuilding the ordinal class distribution. The
candidate is an analytic consequence of the weighted population loss, but a
finite trained network need not satisfy that population optimum.

The APTOS fold-0 audit reproduced the historical checkpoint exactly and found:

- `rounded_expected`: 82.25% accuracy, 0.9030 QWK, 0.2150 MAE;
- raw `posterior_median`: 83.62% accuracy, 0.8929 QWK, 0.2116 MAE;
- raw `class_map`: 83.96% accuracy, but 0.8719 QWK and 0.2253 MAE; and
- `deweighted_class_map`: 81.23% accuracy, 0.8671 QWK, 0.2526 MAE.

All three deweighted variants were worse than `rounded_expected` on every
reported APTOS metric. Deweighting was therefore rejected. Raw class MAP was
not promoted: it traded away ordinal and imbalance-sensitive quality, and
choosing it after seeing this fold would have been post-hoc. Raw posterior
median had help/harm 6/2 (exact paired `p=0.289`), so APTOS alone did not
justify changing the operating rule.

The independent EyePACS fold-0 audit then found:

| Rule | Acc. | Bal. Acc. | Macro-F1 | QWK | MAE |
|---|---:|---:|---:|---:|---:|
| `rounded_expected` | 81.63 | 52.08 | 0.5225 | **0.8093** | 0.2283 |
| `class_map` | **84.38** | 51.44 | **0.5635** | 0.7865 | 0.2201 |
| `posterior_median` | 83.93 | **52.92** | 0.5593 | 0.7999 | **0.2170** |
| `deweighted_mean_round` | 81.91 | 50.74 | 0.5076 | 0.8070 | 0.2255 |
| `deweighted_class_map` | 84.31 | 49.39 | 0.5450 | 0.7848 | 0.2207 |
| `deweighted_posterior_median` | 83.97 | 50.82 | 0.5466 | 0.7931 | 0.2185 |

Raw posterior median corrected 95 predictions while making 22 newly wrong
(exact paired `p=5.3144e-12`). Across APTOS and EyePACS it consistently raises
accuracy and lowers MAE, while lowering QWK by about 0.01. It is therefore
locked prospectively for untouched folds, with the QWK cost reported. Raw MAP
is rejected: it adds only 0.34 APTOS and 0.45 EyePACS accuracy points over the
median while worsening balanced accuracy, QWK, and MAE on both. Deweighting is
rejected for its lack of consistent cross-dataset benefit and large APTOS
degradation. The remaining five rules stay diagnostic only. Historical
checkpoints retain their serialized rule. Every rule receives only
selected-proof transitions, so the proof remains the exclusive grade path.

The median rule is conventional and is not a novelty claim. It minimizes
absolute error under the model-implied raw cascade law; the `>= 0.5`
convention selects the upper grade on an exact tie. It has zero learned or
validation-fitted parameters.

All configured grades must occur in the training fold. Startup rejects any
fold for which a boundary lacks either stop or advance examples; such a fold
cannot identify the complete declared ordinal model or support the full
raw/deweighted decoder audit.

The completed audits can be reproduced without touching either outer test
split:

```bash
sbatch submit_mosaic_decoder_audit.sh aptos \
  runs/mosaic_aptos_f0_ampfix_policy_v2/fold0/best.pth
```

For EyePACS:

```bash
sbatch submit_mosaic_decoder_audit.sh dr \
  runs/mosaic_dr_f0_ampfix_policy_v2/fold0/best.pth
```

The audit must exactly reproduce the checkpoint's historical rounded-mean
accuracy/QWK/MAE before any comparison is trusted. It writes
`decoder_audit/summary.json` and per-image `predictions.csv`. The cross-dataset
decision is now frozen: future folds use `posterior_median`, and no row may be
selected retrospectively per fold. The audit also reports empty selected
proofs among positive/advance targets. Those cases expose the hard-projection
dead-gradient region and determine whether a later training-objective change
is justified.

The APTOS audit found no empty proof on an advance target at any boundary; the
EyePACS audit found only 5/844 (0.59%) at boundary 0 and none at boundaries
1--3. Empty-proof dead gradients therefore do not explain the weak late-grade
recall. The historical sample-mean loss instead gave each boundary total mass
proportional to its risk-set size: the final boundary had only 351/2,636
(13.3%) of boundary 0's training support on APTOS and 1,278/28,446 (4.5%) on
EyePACS. New runs default to `--transition_reduction boundary_mean`, which uses
fixed complete-training-fold counts to average the four boundary losses
equally in expectation. Both projected and dense likelihoods use the same
reduction. `--transition_reduction sample_mean` is retained only to reproduce
the historical objective; never resume an older run under the new default.

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
replay them. Certificate export requires the checkpoint's implementation
signature to match the active source exactly; pre-audit checkpoints are valid
for the decoder audit above, but must not be relabelled as certificates produced
by the audited implementation.

```bash
python export_mosaic_certificates.py \
  --checkpoint runs/mosaic_aptos_f0_median_v1/fold0/best.pth \
  --dataset aptos \
  --data_root Datasets/aptos2019-blindness-detection \
  --fold 0 \
  --split validation \
  --per_grade_limit 5 \
  --output_dir runs/mosaic_aptos_f0_median_v1/fold0/validation_certificates
```

Every manifest row must have `replay_ok=True` and `replay_status=passed`.
Using `--no_verify` records `null/not_run`, never a false success. Twenty-five
certificates are a replay smoke test, not evidence-quality evaluation.

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
uses inner-validation accuracy, and `--skip_test` keeps the outer folds
untouched.

The audited decoder implementation is resume-critical and changes the
implementation signature. Do not resume a pre-audit run under the audited
source. Use the validation-only decoder audit on its `best.pth`, then start any
new training comparison in a fresh run directory. The launchers use the
prospectively locked `MOSAIC_DECISION_RULE=posterior_median`. The other five
rules are diagnostic comparisons, not dataset-specific training
configurations.

## Only after the pilot passes

Implement/run formal Gate 1: a frozen cached-map, three-seed matched comparison
between MOSAIC and the closest corrected baselines on the identical validation
split. The preregistered promotion rule is mean validation QWK no more than
0.01 below the corrected baseline, with accuracy then MAE as tie-breakers, plus
nontrivial proof compression. Next run the RF-small/medium/large locality
frontier and anti-cheating tests. EyePACS is last, after architecture and
hyperparameters are frozen.

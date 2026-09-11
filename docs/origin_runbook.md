# ORIGIN-v3 bounded-rate first-experiment runbook

## Scientific status

ORIGIN is implemented as a new, isolated experiment. It does not modify
OPTIC-C or MOSAIC. ORIGIN-v3 is a fresh viability test of the
generator-exclusive prediction path, not evidence of SOTA or a
foundation-model claim. ORIGIN-v1 and v2 artifacts are retained for failure
analysis but are not resumable as v3 checkpoints.

The prospective configuration is frozen before seeing ORIGIN validation
results:

- ConvNeXt-Tiny, ImageNet initialization, 640 px input;
- native feature stages at strides 4, 8, 16, and 32;
- cumulative prerequisite atoms;
- reference exposure 4096, atom initialization `1e-6`, null-prior
  initialization `1e-4`;
- bounded cumulative/hybrid atom mass with an explicit null category;
- uncoupled bounded sigmoid atoms only for the matched independent-mode
  ablation;
- total-rate ceiling 64, null-prior cap 1, boundary-scale cap 2, and a
  one-rate-unit ledger-roundoff reserve;
- class MAP for the primary accuracy decision;
- unweighted categorical NLL + 0.25 ranked probability score;
- no rate-magnitude penalty and no class oversampling;
- ReduceLROnPlateau on validation loss;
- outer test fold locked;
- an audited FP64 Taylor scaling-and-squaring generator decoder;
- a one-time AMP loss-scale reset to 256 when the encoder unfreezes.

For the default five-grade model, the derived active atom-mass cap is

```text
(64 - 1 - 1) / (4096 * 2) = 0.007568359375
```

The valid geometry weights at each nonempty scale sum to 4096 and the learned
scale weights sum to one. Active cumulative atom mass is at most the derived
cap, boundary calibration is at most 2, and the prior is at most 1. Hence the
usable real-arithmetic rate is at most 63. Stored local maps remain FP32, while
image totals and removed-rate sums are accumulated in FP64. The forward pass
fails closed above 64. This bound is part of the architecture and does not
depend on gradient clipping, the optional evidence penalty, or a decoder-side
clamp.

## Cluster preparation

On the cluster, switch to the exact branch and install the development-only
test dependency once if needed:

```bash
cd /dpc/kuin0170/ESPACOL-New
git fetch origin
git switch severity-foundation-model
git pull --ff-only
python -m pip install -r requirements-dev.txt
```

Do not install packages from inside a training job.

## Gate 0: structural GPU preflight

```bash
sbatch scripts/submit_origin_v3_preflight.sh
```

The preflight must finish with `ORIGIN-v3 bounded-rate preflight passed.` It
checks all ORIGIN unit tests, including adversarial bounded-ledger tests and
high-rate decoder-gradient tests, the exact pretrained ConvNeXt
initialization, the FP64 structural decoder, normalized posterior, a real CUDA
autocast/GradScaler optimizer step, finite backward gradients, architecture
metadata, and CUDA availability. If this job fails, do not submit training.

## Gate 1: APTOS fold-0 inner validation

After Gate 0 passes:

```bash
sbatch scripts/submit_origin_v3_aptos_f0.sh
```

The run must use the fresh default directory
`runs/origin_aptos_f0_v3_bounded/fold0`. Do not point the v3 script at a v1 or
v2 run directory. Checkpoint schema, implementation signature, architecture
signature, and configuration signature prevent an incompatible resume. The
outer fold is constructed and hashed byte-for-byte for provenance but is not
evaluated. Monitor with:

```bash
squeue -u "$USER"
tail -f origin_v3_aptos_f0_<JOB_ID>.out
tail -f origin_v3_aptos_f0_<JOB_ID>.err
```

The initial gate passes when all of the following hold on the fixed inner
validation split:

- accuracy at least 82%;
- QWK at least 0.86;
- no persistent AMP overflow or non-finite value;
- no architectural rate-cap violation and no persistent concentration near
  the logged 80% cap-utilization warning threshold;
- exact certificate replay error remains within tolerance.

Inspect:

```bash
cat runs/origin_aptos_f0_v3_bounded/fold0/result.json
cat runs/origin_aptos_f0_v3_bounded/fold0/validation_certificates.json
```

Inspect `max_total_rate`, `total_rate_cap`, and
`max_total_rate_cap_fraction` in both training history and validation output.
The 80% warning is diagnostic, not a second hidden clamp: persistent use near
the ceiling may indicate atom saturation or insufficient bounded capacity and
must be analyzed rather than silently raising the cap.

Certificates are exact interventions on stored generator ledger units. Each
records the unit's stride and receptive-field size. They are not claims that
the corresponding pixels were causally deleted; fine localization must be
reported per scale and validated separately.

## Gate 2: EyePACS/DR fold-0

Only after the APTOS mechanism and numerical gate passes:

```bash
sbatch scripts/submit_origin_v3_dr_f0.sh
```

The fresh default directory is `runs/origin_dr_f0_v3_bounded/fold0`; v2
checkpoints must not be copied into it or resumed through another entry point.

The target is at least 85% inner-validation accuracy and QWK above 0.82. A
result below this target is not repaired by repeatedly reading the locked outer
test. Diagnose only from training and inner-validation histories.

The completed bounded fold-0 run meets this performance target: inner-
validation accuracy is 85.7369% and QWK is 0.82007. This is a performance-gate
result, not yet authorization to launch full cross-validation.

## Post-Gate 2: full inner-validation structural audit

Before full EyePACS cross-validation, run the validation-wide audit against the
selected checkpoint:

```bash
sbatch scripts/submit_origin_v3_dr_validation_audit.sh
```

The default artifact is separate from the training result and smoke
certificates:

```text
runs/origin_dr_f0_v3_bounded/fold0/audits/full_validation_audit_v2.json
```

Do not overwrite `result.json` or `validation_certificates.json`. Keep any
additional audit configuration as a separately named immutable artifact, for
example `full_validation_audit_<audit-tag>.json`, with its own checksum.

Proceed to full CV only if the audit covers every inner-validation sample,
passes its stored-ledger conservation and FP64 replay tolerances, shows no rate
above the architectural cap, reports the boundary and scale concentration
behind any near-cap rate, and provides grade-stratified effects including the
exact grade-0 local-versus-prior factorization. Persistent boundary saturation
or unexplained single-scale dominance requires a targeted validation-only
diagnosis before CV; it is not repaired by raising the cap.

These interventions delete already-computed stored-ledger entries without
renormalization. They are not causal pixel masking or image re-encoding.
Raw-rate top-k deletion is likewise neither a posterior-impact ranking nor a
proof of a minimal necessary or sufficient region. Locality claims must be
scoped by each unit's theoretical receptive field: a unit whose receptive
field covers the input is global-support evidence even when it has a spatial
index. Locked outer images are read only as opaque bytes to verify the
pre-existing split signature; they are never decoded, transformed, inferred
on, or evaluated, and are not used to choose an architectural change,
threshold, audit setting, or CV decision.

## Post-audit controlled experiment: RF-sealed local evidence

The completed v3 audit passes its exact ledger, replay, grade-0 factorization,
and bounded-rate checks. It does not pass the intended fine-local explanation
gate: `s16+s32` account for 96.57% of the raw evidence rate, while their
theoretical receptive fields (1096 and 1688 pixels) exceed the 640-pixel input.
The v3 result remains an immutable performance and mechanism baseline; do not
launch full CV from it for a fine-local interpretability claim.

The next registered comparison changes only the active evidence scales from
`s4,s8,s16,s32` to `s4,s8`. The encoder, initialization, likelihood, RPS,
optimizer, learning rates, schedule, epochs, split, seed, decision rule, rate
caps, and locked-test policy remain identical. Every active evidence unit then
has a theoretical receptive field no larger than 224 pixels. This is an
RF-sealed evidence experiment, not yet proof of pixel causality or lesion
semantics.

Run the inexpensive APTOS screen and the target EyePACS fold in separate new
directories:

```bash
sbatch scripts/submit_origin_v4_local_aptos_f0.sh
sbatch scripts/submit_origin_v4_local_dr_f0.sh
```

Default artifacts are written under:

```text
runs/origin_aptos_f0_v4_local_s4s8/fold0
runs/origin_dr_f0_v4_local_s4s8/fold0
```

Do not resume either run from a v3 all-scale checkpoint. Before full CV,
require the EyePACS local model to meet the pre-registered performance target
(at least 85% accuracy and QWK above 0.82), then repeat the validation-wide
structural audit against its own checkpoint. Interpretability assessment must
use set-valued deletion curves as well as single-cell effects: distributed DR
evidence need not be reducible to one necessary cell.

After a successful EyePACS training job, submit its matching audit with:

```bash
sbatch scripts/submit_origin_v4_local_dr_validation_audit.sh
```

## ORIGIN-v5: Spatial Receptive-Field Firewall (SRFF)

The v4 experiment isolates the remaining architectural problem. Removing the
full-support `s16` and `s32` ledgers establishes bounded support, but its
completed APTOS result falls to 82.25% accuracy and 0.8723 QWK; the EyePACS
trajectory likewise remains below v3. Shallow features provide locality but
do not retain enough pretrained semantic depth. This is an architectural
deficit, not a decoder, learning-rate, or numerical-stability failure.

V5 introduces one replacement dependency path. The `convnext_tiny_srff`
encoder preserves the ordinary `s4` and `s8` feature ledgers. It then applies
an immutable-validity mask to the `80x80` stride-8 map, partitions that map
into non-overlapping `16x16` windows, and runs the pretrained ConvNeXt deep
suffix independently inside every window. No suffix operation may communicate
across a window boundary. Average pooling produces one descriptor per window,
forming a `5x5` `s128` regional ledger. Its conservative input receptive-field
bound is

```text
RF(s128) = RF(s8) + (16 - 1) * stride(s8)
          = 224 + 15 * 8 = 344 pixels.
```

Thus v5 recovers deep semantic processing without restoring v3's global-support
evidence cells. The only active prediction path remains
`s4,s8,s128 -> conserved rates -> pure-birth generator -> grade posterior`.
The deepest evidence unit is a 344-pixel regional unit, not a pixel-level
lesion mask. The `s4` and `s8` ledgers retain finer-resolution evidence.

This comparison keeps the v3 split, preprocessing, seed, ImageNet
initialization, generator, cumulative atoms, proper NLL plus RPS objective,
rate caps, optimizer, learning rates, batch size, freeze schedule, LR schedule,
epoch budget, checkpoint rule, and MAP decision fixed. Class weighting,
evidence-budget regularization, and outer-test evaluation remain disabled.

Run the structural preflight first, then both fold-0 inner-validation jobs if
resources are available:

```bash
PREFLIGHT_JOB=$(sbatch --parsable scripts/submit_origin_v5_srff_preflight.sh | cut -d';' -f1)
APTOS_JOB=$(sbatch --parsable --dependency=afterok:${PREFLIGHT_JOB} scripts/submit_origin_v5_srff_aptos_f0.sh | cut -d';' -f1)
DR_JOB=$(sbatch --parsable --dependency=afterok:${PREFLIGHT_JOB} scripts/submit_origin_v5_srff_dr_f0.sh | cut -d';' -f1)
printf 'preflight=%s aptos=%s eyepacs=%s\n' "${PREFLIGHT_JOB}" "${APTOS_JOB}" "${DR_JOB}"
```

Default artifacts are isolated under:

```text
runs/origin_aptos_f0_v5_srff_w16/fold0
runs/origin_dr_f0_v5_srff_w16/fold0
```

APTOS is a screening diagnostic: 84% accuracy and 0.90 QWK is the prospective
promising threshold, while the run must at minimum improve upon v4's
82.25%/0.8723 result. The target-dataset gate is stricter. On the same EyePACS
fold-0 validation split and the same selected checkpoint, v5 must exceed both
v3 metrics (85.7369% accuracy and 0.82007 QWK) without falling below v3's
56.86% balanced accuracy or 0.6118 macro-F1. Do not launch full cross-validation
from a checkpoint that fails this joint gate.

Only after the EyePACS gate passes, run the complete 3,162-image validation
audit:

```bash
sbatch scripts/submit_origin_v5_srff_dr_validation_audit.sh
```

The audit must reconstruct the `convnext_tiny_srff`/`s4,s8,s128` checkpoint,
verify conservation and exact replay, report no evidence unit with receptive
field above 344 pixels, and evaluate set-valued deletion curves. Single-cell
necessity is not required because DR evidence can be distributed across
multiple retinal regions. At the observed v3 throughput, expected V100 runtime
is approximately 50 minutes for APTOS and 18--21 hours at the full EyePACS
epoch budget; the packed suffix has approximately the same aggregate spatial
area as the ordinary suffix, so batch size 8 and 32 GB GPU memory should remain
adequate.

## Historical failed runs

The original September 8 v1 runs used the generic `torch.matrix_exp` backward
in FP32. Their forward predictions are diagnostic only: adversarial replay of
their logged rate vectors revealed incorrect backward derivatives. Do not
resume those checkpoints or report their metrics as final evidence.

ORIGIN-v2 replaced that kernel with the audited FP64 Taylor
scaling-and-squaring decoder, but retained unbounded softplus atoms, boundary
scales, and prior rates. Its APTOS fold-0 run reached 86.35% validation
accuracy and 0.9266 QWK, while also reaching a maximum rate of 484.553 and a
mean boundary-0 rate of 80.312. Those metrics are useful viability evidence,
not a passed numerical/mechanism gate.

The corrected v2 EyePACS run later failed closed when one pure-birth boundary
rate reached 720.012, beyond the decoder's audited limit of 700. Raising that
limit would not have been an honest repair: for example,
`exp(-720) ≈ 2.03e-313` is below the smallest normal FP64 value, and the
then-current probability-floor log path would flatten the exact grade-0 NLL
gradient. A loss penalty also could not guarantee safety because it is applied
only after decoding. ORIGIN-v3 therefore bounds the emitted ledger, prior, and
boundary scales before the generator is formed. It is a new architecture and
requires fresh runs.

## Mandatory publication controls

Before any CVPR-level claim, run matched ConvNeXt-Tiny softmax, CORAL, and CORN
heads under the identical split, preprocessing, and training budget. Then
ablate scale set, cumulative versus independent atoms, CTMC versus non-CTMC
decoding, null prior, and decision rule. Interpretation claims additionally
require per-scale deletion, pixel-mask deletion/re-encoding, background and
coordinate controls, and external mask evaluation. Call ORIGIN
`foundation-compatible` until broad pretraining and transfer experiments have
actually been completed.

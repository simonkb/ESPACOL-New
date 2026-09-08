# ORIGIN first-experiment runbook

## Scientific status

ORIGIN is implemented as a new, isolated experiment. It does not modify
OPTIC-C or MOSAIC. The first run is a viability test of the generator-exclusive
prediction path, not evidence of SOTA or a foundation-model claim.

The prospective configuration is frozen before seeing ORIGIN validation
results:

- ConvNeXt-Tiny, ImageNet initialization, 640 px input;
- native feature stages at strides 4, 8, 16, and 32;
- cumulative prerequisite atoms;
- reference exposure 4096, atom initialization `1e-6`, null-prior
  initialization `1e-4`;
- class MAP for the primary accuracy decision;
- unweighted categorical NLL + 0.25 ranked probability score;
- no rate-magnitude penalty and no class oversampling;
- ReduceLROnPlateau on validation loss;
- outer test fold locked;
- an audited FP64 Taylor scaling-and-squaring generator decoder;
- a one-time AMP loss-scale reset to 256 when the encoder unfreezes.

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
sbatch scripts/submit_origin_preflight.sh
```

The preflight must finish with `ORIGIN preflight passed.` It checks all ORIGIN
unit tests (including adversarial high-rate gradient checks), the exact
pretrained ConvNeXt initialization, the FP64 structural decoder, normalized
posterior, a real CUDA autocast/GradScaler optimizer step, finite backward
gradients, and CUDA availability. If this job fails, do not submit training.

## Gate 1: APTOS fold-0 inner validation

After Gate 0 passes:

```bash
sbatch scripts/submit_origin_aptos_f0.sh
```

The repaired run must use a new directory such as
`runs/origin_aptos_f0_v2_stable/fold0`. The outer fold
is constructed and hashed byte-for-byte for provenance but is not evaluated.
Monitor with:

```bash
squeue -u "$USER"
tail -f origin_aptos_f0_<JOB_ID>.out
tail -f origin_aptos_f0_<JOB_ID>.err
```

The initial gate passes when all of the following hold on the fixed inner
validation split:

- accuracy at least 82%;
- QWK at least 0.86;
- no persistent AMP overflow or non-finite value;
- transition rates remain out of the logged saturation-risk regime;
- exact certificate replay error remains within tolerance.

Inspect:

```bash
cat runs/origin_aptos_f0_v2_stable/fold0/result.json
cat runs/origin_aptos_f0_v2_stable/fold0/validation_certificates.json
```

Certificates are exact interventions on stored generator ledger units. Each
records the unit's stride and receptive-field size. They are not claims that
the corresponding pixels were causally deleted; fine localization must be
reported per scale and validated separately.

## Gate 2: EyePACS/DR fold-0

Only after the APTOS mechanism and numerical gate passes:

```bash
sbatch scripts/submit_origin_dr_f0.sh
```

The target is at least 85% inner-validation accuracy and QWK above 0.82. A
result below this target is not repaired by repeatedly reading the locked outer
test. Diagnose only from training and inner-validation histories.

The original September 8 v1 runs used the generic `torch.matrix_exp` backward
in FP32. Their forward predictions are diagnostic only: adversarial replay of
their logged rate vectors revealed incorrect backward derivatives. Do not
resume those checkpoints or report their metrics as final evidence.

## Mandatory publication controls

Before any CVPR-level claim, run matched ConvNeXt-Tiny softmax, CORAL, and CORN
heads under the identical split, preprocessing, and training budget. Then
ablate scale set, cumulative versus independent atoms, CTMC versus non-CTMC
decoding, null prior, and decision rule. Interpretation claims additionally
require per-scale deletion, pixel-mask deletion/re-encoding, background and
coordinate controls, and external mask evaluation. Call ORIGIN
`foundation-compatible` until broad pretraining and transfer experiments have
actually been completed.

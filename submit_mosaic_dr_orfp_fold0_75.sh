#!/bin/bash
# GATED MOSAIC-v5 EyePACS experiment: RF-medium + Ordinal Receptive-Field
# Packing (ORFP), fold 0, inner validation only. Ordinarily this follows the
# APTOS viability gate; MOSAIC_DR_PARALLEL_PILOT=1 records an explicit
# compute-availability override without falsely declaring that gate passed.
# EyePACS fold 0 is promoted beyond this run only at >=2688/3162 correct
# (85.01%), QWK >=.82, G3 recall >=41/83, G4 recall >=16/64, and complete
# packing/proof/certificate invariants. The recall floors match the prior
# uncompiled posterior-median audit. This script never reads the outer test.
#SBATCH --job-name=mosaic_dr_f0_orfp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/mosaic_dr_orfp_f0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/mosaic_dr_orfp_f0_%j.err
#SBATCH --account=kuin0170

source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate G || exit 1

set -euo pipefail
cd /dpc/kuin0170/ESPACOL-New

if [[ "${MOSAIC_APTOS_ORFP_GATE:-}" == "passed" ]]; then
  LAUNCH_AUTHORIZATION="aptos_gate_passed"
elif [[ "${MOSAIC_DR_PARALLEL_PILOT:-0}" == "1" ]]; then
  LAUNCH_AUTHORIZATION="parallel_resource_pilot_before_aptos_gate"
  echo "WARNING: launching the fixed EyePACS fold-0 pilot before the APTOS result." >&2
  echo "This run cannot authorize additional EyePACS folds by itself." >&2
else
  echo "EyePACS ORFP is gated and was not launched." >&2
  echo "First verify the fixed APTOS gate in docs/mosaic_plan.md." >&2
  echo "After it passes, submit with:" >&2
  echo "  sbatch --export=ALL,MOSAIC_APTOS_ORFP_GATE=passed submit_mosaic_dr_orfp_fold0_75.sh" >&2
  echo "For an explicitly authorized concurrent pilot, use:" >&2
  echo "  sbatch --export=ALL,MOSAIC_DR_PARALLEL_PILOT=1 submit_mosaic_dr_orfp_fold0_75.sh" >&2
  exit 2
fi
export MOSAIC_DR_LAUNCH_AUTHORIZATION="${LAUNCH_AUTHORIZATION}"

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="${MOSAIC_RUN_DIR:-runs/mosaic_dr_f0_orfp_v5}"
FOLD=0
TOTAL_EPOCHS=75
FOLD_DIR="${RUN_DIR}/fold${FOLD}"
mkdir -p "${FOLD_DIR}"

exec 9>"${FOLD_DIR}/.writer.lock"
if ! flock -n 9; then
  echo "Another process is already writing ${FOLD_DIR}." >&2
  exit 3
fi
RESUME_ARGS=()
if [[ "${MOSAIC_RESUME:-0}" == "1" ]]; then
  if [[ ! -f "${FOLD_DIR}/last.pth" ]]; then
    echo "Cannot resume: ${FOLD_DIR}/last.pth does not exist." >&2
    exit 4
  fi
  if [[ -f "${FOLD_DIR}/best_validation_metrics.json" ]]; then
    echo "Refusing to resume a completed/early-stopped run: ${FOLD_DIR}." >&2
    echo "Run the checkpoint audit/export directly if post-training work was interrupted." >&2
    exit 5
  fi
  RESUME_ARGS+=(--resume)
elif [[ -e "${FOLD_DIR}/last.pth" || -e "${FOLD_DIR}/history.csv" ]]; then
  echo "Fresh ORFP run refused: ${FOLD_DIR} contains training artifacts." >&2
  echo "Choose a new MOSAIC_RUN_DIR or set MOSAIC_RESUME=1 deliberately." >&2
  exit 6
fi

echo "=== MOSAIC-v5 ORFP EyePACS fold 0 / 75 epochs (inner validation only) ==="
echo "launch_authorization=${MOSAIC_DR_LAUNCH_AUTHORIZATION}"
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

if ! python -c 'import pytest' >/dev/null 2>&1; then
  echo "Missing pytest in conda environment G." >&2
  echo "Install once: python -m pip install -r requirements-dev.txt" >&2
  exit 7
fi

python -m pytest -q \
  tests/test_mosaic_packing.py \
  tests/test_mosaic.py \
  tests/test_mosaic_loss.py \
  tests/test_mosaic_decoder.py \
  tests/test_mosaic_decoder_audit.py \
  tests/test_local_efficientnet.py \
  tests/test_mosaic_certificate.py \
  tests/test_mosaic_certificate_export.py \
  tests/test_mosaic_integration.py

# Validate the patient-disjoint split before any training. The outer fold is
# counted for integrity only and is never loaded by the training invocation.
python - <<'PY'
import torch
from Datasets.mosaic_data import class_histogram, eyepacs_fold, load_eyepacs_items

items = load_eyepacs_items("Datasets/DR")
if len(items) != 35126:
    raise RuntimeError(f"expected 35126 EyePACS images, found {len(items)}")
train, validation, test = eyepacs_fold(items, 0, n_folds=10, seed=42)
if (len(train), len(validation), len(test)) != (28446, 3162, 3518):
    raise RuntimeError(
        "EyePACS fold-0 identity changed: "
        f"{(len(train), len(validation), len(test))}"
    )
if class_histogram(train) != [20918, 1979, 4271, 702, 576]:
    raise RuntimeError(
        f"EyePACS fold-0 training classes changed: {class_histogram(train)}"
    )
if class_histogram(validation) != [2318, 216, 481, 83, 64]:
    raise RuntimeError(
        "EyePACS fold-0 validation classes changed: "
        f"{class_histogram(validation)}"
    )
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
print("gpu", torch.cuda.get_device_name(0))
print("eyepacs_images", len(items), "classes", class_histogram(items))
print(
    "fold0",
    {"train": len(train), "validation": len(validation), "locked_test": len(test)},
    "train_classes",
    class_histogram(train),
)
PY

python tools/audit_mosaic_shortcuts.py \
  --dataset dr \
  --data_root Datasets/DR \
  --folds "${FOLD}" \
  --image_size 896 \
  --output_stride 8 \
  --json_out "${RUN_DIR}/shortcut_audit.json"

# Independent CUDA canary for the exact ORFP architecture and EyePACS
# boundary weights. It is intentionally small; the training batch remains 16.
python - <<'PY'
import torch

from inference.mosaic_certificate import (
    build_mosaic_certificate,
    verify_mosaic_certificate,
)
from losses.mosaic import MosaicLoss
from models.mosaic_model import build_mosaic_model
from utils.spatial_mask import centered_ellipse_mask

max_overlap = 0.5
model = build_mosaic_model(
    num_classes=5,
    image_size=896,
    local_stage="rf_medium",
    local_dim=128,
    pretrained=True,
    max_count=32,
    sufficiency_tolerance=0.02,
    complement_suppression=0.5,
    region_grid_size=0,
    rf_packing_max_overlap=max_overlap,
).cuda().train()
labels = torch.tensor([0, 1, 3, 4], device="cuda")
train_labels = [0] * 20918 + [1] * 1979 + [2] * 4271 + [3] * 702 + [4] * 576
criterion = MosaicLoss.from_training_labels(
    train_labels,
    5,
    weight_method="effective_num",
    weight_beta=0.999,
    max_transition_weight=10.0,
    dense_weight=0.1,
    transition_reduction="boundary_mean",
).cuda()
trunk_parameters = list(model.encoder.trunk.parameters())
trunk_ids = {id(parameter) for parameter in trunk_parameters}
head_parameters = [
    parameter for parameter in model.parameters()
    if id(parameter) not in trunk_ids
]
optimizer = torch.optim.AdamW(
    [
        {"params": trunk_parameters, "lr": 1e-4},
        {"params": head_parameters, "lr": 5e-4},
    ],
    weight_decay=1e-5,
)
images = torch.zeros(4, 3, 896, 896, device="cuda")
masks = centered_ellipse_mask(896, 896, batch_size=4, device=torch.device("cuda"))
optimizer.zero_grad(set_to_none=True)
with torch.amp.autocast("cuda"):
    output = model(images, masks, project=True)
    loss, diagnostics = criterion(
        output.transitions,
        labels,
        projected_stop_probabilities=output.stop_probabilities,
        projected_log_transition_probabilities=output.log_transition_probabilities,
        projected_log_stop_probabilities=output.log_stop_probabilities,
        dense_transitions=output.dense_transitions,
        dense_stop_probabilities=output.dense_stop_probabilities,
        dense_log_transition_probabilities=output.dense_log_transition_probabilities,
        dense_log_stop_probabilities=output.dense_log_stop_probabilities,
        proof_sizes=output.proof.proof_size,
    )

packed = output.evidence.rf_packing_mask
source_valid = output.source_valid_mask
if not model.rf_packing_enabled or model.rf_packing_max_overlap != max_overlap:
    raise RuntimeError("ORFP model configuration contract failed")
if model.region_grid_size != 0:
    raise RuntimeError("ORFP must use the uncompiled source lattice")
if packed is None or source_valid is None or output.source_lattice is None:
    raise RuntimeError("ORFP source/packing provenance is missing")
if output.source_lattice.lattice_size != (112, 112):
    raise RuntimeError("ORFP source lattice contract failed")
if not torch.equal(packed, output.valid_mask):
    raise RuntimeError("the count circuit did not receive the declared packed mask")
if bool((packed & ~source_valid).any()):
    raise RuntimeError("ORFP retained an invalid retinal site")
if bool((output.proof.selected_mask & ~packed.unsqueeze(-1)).any()):
    raise RuntimeError("a reported proof contains a site excluded by ORFP")
if bool(
    (
        output.evidence.witness_probabilities[..., 1:]
        > output.evidence.witness_probabilities[..., :-1] + 1e-7
    ).any()
):
    raise RuntimeError("ORFP violated cumulative ordinal nesting")

height, width = output.source_lattice.lattice_size
stride = float(model.output_stride)
receptive_field = float(model.receptive_field)
observed_max_overlap = 0.0
for sample_mask in packed:
    indices = torch.nonzero(sample_mask, as_tuple=False).flatten()
    if indices.numel() < 1:
        raise RuntimeError("ORFP produced an empty event ledger")
    rows = torch.div(indices, width, rounding_mode="floor").float()
    columns = torch.remainder(indices, width).float()
    dy = (rows[:, None] - rows[None, :]).abs() * stride
    dx = (columns[:, None] - columns[None, :]).abs() * stride
    overlap = (
        (receptive_field - dy).clamp_min(0.0)
        * (receptive_field - dx).clamp_min(0.0)
        / (receptive_field * receptive_field)
    )
    overlap.fill_diagonal_(0.0)
    observed_max_overlap = max(observed_max_overlap, float(overlap.max()))
if observed_max_overlap > max_overlap + 1e-6:
    raise RuntimeError(
        f"ORFP overlap contract failed: {observed_max_overlap} > {max_overlap}"
    )
if not torch.isfinite(loss):
    raise RuntimeError("ORFP canary loss is non-finite")
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(), 5.0, error_if_nonfinite=True
)
optimizer.step()
certificate = build_mosaic_certificate(
    output,
    sample_index=0,
    sample_id="eyepacs_orfp_cuda_canary",
    sufficiency_tolerance=0.02,
    complement_suppression=0.5,
)
replay = verify_mosaic_certificate(certificate)
if not replay["ok"]:
    raise RuntimeError(f"ORFP certificate canary failed: {replay}")
print(
    "orfp_dr_canary",
    {
        "gpu": torch.cuda.get_device_name(0),
        "packed_event_counts": packed.sum(dim=1).tolist(),
        "max_allowed_overlap": max_overlap,
        "max_observed_overlap": observed_max_overlap,
        "loss": float(loss.detach()),
        "grad_norm": float(grad_norm),
        "dead_positive_count": float(diagnostics["dead_positive_count"]),
        "certificate_replay": replay["ok"],
    },
)
PY

python train_mosaic.py \
  --dataset dr \
  --data_root Datasets/DR \
  --run_dir "${RUN_DIR}" \
  --folds "${FOLD}" \
  --seed 42 \
  "${RESUME_ARGS[@]}" \
  --skip_test \
  --local_stage rf_medium \
  --evidence_dim 128 \
  --region_grid_size 0 \
  --rf_packing_max_overlap 0.5 \
  --image_size 896 \
  --batch_size 16 \
  --num_workers 8 \
  --epochs "${TOTAL_EPOCHS}" \
  --early_stop_patience "${TOTAL_EPOCHS}" \
  --decision_rule posterior_median \
  --transition_reduction boundary_mean \
  --max_count 32 \
  --count_block_size 64 \
  --normal_expected_count 0.5 \
  --dense_warmup_epochs 4 \
  --proof_ramp_epochs 4 \
  --proof_epsilon 0.02 \
  --necessity_fraction 0.5 \
  --dense_loss_weight 0.1 \
  --transition_weighting effective_num \
  --effective_num_beta 0.999 \
  --transition_weight_cap 10.0 \
  --lr 1e-4 \
  --head_lr 5e-4 \
  --weight_decay 1e-5 \
  --amp_init_scale 8192 \
  --amp_growth_interval 2000 \
  --amp_max_consecutive_skips 8

# Reproduce and decompose the trained proof decoder on inner validation only.
# Diagnostic alternatives are never used to replace the fixed checkpoint rule.
python tools/audit_mosaic_decoders.py \
  --checkpoint "${FOLD_DIR}/best.pth" \
  --data_root Datasets/DR \
  --batch_size 16 \
  --num_workers 8 \
  --output_dir "${FOLD_DIR}/decoder_audit" \
  --require_implementation_match

# Replay one certificate per true grade from the trained model before deciding
# whether any further EyePACS fold is warranted. The outer fold remains locked.
python export_mosaic_certificates.py \
  --checkpoint "${FOLD_DIR}/best.pth" \
  --dataset dr \
  --data_root Datasets/DR \
  --fold "${FOLD}" \
  --split validation \
  --per_grade_limit 1 \
  --output_dir "${FOLD_DIR}/certificate_smoke"

# Report the prospectively fixed fold-0 gate. Passing does not launch another
# fold and does not unlock or evaluate the outer test split.
MOSAIC_GATE_METRICS="${FOLD_DIR}/best_validation_metrics.json" \
MOSAIC_GATE_OUTPUT="${FOLD_DIR}/orfp_gate_metrics.json" \
python - <<'PY'
import json
import os
from pathlib import Path

metrics = json.loads(Path(os.environ["MOSAIC_GATE_METRICS"]).read_text())
n = 3162
accuracy = float(metrics["acc"])
correct = int(round(accuracy * n / 100.0))
qwk = float(metrics["qwk"])
recalls = [float(value) for value in metrics["per_grade_recall"]]
proof_fraction = float(metrics["proof_fraction_mean"])
promotion = (
    correct >= 2688
    and qwk >= 0.82
    and recalls[3] >= 41.0 / 83.0 - 1e-8
    and recalls[4] >= 16.0 / 64.0 - 1e-8
)
assessment = {
    "scope": "EyePACS fold-0 inner validation only",
    "n": n,
    "correct": correct,
    "accuracy_percent": accuracy,
    "qwk": qwk,
    "mae": float(metrics["mae"]),
    "grade_3_recall": recalls[3],
    "grade_4_recall": recalls[4],
    "proof_fraction_mean": proof_fraction,
    "metric_promotion_beyond_fold_0": promotion,
    "launch_authorization": os.environ["MOSAIC_DR_LAUNCH_AUTHORIZATION"],
    "structural_gate": (
        "preflight passed; complete validation-certificate/anti-cheating "
        "analysis remains required before promotion"
    ),
    "outer_test_evaluated": False,
}
Path(os.environ["MOSAIC_GATE_OUTPUT"]).write_text(
    json.dumps(assessment, indent=2) + "\n"
)
print("=== Fixed ORFP EyePACS fold-0 metric gate ===")
print(json.dumps(assessment, indent=2))
if promotion:
    print(
        "Metric promotion passed. Do not launch more folds until the "
        "structural/certificate analysis has also passed."
    )
else:
    print("Promotion failed. Do not launch additional EyePACS folds.")
PY

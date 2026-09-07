#!/bin/bash
# Controlled MOSAIC-v5 experiment: RF-medium + Ordinal Receptive-Field
# Packing (ORFP). APTOS fold 0, inner validation only; outer test stays locked.
# Apart from ORFP, this matches runs/mosaic_aptos_f0_boundary_mean_v1.
# Predeclared read-out (293 validation images; never inspect the outer test):
#   viability -> EyePACS: >=235 correct (80.20%), QWK >=.85, G3 recall >=4/15,
#     G4 recall >=8/24, proof fraction <=.25, and every structural check passes;
#   reference recovery: >=246 correct (83.96%), QWK >=.8808, MAE <=.2284;
#   stretch: >=250 correct (85.32%) while retaining the viability invariants.
#SBATCH --job-name=mosaic_a_f0_orfp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/mosaic_aptos_orfp_f0_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/mosaic_aptos_orfp_f0_%j.err
#SBATCH --account=kuin0170

source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate G || exit 1

set -euo pipefail
cd /dpc/kuin0170/ESPACOL-New

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="${MOSAIC_RUN_DIR:-runs/mosaic_aptos_f0_orfp_v5}"
FOLD=0
TOTAL_EPOCHS=50
FOLD_DIR="${RUN_DIR}/fold${FOLD}"
mkdir -p "${FOLD_DIR}"

exec 9>"${FOLD_DIR}/.writer.lock"
if ! flock -n 9; then
  echo "Another process is already writing ${FOLD_DIR}." >&2
  exit 2
fi
RESUME_ARGS=()
if [[ "${MOSAIC_RESUME:-0}" == "1" ]]; then
  if [[ ! -f "${FOLD_DIR}/last.pth" ]]; then
    echo "Cannot resume: ${FOLD_DIR}/last.pth does not exist." >&2
    exit 3
  fi
  if [[ -f "${FOLD_DIR}/best_validation_metrics.json" ]]; then
    echo "Refusing to resume a completed/early-stopped run: ${FOLD_DIR}." >&2
    echo "Run the checkpoint audit/export directly if post-training work was interrupted." >&2
    exit 4
  fi
  RESUME_ARGS+=(--resume)
elif [[ -e "${FOLD_DIR}/last.pth" || -e "${FOLD_DIR}/history.csv" ]]; then
  echo "Fresh ORFP run refused: ${FOLD_DIR} contains training artifacts." >&2
  echo "Choose a new MOSAIC_RUN_DIR or set MOSAIC_RESUME=1 deliberately." >&2
  exit 5
fi

echo "=== MOSAIC-v5 ORFP APTOS fold 0 / 50 epochs (inner validation only) ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version

if ! python -c 'import pytest' >/dev/null 2>&1; then
  echo "Missing pytest in conda environment G." >&2
  echo "Install once: python -m pip install -r requirements-dev.txt" >&2
  exit 6
fi

# Freeze the exact development split used by the uncompiled RF-medium
# reference. The 733-image outer fold is counted but never evaluated.
python - <<'PY'
import torch
from Datasets.mosaic_data import aptos_fold, class_histogram, load_aptos_items

items = load_aptos_items("Datasets/aptos2019-blindness-detection")
if len(items) != 3662:
    raise RuntimeError(f"expected 3662 APTOS images, found {len(items)}")
train, validation, test = aptos_fold(items, 0, n_folds=5, seed=42)
if (len(train), len(validation), len(test)) != (2636, 293, 733):
    raise RuntimeError(
        "APTOS fold-0 identity changed: "
        f"{(len(train), len(validation), len(test))}"
    )
if class_histogram(train) != [1300, 266, 719, 139, 212]:
    raise RuntimeError(
        f"APTOS fold-0 training classes changed: {class_histogram(train)}"
    )
if class_histogram(validation) != [144, 30, 80, 15, 24]:
    raise RuntimeError(
        "APTOS fold-0 validation classes changed: "
        f"{class_histogram(validation)}"
    )
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
print("gpu", torch.cuda.get_device_name(0))
print("aptos_images", len(items), "classes", class_histogram(items))
print(
    "fold0",
    {"train": len(train), "validation": len(validation), "locked_test": len(test)},
    "train_classes",
    class_histogram(train),
    "validation_classes",
    class_histogram(validation),
)
PY

python tools/audit_mosaic_shortcuts.py \
  --dataset aptos \
  --data_root Datasets/aptos2019-blindness-detection \
  --folds "${FOLD}" \
  --image_size 896 \
  --output_stride 8 \
  --json_out "${RUN_DIR}/shortcut_audit.json"

# The full structural suite covers packing geometry, shared-boundary masks,
# ordinal nesting, the exact count/proof circuit, intervention, and replay.
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

# Exercise the exact 896-pixel model, ORFP compiler, loss, and optimizer once
# on CUDA. The pairwise check uses the same unclipped RF-square intersection
# fraction declared by ORFP, not feature-map distance or clipped image boxes.
python - <<'PY'
import torch

from inference.mosaic_certificate import (
    build_mosaic_certificate,
    verify_mosaic_certificate,
)
from losses.mosaic import MosaicLoss
from models.mosaic_model import build_mosaic_model
from utils.spatial_mask import centered_ellipse_mask

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")

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
train_labels = [0] * 1300 + [1] * 266 + [2] * 719 + [3] * 139 + [4] * 212
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
    sample_id="aptos_orfp_cuda_canary",
    sufficiency_tolerance=0.02,
    complement_suppression=0.5,
)
replay = verify_mosaic_certificate(certificate)
if not replay["ok"]:
    raise RuntimeError(f"ORFP certificate canary failed: {replay}")
print(
    "orfp_canary",
    {
        "gpu": torch.cuda.get_device_name(0),
        "source_stride": model.output_stride,
        "source_receptive_field": model.receptive_field,
        "source_valid_cells": model.expected_valid_cells,
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

# This changes exactly one architectural mechanism relative to the matched
# uncompiled RF-medium reference: ORFP is enabled with omega_RF=0.5.
python train_mosaic.py \
  --dataset aptos \
  --data_root Datasets/aptos2019-blindness-detection \
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
  --batch_size 4 \
  --num_workers 4 \
  --epochs "${TOTAL_EPOCHS}" \
  --early_stop_patience 30 \
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

# Audit the trained checkpoint on the same inner validation set. This is a
# diagnostic decomposition only: the checkpoint decoder remains prospectively
# fixed to posterior_median and the outer fold stays untouched.
python tools/audit_mosaic_decoders.py \
  --checkpoint "${FOLD_DIR}/best.pth" \
  --data_root Datasets/aptos2019-blindness-detection \
  --batch_size 4 \
  --num_workers 4 \
  --output_dir "${FOLD_DIR}/decoder_audit" \
  --require_implementation_match

# Export one deterministic validation example per true grade and independently
# replay every trained certificate. This tests the real learned ORFP ledger,
# not only the synthetic CUDA preflight above.
python export_mosaic_certificates.py \
  --checkpoint "${FOLD_DIR}/best.pth" \
  --dataset aptos \
  --data_root Datasets/aptos2019-blindness-detection \
  --fold "${FOLD}" \
  --split validation \
  --per_grade_limit 1 \
  --output_dir "${FOLD_DIR}/certificate_smoke"

# This is a label-metric read-out only. Structural viability also requires the
# preflight suite/canary above and the later certificate analysis; this block
# never launches EyePACS automatically.
MOSAIC_GATE_METRICS="${FOLD_DIR}/best_validation_metrics.json" \
MOSAIC_GATE_OUTPUT="${FOLD_DIR}/orfp_gate_metrics.json" \
python - <<'PY'
import json
import os
from pathlib import Path

metrics_path = Path(os.environ["MOSAIC_GATE_METRICS"])
metrics = json.loads(metrics_path.read_text())
n = 293
accuracy = float(metrics["acc"])
correct = int(round(accuracy * n / 100.0))
qwk = float(metrics["qwk"])
mae = float(metrics["mae"])
recalls = [float(value) for value in metrics["per_grade_recall"]]
proof_fraction = float(metrics["proof_fraction_mean"])

viability = (
    correct >= 235
    and qwk >= 0.85
    and recalls[3] >= 4.0 / 15.0 - 1e-8
    and recalls[4] >= 8.0 / 24.0 - 1e-8
    and proof_fraction <= 0.25
)
reference = viability and correct >= 246 and qwk >= 0.8808 and mae <= 0.2284
stretch = viability and correct >= 250
assessment = {
    "scope": "APTOS fold-0 inner validation only",
    "n": n,
    "correct": correct,
    "accuracy_percent": accuracy,
    "qwk": qwk,
    "mae": mae,
    "grade_3_recall": recalls[3],
    "grade_4_recall": recalls[4],
    "proof_fraction_mean": proof_fraction,
    "metric_viability_to_eyepacs": viability,
    "metric_reference_recovery": reference,
    "metric_stretch_success": stretch,
    "structural_gate": (
        "preflight_passed; complete validation-certificate/anti-cheating "
        "analysis remains required before a paper claim"
    ),
    "outer_test_evaluated": False,
}
Path(os.environ["MOSAIC_GATE_OUTPUT"]).write_text(
    json.dumps(assessment, indent=2) + "\n"
)
print("=== Fixed ORFP APTOS metric gate (not an automatic DR launch) ===")
print(json.dumps(assessment, indent=2))
if viability:
    print(
        "Metric viability passed. Review the structural outputs, then the "
        "gated EyePACS launcher may be authorized explicitly."
    )
else:
    print("Metric viability failed. Do not launch the EyePACS ORFP run.")
PY

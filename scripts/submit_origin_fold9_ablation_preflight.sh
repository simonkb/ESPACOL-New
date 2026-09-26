#!/bin/bash
#SBATCH --job-name=of9_ab_pre
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_fold9_ablation_logs/preflight_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_fold9_ablation_logs/preflight_%j.err
#SBATCH --account=kuin0170

set -euo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:?ORIGIN_REPO_ROOT is required}"
LAUNCH_COMMIT="${ORIGIN_LAUNCH_COMMIT:?ORIGIN_LAUNCH_COMMIT is required}"
EXPERIMENT_ROOT="${ORIGIN_FOLD9_ABLATION_ROOT:?ORIGIN_FOLD9_ABLATION_ROOT is required}"
DATA_ROOT="${ORIGIN_DATA_ROOT:?ORIGIN_DATA_ROOT is required}"
PROTOCOL="${ORIGIN_FOLD9_ABLATION_PROTOCOL:?ORIGIN_FOLD9_ABLATION_PROTOCOL is required}"

cd "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[[ "$(git rev-parse HEAD)" == "${LAUNCH_COMMIT}" ]] || {
  echo "Immutable launch commit mismatch." >&2; exit 2;
}
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
  echo "Tracked files changed in the immutable worktree." >&2; exit 3;
}

echo "=== ORIGIN fold-9 ablation structural preflight ==="
date --iso-8601=seconds
echo "commit=${LAUNCH_COMMIT}"
python --version

if ! python -c 'import pytest' >/dev/null 2>&1; then
  echo "Missing pytest. Install requirements-dev.txt in the G environment." >&2
  exit 4
fi
python -m pytest -q tests/test_origin*.py
python train_origin_ablation.py --help >/dev/null

ORIGIN_PREFLIGHT_PROTOCOL="${PROTOCOL}" \
ORIGIN_PREFLIGHT_COMMIT="${LAUNCH_COMMIT}" \
ORIGIN_PREFLIGHT_ROOT="${EXPERIMENT_ROOT}" \
ORIGIN_PREFLIGHT_DATA="${DATA_ROOT}" \
python - <<'PY'
import os
import gc
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import torch

from configs.origin_ablation_config import OriginAblationConfig
from Datasets.origin_data import class_histogram, load_origin_items, split_origin_items
from models.origin_ablation import build_origin_ablation_model
from scripts.origin_fold9_ablation_common import (
    DATASET,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SPLIT_HISTOGRAMS,
    FOLD,
    N_FOLDS,
    PREFLIGHT_MARKER_SCHEMA,
    canonical_sha256,
    frozen_config_for_variant,
    read_json,
    verify_protocol,
    write_json_atomic,
)
from training.origin_ablation_trainer import origin_ablation_implementation_signature

protocol = read_json(os.environ["ORIGIN_PREFLIGHT_PROTOCOL"])
selected = verify_protocol(
    protocol,
    expected_launch_commit=os.environ["ORIGIN_PREFLIGHT_COMMIT"],
    expected_root=os.environ["ORIGIN_PREFLIGHT_ROOT"],
    expected_data_root=os.environ["ORIGIN_PREFLIGHT_DATA"],
)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the fold-9 ablation")

items = load_origin_items(DATASET, os.environ["ORIGIN_PREFLIGHT_DATA"])
train, validation, locked = split_origin_items(
    DATASET, items, FOLD, n_folds=N_FOLDS, val_fraction=0.1, seed=42
)
subsets = {"train": train, "validation": validation, "locked_test": locked}
counts = {name: len(values) for name, values in subsets.items()}
histograms = {name: class_histogram(values, 5) for name, values in subsets.items()}
if counts != EXPECTED_SPLIT_COUNTS or histograms != EXPECTED_SPLIT_HISTOGRAMS:
    raise RuntimeError(
        f"EyePACS fold-9 identity changed: counts={counts}, histograms={histograms}"
    )
patient_sets = [
    {Path(path).stem.rsplit("_", 1)[0] for path, _ in subset}
    for subset in (train, validation, locked)
]
if any(patient_sets[i] & patient_sets[j] for i in range(3) for j in range(i + 1, 3)):
    raise RuntimeError("EyePACS fold 9 leaks a patient across partitions")

print("gpu", torch.cuda.get_device_name(0))
print("selected_variants", selected)
print("fold9_counts", counts)
print("fold9_histograms", histograms)

# Exercise every registered prediction path under the same CUDA/AMP regime as
# training before allocating multi-day jobs. This catches posterior dtype,
# gradient, state-dict, and reconstruction failures without opening any image
# from the locked outer partition.
allowed_config_fields = {field.name for field in fields(OriginAblationConfig)}
for variant_index, variant in enumerate(selected):
    config_values = frozen_config_for_variant(variant)
    config = OriginAblationConfig(
        **{
            key: value
            for key, value in config_values.items()
            if key in allowed_config_fields
        }
    )
    torch.manual_seed(17_000 + variant_index)
    torch.cuda.manual_seed_all(17_000 + variant_index)
    model = build_origin_ablation_model(config, pretrained=False).cuda().eval()
    images = torch.randn(2, 3, 64, 64, device="cuda")
    masks = torch.ones(2, 64, 64, dtype=torch.bool, device="cuda")
    with torch.autocast(device_type="cuda", enabled=True):
        output = model(images, pixel_valid_mask=masks, force_decoder_fp64=True)
    for field_name in (
        "class_probs",
        "log_class_probs",
        "cumulative_probs",
        "expected_grade",
    ):
        value = getattr(output, field_name)
        if value.dtype != torch.float64 or not bool(torch.isfinite(value).all()):
            raise RuntimeError(
                f"{variant} canary produced invalid FP64 posterior field {field_name}"
            )
    loss = -output.log_class_probs[:, 0].mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
        raise RuntimeError(f"{variant} canary produced non-finite gradients")

    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    architecture = model.architecture_metadata()
    del output, loss, gradients, model
    torch.cuda.empty_cache()
    reconstructed = build_origin_ablation_model(config, pretrained=False)
    reconstructed.load_state_dict(state, strict=True)
    if reconstructed.architecture_metadata() != architecture:
        raise RuntimeError(f"{variant} checkpoint reconstruction changed architecture")
    del reconstructed, state
    gc.collect()
    print("canary_passed", variant)

marker = {
    "schema": PREFLIGHT_MARKER_SCHEMA,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "launch_commit": os.environ["ORIGIN_PREFLIGHT_COMMIT"],
    "protocol_checksum_sha256": protocol["content_checksum_sha256"],
    "selected_variants": list(selected),
    "implementation_signature": origin_ablation_implementation_signature(),
    "gpu_canaries_completed": True,
}
marker["content_checksum_sha256"] = canonical_sha256(marker)
marker_path = Path(os.environ["ORIGIN_PREFLIGHT_ROOT"]) / "PREFLIGHT_COMPLETE.json"
if marker_path.exists():
    raise FileExistsError(f"refusing to overwrite preflight marker: {marker_path}")
write_json_atomic(marker_path, marker)
print("preflight_marker", marker_path)
PY

echo "ORIGIN fold-9 ablation preflight passed."

#!/bin/bash
# Dataset-free structural preflight for the ORIGIN-v6 relational transition
# ledger. The SHA-256-bound v3 source checkpoint is read but never modified.
#SBATCH --job-name=origin_v6_rpre
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v6_relation_preflight_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v6_relation_preflight_%j.err
#SBATCH --account=kuin0170

set -eo pipefail
set +u
source /etc/profile.d/lmod.sh || exit 1
module load miniconda/3 || exit 1
module load cuda/12.6 || exit 1
source activate "${ORIGIN_CONDA_ENV:-G}" || exit 1
set -u

REPO_ROOT="${ORIGIN_REPO_ROOT:-/dpc/kuin0170/ESPACOL-New}"
cd "${REPO_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

: "${ORIGIN_V3_CHECKPOINT:?Set ORIGIN_V3_CHECKPOINT to one completed v3 best.pth}"
: "${ORIGIN_V3_CHECKPOINT_SHA256:?Set ORIGIN_V3_CHECKPOINT_SHA256 to its expected sha256sum}"

echo "=== ORIGIN-v6 relational transition ledger preflight ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version
python -m pytest -q tests/test_origin*.py

python - <<'PY'
import os
from dataclasses import fields

import torch

from configs.origin_config import OriginConfig
from losses.origin import OriginLoss
from models.origin import build_origin_model
from training.origin_trainer import load_origin_v3_relation_warm_start

checkpoint = os.environ["ORIGIN_V3_CHECKPOINT"]
checksum = os.environ["ORIGIN_V3_CHECKPOINT_SHA256"]
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
allowed = {field.name for field in fields(OriginConfig)}
values = {key: value for key, value in state["config"].items() if key in allowed}
values.update(
    relation_enabled=True,
    relation_source_scale="s8",
    relation_grid_size=10,
    relation_dim=64,
    relation_head_dim=16,
    relation_delta_cap=2.0,
    warm_start_checkpoint=checkpoint,
    warm_start_sha256=checksum,
    warm_start_metric_floor_tolerance=1e-6,
    relation_only_epochs=1,
    epochs=max(1, int(values.get("epochs", 1))),
    resume=False,
)
cfg = OriginConfig(**values)
model = build_origin_model(
    num_classes=cfg.n_classes,
    encoder_name=cfg.encoder,
    pretrained=False,
    evidence_scales=cfg.evidence_scales,
    projection_dim=cfg.projection_dim,
    reference_count=cfg.reference_count,
    atom_rate_init=cfg.atom_rate_init,
    prior_rate_init=cfg.prior_rate_init,
    boundary_scale_init=cfg.boundary_scale_init,
    total_rate_cap=cfg.total_rate_cap,
    prior_rate_cap=cfg.prior_rate_cap,
    boundary_scale_cap=cfg.boundary_scale_cap,
    rate_roundoff_margin=cfg.rate_roundoff_margin,
    atom_mode=cfg.atom_mode,
    hybrid_cumulative_init=cfg.hybrid_cumulative_init,
    evidence_dropout=cfg.evidence_dropout,
    mask_valid_fraction=cfg.mask_valid_fraction,
    relation_enabled=True,
    relation_source_scale=cfg.relation_source_scale,
    relation_grid_size=cfg.relation_grid_size,
    relation_dim=cfg.relation_dim,
    relation_head_dim=cfg.relation_head_dim,
    relation_delta_cap=cfg.relation_delta_cap,
)
provenance = load_origin_v3_relation_warm_start(
    model,
    cfg,
    fold=int(state["fold"]),
    split_signature=str(state["split_signature"]),
)
if provenance is None:
    raise RuntimeError("strict v3-to-v6 migration returned no provenance")
if not torch.cuda.is_available():
    raise RuntimeError("ORIGIN-v6 preflight requires CUDA")
device = torch.device("cuda")
model.to(device).eval()
images = torch.zeros(1, 3, cfg.img_size, cfg.img_size, device=device)
mask = torch.ones(1, 1, cfg.img_size, cfg.img_size, dtype=torch.bool, device=device)
with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=cfg.amp):
    output = model(images, mask, force_decoder_fp64=True)
if output.relation_evidence is None:
    raise RuntimeError("v6 model emitted no relation evidence")
identity_error = float((output.total_rates - output.base_total_rates).abs().max().cpu())
if identity_error != 0.0:
    raise RuntimeError(f"zero-init relation field is not an exact v3 identity: {identity_error}")
all_edges = output.relation_evidence.edge_valid_mask
replayed = model.replay_without_relations(output, all_edges).output
replay_error = float((replayed.total_rates - output.base_total_rates).abs().max().cpu())
if replay_error > 2e-10:
    raise RuntimeError(f"all-edge deletion failed to recover v3 rates: {replay_error}")

# CUDA/AMP first-order learnability canary.  Zero initialization must preserve
# the v3 function while retaining a live gradient into the sole pair-output
# parameter.  Mirror the registered relation-only phase: the hash-bound v3
# base stays frozen and in evaluation mode while the relation field trains.
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith("generator.relation_field."))
model.eval()
relation_field = model.generator.relation_field
if relation_field is None:
    raise RuntimeError("relation field disappeared before gradient canary")
relation_field.train()
model.zero_grad(set_to_none=True)
relation_optimizer = torch.optim.AdamW(
    relation_field.parameters(), lr=cfg.head_lr, weight_decay=cfg.weight_decay
)
gradient_scaler = torch.amp.GradScaler(
    "cuda",
    enabled=cfg.amp,
    init_scale=cfg.amp_init_scale,
    growth_interval=cfg.amp_growth_interval,
)
with torch.autocast(device_type="cuda", enabled=cfg.amp):
    gradient_output = model(images, mask, force_decoder_fp64=True)
labels = torch.tensor([cfg.n_classes - 1], dtype=torch.long, device=device)
criterion = OriginLoss(
    cfg.n_classes,
    rps_weight=cfg.rps_weight,
    evidence_budget_weight=0.0,
).to(device)
with torch.autocast(device_type="cuda", enabled=False):
    canary_loss, canary_diagnostics = criterion(gradient_output, labels, epoch=0)
if not bool(torch.isfinite(canary_loss)):
    raise RuntimeError("ORIGIN NLL+RPS gradient-canary loss is non-finite")
gradient_scaler.scale(canary_loss).backward()
gradient_scaler.unscale_(relation_optimizer)
gradient = relation_field.trilinear_weight.grad
if gradient is None:
    raise RuntimeError("zero-init relation trilinear weight has no gradient")
if not bool(torch.isfinite(gradient).all()):
    raise RuntimeError("zero-init relation trilinear gradient is non-finite")
if int(torch.count_nonzero(gradient)) == 0:
    raise RuntimeError("zero-init relation trilinear gradient is identically zero")
base_gradient_names = [
    name
    for name, parameter in model.named_parameters()
    if not name.startswith("generator.relation_field.") and parameter.grad is not None
]
if base_gradient_names:
    raise RuntimeError(f"frozen v3 parameters received gradients: {base_gradient_names[:8]}")
print("gpu", torch.cuda.get_device_name(0))
print("source_sha256", provenance["source_checkpoint_sha256"])
print("migrated_keys", provenance["loaded_key_count"])
print("new_relation_keys", len(provenance["initialized_relation_keys"]))
print("zero_init_identity_error", identity_error)
print("all_edge_replay_error", replay_error)
print("nll_rps_canary_loss", float(canary_loss.detach().cpu()))
print(
    "nll_rps_canary_diagnostics",
    {key: float(value.detach().cpu()) if torch.is_tensor(value) else value
     for key, value in canary_diagnostics.items()},
)
print("relation_trilinear_grad_abs_max", float(gradient.abs().max().cpu()))
PY

echo "ORIGIN-v6 relation preflight passed."

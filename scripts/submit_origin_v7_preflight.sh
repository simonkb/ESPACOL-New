#!/bin/bash
# Dataset-free structural preflight for the ORIGIN-v7 identified sparse
# relation ledger. The SHA-256-bound v3 checkpoint is read but never modified.
#SBATCH --job-name=origin_v7_pre
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v7_preflight_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v7_preflight_%j.err
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

echo "=== ORIGIN-v7 identified sparse relation preflight ==="
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
    relation_variant="identified_sparse_v1",
    relation_edge_budget=8,
    relation_permutation_seed=617,
    warm_start_checkpoint=checkpoint,
    warm_start_sha256=checksum,
    warm_start_metric_floor_tolerance=1e-6,
    relation_only_epochs=1,
    epochs=max(1, int(values.get("epochs", 1))),
    resume=False,
)
cfg = OriginConfig(**values)


def build(variant: str):
    return build_origin_model(
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
        relation_variant=variant,
        relation_edge_budget=cfg.relation_edge_budget,
        relation_permutation_seed=cfg.relation_permutation_seed,
    )


model = build(cfg.relation_variant)
provenance = load_origin_v3_relation_warm_start(
    model,
    cfg,
    fold=int(state["fold"]),
    split_signature=str(state["split_signature"]),
)
if provenance is None or provenance.get("target_protocol") != "v7":
    raise RuntimeError("strict v3-to-v7 migration returned invalid provenance")
contract = model.architecture_metadata().get("relation_contract")
if not isinstance(contract, dict) or contract.get("kind") != (
    "identified_sparse_cumulative_ordinal_pair_log_odds_v1"
):
    raise RuntimeError("model does not declare the registered v7 relation contract")
if contract.get("pair_identification") != (
    "pre_geometry_anova_then_geometry_then_post_geometry_anova"
):
    raise RuntimeError("model does not declare the registered two-stage identification")
if contract.get("constant_endpoint_field") != "exact_zero_relation_evidence":
    raise RuntimeError("model does not declare the constant-field null contract")

# The target and both preregistered null controls must expose exactly the same
# trainable relation parameter names and shapes.
relation_signatures = {}
for variant in (
    "identified_sparse_v1",
    "additive_endpoint_control_v1",
    "shuffled_pair_control_v1",
):
    candidate = build(variant)
    relation_signatures[variant] = tuple(
        (name, tuple(parameter.shape), parameter.numel())
        for name, parameter in candidate.named_parameters()
        if name.startswith("generator.relation_field.")
    )
if len(set(relation_signatures.values())) != 1:
    raise RuntimeError(f"v7 relation controls are not parameter matched: {relation_signatures}")

if not torch.cuda.is_available():
    raise RuntimeError("ORIGIN-v7 preflight requires CUDA")
device = torch.device("cuda")
model.to(device).eval()
images = torch.zeros(1, 3, cfg.img_size, cfg.img_size, device=device)
mask = torch.ones(1, 1, cfg.img_size, cfg.img_size, dtype=torch.bool, device=device)
with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=cfg.amp):
    output = model(images, mask, force_decoder_fp64=True)
relation = output.relation_evidence
if relation is None or relation.selected_edge_mask is None:
    raise RuntimeError("v7 model emitted no selected-edge relation evidence")
if relation.candidate_edge_mask is None or not torch.equal(
    relation.candidate_edge_mask, relation.edge_valid_mask
):
    raise RuntimeError("v7 model emitted an invalid candidate-edge ledger")
identity_error = float((output.total_rates - output.base_total_rates).abs().max().cpu())
if identity_error != 0.0:
    raise RuntimeError(f"zero-init relation field is not an exact v3 identity: {identity_error}")
selected = relation.selected_edge_mask.bool()
valid = relation.edge_valid_mask[:, None].expand_as(selected)
if bool((selected & ~valid).any()):
    raise RuntimeError("v7 selected an invalid relation edge")
selected_counts = selected.flatten(2).sum(-1)
valid_counts = valid.flatten(2).sum(-1)
expected_counts = torch.minimum(
    valid_counts,
    torch.full_like(valid_counts, cfg.relation_edge_budget),
)
if not torch.equal(selected_counts, expected_counts):
    raise RuntimeError(
        f"v7 fixed edge budget failed: selected={selected_counts}, expected={expected_counts}"
    )
residuals = relation.interaction_residuals
if residuals is None:
    raise RuntimeError("v7 did not expose its identified interaction residual")
masked_residuals = torch.where(valid, residuals, torch.zeros_like(residuals))
row_error = float(masked_residuals.sum(-1).abs().max().cpu())
column_error = float(masked_residuals.sum(-2).abs().max().cpu())
anova_tolerance = 2e-5
if max(row_error, column_error) > anova_tolerance:
    raise RuntimeError(
        "masked two-way identification failed: "
        f"row_error={row_error}, column_error={column_error}"
    )
replayed = model.replay_without_relations(output, selected).output
replay_error = float((replayed.total_rates - output.base_total_rates).abs().max().cpu())
if replay_error > 2e-10:
    raise RuntimeError(f"selected-edge deletion failed to recover v3 rates: {replay_error}")

# CUDA/AMP learnability canary. The v3 base is both gradient-frozen and in
# evaluation mode; only the exactly zero-initialized relation output trains.
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith("generator.relation_field."))
model.eval()
relation_field = model.generator.relation_field
if relation_field is None:
    raise RuntimeError("relation field disappeared before gradient canary")
relation_field.train()
model.zero_grad(set_to_none=True)
optimizer = torch.optim.AdamW(
    relation_field.parameters(), lr=cfg.head_lr, weight_decay=cfg.weight_decay
)
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=cfg.amp,
    init_scale=cfg.amp_init_scale,
    growth_interval=cfg.amp_growth_interval,
)
# A constant image can correctly collapse under the identifiable interaction
# projection. Use a deterministic spatially varying image to test that the
# zero-output identity still has a live first-order learning path.
axis = torch.linspace(-1.0, 1.0, cfg.img_size, device=device)
yy, xx = torch.meshgrid(axis, axis, indexing="ij")
gradient_images = torch.stack((xx, yy, 0.5 * (xx + yy)), dim=0).unsqueeze(0)
with torch.autocast(device_type="cuda", enabled=cfg.amp):
    gradient_output = model(gradient_images, mask, force_decoder_fp64=True)
labels = torch.tensor([cfg.n_classes - 1], dtype=torch.long, device=device)
criterion = OriginLoss(
    cfg.n_classes,
    rps_weight=cfg.rps_weight,
    evidence_budget_weight=0.0,
).to(device)
with torch.autocast(device_type="cuda", enabled=False):
    loss, diagnostics = criterion(gradient_output, labels, epoch=0)
if not bool(torch.isfinite(loss)):
    raise RuntimeError("ORIGIN-v7 NLL+RPS canary loss is non-finite")
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
gradient = relation_field.trilinear_weight.grad
if gradient is None or not bool(torch.isfinite(gradient).all()):
    raise RuntimeError("zero-init relation output has no finite gradient")
if int(torch.count_nonzero(gradient)) == 0:
    raise RuntimeError("zero-init relation output gradient is identically zero")
base_gradients = [
    name
    for name, parameter in model.named_parameters()
    if not name.startswith("generator.relation_field.") and parameter.grad is not None
]
if base_gradients:
    raise RuntimeError(f"frozen v3 parameters received gradients: {base_gradients[:8]}")

print("gpu", torch.cuda.get_device_name(0))
print("source_sha256", provenance["source_checkpoint_sha256"])
print("relation_parameter_count", sum(item[2] for item in next(iter(relation_signatures.values()))))
print("zero_init_identity_error", identity_error)
print("selected_edges_per_boundary", selected_counts.detach().cpu().tolist())
print("anova_row_sum_error", row_error)
print("anova_column_sum_error", column_error)
print("all_selected_edge_replay_error", replay_error)
print("nll_rps_canary_loss", float(loss.detach().cpu()))
print("relation_output_gradient_abs_max", float(gradient.abs().max().cpu()))
PY

echo "ORIGIN-v7 preflight passed."

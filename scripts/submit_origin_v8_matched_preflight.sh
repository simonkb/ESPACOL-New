#!/bin/bash
# Dataset-free structural preflight for the three registered ORIGIN-v8 arms.
# The immutable APTOS v3 checkpoint is read but never modified.
#SBATCH --job-name=origin_v8_pre
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/dpc/kuin0170/ESPACOL-New/origin_v8_preflight_%j.out
#SBATCH --error=/dpc/kuin0170/ESPACOL-New/origin_v8_preflight_%j.err
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

: "${ORIGIN_EXPECTED_COMMIT:?Set ORIGIN_EXPECTED_COMMIT to the commit being preflighted}"
: "${ORIGIN_V3_CHECKPOINT:?Set ORIGIN_V3_CHECKPOINT to the immutable APTOS v3 best.pth}"
: "${ORIGIN_V3_CHECKPOINT_SHA256:?Set its expected sha256sum}"
[[ "$(git rev-parse HEAD)" == "${ORIGIN_EXPECTED_COMMIT}" ]] || {
  echo "Preflight commit differs from ORIGIN_EXPECTED_COMMIT." >&2
  exit 2
}

echo "=== ORIGIN-v8 conserved-witness matched-control preflight ==="
date --iso-8601=seconds
git rev-parse HEAD
git status --short
python --version
python -m pytest -q tests/test_origin*.py tests/test_compare_origin_v8_controls.py

python - <<'PY'
import hashlib
import os
from dataclasses import fields

import torch

from configs.origin_config import OriginConfig
from models.origin import build_origin_model
from training.origin_trainer import load_origin_v3_relation_warm_start


VARIANTS = (
    "identified_conserved_witness_v1",
    "additive_conserved_witness_control_v1",
    "shuffled_conserved_witness_control_v1",
)
checkpoint = os.environ["ORIGIN_V3_CHECKPOINT"]
expected_sha = os.environ["ORIGIN_V3_CHECKPOINT_SHA256"]
digest = hashlib.sha256()
with open(checkpoint, "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
observed_sha = digest.hexdigest()
if observed_sha != expected_sha:
    raise RuntimeError("immutable v3 source checkpoint SHA-256 mismatch")
source = torch.load(checkpoint, map_location="cpu", weights_only=False)
allowed = {field.name for field in fields(OriginConfig)}
values = {key: value for key, value in source["config"].items() if key in allowed}
values.update(
    relation_enabled=True,
    relation_source_scale="s8",
    relation_grid_size=10,
    relation_dim=64,
    relation_head_dim=16,
    relation_delta_cap=2.0,
    relation_edge_budget=8,
    relation_allocation_temperature=1.0,
    relation_permutation_seed=617,
    warm_start_checkpoint=checkpoint,
    warm_start_sha256=expected_sha,
    warm_start_metric_floor_tolerance=1e-6,
    relation_only_epochs=15,
    epochs=15,
    resume=False,
)


def build(variant: str):
    # Resetting the global seed proves that initialization, not merely shape,
    # is identical across the three separately launched jobs.
    torch.manual_seed(42)
    cfg = OriginConfig(**{**values, "relation_variant": variant})
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
        relation_variant=cfg.relation_variant,
        relation_edge_budget=cfg.relation_edge_budget,
        relation_allocation_temperature=cfg.relation_allocation_temperature,
        relation_permutation_seed=cfg.relation_permutation_seed,
    )
    provenance = load_origin_v3_relation_warm_start(
        model,
        cfg,
        fold=int(source["fold"]),
        split_signature=str(source["split_signature"]),
    )
    if provenance is None or provenance.get("target_protocol") != "v8":
        raise RuntimeError(f"{variant} did not perform a strict v3-to-v8 migration")
    return cfg, model


arms = {variant: build(variant) for variant in VARIANTS}
normalized_configs = {}
for variant, (cfg, _) in arms.items():
    normalized = dict(vars(cfg))
    normalized.pop("relation_variant")
    normalized_configs[variant] = normalized
if len({repr(sorted(config.items())) for config in normalized_configs.values()}) != 1:
    raise RuntimeError("V8 arm configurations differ beyond relation_variant")
parameter_signatures = {}
initial_value_hashes = {}
for variant, (_, model) in arms.items():
    entries = []
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not name.startswith("generator.relation_field."):
            continue
        entries.append((name, tuple(parameter.shape), parameter.numel(), str(parameter.dtype)))
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    parameter_signatures[variant] = tuple(entries)
    initial_value_hashes[variant] = digest.hexdigest()
if len(set(parameter_signatures.values())) != 1:
    raise RuntimeError(f"V8 arms are not parameter matched: {parameter_signatures}")
if len(set(initial_value_hashes.values())) != 1:
    raise RuntimeError(f"V8 relation initializations differ: {initial_value_hashes}")

if not torch.cuda.is_available():
    raise RuntimeError("ORIGIN-v8 preflight requires CUDA")
device = torch.device("cuda")
cfg = arms[VARIANTS[0]][0]
axis = torch.linspace(-1.0, 1.0, cfg.img_size, device=device)
yy, xx = torch.meshgrid(axis, axis, indexing="ij")
images = torch.stack((xx, yy, 0.5 * (xx + yy)), dim=0).unsqueeze(0)
mask = torch.ones(1, 1, cfg.img_size, cfg.img_size, dtype=torch.bool, device=device)
summaries = {}
candidate_count_reference = None
for variant, (_, model) in arms.items():
    model.to(device).eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=cfg.amp):
        output = model(images, mask, force_decoder_fp64=True)
    relation = output.relation_evidence
    if relation is None or relation.witness_allocation is None:
        raise RuntimeError(f"{variant} emitted no conserved witness ledger")
    identity_error = float((output.total_rates - output.base_total_rates).abs().max().cpu())
    if identity_error != 0.0:
        raise RuntimeError(f"{variant} is not an exact zero-start v3 identity")
    shortlist = relation.shortlist_edge_mask.bool()
    allocation = relation.witness_allocation
    candidate_counts = relation.candidate_edge_mask.flatten(1).sum(-1)
    if candidate_count_reference is None:
        candidate_count_reference = candidate_counts
    elif not torch.equal(candidate_counts, candidate_count_reference):
        raise RuntimeError(f"{variant} candidate relation budget differs")
    shortlist_counts = shortlist.flatten(2).sum(-1)
    if not bool(shortlist_counts.eq(8).all()):
        raise RuntimeError(f"{variant} did not enforce the registered max-witness cap")
    allocation_sums = allocation.flatten(2).sum(-1)
    if not torch.allclose(allocation_sums, torch.ones_like(allocation_sums), atol=2e-6, rtol=2e-6):
        raise RuntimeError(f"{variant} did not conserve unit witness allocation")
    contract = model.architecture_metadata().get("relation_contract")
    if not isinstance(contract, dict):
        raise RuntimeError(f"{variant} has no relation contract")
    expected = {
        "kind": "identified_conserved_sparse_witness_ordinal_log_odds_v1",
        "variant": variant,
        "max_witnesses": 8,
        "allocation": "capped_entmax15_unit_simplex",
        "allocation_temperature": 1.0,
        "allocation_ledger_dtype": "float64",
        "single_score_contract": (
            "same_identified_score_drives_shortlist_allocation_and_edgewise_value"
        ),
        "global_strength_orientation": (
            "one_zero_start_scalar_per_ordinal_boundary"
        ),
        "incremental_compilation": "conserved_sparse_stored_witness_log_odds",
        "zero_initialization": "zero_strength_exact_v3_function_identity",
        "relation_replay": (
            "stored_contribution_deletion_without_reselection_or_reallocation"
        ),
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(
                f"{variant} contract mismatch for {key}: {contract.get(key)!r} != {value!r}"
            )
    expected_null = (
        "fixed_endpoint_to_geometry_correspondence_permutation"
        if variant == "shuffled_conserved_witness_control_v1"
        else None
    )
    if contract.get("spatial_null_semantics") != expected_null:
        raise RuntimeError(f"{variant} declares the wrong spatial-null semantics")
    summaries[variant] = {
        "identity_error": identity_error,
        "candidate_counts": candidate_counts.detach().cpu().tolist(),
        "shortlist_counts": shortlist_counts.detach().cpu().tolist(),
        "allocation_sums": allocation_sums.detach().cpu().tolist(),
        "spatial_null_semantics": contract.get("spatial_null_semantics"),
    }
    model.cpu()
    del model, output
    torch.cuda.empty_cache()

parameter_count = sum(item[2] for item in next(iter(parameter_signatures.values())))
print("gpu", torch.cuda.get_device_name(0))
print("source_sha256", observed_sha)
print("relation_parameter_count", parameter_count)
print("relation_parameter_signature", parameter_signatures[VARIANTS[0]])
print("identical_relation_initialization_sha256", next(iter(initial_value_hashes.values())))
print("arm_summaries", summaries)
PY

echo "ORIGIN-v8 matched-control preflight passed."

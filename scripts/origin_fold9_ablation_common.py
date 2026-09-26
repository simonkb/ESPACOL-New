#!/usr/bin/env python3
"""Shared, fail-closed protocol helpers for the EyePACS fold-9 ablation.

The ablation has two deliberately separate phases.  Every candidate is first
trained and selected using the *same* fold-9 inner validation split.  Only
after every requested candidate has produced an audited completion marker may
one release job evaluate the locked outer partition.  This module contains no
model code; it defines the immutable experimental identities used by the
launcher, workers, release job, and aggregator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_SCHEMA = "origin-fold9-ablation-protocol-v1"
PREFLIGHT_MARKER_SCHEMA = "origin-fold9-ablation-preflight-complete-v1"
TRAIN_MARKER_SCHEMA = "origin-fold9-ablation-train-complete-v1"
SPLIT_SCHEMA = "origin-fold9-ablation-split-v1"
DATASET = "dr"
N_FOLDS = 10
N_IMAGES = 35_126
FOLD = 9
TRAINING_STREAM_SEED = 100_051
EXPECTED_SPLIT_SIGNATURE = (
    "af92dbf28308ebd8694be923544e5ddd42046afc79ac26a730cce9a6dcdf4477"
)
EXPECTED_SPLIT_COUNTS = {
    "train": 28_456,
    "validation": 3_162,
    "locked_test": 3_508,
}
EXPECTED_SPLIT_HISTOGRAMS = {
    "train": [20_866, 2_000, 4_299, 709, 582],
    "validation": [2_340, 210, 476, 75, 61],
    "locked_test": [2_604, 233, 517, 89, 65],
}

# The ablation branch starts from the code that produced and audited the final
# ten-fold result.  LAUNCH_COMMIT is recorded separately because the new
# matched controls necessarily add scientific implementation files.
V3_REFERENCE_COMMIT = "b0f4609b8847d862554c84b29dec59bc562d256f"
V3_IMPLEMENTATION_SIGNATURE = (
    "0d9734495c4aeccb2a0038f2b9672f88c445a13b276d7d1073cae1cc0c86909b"
)
V3_ARCHITECTURE_SIGNATURE = (
    "02d2e5455d89b9dbacd1522a22fa7da4a25e91ebcaf0ce0c60de6aea214b44e6"
)


# Keep this order stable: Slurm array indices are protocol identities, not a
# convenience ordering that may be changed after results are observed.
FULL_VARIANTS: tuple[str, ...] = (
    "origin_full",
    "pooled_softmax",
    "pooled_cumulative_logit",
    "ledger_sequential_hazard",
    "origin_simplex_direct",
    "origin_nll_only",
    "origin_fine_only",
    "origin_coarse_only",
    "origin_independent_sigmoid",
    "pooled_conditional",
)
CORE_VARIANTS: tuple[str, ...] = FULL_VARIANTS[:5]

_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "origin_full": {
        "family": "origin_generator",
        "description": "Complete bounded cumulative multi-scale ORIGIN-v3.",
        "config_overrides": {},
    },
    "pooled_softmax": {
        "family": "matched_pooled_control",
        "description": "Matched ConvNeXt multi-scale pooled categorical softmax head.",
        "config_overrides": {},
    },
    "pooled_cumulative_logit": {
        "family": "matched_pooled_control",
        "description": "Matched pooled cumulative-threshold ordinal-logit head.",
        "config_overrides": {},
    },
    "ledger_sequential_hazard": {
        "family": "matched_ledger_control",
        "description": "Additive local ledger decoded by sequential hazards, without the CTMC generator.",
        "config_overrides": {},
    },
    "origin_simplex_direct": {
        "family": "origin_mechanism_ablation",
        "description": "Direct boundary-simplex atoms without cumulative prerequisite sharing.",
        "config_overrides": {"atom_mode": "simplex_direct"},
    },
    "origin_nll_only": {
        "family": "origin_objective_ablation",
        "description": "Full ORIGIN with the ordinal RPS term removed.",
        "config_overrides": {"rps_weight": 0.0},
    },
    "origin_fine_only": {
        "family": "origin_scale_ablation",
        "description": "ORIGIN ledger restricted to fine strides 4 and 8.",
        "config_overrides": {"evidence_scales": ["s4", "s8"]},
    },
    "origin_coarse_only": {
        "family": "origin_scale_ablation",
        "description": "ORIGIN ledger restricted to coarse strides 16 and 32.",
        "config_overrides": {"evidence_scales": ["s16", "s32"]},
    },
    "origin_independent_sigmoid": {
        "family": "origin_mechanism_ablation",
        "description": "Independent sigmoid boundary atoms in place of cumulative atoms.",
        "config_overrides": {"atom_mode": "independent"},
    },
    "pooled_conditional": {
        "family": "matched_pooled_control",
        "description": "Matched pooled conditional continuation-ratio ordinal head.",
        "config_overrides": {},
    },
}


FROZEN_V3_CONFIG: dict[str, Any] = {
    "dataset": DATASET,
    "n_classes": 5,
    "n_folds": N_FOLDS,
    "val_fraction": 0.1,
    "preprocessing_version": "canonical-square-fixed-ellipse-v1",
    "seed": 42,
    "img_size": 640,
    "encoder": "convnext_tiny",
    "pretrained": True,
    "evidence_scales": ["s4", "s8", "s16", "s32"],
    "projection_dim": 128,
    "grad_checkpoint": False,
    "mask_valid_fraction": 0.5,
    "reference_count": 4096.0,
    "atom_rate_init": 1e-6,
    "prior_rate_init": 1e-4,
    "boundary_scale_init": 1.0,
    "total_rate_cap": 64.0,
    "prior_rate_cap": 1.0,
    "boundary_scale_cap": 2.0,
    "rate_roundoff_margin": 1.0,
    "atom_mode": "cumulative",
    "hybrid_cumulative_init": 0.9,
    "evidence_dropout": 0.0,
    "force_decoder_fp64": True,
    "decision_rule": "class_map",
    "rps_weight": 0.25,
    "evidence_budget_weight": 0.0,
    "evidence_budget_delay_epochs": 5,
    "class_weighting": "none",
    "effective_num_beta": 0.999,
    "class_weight_cap": 10.0,
    "allow_weighted_likelihood": False,
    "epochs": 75,
    "batch_size": 8,
    "num_workers": 8,
    "pin_memory": True,
    "lr": 1e-4,
    "head_lr": 5e-4,
    "weight_decay": 1e-5,
    "grad_clip_norm": 5.0,
    "encoder_freeze_epochs": 2,
    "scheduler": "plateau",
    "lr_factor": 0.2,
    "lr_patience": 5,
    "lr_min": 1e-6,
    "early_stopping_patience": 15,
    "checkpoint_selection": "acc_then_qwk",
    "selection_qwk_weight": 0.1,
    "amp": True,
    "amp_init_scale": 4096.0,
    "amp_unfreeze_scale": 256.0,
    "amp_growth_interval": 2000,
    "amp_max_consecutive_skips": 8,
    "resume": False,
    "stratified_batches": False,
    "labels_csv": None,
    "image_column": None,
    "label_column": None,
    "image_dir": None,
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_checksummed_payload(
    payload: Mapping[str, Any], *, schema: str | None = None
) -> None:
    if schema is not None and payload.get("schema") != schema:
        raise ValueError(
            f"unexpected schema {payload.get('schema')!r}; expected {schema!r}"
        )
    recorded = payload.get("content_checksum_sha256")
    if not isinstance(recorded, str):
        raise ValueError("checksummed payload has no content_checksum_sha256")
    unsigned = dict(payload)
    del unsigned["content_checksum_sha256"]
    observed = canonical_sha256(unsigned)
    if observed != recorded:
        raise ValueError(
            f"payload checksum mismatch: recorded={recorded}, observed={observed}"
        )


def values_close(left: Any, right: Any, *, atol: float = 2e-5) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return (
            math.isfinite(float(left))
            and math.isfinite(float(right))
            and math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=atol)
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            values_close(a, b, atol=atol) for a, b in zip(left, right)
        )
    return left == right


def get_variant_spec(name: str) -> dict[str, Any]:
    """Return an isolated copy so callers cannot mutate the protocol table."""

    try:
        spec = _VARIANT_SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown fold-9 ablation variant {name!r}; expected one of {FULL_VARIANTS}"
        ) from exc
    return {"name": name, **deepcopy(spec)}


def variants_for_group(group: str) -> tuple[str, ...]:
    normalized = group.strip().lower()
    if normalized == "full":
        return FULL_VARIANTS
    if normalized == "core":
        return CORE_VARIANTS
    raise ValueError("ablation group must be 'full' or 'core'")


def frozen_config_for_variant(name: str) -> dict[str, Any]:
    config = deepcopy(FROZEN_V3_CONFIG)
    config.update(get_variant_spec(name)["config_overrides"])
    config["ablation_variant"] = name
    return config


def config_signature_for_variant(name: str) -> str:
    return canonical_sha256(frozen_config_for_variant(name))


def validate_variant_config(config: Mapping[str, Any], name: str) -> None:
    expected = frozen_config_for_variant(name)
    # ``resume`` is an invocation policy, not part of ORIGIN's numerical
    # identity (training/origin_trainer.py excludes it from the checkpoint
    # config signature).  A checkpoint written after a legitimate resume
    # records True, whereas its preemption predecessor records False.  Accept
    # either boolean while keeping every scientific field frozen.
    if not isinstance(config.get("resume"), bool):
        raise ValueError("checkpoint resume policy must be boolean")
    mismatches = {
        key: {"expected": expected_value, "observed": config.get(key)}
        for key, expected_value in expected.items()
        if key != "resume"
        if not values_close(config.get(key), expected_value)
    }
    if mismatches:
        raise ValueError(
            f"checkpoint violates the locked {name} configuration: {mismatches}"
        )


def validate_split_manifest(split: Mapping[str, Any]) -> None:
    expected = {
        "schema": SPLIT_SCHEMA,
        "dataset": DATASET,
        "fold": FOLD,
        "signature": EXPECTED_SPLIT_SIGNATURE,
        "evaluation_scope": "inner_validation_only_outer_locked",
        "counts": EXPECTED_SPLIT_COUNTS,
        "histograms": EXPECTED_SPLIT_HISTOGRAMS,
    }
    mismatches = {
        key: {"expected": value, "observed": split.get(key)}
        for key, value in expected.items()
        if split.get(key) != value
    }
    if mismatches:
        raise ValueError(f"fold-9 split manifest mismatch: {mismatches}")


def make_protocol_payload(
    *,
    selected_variants: Sequence[str],
    group: str,
    launch_commit: str,
    experiment_root: str | Path,
    data_root: str | Path,
    immutable_worktree: str | Path,
    tag: str,
) -> dict[str, Any]:
    variants = tuple(selected_variants)
    if variants != variants_for_group(group):
        raise ValueError(
            f"variant order does not match locked group {group!r}: {variants}"
        )
    payload: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_status": (
            "retrospective_observed_fold9_mechanism_stress_test_"
            "frozen_before_ablation_training"
        ),
        "tag": str(tag),
        "dataset": DATASET,
        "n_folds": N_FOLDS,
        "n_images": N_IMAGES,
        "fold": FOLD,
        "ablation_group": group,
        "selected_variants": list(variants),
        "variant_specs": {name: get_variant_spec(name) for name in variants},
        "variant_config_signatures": {
            name: config_signature_for_variant(name) for name in variants
        },
        "frozen_v3_config": deepcopy(FROZEN_V3_CONFIG),
        "expected_split": {
            "schema": SPLIT_SCHEMA,
            "signature": EXPECTED_SPLIT_SIGNATURE,
            "counts": deepcopy(EXPECTED_SPLIT_COUNTS),
            "histograms": deepcopy(EXPECTED_SPLIT_HISTOGRAMS),
        },
        "evaluation_policy": {
            "training_scope": "inner_validation_only_outer_locked",
            "checkpoint_selection": "acc_then_qwk",
            "outer_test_release": (
                "single_nonadaptive_release_after_all_training_markers"
            ),
            "outer_test_reuse": "same_locked_fold9_partition_for_every_variant",
            "fold9_prior_observation": (
                "the full-model fold9 result was observed before this ablation; "
                "this suite is not an untouched confirmatory benchmark"
            ),
            "outer_test_forbidden_during_training": True,
            "variant_independent_training_stream_seed": TRAINING_STREAM_SEED,
            "paired_patient_cluster_bootstrap_samples": 10_000,
            "paired_patient_cluster_bootstrap_seed": 26_092_026,
        },
        "v3_reference_commit": V3_REFERENCE_COMMIT,
        "v3_implementation_signature": V3_IMPLEMENTATION_SIGNATURE,
        "v3_architecture_signature": V3_ARCHITECTURE_SIGNATURE,
        "launch_commit": launch_commit,
        "experiment_root": str(Path(experiment_root).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "immutable_worktree": str(Path(immutable_worktree).resolve()),
    }
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    return payload


def verify_protocol(
    protocol: Mapping[str, Any],
    *,
    expected_launch_commit: str | None = None,
    expected_root: str | Path | None = None,
    expected_data_root: str | Path | None = None,
) -> tuple[str, ...]:
    verify_checksummed_payload(protocol, schema=PROTOCOL_SCHEMA)
    for key, expected in {
        "dataset": DATASET,
        "n_folds": N_FOLDS,
        "n_images": N_IMAGES,
        "fold": FOLD,
        "development_status": (
            "retrospective_observed_fold9_mechanism_stress_test_"
            "frozen_before_ablation_training"
        ),
        "v3_reference_commit": V3_REFERENCE_COMMIT,
        "v3_implementation_signature": V3_IMPLEMENTATION_SIGNATURE,
        "v3_architecture_signature": V3_ARCHITECTURE_SIGNATURE,
    }.items():
        if protocol.get(key) != expected:
            raise ValueError(
                f"ablation protocol {key} mismatch: {protocol.get(key)!r} != {expected!r}"
            )
    group = str(protocol.get("ablation_group", ""))
    selected = tuple(protocol.get("selected_variants", ()))
    if selected != variants_for_group(group):
        raise ValueError("protocol variant set/order is not the locked group")
    expected_specs = {name: get_variant_spec(name) for name in selected}
    if protocol.get("variant_specs") != expected_specs:
        raise ValueError("protocol variant specifications changed")
    expected_signatures = {
        name: config_signature_for_variant(name) for name in selected
    }
    if protocol.get("variant_config_signatures") != expected_signatures:
        raise ValueError("protocol variant configuration signatures changed")
    split = protocol.get("expected_split")
    if not isinstance(split, Mapping):
        raise ValueError("protocol expected_split is missing")
    for key, expected in {
        "schema": SPLIT_SCHEMA,
        "signature": EXPECTED_SPLIT_SIGNATURE,
        "counts": EXPECTED_SPLIT_COUNTS,
        "histograms": EXPECTED_SPLIT_HISTOGRAMS,
    }.items():
        if split.get(key) != expected:
            raise ValueError(f"protocol expected_split {key} changed")
    policy = protocol.get("evaluation_policy")
    if not isinstance(policy, Mapping) or policy.get("outer_test_forbidden_during_training") is not True:
        raise ValueError("protocol does not seal the outer test during training")
    if policy.get("variant_independent_training_stream_seed") != TRAINING_STREAM_SEED:
        raise ValueError("protocol training-stream seed changed")
    if policy.get("paired_patient_cluster_bootstrap_samples") != 10_000:
        raise ValueError("protocol bootstrap sample count changed")
    if policy.get("paired_patient_cluster_bootstrap_seed") != 26_092_026:
        raise ValueError("protocol bootstrap seed changed")
    if expected_launch_commit is not None and protocol.get("launch_commit") != expected_launch_commit:
        raise ValueError("protocol launch commit mismatch")
    if expected_root is not None and Path(str(protocol.get("experiment_root"))).resolve() != Path(expected_root).resolve():
        raise ValueError("protocol experiment root mismatch")
    if expected_data_root is not None and Path(str(protocol.get("data_root"))).resolve() != Path(expected_data_root).resolve():
        raise ValueError("protocol data root mismatch")
    return selected

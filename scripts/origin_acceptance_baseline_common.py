"""Immutable registry and task map for the acceptance baseline suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


PROTOCOL_ID = "origin-acceptance-baselines-v1"
BASELINE_ORDER = (
    "ledger_sequential_hazard",
    "pooled_conditional",
    "ordinal_additive_mil",
    "sparse_bagnet",
    "origin_ctmc",
)
DATASET_FOLDS = {"aptos": tuple(range(5)), "dr": tuple(range(10))}
REPLICATION_SEEDS = (42, 31415, 27182)
SPLIT_SEED = 42
SPARSE_L1_WEIGHT = 1e-4
SPARSE_L1_DELAY_EPOCHS = 0
PAIRED_BOOTSTRAP_SAMPLES = 10_000
PAIRED_BOOTSTRAP_SEED = 20_261_001
RELEASE_IDENTIFIER_SCHEMA = "origin-dataset-scoped-sha256-v1"

BASELINE_SPECS: Mapping[str, Mapping[str, Any]] = {
    "ledger_sequential_hazard": {
        "description": "same conserved ORIGIN ledger decoded by adjacent conditional hazards",
        "implementation_origin": (
            "matched_in_repo_analogue_not_official_author_implementation"
        ),
        "sparse_l1_weight": 0.0,
    },
    "pooled_conditional": {
        "description": "all-scale masked global pooling with sequential continuation head",
        "implementation_origin": (
            "matched_in_repo_analogue_not_official_author_implementation"
        ),
        "sparse_l1_weight": 0.0,
    },
    "ordinal_additive_mil": {
        "description": "signed boundary-local Additive-MIL with fixed original-valid-count mean",
        "implementation_origin": (
            "matched_in_repo_analogue_not_official_author_implementation"
        ),
        "sparse_l1_weight": 0.0,
    },
    "sparse_bagnet": {
        "description": "nonnegative multiclass local evidence with masked mean and activation L1",
        "implementation_origin": (
            "matched_in_repo_analogue_not_official_author_implementation"
        ),
        "sparse_l1_weight": SPARSE_L1_WEIGHT,
    },
    "origin_ctmc": {
        "description": "standard ORIGIN-v3 conserved local ledger and FP64 pure-birth CTMC decoder",
        "implementation_origin": "in_repo_proposed_method",
        "sparse_l1_weight": 0.0,
    },
}


@dataclass(frozen=True, order=True)
class AcceptanceTrainingTask:
    dataset: str
    fold: int
    baseline_variant: str
    training_seed: int

    def __post_init__(self) -> None:
        if self.dataset not in DATASET_FOLDS:
            raise ValueError(f"unknown protocol dataset {self.dataset!r}")
        if self.fold not in DATASET_FOLDS[self.dataset]:
            raise ValueError(f"invalid {self.dataset} fold {self.fold}")
        if self.baseline_variant not in BASELINE_ORDER:
            raise ValueError(f"unknown baseline {self.baseline_variant!r}")
        if self.training_seed not in REPLICATION_SEEDS:
            raise ValueError(f"unregistered training seed {self.training_seed}")

    @property
    def key(self) -> str:
        return (
            f"{self.dataset}__fold{self.fold}__{self.baseline_variant}"
            f"__seed{self.training_seed}"
        )


def canary_tasks() -> tuple[AcceptanceTrainingTask, ...]:
    return tuple(
        AcceptanceTrainingTask("aptos", 0, variant, REPLICATION_SEEDS[0])
        for variant in BASELINE_ORDER
    )


def full_tasks() -> tuple[AcceptanceTrainingTask, ...]:
    return tuple(
        AcceptanceTrainingTask(dataset, fold, variant, seed)
        for dataset in DATASET_FOLDS
        for fold in DATASET_FOLDS[dataset]
        for seed in REPLICATION_SEEDS
        for variant in BASELINE_ORDER
    )


def tasks_for_scope(scope: str) -> tuple[AcceptanceTrainingTask, ...]:
    scope = str(scope).lower()
    if scope == "canary":
        return canary_tasks()
    if scope == "full":
        return full_tasks()
    raise ValueError("scope must be 'canary' or 'full'")


def task_at(scope: str, index: int) -> AcceptanceTrainingTask:
    tasks = tasks_for_scope(scope)
    if index < 0 or index >= len(tasks):
        raise IndexError(f"task index {index} outside 0..{len(tasks) - 1}")
    return tasks[index]


def worker_run_dir(root: str | Path, scope: str, task: AcceptanceTrainingTask) -> Path:
    return Path(root) / str(scope).lower() / "training" / task.key


def protocol_payload() -> dict[str, Any]:
    return {
        "schema": "origin-acceptance-protocol-v1",
        "protocol_id": PROTOCOL_ID,
        "baseline_order": list(BASELINE_ORDER),
        "baseline_specs": {key: dict(value) for key, value in BASELINE_SPECS.items()},
        "dataset_folds": {key: list(value) for key, value in DATASET_FOLDS.items()},
        "replication_seeds": list(REPLICATION_SEEDS),
        "split_seed": SPLIT_SEED,
        "paired_cluster_bootstrap_samples": PAIRED_BOOTSTRAP_SAMPLES,
        "paired_cluster_bootstrap_seed": PAIRED_BOOTSTRAP_SEED,
        "paired_cluster_bootstrap_unit": "EyePACS_patient_stem_APTOS_image",
        "canary_tasks": [asdict(task) for task in canary_tasks()],
        "full_task_count": len(full_tasks()),
        "selection_scope": "inner_validation_only",
        "outer_release": "one_post_freeze_suite_level_pass",
        "outer_release_result_schema": "origin-acceptance-outer-result-v2",
        "posterior_quality_contract": {
            "reliability_bins": 15,
            "reliability_binning": "equal_width_[0,1]",
            "multiclass_brier": "mean_sum_k_(p_k-onehot_k)^2",
            "threshold_binary_brier": (
                "per_boundary_mean_(P(Y>k)-1[Y>k])^2_then_unweighted_boundary_mean"
            ),
            "threshold_ece": (
                "per_boundary_binary_ECE_then_unweighted_boundary_mean"
            ),
            "classwise_ece": (
                "one_vs_rest_per_class_ECE_then_unweighted_class_mean"
            ),
            "aggregate_reliability_scope": (
                "pooled_out_of_fold_predictions_per_training_seed"
            ),
            "bootstrap": (
                "paired_cluster_bootstrap_for_each_boundary_brier_ECE_and_class_ECE"
            ),
        },
        "outer_release_identifier_policy": {
            "schema": RELEASE_IDENTIFIER_SCHEMA,
            "image_identifier": "sha256(dataset,namespace,dataset_relative_path)",
            "cluster_identifier": "sha256(dataset,namespace,raw_cluster_key)",
            "raw_paths_or_patient_identifiers_exported": False,
        },
        "comparator_implementation_scope": {
            "kind": "matched_in_repo_analogues",
            "official_author_implementations": False,
            "claim_boundary": (
                "architecture-matched controlled comparisons only; results must not "
                "be attributed to official author implementations"
            ),
        },
        "historical_origin_oof_role": (
            "external_sanity_reference_only_not_a_paired_arm_or_selection_input"
        ),
    }


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


PROTOCOL_SHA256 = canonical_sha256(protocol_payload())


def dataset_scoped_identifier(dataset: str, namespace: str, raw_value: str) -> str:
    """One-way identifier for public release artifacts.

    Namespaces prevent the same raw token from linking image and cluster IDs;
    dataset scoping prevents cross-dataset linkage.  Raw values remain only in
    release-worker memory and are never serialized.
    """

    if dataset not in DATASET_FOLDS:
        raise ValueError(f"unknown protocol dataset {dataset!r}")
    namespace = str(namespace).strip()
    raw_value = str(raw_value)
    if not namespace or not raw_value:
        raise ValueError("identifier namespace and raw value must be non-empty")
    material = (
        f"{RELEASE_IDENTIFIER_SCHEMA}\0{dataset}\0{namespace}\0{raw_value}"
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def release_identifier_policy(dataset: str) -> dict[str, Any]:
    if dataset not in DATASET_FOLDS:
        raise ValueError(f"unknown protocol dataset {dataset!r}")
    return {
        "schema": RELEASE_IDENTIFIER_SCHEMA,
        "dataset_scope": dataset,
        "algorithm": "sha256",
        "image_namespace": "image",
        "cluster_namespace": "patient_cluster" if dataset == "dr" else "image_cluster",
        "image_raw_key": "dataset_relative_posix_path",
        "cluster_raw_key": (
            "EyePACS_filename_stem_without_eye_suffix"
            if dataset == "dr"
            else "dataset_relative_posix_path"
        ),
        "raw_image_paths_exported": False,
        "raw_patient_or_cluster_identifiers_exported": False,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(
    directory: str | Path,
    *,
    required_names: Iterable[str],
    task: AcceptanceTrainingTask,
    scope: str,
) -> dict[str, Any]:
    directory = Path(directory)
    artifacts: dict[str, dict[str, Any]] = {}
    for name in required_names:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"required worker artifact is missing: {path}")
        artifacts[name] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    payload: dict[str, Any] = {
        "schema": "origin-acceptance-training-artifact-manifest-v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "scope": str(scope).lower(),
        "task": asdict(task),
        "test_evaluated": False,
        "artifacts": artifacts,
    }
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    return payload


__all__ = [
    "AcceptanceTrainingTask",
    "BASELINE_ORDER",
    "BASELINE_SPECS",
    "DATASET_FOLDS",
    "PROTOCOL_ID",
    "PROTOCOL_SHA256",
    "PAIRED_BOOTSTRAP_SAMPLES",
    "PAIRED_BOOTSTRAP_SEED",
    "REPLICATION_SEEDS",
    "RELEASE_IDENTIFIER_SCHEMA",
    "SPLIT_SEED",
    "SPARSE_L1_DELAY_EPOCHS",
    "SPARSE_L1_WEIGHT",
    "artifact_manifest",
    "canary_tasks",
    "canonical_sha256",
    "file_sha256",
    "full_tasks",
    "task_at",
    "tasks_for_scope",
    "dataset_scoped_identifier",
    "release_identifier_policy",
    "worker_run_dir",
]

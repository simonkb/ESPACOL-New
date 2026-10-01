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

BASELINE_SPECS: Mapping[str, Mapping[str, Any]] = {
    "ledger_sequential_hazard": {
        "description": "same conserved ORIGIN ledger decoded by adjacent conditional hazards",
        "sparse_l1_weight": 0.0,
    },
    "pooled_conditional": {
        "description": "all-scale masked global pooling with sequential continuation head",
        "sparse_l1_weight": 0.0,
    },
    "ordinal_additive_mil": {
        "description": "signed boundary-local Additive-MIL with fixed original-valid-count mean",
        "sparse_l1_weight": 0.0,
    },
    "sparse_bagnet": {
        "description": "nonnegative multiclass local evidence with masked mean and activation L1",
        "sparse_l1_weight": SPARSE_L1_WEIGHT,
    },
    "origin_ctmc": {
        "description": "standard ORIGIN-v3 conserved local ledger and FP64 pure-birth CTMC decoder",
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
    "worker_run_dir",
]

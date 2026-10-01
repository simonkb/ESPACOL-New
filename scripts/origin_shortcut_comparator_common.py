"""Frozen task registry for the shortcut comparator follow-up (protocol v2).

The 21 ORIGIN-v1 workers are imported as a sealed reference.  This module
enumerates exactly 84 *additional* workers: four matched in-repository
analogues times the same 21 arm/family/seed cells.  Comparator names must not
be presented as results from official author implementations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from benchmarks.shortcut_metrics import GateBThresholds


COMPARATOR_VARIANTS = (
    "ledger_sequential_hazard",
    "pooled_conditional",
    "ordinal_additive_mil",
    "sparse_bagnet",
)
FAMILIES = ("localized", "border", "diffuse")
TRAINING_SEEDS = (1701, 2603, 3907)
TASKS_PER_VARIANT = 21
TASK_COUNT = len(COMPARATOR_VARIANTS) * TASKS_PER_VARIANT
PROTOCOL_ID = "origin-ordinal-shortcut-comparator-followup-v2"


@dataclass(frozen=True)
class ShortcutComparatorTask:
    model_variant: str
    arm: str
    family: str
    training_seed: int

    @property
    def key(self) -> str:
        return (
            f"{self.model_variant}__{self.arm}__{self.family}"
            f"__seed{self.training_seed}"
        )


def _origin_cell(index: int) -> tuple[str, str, int]:
    if index < 0 or index >= TASKS_PER_VARIANT:
        raise IndexError("origin shortcut cell must lie in 0..20")
    if index < 9:
        cell = index
        return "shortcut", FAMILIES[cell // 3], TRAINING_SEEDS[cell % 3]
    if index < 18:
        cell = index - 9
        return "cue_only", FAMILIES[cell // 3], TRAINING_SEEDS[cell % 3]
    # One clean checkpoint per seed is audited against all three families.
    return "clean", "localized", TRAINING_SEEDS[index - 18]


def task_at(index: int) -> ShortcutComparatorTask:
    if index < 0 or index >= TASK_COUNT:
        raise IndexError(f"task index {index} outside 0..{TASK_COUNT - 1}")
    variant = COMPARATOR_VARIANTS[index // TASKS_PER_VARIANT]
    arm, family, seed = _origin_cell(index % TASKS_PER_VARIANT)
    return ShortcutComparatorTask(variant, arm, family, seed)


def all_tasks() -> tuple[ShortcutComparatorTask, ...]:
    return tuple(task_at(index) for index in range(TASK_COUNT))


def protocol_core() -> dict[str, Any]:
    return {
        "schema": "origin-ordinal-shortcut-comparator-protocol-v2",
        "protocol_id": PROTOCOL_ID,
        "development_status": (
            "comparator_followup_frozen_before_comparator_result_inspection"
        ),
        "dataset": "aptos",
        "fold": 0,
        "split_seed": 42,
        "shortcut_seed": 271828,
        "training_seeds": list(TRAINING_SEEDS),
        "families": list(FAMILIES),
        "arms": ["shortcut", "cue_only", "clean"],
        "model_variants": list(COMPARATOR_VARIANTS),
        "additional_training_workers": TASK_COUNT,
        "execution_topology": {
            "cpu_preflight_jobs": 1,
            "gpu_training_and_blind_audit_workers": TASK_COUNT,
            "cpu_expanded_aggregate_jobs": 1,
            "sealed_origin_v1_workers_reused": 21,
        },
        "tasks": [asdict(task) | {"key": task.key} for task in all_tasks()],
        "origin_reference": {
            "model_variant": "origin_ctmc",
            "worker_count": 21,
            "reuse_policy": "import sealed v1 artifacts without mutation or retraining",
        },
        "implementation_scope": {
            "kind": "matched_in_repository_analogues",
            "official_author_implementations": False,
            "claim_boundary": (
                "controlled architecture-matched comparisons only; no result may "
                "be attributed to official author code"
            ),
        },
        "test_domain": {"position": "unseen", "appearance": "unseen"},
        "factorial_marker_states": 16,
        "localization_permutations": 999,
        "effect_bootstrap_replicates": 2000,
        "effect_bootstrap_unit": "image_cluster_all_four_boundaries_retained",
        "gate_b_thresholds": GateBThresholds().as_dict(),
        "cue_only_positive_control_role": "necessary_but_not_sufficient",
    }


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


PROTOCOL_CORE_SHA256 = canonical_sha256(protocol_core())


__all__ = [
    "COMPARATOR_VARIANTS",
    "FAMILIES",
    "PROTOCOL_CORE_SHA256",
    "PROTOCOL_ID",
    "ShortcutComparatorTask",
    "TASK_COUNT",
    "TRAINING_SEEDS",
    "all_tasks",
    "canonical_sha256",
    "protocol_core",
    "task_at",
]

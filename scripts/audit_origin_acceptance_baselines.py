#!/usr/bin/env python3
"""Fail-closed audit and gate creation for acceptance baseline workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from scripts.origin_acceptance_baseline_common import (
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    canonical_sha256,
    file_sha256,
    tasks_for_scope,
    worker_run_dir,
)
from train_origin import write_json_atomic


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object at {path}")
    return value


def _check_content_checksum(payload: dict[str, Any], path: Path) -> None:
    expected = payload.get("content_checksum_sha256")
    unsigned = dict(payload)
    unsigned.pop("content_checksum_sha256", None)
    if expected != canonical_sha256(unsigned):
        raise ValueError(f"content checksum mismatch: {path}")


def audit_scope(root: Path, scope: str) -> dict[str, Any]:
    tasks = tasks_for_scope(scope)
    audited: list[dict[str, Any]] = []
    split_by_fold: dict[tuple[str, int], str] = {}
    for task in tasks:
        fold_dir = worker_run_dir(root, scope, task) / f"fold{task.fold}"
        manifest_path = fold_dir / "artifact_manifest.json"
        manifest = _load_json(manifest_path)
        _check_content_checksum(manifest, manifest_path)
        if manifest.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ValueError(f"protocol mismatch in {manifest_path}")
        if manifest.get("test_evaluated") is not False:
            raise ValueError(f"training worker touched test scope: {manifest_path}")
        for name, record in manifest.get("artifacts", {}).items():
            path = fold_dir / name
            if file_sha256(path) != record.get("sha256"):
                raise ValueError(f"artifact hash mismatch: {path}")
            if path.stat().st_size != int(record.get("size_bytes", -1)):
                raise ValueError(f"artifact size mismatch: {path}")

        completion = _load_json(fold_dir / "TRAINING_COMPLETE.json")
        if completion.get("artifact_manifest_checksum") != manifest.get(
            "content_checksum_sha256"
        ):
            raise ValueError(f"completion/manifest mismatch: {fold_dir}")
        result = _load_json(fold_dir / "result.json")
        if result.get("test_evaluated") is not False:
            raise ValueError(f"result contains training-time test evaluation: {fold_dir}")
        if result.get("warm_start_eligible_for_selection") is not False:
            raise ValueError(f"warm start is selection-eligible: {fold_dir}")
        if int(result.get("best_epoch", 0)) < 1:
            raise ValueError(f"best learned checkpoint is not post-update: {fold_dir}")
        warm = _load_json(fold_dir / "warm_start_floor.json")
        _check_content_checksum(warm, fold_dir / "warm_start_floor.json")
        if warm.get("epoch") != 0 or warm.get("eligible_for_model_selection") is not False:
            raise ValueError(f"invalid warm-start floor semantics: {fold_dir}")

        checkpoint_path = fold_dir / "best_learned.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("acceptance_schema") != "origin-acceptance-learned-checkpoint-v1":
            raise ValueError(f"wrong learned checkpoint schema: {checkpoint_path}")
        if int(checkpoint.get("epoch", 0)) < 1:
            raise ValueError(f"epoch-zero checkpoint selected: {checkpoint_path}")
        if checkpoint.get("split_signature") != result.get("split_signature"):
            raise ValueError(f"checkpoint/result split mismatch: {fold_dir}")

        split = _load_json(fold_dir / "split_manifest.json")
        signature = str(split.get("signature"))
        fold_key = (task.dataset, task.fold)
        previous = split_by_fold.setdefault(fold_key, signature)
        if previous != signature:
            raise ValueError(
                f"outer/inner split differs between baselines or seeds for {fold_key}"
            )
        audited.append(
            {
                "task": manifest["task"],
                "fold_dir": str(fold_dir),
                "manifest_sha256": file_sha256(manifest_path),
                "best_learned_sha256": file_sha256(checkpoint_path),
                "split_signature": signature,
                "best_epoch": int(checkpoint["epoch"]),
            }
        )

    payload: dict[str, Any] = {
        "schema": "origin-acceptance-training-audit-v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "scope": scope,
        "status": "passed",
        "worker_count": len(audited),
        "workers": audited,
        "outer_test_released": False,
    }
    payload["content_checksum_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_root", required=True)
    parser.add_argument("--scope", choices=("canary", "full"), required=True)
    parser.add_argument("--write_gate", action="store_true")
    args = parser.parse_args()
    root = Path(args.experiment_root)
    payload = audit_scope(root, args.scope)
    output = root / args.scope / "training_audit.json"
    write_json_atomic(output, payload)
    if args.write_gate:
        gate_name = "CANARY_PASSED.json" if args.scope == "canary" else "TRAINING_FROZEN.json"
        write_json_atomic(root / args.scope / gate_name, payload)
    print(json.dumps({"status": "passed", "workers": payload["worker_count"], "path": str(output)}))


if __name__ == "__main__":
    main()

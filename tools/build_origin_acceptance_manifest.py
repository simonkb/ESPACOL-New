#!/usr/bin/env python3
"""Build an independently checkable ORIGIN V3 checkpoint/split manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Datasets.origin_data import load_origin_items, split_origin_items


SCHEMA = "origin-acceptance-artifact-manifest-v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _validated_cv_submission(cv_root: Path) -> dict[str, Any]:
    path = cv_root / "SUBMISSION.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    submission = json.loads(path.read_text(encoding="utf-8"))
    if submission.get("schema") != "origin-v3-full-cv-submission-v1":
        raise ValueError("unexpected full-CV submission schema")
    expected = submission.get("content_checksum_sha256")
    checksum_payload = dict(submission)
    checksum_payload.pop("content_checksum_sha256", None)
    observed = _canonical_sha256(checksum_payload)
    if expected != observed:
        raise ValueError("full-CV submission content checksum mismatch")
    commit = str(submission.get("launch_commit", ""))
    if len(commit) != 40:
        raise ValueError("full-CV submission launch commit is missing")
    return {
        "relative_path": str(path.relative_to(cv_root)),
        "sha256": _file_sha256(path),
        "content_checksum_sha256": expected,
        "training_launch_commit": commit,
        "tag": submission.get("tag"),
    }


def _audit_source_provenance(repo_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise ValueError("artifact-audit source commit is invalid")
    return {
        "audit_source_commit": commit,
        "tracked_tree_dirty": bool(status),
    }


def _anonymous_id(dataset: str, value: str) -> str:
    return _sha256_bytes(f"{dataset}\0{value}".encode("utf-8"))


def _patient_key(dataset: str, path: str) -> str:
    stem = Path(path).stem
    if dataset == "dr":
        return stem.rsplit("_", 1)[0]
    return stem


def _membership_rows(
    dataset: str,
    named_splits: Iterable[tuple[str, list[tuple[str, int]]]],
) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for split, items in named_splits:
        for path, label in items:
            rows.append(
                {
                    "image_id_sha256": _anonymous_id(dataset, Path(path).name),
                    "patient_cluster_sha256": _anonymous_id(
                        dataset, _patient_key(dataset, path)
                    ),
                    "label": int(label),
                    "split": split,
                }
            )
    return sorted(rows, key=lambda row: (str(row["split"]), str(row["image_id_sha256"])))


def _write_deterministic_csv_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                fields = ["image_id_sha256", "patient_cluster_sha256", "label", "split"]
                writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    temporary.replace(path)


def _checkpoint_record(path: Path, *, dataset: str, fold: int) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema") != "origin-checkpoint-v3":
        raise ValueError(f"unexpected checkpoint schema: {path}")
    if int(state.get("fold", -1)) != fold:
        raise ValueError(f"checkpoint fold mismatch: {path}")
    config = state.get("config", {})
    if config.get("dataset") != dataset:
        raise ValueError(f"checkpoint dataset mismatch: {path}")
    return {
        "dataset": dataset,
        "fold": fold,
        "relative_path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "checkpoint_schema": state["schema"],
        "split_signature": state.get("split_signature"),
        "implementation_signature": state.get("implementation_signature"),
        "architecture_signature": state.get("architecture_signature"),
        "config_signature": state.get("config_signature"),
        "best_epoch": state.get("epoch"),
    }


def build_manifest(
    *,
    cv_root: Path,
    dr_root: Path,
    aptos_root: Path,
    output_dir: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    source_provenance = _audit_source_provenance(repo_root)
    if source_provenance["tracked_tree_dirty"]:
        raise RuntimeError("artifact audit requires a clean tracked source tree")
    cv_submission = _validated_cv_submission(cv_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "dr": {"root": dr_root, "folds": 10, "val_fraction": 0.1, "seed": 42},
        "aptos": {"root": aptos_root, "folds": 5, "val_fraction": 0.1, "seed": 42},
    }
    checkpoint_records: list[dict[str, Any]] = []
    membership_records: list[dict[str, Any]] = []
    for dataset, spec in datasets.items():
        items = load_origin_items(dataset, str(spec["root"]))
        seen_outer: set[str] = set()
        for fold in range(int(spec["folds"])):
            train, validation, test = split_origin_items(
                dataset,
                items,
                fold,
                n_folds=int(spec["folds"]),
                val_fraction=float(spec["val_fraction"]),
                seed=int(spec["seed"]),
            )
            current_outer = {str(Path(path).resolve()) for path, _ in test}
            if seen_outer.intersection(current_outer):
                raise AssertionError(f"{dataset} outer folds overlap at fold {fold}")
            seen_outer.update(current_outer)
            rows = _membership_rows(
                dataset,
                (("train", train), ("validation", validation), ("outer_test", test)),
            )
            membership_path = output_dir / "split_memberships" / f"{dataset}_fold{fold}.csv.gz"
            _write_deterministic_csv_gz(membership_path, rows)
            membership_records.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "relative_path": str(membership_path.relative_to(output_dir)),
                    "rows": len(rows),
                    "counts": {
                        "train": len(train),
                        "validation": len(validation),
                        "outer_test": len(test),
                    },
                    "sha256": _file_sha256(membership_path),
                }
            )
            checkpoint = (
                cv_root / dataset / "workers" / f"fold{fold}" / f"fold{fold}" / "best.pth"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            checkpoint_records.append(
                _checkpoint_record(checkpoint, dataset=dataset, fold=fold)
            )
        if len(seen_outer) != len(items):
            raise AssertionError(
                f"{dataset} outer-fold union has {len(seen_outer)} images, expected {len(items)}"
            )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "cv_root": str(cv_root),
        "source_provenance": source_provenance,
        "full_cv_submission": cv_submission,
        "raw_patient_data_redistributed": False,
        "identifier_contract": (
            "SHA256(dataset + NUL + image basename/patient key); labels and split roles retained"
        ),
        "checkpoints": checkpoint_records,
        "split_memberships": membership_records,
    }
    payload["content_checksum_sha256"] = _canonical_sha256(payload)
    destination = output_dir / "manifest.json"
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(destination)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--dr-root", type=Path, required=True)
    parser.add_argument("--aptos-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_manifest(
        cv_root=args.cv_root,
        dr_root=args.dr_root,
        aptos_root=args.aptos_root,
        output_dir=args.output_dir,
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output_dir / "manifest.json"),
                "checkpoints": len(payload["checkpoints"]),
                "split_memberships": len(payload["split_memberships"]),
                "checksum": payload["content_checksum_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

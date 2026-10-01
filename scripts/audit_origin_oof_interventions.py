#!/usr/bin/env python3
"""Run the frozen exact grouped-intervention census on one ORIGIN OOF fold."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from datetime import datetime, timezone
import fcntl
import gc
import gzip
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from configs.origin_config import OriginConfig
from Datasets.origin_data import (
    OriginFundusTransform,
    OriginImageDataset,
    class_histogram,
    load_origin_items,
    split_origin_items,
    split_paths_are_disjoint,
)
from models.origin import build_origin_model
from scripts.origin_oof_intervention_common import (
    FOLD_SCHEMA,
    ROW_SCHEMA,
    audit_image_ledger,
    load_audit_protocol,
)
from scripts.origin_v3_cv_common import (
    ARCHITECTURE_SIGNATURE,
    BASE_COMMIT,
    DATASET_SPECS,
    IMPLEMENTATION_SIGNATURE,
    canonical_sha256,
    file_sha256,
    read_json,
    validate_pinned_config,
    verify_checksummed_payload,
    write_json_atomic,
)
from train_origin import split_signature
from training.origin_trainer import (
    _architecture_record,
    _canonical_sha256,
    _critical_config,
    origin_implementation_signature,
)


class AtomicJsonlGzipWriter:
    """Deterministic gzip JSONL writer published only after clean close."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        self.raw: io.BufferedWriter | None = None
        self.gzip_stream: gzip.GzipFile | None = None
        self.text_stream: io.TextIOWrapper | None = None
        self.rows = 0

    def __enter__(self) -> "AtomicJsonlGzipWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.raw = self.temporary.open("wb")
        self.gzip_stream = gzip.GzipFile(
            filename="", fileobj=self.raw, mode="wb", compresslevel=6, mtime=0
        )
        self.text_stream = io.TextIOWrapper(self.gzip_stream, encoding="utf-8", newline="\n")
        return self

    def write(self, row: Mapping[str, Any]) -> None:
        if self.text_stream is None:
            raise RuntimeError("writer is not open")
        json.dump(
            dict(row), self.text_stream, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        )
        self.text_stream.write("\n")
        self.rows += 1

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self.text_stream is not None:
                self.text_stream.flush()
                self.text_stream.close()
            elif self.gzip_stream is not None:
                self.gzip_stream.close()
            if self.raw is not None and not self.raw.closed:
                self.raw.close()
            if exc_type is None:
                os.replace(self.temporary, self.path)
        finally:
            if self.temporary.exists():
                self.temporary.unlink()


def iter_jsonl_gzip(path: str | Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from error


def _relative_image_path(path: str, data_root: Path) -> str:
    resolved = Path(path).expanduser().resolve(strict=False)
    try:
        return resolved.relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _field(output: object, name: str) -> Any:
    return output.get(name) if isinstance(output, Mapping) else getattr(output, name)


def _load_prediction_rows(path: Path, expected_n: int, fold: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if len(rows) != expected_n:
        raise AssertionError("outer prediction export row count changed")
    for index, row in enumerate(rows):
        if int(row["fold"]) != fold or int(row["outer_index"]) != index:
            raise AssertionError("outer prediction export order changed")
    return rows


def _verify_and_reconstruct(
    *,
    dataset: str,
    fold: int,
    cv_root: Path,
    data_root: Path,
    cv_protocol_path: Path,
) -> tuple[OriginConfig, torch.nn.Module, list[tuple[str, int]], list[dict[str, str]], dict[str, Any]]:
    spec = DATASET_SPECS[dataset]
    if fold not in range(int(spec["n_folds"])):
        raise ValueError(f"invalid {dataset} fold {fold}")
    cv_protocol = read_json(cv_protocol_path)
    verify_checksummed_payload(cv_protocol, schema="origin-v3-full-cv-protocol-v1")
    for key, expected in {
        "dataset": dataset,
        "n_folds": spec["n_folds"],
        "n_images": spec["n_images"],
        "base_commit": BASE_COMMIT,
        "implementation_signature": IMPLEMENTATION_SIGNATURE,
        "architecture_signature": ARCHITECTURE_SIGNATURE,
        "config_signature": spec["config_signature"],
        "evaluation_scope": "outer_test_after_selection",
    }.items():
        if cv_protocol.get(key) != expected:
            raise AssertionError(f"CV protocol {key} mismatch")
    validate_pinned_config(cv_protocol.get("frozen_config", {}), dataset)
    if Path(cv_protocol["cv_root"]).resolve() != cv_root.resolve():
        raise AssertionError("CV root differs from the frozen protocol")
    if Path(cv_protocol["data_root"]).resolve() != data_root.resolve():
        raise AssertionError("data root differs from the frozen protocol")
    fold_dir = cv_root / "workers" / f"fold{fold}" / f"fold{fold}"
    paths = {
        "checkpoint": fold_dir / "best.pth",
        "split_manifest": fold_dir / "split_manifest.json",
        "outer_predictions_csv": fold_dir / "outer_predictions.csv",
        "outer_predictions_manifest": fold_dir / "outer_predictions_manifest.json",
        "fold_marker": fold_dir / "CV_FOLD_COMPLETE.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"OOF intervention audit requires complete fold artifacts: {missing}")
    marker = read_json(paths["fold_marker"])
    verify_checksummed_payload(marker, schema="origin-v3-cv-fold-complete-v1")
    prediction_manifest = read_json(paths["outer_predictions_manifest"])
    verify_checksummed_payload(prediction_manifest, schema="origin-v3-outer-predictions-v1")
    split_manifest = read_json(paths["split_manifest"])
    state = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    if state.get("schema") != "origin-checkpoint-v3":
        raise ValueError("OOF census requires an ORIGIN-v3 checkpoint")
    if origin_implementation_signature() != IMPLEMENTATION_SIGNATURE:
        raise AssertionError("active ORIGIN implementation differs from the frozen checkpoint code")
    for key, expected in {
        "dataset": dataset,
        "fold": fold,
        "protocol_checksum_sha256": cv_protocol["content_checksum_sha256"],
        "implementation_signature": IMPLEMENTATION_SIGNATURE,
        "architecture_signature": ARCHITECTURE_SIGNATURE,
        "config_signature": spec["config_signature"],
    }.items():
        if marker.get(key) != expected:
            raise AssertionError(f"fold-completion marker {key} mismatch")
    for key, expected in {
        "dataset": dataset,
        "fold": fold,
        "scope": "locked_outer_test_after_inner_validation_selection",
        "protocol_checksum_sha256": cv_protocol["content_checksum_sha256"],
        "split_signature": split_manifest.get("signature"),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "csv_sha256": file_sha256(paths["outer_predictions_csv"]),
    }.items():
        if prediction_manifest.get(key) != expected:
            raise AssertionError(f"outer prediction manifest {key} mismatch")
    if split_manifest.get("schema") != "origin-split-v2":
        raise AssertionError("unexpected split manifest schema")
    if split_manifest.get("evaluation_scope") != "outer_test_after_selection":
        raise AssertionError("split manifest does not authorize locked-outer evaluation")
    config_values = dict(state.get("config", {}))
    allowed = {field.name for field in fields(OriginConfig)}
    cfg = OriginConfig(**{key: value for key, value in config_values.items() if key in allowed})
    validate_pinned_config(config_values, dataset)
    for key, expected in {
        "fold": fold,
        "split_signature": split_manifest.get("signature"),
        "implementation_signature": IMPLEMENTATION_SIGNATURE,
        "architecture_signature": ARCHITECTURE_SIGNATURE,
        "config_signature": spec["config_signature"],
    }.items():
        if state.get(key) != expected:
            raise AssertionError(f"checkpoint {key} mismatch")

    items = load_origin_items(
        dataset,
        str(data_root),
        labels_csv=cfg.labels_csv,
        image_column=cfg.image_column,
        label_column=cfg.label_column,
        image_dir=cfg.image_dir,
    )
    if len(items) != int(spec["n_images"]):
        raise AssertionError("dataset size changed")
    train_items, validation_items, outer_items = split_origin_items(
        dataset, items, fold, n_folds=cfg.n_folds,
        val_fraction=cfg.val_fraction, seed=cfg.seed,
    )
    if not split_paths_are_disjoint(train_items, validation_items, outer_items):
        raise AssertionError("reconstructed split overlaps")
    expected_counts = {
        "train": len(train_items), "validation": len(validation_items),
        "locked_test": len(outer_items),
    }
    expected_histograms = {
        "train": class_histogram(train_items, cfg.n_classes),
        "validation": class_histogram(validation_items, cfg.n_classes),
        "locked_test": class_histogram(outer_items, cfg.n_classes),
    }
    if split_manifest.get("counts") != expected_counts:
        raise AssertionError("split counts changed")
    if split_manifest.get("histograms") != expected_histograms:
        raise AssertionError("split histograms changed")
    signature = split_signature(
        ("train", train_items), ("validation", validation_items),
        ("locked_test", outer_items),
    )
    if signature != split_manifest.get("signature") or signature != state.get("split_signature"):
        raise AssertionError("split content signature changed")

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
        grad_checkpoint=False,
    )
    architecture = _architecture_record(model)
    if state.get("architecture") != architecture:
        raise AssertionError("reconstructed architecture differs from checkpoint")
    if _canonical_sha256(architecture) != ARCHITECTURE_SIGNATURE:
        raise AssertionError("reconstructed architecture signature changed")
    critical_config = _critical_config(cfg)
    if state.get("critical_config") != critical_config:
        raise AssertionError("reconstructed critical config differs from checkpoint")
    model.load_state_dict(state["model_state"], strict=True)
    prediction_rows = _load_prediction_rows(paths["outer_predictions_csv"], len(outer_items), fold)
    del state, train_items, validation_items, items
    gc.collect()
    provenance = {
        "cv_protocol": cv_protocol,
        "split_manifest": split_manifest,
        "prediction_manifest": prediction_manifest,
        "fold_marker": marker,
        "paths": paths,
    }
    return cfg, model, outer_items, prediction_rows, provenance


def audit_fold(
    *,
    dataset: str,
    fold: int,
    cv_root: Path,
    data_root: Path,
    cv_protocol_path: Path,
    audit_protocol_path: Path,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    device: str = "cuda",
) -> dict[str, Any]:
    audit_protocol, audit_config = load_audit_protocol(audit_protocol_path)
    cfg, model, outer_items, prediction_rows, provenance = _verify_and_reconstruct(
        dataset=dataset, fold=fold, cv_root=cv_root, data_root=data_root,
        cv_protocol_path=cv_protocol_path,
    )
    fold_dir = cv_root / "workers" / f"fold{fold}" / f"fold{fold}"
    if output_dir.resolve() != (fold_dir / "oof_intervention_audit").resolve():
        raise AssertionError("audit output must stay in its locked fold directory")
    manifest_path = output_dir / "audit_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("refusing to overwrite a completed OOF intervention audit")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("invalid loader configuration")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = (output_dir / ".audit.lock").open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_stream.close()
        raise RuntimeError("another process owns this fold audit") from None

    checkpoint_path = provenance["paths"]["checkpoint"]
    checkpoint_hash_before = file_sha256(checkpoint_path)
    transform = OriginFundusTransform(cfg.img_size, augment=False)
    loader = DataLoader(
        OriginImageDataset(outer_items, transform), batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=selected_device.type == "cuda", persistent_workers=False,
    )
    model.to(selected_device).eval()
    artifact_paths = {
        "image_rows": output_dir / "image_boundary_census.jsonl.gz",
        "curve_rows": output_dir / "image_boundary_curves.jsonl.gz",
        "summary_rows": output_dir / "image_boundary_curve_summaries.jsonl.gz",
    }
    seen_indices: list[int] = []
    strata: dict[str, int] = {}
    eligible_boundary_count = 0
    maximum_reconstruction_error = 0.0
    maximum_decoder_error = 0.0
    try:
        with (
            AtomicJsonlGzipWriter(artifact_paths["image_rows"]) as image_writer,
            AtomicJsonlGzipWriter(artifact_paths["curve_rows"]) as curve_writer,
            AtomicJsonlGzipWriter(artifact_paths["summary_rows"]) as summary_writer,
            torch.inference_mode(),
        ):
            for images, pixel_masks, labels, indices in loader:
                images = images.to(selected_device, non_blocking=True)
                pixel_masks = pixel_masks.to(selected_device, non_blocking=True)
                with torch.autocast(
                    device_type=selected_device.type,
                    enabled=bool(cfg.amp and selected_device.type == "cuda"),
                ):
                    output = model(
                        images, pixel_valid_mask=pixel_masks, force_decoder_fp64=True
                    )
                local_maps = dict(_field(output, "local_rate_maps"))
                valid_masks = dict(_field(output, "valid_masks"))
                metadata = dict(_field(output, "metadata"))
                probabilities = _field(output, "class_probs").detach().cpu().double()
                total_rates = _field(output, "total_rates").detach().cpu().double()
                prior_rates = _field(output, "prior_rates").detach().cpu().double()
                predictions = _field(output, "class_map").detach().cpu().long()
                for row_index, raw_index in enumerate(indices):
                    outer_index = int(raw_index)
                    if outer_index != len(seen_indices):
                        raise AssertionError("outer loader order or coverage changed")
                    seen_indices.append(outer_index)
                    path, expected_label = outer_items[outer_index]
                    label = int(labels[row_index])
                    if label != int(expected_label):
                        raise AssertionError("outer loader label differs from locked split")
                    prediction_row = prediction_rows[outer_index]
                    expected_probs = torch.tensor(
                        [float(prediction_row[f"p_grade_{grade}"]) for grade in range(cfg.n_classes)],
                        dtype=torch.float64,
                    )
                    prediction_error = float((probabilities[row_index] - expected_probs).abs().max())
                    if prediction_error > 2e-5:
                        raise AssertionError("audit inference differs from frozen outer prediction export")
                    if int(predictions[row_index]) != int(prediction_row["primary_prediction"]):
                        raise AssertionError("audit MAP decision differs from frozen outer export")
                    relative_path = _relative_image_path(path, data_root)
                    patient_id = Path(path).stem.rsplit("_", 1)[0] if dataset == "dr" else ""
                    image_key = f"{dataset}:fold{fold}:{outer_index}"
                    identity = {
                        "dataset": dataset,
                        "fold": fold,
                        "outer_index": outer_index,
                        "image_key": image_key,
                        "image_path_relative": relative_path,
                        "image_id": Path(path).name,
                        "patient_id": patient_id,
                        "cluster_id": patient_id or image_key,
                    }
                    predicted_grade = int(predictions[row_index])
                    eligible_boundaries = {
                        boundary
                        for boundary in range(cfg.n_classes - 1)
                        if label > boundary or predicted_grade > boundary
                    }
                    result = audit_image_ledger(
                        rate_maps={name: value[row_index] for name, value in local_maps.items()},
                        valid_masks={name: value[row_index] for name, value in valid_masks.items()},
                        prior_rates=prior_rates[row_index],
                        metadata=metadata,
                        identity=identity,
                        true_grade=label,
                        config=audit_config,
                        baseline_total_rates=total_rates[row_index],
                        baseline_probabilities=probabilities[row_index],
                        curve_eligible_boundaries=eligible_boundaries,
                    )
                    eligible_boundary_count += len(eligible_boundaries)
                    for record in result["image_rows"]:
                        image_writer.write(record)
                    for record in result["curve_rows"]:
                        curve_writer.write(record)
                    for record in result["summary_rows"]:
                        summary_writer.write(record)
                    diagnostics = result["diagnostics"]
                    maximum_reconstruction_error = max(
                        maximum_reconstruction_error,
                        float(diagnostics["ledger_total_rate_reconstruction_max_abs_error"]),
                    )
                    maximum_decoder_error = max(
                        maximum_decoder_error,
                        float(diagnostics["baseline_decoder_replay_max_abs_probability_error"]),
                    )
                    key = f"true={label}|pred={int(predictions[row_index])}|correct={label == int(predictions[row_index])}"
                    strata[key] = strata.get(key, 0) + 1
        if seen_indices != list(range(len(outer_items))):
            raise AssertionError("OOF audit did not cover every locked outer image exactly once")
        if checkpoint_hash_before != file_sha256(checkpoint_path):
            raise AssertionError("checkpoint changed during read-only intervention audit")
        artifacts = {
            name: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "rows": {
                    "image_rows": image_writer.rows,
                    "curve_rows": curve_writer.rows,
                    "summary_rows": summary_writer.rows,
                }[name],
                "row_schema": ROW_SCHEMA,
            }
            for name, path in artifact_paths.items()
        }
        manifest: dict[str, Any] = {
            "schema": FOLD_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "fold": fold,
            "scope": "every_locked_outer_fold_image_exact_stored_ledger_interventions",
            "n_images": len(outer_items),
            "n_boundaries": cfg.n_classes - 1,
            "n_image_boundaries": len(outer_items) * (cfg.n_classes - 1),
            "n_curve_eligible_image_boundaries": eligible_boundary_count,
            "curve_eligibility_rule": (
                "true_grade > boundary or class_map_prediction > boundary; "
                "fixed before interventions and independent of intervention outcomes"
            ),
            "decision_rule": "class_map",
            "split_signature": provenance["split_manifest"]["signature"],
            "cv_protocol_checksum_sha256": provenance["cv_protocol"]["content_checksum_sha256"],
            "audit_protocol_path": str(audit_protocol_path.resolve()),
            "audit_protocol_checksum_sha256": audit_protocol["content_checksum_sha256"],
            "audit_config": audit_config.record(),
            "audit_config_sha256": canonical_sha256(audit_config.record()),
            "implementation_signature": IMPLEMENTATION_SIGNATURE,
            "architecture_signature": ARCHITECTURE_SIGNATURE,
            "checkpoint_sha256": checkpoint_hash_before,
            "outer_prediction_manifest_sha256": file_sha256(
                provenance["paths"]["outer_predictions_manifest"]
            ),
            "audit_implementation_sha256": file_sha256(Path(__file__).resolve()),
            "audit_common_implementation_sha256": file_sha256(
                REPO_ROOT / "scripts" / "origin_oof_intervention_common.py"
            ),
            "strata_image_counts": strata,
            "diagnostics": {
                "max_ledger_total_rate_reconstruction_abs_error": maximum_reconstruction_error,
                "max_baseline_decoder_replay_probability_abs_error": maximum_decoder_error,
                "selected_groups_per_eligible_image_boundary": len(
                    audit_config.cell_budget_fractions
                ) * (2 + len(audit_protocol["controls"]) * audit_config.random_repeats),
                "decoded_intervention_rate_vectors_per_eligible_image_boundary": 2
                * len(audit_config.cell_budget_fractions)
                * (2 + len(audit_protocol["controls"]) * audit_config.random_repeats),
            },
            "artifacts": artifacts,
            "interpretation_scope": "exact_frozen_stored_ledger_intervention_not_causal_pixel_intervention",
        }
        manifest["content_checksum_sha256"] = canonical_sha256(manifest)
        write_json_atomic(manifest_path, manifest)
        return manifest
    finally:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_SPECS), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--cv-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cv-protocol", type=Path, required=True)
    parser.add_argument(
        "--audit-protocol", type=Path,
        default=REPO_ROOT / "scripts" / "protocols" / "origin_oof_intervention_protocol.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    manifest = audit_fold(
        dataset=args.dataset, fold=args.fold, cv_root=args.cv_root,
        data_root=args.data_root, cv_protocol_path=args.cv_protocol,
        audit_protocol_path=args.audit_protocol, output_dir=args.output_dir,
        batch_size=args.batch_size, num_workers=args.num_workers, device=args.device,
    )
    print(
        "ORIGIN OOF intervention fold complete: "
        f"dataset={manifest['dataset']} fold={manifest['fold']} n={manifest['n_images']} "
        f"manifest={args.output_dir / 'audit_manifest.json'}"
    )


if __name__ == "__main__":
    main()

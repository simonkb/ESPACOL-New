from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from configs.origin_config import OriginConfig
from Datasets.origin_data import (
    BUSI_CLASS_TO_LABEL,
    ORIGIN_PREPROCESSING_VERSION,
    OriginFundusTransform,
    OriginGenericTransform,
    aptos_fold,
    eyepacs_fold,
    load_busi_items,
    load_csv_items,
    make_origin_loaders,
    split_paths_are_disjoint,
    validate_ordinal_labels,
)
from train_origin import (
    build_parser,
    guard_and_write_split_manifest,
    parse_scales,
    split_signature,
)


def _image(path: Path, colour: tuple[int, int, int] = (80, 120, 160)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (37, 29), colour).save(path)


def _path_set(items):
    return {str(Path(path).resolve()) for path, _ in items}


def _patient_set(items):
    return {Path(path).stem.rsplit("_", 1)[0] for path, _ in items}


def test_origin_config_defaults_are_internally_valid() -> None:
    cfg = OriginConfig()
    assert cfg.img_size == 640
    assert cfg.encoder == "convnext_tiny"
    assert cfg.evidence_scales == ("s4", "s8", "s16", "s32")
    assert cfg.reference_count == 4096.0
    assert cfg.atom_rate_init == 1e-6
    assert cfg.prior_rate_init == 1e-4
    assert cfg.decision_rule == "class_map"
    assert cfg.preprocessing_version == ORIGIN_PREPROCESSING_VERSION
    assert cfg.class_weighting == "none"


def test_origin_config_requires_explicit_weighted_likelihood_acknowledgement() -> None:
    with pytest.raises(ValueError, match="population posterior"):
        OriginConfig(class_weighting="effective_num")


def test_origin_cli_defaults_to_locked_outer_test() -> None:
    args = build_parser().parse_args([])
    assert not args.include_test
    assert not args.skip_test
    assert args.decision_rule == "class_map"
    assert args.scales == ("s4", "s8", "s16", "s32")


def test_scale_parser_accepts_stride_shorthand_and_rejects_duplicates() -> None:
    assert parse_scales("4,s8,16") == ("s4", "s8", "s16")
    with pytest.raises(argparse.ArgumentTypeError, match="unique"):
        parse_scales("s8,8")


def test_split_signature_keeps_relative_directories_but_ignores_machine_root() -> None:
    first = split_signature(
        ("train", [("/machine_a/data/patient_1/image.png", 0)]),
        ("validation", [("/machine_a/data/patient_2/image.png", 0)]),
    )
    relocated = split_signature(
        ("train", [("/machine_b/data/patient_1/image.png", 0)]),
        ("validation", [("/machine_b/data/patient_2/image.png", 0)]),
    )
    changed_patient = split_signature(
        ("train", [("/machine_a/data/patient_3/image.png", 0)]),
        ("validation", [("/machine_a/data/patient_2/image.png", 0)]),
    )
    assert first == relocated
    assert first != changed_patient


def test_split_signature_binds_image_bytes(tmp_path: Path) -> None:
    path = tmp_path / "patient" / "image.png"
    _image(path, (10, 20, 30))
    before = split_signature(("train", [(str(path), 0)]))
    _image(path, (200, 100, 50))
    after = split_signature(("train", [(str(path), 0)]))
    assert before != after


def test_split_manifest_guard_preserves_existing_run_provenance(tmp_path: Path) -> None:
    original = {
        "dataset": "aptos",
        "fold": 0,
        "signature": "original",
        "counts": {"train": 1, "validation": 1, "locked_test": 1},
        "histograms": {"train": [1], "validation": [1], "locked_test": [1]},
    }
    guard_and_write_split_manifest(tmp_path, original, resume=False)
    (tmp_path / "history.csv").write_text("epoch\n1\n")
    changed = {**original, "signature": "changed"}
    with pytest.raises(FileExistsError, match="history.csv"):
        guard_and_write_split_manifest(tmp_path, changed, resume=False)
    with pytest.raises(ValueError, match="signature"):
        guard_and_write_split_manifest(tmp_path, changed, resume=True)
    assert (tmp_path / "split_manifest.json").read_text().find("original") >= 0


def test_generic_transform_returns_full_valid_mask() -> None:
    transform = OriginGenericTransform(size=32, augment=False)
    image, mask = transform(Image.new("RGB", (43, 21), (40, 50, 60)))
    assert image.shape == (3, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert mask.dtype == torch.bool
    assert bool(mask.all())


def test_fundus_transform_uses_image_independent_canonical_support() -> None:
    transform = OriginFundusTransform(size=32, augment=False)
    _, dark_mask = transform(Image.new("RGB", (51, 33), 0))
    _, bright_mask = transform(Image.new("RGB", (27, 61), 255))
    assert torch.equal(dark_mask, bright_mask)
    assert dark_mask.shape == (1, 32, 32)
    assert bool(dark_mask.any()) and not bool(dark_mask.all())


def test_busi_folder_adapter_excludes_segmentation_masks(tmp_path: Path) -> None:
    for class_name in BUSI_CLASS_TO_LABEL:
        _image(tmp_path / class_name / f"{class_name} (1).png")
        _image(tmp_path / class_name / f"{class_name} (1)_mask.png")
    items = load_busi_items(tmp_path)
    assert len(items) == 3
    assert sorted(label for _, label in items) == [0, 1, 2]
    assert all("mask" not in Path(path).name.lower() for path, _ in items)


def test_generic_csv_adapter_resolves_extensionless_identifiers(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    _image(image_root / "a.png")
    _image(image_root / "b.jpg")
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"case": ["a", "b.jpg"], "severity": [0, 1]}).to_csv(
        csv_path, index=False
    )
    items = load_csv_items(
        tmp_path,
        csv_path,
        image_column="case",
        label_column="severity",
        image_dir="images",
    )
    assert [Path(path).name for path, _ in items] == ["a.png", "b.jpg"]
    assert [label for _, label in items] == [0, 1]


def test_aptos_wrapper_is_deterministic_and_disjoint() -> None:
    items = [(f"/virtual/g{grade}_{index}.png", grade) for grade in range(5) for index in range(20)]
    first = aptos_fold(items, fold=2, n_folds=5, seed=19)
    second = aptos_fold(items, fold=2, n_folds=5, seed=19)
    assert first == second
    assert split_paths_are_disjoint(*first)
    assert set().union(*(_path_set(split) for split in first)) == _path_set(items)
    assert all({label for _, label in split} == set(range(5)) for split in first)


def test_eyepacs_wrapper_keeps_paired_eyes_patient_disjoint() -> None:
    items = []
    for grade in range(5):
        for patient_index in range(15):
            patient = f"g{grade}p{patient_index}"
            items.extend(
                [
                    (f"/virtual/{patient}_left.jpeg", grade),
                    (f"/virtual/{patient}_right.jpeg", grade),
                ]
            )
    train, validation, test = eyepacs_fold(
        items, fold=0, n_folds=5, val_fraction=0.2, seed=7
    )
    assert split_paths_are_disjoint(train, validation, test)
    train_patients = _patient_set(train)
    validation_patients = _patient_set(validation)
    test_patients = _patient_set(test)
    assert train_patients.isdisjoint(validation_patients)
    assert train_patients.isdisjoint(test_patients)
    assert validation_patients.isdisjoint(test_patients)
    assert train_patients | validation_patients | test_patients == _patient_set(items)


def test_origin_loader_preserves_four_field_sample_contract(tmp_path: Path) -> None:
    items = []
    for label in range(3):
        for index in range(4):
            path = tmp_path / str(label) / f"{index}.png"
            _image(path)
            items.append((str(path), label))
    train, validation, test = aptos_fold(
        items, fold=0, n_folds=2, val_fraction=0.25, seed=3
    )
    loaders = make_origin_loaders(
        train,
        validation,
        test,
        image_size=32,
        batch_size=3,
        num_workers=0,
        pin_memory=False,
        seed=3,
        fundus=False,
        stratified=False,
    )
    batch = next(iter(loaders[1]))
    assert len(batch) == 4
    images, masks, labels, indices = batch
    assert images.ndim == 4 and images.shape[1:] == (3, 32, 32)
    assert masks.ndim == 4 and masks.shape[1:] == (1, 32, 32)
    assert labels.dtype == torch.long
    assert indices.dtype == torch.long


def test_validate_ordinal_labels_rejects_gaps() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        validate_ordinal_labels([("a", 0), ("b", 2)])

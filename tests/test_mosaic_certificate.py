"""Focused tests for replayable MOSAIC inference certificates."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from inference.mosaic_certificate import (
    CertificateReplayError,
    SCHEMA_VERSION,
    build_mosaic_certificate,
    certificate_to_json,
    load_mosaic_certificate,
    save_mosaic_certificate,
    verify_mosaic_certificate,
)
from models.mosaic import MOSAICOrdinalCore


def _example_output():
    torch.manual_seed(11)
    # Two samples, six micro-regions, three ordinal grades.
    logits = torch.randn(2, 6, 3)
    valid = torch.tensor(
        [[True, True, True, True, True, False], [True, True, True, False, False, False]]
    )
    core = MOSAICOrdinalCore(
        num_classes=3,
        max_count=3,
        sufficiency_tolerance=0.05,
        complement_suppression=0.5,
        implementation="serial",
        block_size=3,
    )
    with torch.no_grad():
        output = core(logits, valid, project=True, return_pivotality=True)
    metadata = {
        "input_size": [16, 24],
        "lattice_size": [2, 3],
        "local_dim": 8,
        "receptive_field": {
            "tap": "unit_test",
            "feature_index": 0,
            "channels": 8,
            "output_stride": 8,
            "receptive_field": 7,
            "center_offset": 0.5,
            "squeeze_excitation_removed": False,
            "globally_mixed": False,
        },
    }
    return output, valid, metadata


class MosaicCertificateTests(unittest.TestCase):
    def test_certificate_contains_required_trace_and_replays(self) -> None:
        output, valid, metadata = _example_output()
        certificate = build_mosaic_certificate(
            output,
            lattice_metadata=metadata,
            valid_mask=valid,
            sample_index=0,
            sample_id="aptos-example",
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        self.assertEqual(certificate["schema_version"], SCHEMA_VERSION)
        self.assertEqual(certificate["sample_id"], "aptos-example")
        self.assertIn("witness_probabilities", certificate["dense_ledger"])
        self.assertIn("selected_indices", certificate["proof"])
        self.assertIn("fixed_proof_pivotality", certificate["proof"])
        self.assertIn("receptive_field", certificate["receptive_field_metadata"])
        self.assertIn(
            "projected_log_stop_probabilities", certificate["prediction"]
        )
        self.assertIn(
            "dense_log_conditional_low_distribution", certificate["cardinality"]
        )
        self.assertEqual(
            certificate["numerical_contract"]["arithmetic"],
            "fp32_scaled_log_lower_tail_poisson_binomial",
        )

        report = verify_mosaic_certificate(certificate)
        self.assertTrue(report["ok"], report)
        self.assertTrue(all(report["checks"].values()))

    def test_v1_certificate_is_explicitly_rejected(self) -> None:
        output, valid, metadata = _example_output()
        certificate = build_mosaic_certificate(
            output, lattice_metadata=metadata, valid_mask=valid
        )
        certificate["schema_version"] = "mosaic-certificate-v1"
        with self.assertRaisesRegex(ValueError, "no stable log-stop trace"):
            verify_mosaic_certificate(certificate)

    def test_each_selected_cell_has_support_and_effect(self) -> None:
        output, valid, metadata = _example_output()
        certificate = build_mosaic_certificate(
            output,
            lattice_metadata=metadata,
            valid_mask=valid,
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        for boundary_cells in certificate["proof"]["selected_cells"]:
            for cell in boundary_cells:
                self.assertIn("center_yx", cell)
                self.assertIn("receptive_field_box_yxyx", cell)
                self.assertGreaterEqual(cell["fixed_proof_pivotality"], -1e-7)

    def test_json_round_trip_and_file_helpers(self) -> None:
        output, valid, metadata = _example_output()
        certificate = build_mosaic_certificate(
            output,
            lattice_metadata=metadata,
            valid_mask=valid,
            sample_index=1,
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        encoded = certificate_to_json(certificate)
        decoded = json.loads(encoded)
        self.assertTrue(verify_mosaic_certificate(decoded)["ok"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "certificate.json"
            save_mosaic_certificate(certificate, path)
            loaded = load_mosaic_certificate(path)
            self.assertTrue(verify_mosaic_certificate(loaded)["ok"])

    def test_plain_mapping_without_precomputed_pivotality_is_supported(self) -> None:
        output, valid, metadata = _example_output()
        proof_mapping = {
            name: getattr(output.proof, name)
            for name in (
                "selected_mask",
                "sorted_indices",
                "proof_size",
                "dense_transition",
                "projected_transition",
                "complement_transition",
                "retained_distribution",
                "complement_distribution",
                "sufficiency_gap",
                "complement_drop",
            )
        }
        output_mapping = {
            "local_state_probabilities": output.local_state_probabilities,
            "witness_probabilities": output.witness_probabilities,
            "log_witness_probabilities": output.log_witness_probabilities,
            "log_nonwitness_probabilities": output.log_nonwitness_probabilities,
            "alpha": output.alpha,
            "log_alpha": output.log_alpha,
            "dense_transitions": output.dense_transitions,
            "dense_stop_probabilities": output.dense_stop_probabilities,
            "dense_log_stop_probabilities": output.dense_log_stop_probabilities,
            "transitions": output.transitions,
            "stop_probabilities": output.stop_probabilities,
            "log_stop_probabilities": output.log_stop_probabilities,
            "cumulative_probabilities": output.cumulative_probabilities,
            "class_probabilities": output.class_probabilities,
            "expected_grade": output.expected_grade,
            "predicted_grade": output.predicted_grade,
            "proof": proof_mapping,
            "pivotality": None,
        }
        certificate = build_mosaic_certificate(
            output_mapping,
            lattice_metadata=metadata,
            valid_mask=valid,
            sample_index=1,
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        self.assertTrue(verify_mosaic_certificate(certificate)["ok"])

    def test_tampered_dense_ledger_fails_replay(self) -> None:
        output, valid, metadata = _example_output()
        certificate = build_mosaic_certificate(
            output,
            lattice_metadata=metadata,
            valid_mask=valid,
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        tampered = copy.deepcopy(certificate)
        tampered["dense_ledger"]["witness_probabilities"][0][0] *= 0.2
        report = verify_mosaic_certificate(tampered)
        self.assertFalse(report["ok"])
        self.assertFalse(report["checks"]["dense_transitions"])
        with self.assertRaises(CertificateReplayError):
            verify_mosaic_certificate(tampered, raise_on_error=True)

    def test_invalid_selected_cell_is_detected(self) -> None:
        output, valid, metadata = _example_output()
        certificate = build_mosaic_certificate(
            output,
            lattice_metadata=metadata,
            valid_mask=valid,
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        tampered = copy.deepcopy(certificate)
        # Index 5 is invalid for sample zero.
        tampered["proof"]["selected_indices"][0] = [5]
        tampered["proof"]["proof_sizes"][0] = 1
        report = verify_mosaic_certificate(tampered)
        self.assertFalse(report["ok"])
        self.assertFalse(report["checks"]["selected_indices_valid"])

    def test_builder_requires_receptive_field_metadata(self) -> None:
        output, valid, _ = _example_output()
        with self.assertRaisesRegex(ValueError, "lattice_metadata"):
            build_mosaic_certificate(output, lattice_metadata=None, valid_mask=valid)
        with self.assertRaisesRegex(ValueError, "receptive-field"):
            build_mosaic_certificate(output, lattice_metadata={}, valid_mask=valid)

    def test_human_facing_cell_and_geometry_tampering_is_detected(self) -> None:
        output, valid, metadata = _example_output()
        certificate = build_mosaic_certificate(
            output,
            lattice_metadata=metadata,
            valid_mask=valid,
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        tampered_cell = copy.deepcopy(certificate)
        first_nonempty = next(
            cells for cells in tampered_cell["proof"]["selected_cells"] if cells
        )
        first_nonempty[0]["center_yx"] = [9999.0, 9999.0]
        first_nonempty[0]["witness_probability"] = -123.0
        cell_report = verify_mosaic_certificate(tampered_cell)
        self.assertFalse(cell_report["ok"])
        self.assertFalse(cell_report["checks"]["integrity_sha256"])
        self.assertFalse(cell_report["checks"]["selected_cell_records"])

        tampered_geometry = copy.deepcopy(certificate)
        tampered_geometry["receptive_field_metadata"]["receptive_field"][
            "center_offset"
        ] = 99.5
        geometry_report = verify_mosaic_certificate(tampered_geometry)
        self.assertFalse(geometry_report["ok"])
        self.assertFalse(geometry_report["checks"]["integrity_sha256"])
        self.assertFalse(geometry_report["checks"]["selected_cell_records"])

    def test_full_lattice_near_normal_certificate_replays(self) -> None:
        cells = 112 * 112
        probabilities = torch.tensor([0.99996, 1e-5, 1e-5, 1e-5, 1e-5])
        logits = probabilities.log().view(1, 1, 5).repeat(1, cells, 1)
        core = MOSAICOrdinalCore(
            num_classes=5,
            max_count=32,
            sufficiency_tolerance=0.02,
            complement_suppression=0.5,
            implementation="block_tree",
            block_size=64,
        )
        with torch.no_grad():
            output = core(logits, project=True, return_pivotality=False)
        metadata = {
            "input_size": [896, 896],
            "lattice_size": [112, 112],
            "local_dim": 128,
            "receptive_field": {
                "tap": "rf_medium",
                "feature_index": 3,
                "channels": 64,
                "output_stride": 8,
                "receptive_field": 95,
                "center_offset": 0.5,
                "squeeze_excitation_removed": False,
                "globally_mixed": False,
            },
        }
        certificate = build_mosaic_certificate(
            output,
            lattice_metadata=metadata,
            valid_mask=torch.ones(1, cells, dtype=torch.bool),
            sufficiency_tolerance=0.02,
            complement_suppression=0.5,
        )
        report = verify_mosaic_certificate(certificate)
        self.assertTrue(report["ok"], report)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA replay audit requires a GPU")
    def test_cuda_full_lattice_certificate_replays_canonically_on_cpu(self) -> None:
        """Guard the cross-device FP32 numerical contract used by export."""

        cells = 112 * 112
        probabilities = torch.tensor(
            [0.99996, 1e-5, 1e-5, 1e-5, 1e-5], device="cuda"
        )
        logits = probabilities.log().view(1, 1, 5).repeat(1, cells, 1)
        # Add a few focal severe witnesses so both the near-normal background
        # and non-empty proof paths participate in GPU reductions.
        focal = torch.tensor(
            [0.10, 1e-6, 1e-6, 1e-6, 0.90], device="cuda"
        )
        logits[:, :9] = focal.log()
        core = MOSAICOrdinalCore(
            num_classes=5,
            max_count=32,
            sufficiency_tolerance=0.02,
            complement_suppression=0.5,
            implementation="block_tree",
            block_size=64,
        ).cuda()
        with torch.no_grad():
            output = core(logits, project=True, return_pivotality=False)
        metadata = {
            "input_size": [896, 896],
            "lattice_size": [112, 112],
            "local_dim": 128,
            "receptive_field": {
                "tap": "rf_medium",
                "feature_index": 3,
                "channels": 64,
                "output_stride": 8,
                "receptive_field": 95,
                "center_offset": 0.5,
                "squeeze_excitation_removed": False,
                "globally_mixed": False,
            },
        }
        certificate = build_mosaic_certificate(
            output,
            lattice_metadata=metadata,
            valid_mask=torch.ones(1, cells, device="cuda", dtype=torch.bool),
            sufficiency_tolerance=0.02,
            complement_suppression=0.5,
        )
        report = verify_mosaic_certificate(certificate)
        self.assertTrue(report["ok"], report)

    def test_image_model_wrapper_supplies_mask_and_lattice(self) -> None:
        output, valid, metadata = _example_output()
        wrapped = {"evidence": output, "valid_mask": valid, "lattice": metadata}
        certificate = build_mosaic_certificate(
            wrapped,
            sample_index=0,
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        self.assertEqual(certificate["dense_ledger"]["valid_mask_source"], "provided")
        self.assertTrue(verify_mosaic_certificate(certificate)["ok"])


if __name__ == "__main__":
    unittest.main()

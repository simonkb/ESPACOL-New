"""Focused tests for replayable MOSAIC inference certificates."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch

from inference.mosaic_certificate import (
    CertificateReplayError,
    MAX_REPLAY_ATOL,
    MAX_REPLAY_RTOL,
    SCHEMA_VERSION,
    _payload_sha256,
    build_mosaic_certificate,
    certificate_to_json,
    load_mosaic_certificate,
    save_mosaic_certificate,
    verify_mosaic_certificate,
)
from models.mosaic import (
    MOSAICOrdinalCore,
    nested_witness_probabilities,
    receptive_field_packed_ordinal_evidence,
    regional_logmeanexp_ordinal_evidence,
    regional_max_ordinal_evidence,
)
from models.mosaic_decoder import proof_only_decisions


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


def _regional_example_output(pool_type: str = "normalized_logmeanexp"):
    torch.manual_seed(111)
    source_logits = torch.randn(1, 16, 3)
    # Put a distinct severe peak in every fixed 2x2 source block.
    for index in (0, 2, 8, 10):
        source_logits[0, index] = torch.tensor([-4.0, -2.0, 5.0])
    source_valid = torch.ones(1, 16, dtype=torch.bool)
    source_evidence = nested_witness_probabilities(source_logits, source_valid)
    if pool_type == "normalized_logmeanexp":
        regional = regional_logmeanexp_ordinal_evidence(
            source_evidence, source_valid, (4, 4), (2, 2), temperature=0.25
        )
        pool_tag = "lme"
    elif pool_type == "existential_max":
        regional = regional_max_ordinal_evidence(
            source_evidence, source_valid, (4, 4), (2, 2)
        )
        pool_tag = "existential_max"
    else:
        raise ValueError(pool_type)
    core = MOSAICOrdinalCore(
        num_classes=3,
        max_count=3,
        sufficiency_tolerance=0.05,
        complement_suppression=0.5,
        implementation="serial",
        block_size=3,
    )
    with torch.no_grad():
        output = core.forward_evidence(
            regional.evidence,
            valid_mask=regional.valid_mask,
            project=True,
            return_pivotality=True,
        )
    output.evidence_valid_mask = regional.valid_mask
    output.regional_source_indices = regional.source_indices
    output.source_lattice_size = regional.source_lattice_size
    output.regional_block_size = regional.block_size
    output.regional_pool_temperature = regional.temperature
    output.regional_pool_type = regional.pool_type
    source_metadata = {
        "input_size": [32, 32],
        "lattice_size": [4, 4],
        "local_dim": 8,
        "receptive_field": {
            "tap": "rf_medium",
            "feature_index": 3,
            "channels": 64,
            "output_stride": 8,
            "receptive_field": 7,
            "center_offset": 0.5,
            "squeeze_excitation_removed": False,
            "globally_mixed": False,
        },
    }
    regional_metadata = copy.deepcopy(source_metadata)
    regional_metadata["lattice_size"] = [2, 2]
    regional_metadata["receptive_field"].update(
        {
            "tap": f"rf_medium_regional_{pool_tag}_2x2",
            "output_stride": 16,
            "receptive_field": 15,
            "center_offset": 4.5,
        }
    )
    wrapped = SimpleNamespace(
        evidence=output,
        valid_mask=regional.valid_mask,
        lattice=regional_metadata,
        source_lattice=source_metadata,
        source_valid_mask=source_valid,
        decision_rule="rounded_expected",
        decision_transition_weights=torch.ones(2, 2),
    )
    return wrapped


def _rf_packed_example_output():
    torch.manual_seed(112)
    source_logits = torch.randn(1, 16, 3)
    # Adjacent high-severity cells force the deterministic RF exclusion rule
    # to choose one representative rather than count both overlapping fields.
    source_logits[0, 5] = torch.tensor([-5.0, -2.0, 6.0])
    source_logits[0, 6] = torch.tensor([-4.0, -1.0, 5.0])
    # A finite log probability below FP32's exp range must remain distinct
    # from the certificate's encoded mathematical log-zero sentinel.
    source_logits[0, 14] = torch.tensor([0.0, -1000.0, -1000.0])
    source_valid = torch.ones(1, 16, dtype=torch.bool)
    source_valid[0, 15] = False
    source_evidence = nested_witness_probabilities(source_logits, source_valid)
    packed = receptive_field_packed_ordinal_evidence(
        source_evidence,
        source_valid,
        (4, 4),
        output_stride=4,
        receptive_field=7.0,
        max_overlap=0.25,
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
        output = core.forward_evidence(
            packed.evidence,
            valid_mask=packed.packing_mask,
            project=True,
            return_pivotality=True,
        )
    output.source_state_probabilities = source_evidence.state_probabilities
    output.source_witness_probabilities = source_evidence.witness_probabilities
    output.source_log_witness_probabilities = (
        source_evidence.log_witness_probabilities
    )
    output.source_log_nonwitness_probabilities = (
        source_evidence.log_nonwitness_probabilities
    )
    output.rf_packing_mask = packed.packing_mask
    output.rf_packing_max_overlap = packed.max_overlap
    output.rf_packing_nms_iou = packed.nms_iou_threshold
    output.rf_packing_output_stride = packed.output_stride
    output.rf_packing_receptive_field = packed.receptive_field
    metadata = {
        "input_size": [16, 16],
        "lattice_size": [4, 4],
        "local_dim": 8,
        "receptive_field": {
            "tap": "rf_medium",
            "feature_index": 3,
            "channels": 64,
            "output_stride": 4,
            "receptive_field": 7.0,
            "center_offset": 0.5,
            "squeeze_excitation_removed": False,
            "globally_mixed": False,
        },
    }
    return SimpleNamespace(
        evidence=output,
        valid_mask=packed.packing_mask,
        lattice=metadata,
        source_lattice=metadata,
        source_valid_mask=source_valid,
        decision_rule="rounded_expected",
        decision_transition_weights=torch.ones(2, 2),
    )


class MosaicCertificateTests(unittest.TestCase):
    def test_rf_packing_certificate_replays_raw_ledger_and_exclusion(self) -> None:
        certificate = build_mosaic_certificate(
            _rf_packed_example_output(),
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        packing = certificate["receptive_field_packing"]
        self.assertEqual(certificate["evidence_compiler"], "rf_packing")
        self.assertEqual(
            packing["algorithm"],
            "stable_descending_severity_receptive_field_nms",
        )
        self.assertEqual(
            packing["priority_definition"],
            "sum_boundary_witness_probabilities",
        )
        self.assertEqual(packing["max_overlap_fraction"], 0.25)
        self.assertAlmostEqual(packing["converted_iou_threshold"], 1.0 / 7.0)
        self.assertGreater(
            packing["max_observed_pairwise_overlap_fraction"], 0.0
        )
        self.assertLessEqual(
            packing["max_observed_pairwise_overlap_fraction"], 0.25
        )
        self.assertEqual(packing["geometry"]["lattice_size"], [4, 4])
        self.assertEqual(packing["geometry"]["output_stride"], 4)
        self.assertEqual(packing["geometry"]["receptive_field"], 7.0)
        self.assertIn("raw_local_state_probabilities", packing)
        self.assertIn("raw_witness_probabilities", packing)
        self.assertLess(packing["raw_log_witness_probabilities"][14][1], -900.0)
        self.assertGreater(
            packing["raw_log_witness_probabilities"][14][1], -5.0e29
        )
        report = verify_mosaic_certificate(certificate)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["checks"]["rf_packing_provenance"])
        self.assertTrue(report["checks"]["rf_packing_severity_ranking"])
        self.assertTrue(report["checks"]["rf_packing_mask"])
        self.assertTrue(report["checks"]["rf_packing_ledger"])
        self.assertTrue(report["checks"]["rf_packing_pairwise_overlap"])
        round_tripped = json.loads(certificate_to_json(certificate))
        round_trip_report = verify_mosaic_certificate(round_tripped)
        self.assertTrue(round_trip_report["ok"], round_trip_report)

    def test_rehashed_rf_packing_mask_tampering_fails_semantic_replay(self) -> None:
        certificate = build_mosaic_certificate(
            _rf_packed_example_output(),
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        tampered = copy.deepcopy(certificate)
        packed = tampered["receptive_field_packing"]["packed_mask"]
        excluded = next(index for index, selected in enumerate(packed) if not selected)
        packed[excluded] = True
        tampered["integrity"]["payload_sha256"] = _payload_sha256(tampered)
        report = verify_mosaic_certificate(tampered)
        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"]["integrity_sha256"])
        self.assertFalse(report["checks"]["rf_packing_mask"])
        self.assertFalse(report["checks"]["rf_packing_pairwise_overlap"])

    def test_rehashed_rf_packing_section_deletion_fails_compiler_contract(self) -> None:
        certificate = build_mosaic_certificate(
            _rf_packed_example_output(),
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        tampered = copy.deepcopy(certificate)
        tampered.pop("receptive_field_packing")
        tampered["integrity"]["payload_sha256"] = _payload_sha256(tampered)
        report = verify_mosaic_certificate(tampered)
        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"]["integrity_sha256"])
        self.assertFalse(report["checks"]["evidence_compiler"])

    def test_rehashed_rf_source_log_sentinel_tampering_fails(self) -> None:
        certificate = build_mosaic_certificate(
            _rf_packed_example_output(),
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        tampered = copy.deepcopy(certificate)
        # The source state underflows to probability zero here, but its finite
        # log probability is part of the stable evidence trace.  Replacing it
        # by mathematical log-zero must not be silently accepted.
        tampered["receptive_field_packing"][
            "raw_log_witness_probabilities"
        ][14][1] = -1.0e30
        tampered["integrity"]["payload_sha256"] = _payload_sha256(tampered)
        report = verify_mosaic_certificate(tampered)
        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"]["integrity_sha256"])
        self.assertFalse(report["checks"]["rf_packing_ledger"])

    def test_rf_packed_ledger_requires_exact_source_reconstruction(self) -> None:
        certificate = build_mosaic_certificate(
            _rf_packed_example_output(),
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        tampered = copy.deepcopy(certificate)
        selected = tampered["receptive_field_packing"]["packed_mask"].index(True)
        # This perturbation is below the global numerical replay tolerance.
        # ORFP provenance is nevertheless a deterministic tensor selection and
        # therefore requires bit-exact reconstruction from the source ledger.
        tampered["dense_ledger"]["witness_probabilities"][selected][0] += 1e-6
        tampered["integrity"]["payload_sha256"] = _payload_sha256(tampered)
        report = verify_mosaic_certificate(tampered)
        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"]["integrity_sha256"])
        self.assertFalse(report["checks"]["rf_packing_ledger"])

    def test_legacy_v3_certificate_without_packing_remains_replayable(self) -> None:
        output, valid, metadata = _example_output()
        certificate = build_mosaic_certificate(
            output,
            lattice_metadata=metadata,
            valid_mask=valid,
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        certificate["schema_version"] = "mosaic-certificate-v3"
        certificate.pop("evidence_compiler")
        certificate["integrity"]["payload_sha256"] = _payload_sha256(certificate)
        report = verify_mosaic_certificate(certificate)
        self.assertTrue(report["ok"], report)

    def test_legacy_v3_cannot_make_an_unverified_packing_claim(self) -> None:
        certificate = build_mosaic_certificate(
            _rf_packed_example_output(),
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        certificate["schema_version"] = "mosaic-certificate-v3"
        certificate["integrity"]["payload_sha256"] = _payload_sha256(certificate)
        report = verify_mosaic_certificate(certificate)
        self.assertFalse(report["ok"])
        self.assertFalse(report["checks"]["rf_packing_provenance"])

    def test_regional_wrapper_serializes_fixed_geometry_and_peak_provenance(self) -> None:
        wrapped = _regional_example_output()
        certificate = build_mosaic_certificate(
            wrapped,
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )

        provenance = certificate["regional_envelope_provenance"]
        self.assertEqual(
            certificate["evidence_compiler"],
            "regional_normalized_logmeanexp",
        )
        self.assertEqual(
            provenance["aggregation"],
            "fixed_disjoint_boundarywise_normalized_logmeanexp_logit",
        )
        self.assertEqual(provenance["pool_type"], "normalized_logmeanexp")
        self.assertEqual(provenance["temperature"], 0.25)
        self.assertEqual(provenance["regional_block_size"], [2, 2])
        self.assertEqual(
            provenance["source_lattice_metadata"]["lattice_size"], [4, 4]
        )
        self.assertEqual(len(provenance["regions"]), 4)
        self.assertEqual(provenance["regions"][0]["source_row_range"], [0, 2])
        self.assertEqual(provenance["regions"][0]["source_column_range"], [0, 2])
        self.assertEqual(len(provenance["peak_source_indices"]), 4)
        selected = [
            cell
            for boundary_cells in certificate["proof"]["selected_cells"]
            for cell in boundary_cells
        ]
        self.assertTrue(selected)
        for cell in selected:
            peak = cell["regional_peak_source"]
            self.assertIn("receptive_field_box_yxyx", peak)
            self.assertIn("not a standalone sufficiency claim", peak["provenance_scope"])
        report = verify_mosaic_certificate(certificate)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["checks"]["regional_envelope_provenance"])

    def test_existential_max_certificate_declares_exact_region_semantics(self) -> None:
        certificate = build_mosaic_certificate(
            _regional_example_output("existential_max"),
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        provenance = certificate["regional_envelope_provenance"]
        self.assertEqual(
            certificate["evidence_compiler"], "regional_existential_max"
        )
        self.assertEqual(provenance["pool_type"], "existential_max")
        self.assertEqual(
            provenance["aggregation"],
            "fixed_disjoint_boundarywise_existential_max_logit",
        )
        self.assertNotIn("temperature", provenance)
        selected = [
            cell
            for boundary_cells in certificate["proof"]["selected_cells"]
            for cell in boundary_cells
        ]
        self.assertTrue(selected)
        for cell in selected:
            self.assertIn(
                "exact existential regional witness",
                cell["regional_peak_source"]["provenance_scope"],
            )
        report = verify_mosaic_certificate(certificate)
        self.assertTrue(report["ok"], report)

    def test_legacy_regional_v3_without_pool_type_still_replays(self) -> None:
        certificate = build_mosaic_certificate(
            _regional_example_output(),
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        legacy = copy.deepcopy(certificate)
        legacy["schema_version"] = "mosaic-certificate-v3"
        legacy.pop("evidence_compiler")
        legacy["regional_envelope_provenance"].pop("pool_type")
        legacy["integrity"]["payload_sha256"] = _payload_sha256(legacy)
        report = verify_mosaic_certificate(legacy)
        self.assertTrue(report["ok"], report)

    def test_legacy_v3_without_regional_provenance_still_replays(self) -> None:
        output, valid, metadata = _example_output()
        certificate = build_mosaic_certificate(
            output,
            lattice_metadata=metadata,
            valid_mask=valid,
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        self.assertNotIn("regional_envelope_provenance", certificate)
        report = verify_mosaic_certificate(certificate)
        self.assertTrue(report["ok"], report)
        self.assertTrue(report["checks"]["regional_envelope_provenance"])

    def test_regional_peak_outside_fixed_block_fails_even_if_rehashed(self) -> None:
        certificate = build_mosaic_certificate(
            _regional_example_output(),
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        tampered = copy.deepcopy(certificate)
        # Region zero owns source rows/columns [0,2); source 15 lies in the
        # diagonally opposite block.  Rehash to show this is a semantic check,
        # not merely detection by the payload digest.
        tampered["regional_envelope_provenance"]["peak_source_indices"][0][0] = 15
        tampered["integrity"]["payload_sha256"] = _payload_sha256(tampered)
        report = verify_mosaic_certificate(tampered)
        self.assertFalse(report["ok"])
        self.assertTrue(report["checks"]["integrity_sha256"])
        self.assertFalse(report["checks"]["regional_envelope_provenance"])

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
        self.assertEqual(certificate["evidence_compiler"], "source_lattice")
        self.assertEqual(SCHEMA_VERSION, "mosaic-certificate-v4")
        self.assertEqual(certificate["sample_id"], "aptos-example")
        self.assertIn("witness_probabilities", certificate["dense_ledger"])
        self.assertIn("selected_indices", certificate["proof"])
        self.assertIn("fixed_proof_pivotality", certificate["proof"])
        self.assertIn("receptive_field", certificate["receptive_field_metadata"])
        self.assertIn(
            "projected_log_stop_probabilities", certificate["prediction"]
        )
        self.assertEqual(
            certificate["prediction"]["decision_rule"], "rounded_expected"
        )
        self.assertEqual(
            certificate["prediction"]["transition_weight_order"],
            ["stop", "advance"],
        )
        self.assertEqual(
            certificate["proof_rule"]["score_space"],
            "raw_cardinality_transition_scores",
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

    def test_v2_certificate_is_explicitly_rejected(self) -> None:
        output, valid, metadata = _example_output()
        certificate = build_mosaic_certificate(
            output, lattice_metadata=metadata, valid_mask=valid
        )
        certificate["schema_version"] = "mosaic-certificate-v2"
        with self.assertRaisesRegex(ValueError, "decision rule and outcome weights"):
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

    def test_deweighted_rule_replays_from_proof_and_training_weights(self) -> None:
        output, valid, metadata = _example_output()
        weights = torch.tensor([[0.35, 2.4], [3.1, 0.55]])
        certificate = build_mosaic_certificate(
            output,
            lattice_metadata=metadata,
            valid_mask=valid,
            sample_index=0,
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
            decision_rule="deweighted_class_map",
            transition_weights=weights,
        )
        expected = proof_only_decisions(
            output.transitions[0],
            output.log_stop_probabilities[0],
            weights,
        )
        prediction = certificate["prediction"]
        self.assertEqual(prediction["decision_rule"], "deweighted_class_map")
        self.assertEqual(
            prediction["probability_space"], "analytically_deweighted"
        )
        self.assertEqual(
            prediction["predicted_grade"], int(expected.deweighted_argmax)
        )
        torch.testing.assert_close(
            torch.tensor(prediction["class_probabilities"]),
            expected.deweighted_class_probabilities,
        )
        report = verify_mosaic_certificate(certificate)
        self.assertTrue(report["ok"], report)
        self.assertEqual(
            report["replayed_decision_rule"], "deweighted_class_map"
        )

    def test_wrapper_decoder_metadata_is_inferred_and_conflicts_are_rejected(
        self,
    ) -> None:
        output, valid, metadata = _example_output()
        weights = torch.tensor([[0.35, 2.4], [3.1, 0.55]])
        wrapped = SimpleNamespace(
            evidence=output,
            valid_mask=valid,
            lattice=metadata,
            decision_rule="deweighted_class_map",
            decision_transition_weights=weights,
        )

        certificate = build_mosaic_certificate(
            wrapped,
            sample_index=0,
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
        )
        expected = proof_only_decisions(
            output.transitions[0],
            output.log_stop_probabilities[0],
            weights,
        )
        prediction = certificate["prediction"]
        self.assertEqual(prediction["decision_rule"], "deweighted_class_map")
        self.assertEqual(
            prediction["predicted_grade"], int(expected.deweighted_argmax)
        )
        torch.testing.assert_close(
            torch.tensor(prediction["transition_weights"]), weights
        )
        self.assertTrue(verify_mosaic_certificate(certificate)["ok"])

        with self.assertRaisesRegex(ValueError, "decision_rule conflicts"):
            build_mosaic_certificate(
                wrapped,
                decision_rule="class_map",
                sufficiency_tolerance=0.05,
                complement_suppression=0.5,
            )
        with self.assertRaisesRegex(ValueError, "transition_weights conflict"):
            build_mosaic_certificate(
                wrapped,
                transition_weights=weights * 2.0,
                sufficiency_tolerance=0.05,
                complement_suppression=0.5,
            )

    def test_builder_rejects_nonpositive_transition_weights(self) -> None:
        output, valid, metadata = _example_output()
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            build_mosaic_certificate(
                output,
                lattice_metadata=metadata,
                valid_mask=valid,
                transition_weights=torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
            )

    def test_deweighted_rule_requires_explicit_transition_weights(self) -> None:
        output, valid, metadata = _example_output()
        with self.assertRaisesRegex(ValueError, "provided explicitly"):
            build_mosaic_certificate(
                output,
                lattice_metadata=metadata,
                valid_mask=valid,
                decision_rule="deweighted_class_map",
            )

    def test_replay_tolerances_cannot_exceed_audited_maxima(self) -> None:
        output, valid, metadata = _example_output()
        with self.assertRaisesRegex(ValueError, "audited maxima"):
            build_mosaic_certificate(
                output,
                lattice_metadata=metadata,
                valid_mask=valid,
                replay_atol=MAX_REPLAY_ATOL * 2.0,
            )

        certificate = build_mosaic_certificate(
            output,
            lattice_metadata=metadata,
            valid_mask=valid,
            sufficiency_tolerance=0.05,
        )
        tampered = copy.deepcopy(certificate)
        tampered["numerical_contract"]["replay_rtol"] = MAX_REPLAY_RTOL * 2.0
        tampered["integrity"]["payload_sha256"] = _payload_sha256(tampered)
        with self.assertRaisesRegex(ValueError, "audited maxima"):
            verify_mosaic_certificate(tampered)

    def test_malformed_selected_boundary_count_fails_without_crashing(self) -> None:
        output, valid, metadata = _example_output()
        certificate = build_mosaic_certificate(
            output,
            lattice_metadata=metadata,
            valid_mask=valid,
            sufficiency_tolerance=0.05,
        )
        for malformed in ([], certificate["proof"]["selected_indices"] + [[]]):
            tampered = copy.deepcopy(certificate)
            tampered["proof"]["selected_indices"] = copy.deepcopy(malformed)
            tampered["integrity"]["payload_sha256"] = _payload_sha256(tampered)
            report = verify_mosaic_certificate(tampered)
            self.assertFalse(report["ok"])
            self.assertFalse(report["checks"]["selected_indices_valid"])

    def test_weight_and_rule_tampering_fail_independent_decision_replay(self) -> None:
        output, valid, metadata = _example_output()
        certificate = build_mosaic_certificate(
            output,
            lattice_metadata=metadata,
            valid_mask=valid,
            sample_index=0,
            sufficiency_tolerance=0.05,
            complement_suppression=0.5,
            decision_rule="deweighted_class_map",
            transition_weights=torch.tensor([[0.35, 2.4], [3.1, 0.55]]),
        )

        tampered_weights = copy.deepcopy(certificate)
        tampered_weights["prediction"]["transition_weights"][0][0] *= 4.0
        # Repair the generic payload hash so this test specifically exercises
        # independent mathematical replay of the decoder.
        tampered_weights["integrity"]["payload_sha256"] = _payload_sha256(
            tampered_weights
        )
        weight_report = verify_mosaic_certificate(tampered_weights)
        self.assertFalse(weight_report["ok"])
        self.assertTrue(weight_report["checks"]["integrity_sha256"])
        self.assertFalse(
            weight_report["checks"]["deweighted_class_probabilities"]
        )

        tampered_rule = copy.deepcopy(certificate)
        tampered_rule["prediction"]["decision_rule"] = "class_map"
        tampered_rule["integrity"]["payload_sha256"] = _payload_sha256(
            tampered_rule
        )
        rule_report = verify_mosaic_certificate(tampered_rule)
        self.assertFalse(rule_report["ok"])
        self.assertTrue(rule_report["checks"]["integrity_sha256"])
        self.assertFalse(rule_report["checks"]["probability_space"])

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

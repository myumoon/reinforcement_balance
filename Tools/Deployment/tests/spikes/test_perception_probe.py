from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reinbalance_survivors_contracts.ui_intent import ContractValidationError
from spikes.perception_probe import (
    Box,
    CoordinateTransform,
    decode_frame,
    encode_near_lossless,
    encode_png,
    evaluate_probe,
    load_feasibility_config,
    make_synthetic_fixture,
)
from spikes.survivors_vertical_feasibility import (
    ActionReplayRecord,
    GateEvidence,
    issue_verdict,
    replay_action_displacement,
)


def test_bgra_png_and_near_lossless_round_trip():
    frame = np.arange(8 * 12 * 4, dtype=np.uint8).reshape(8, 12, 4)
    for encoding, payload in (
        ("raw_bgra", frame.tobytes()),
        ("png", encode_png(frame)),
        ("near_lossless", encode_near_lossless(frame)),
    ):
        actual = decode_frame(payload, encoding, width=12, height=8)
        np.testing.assert_array_equal(actual, frame)


@pytest.mark.parametrize(
    "transform",
    [
        CoordinateTransform.roi(100, 50),
        CoordinateTransform.resize((1920, 1080), (640, 360)),
        CoordinateTransform.letterbox((1920, 1080), (640, 640)),
        CoordinateTransform.tile(2, 1, tile_width=960, tile_height=540),
    ],
)
def test_all_coordinate_transforms_are_reversible(transform):
    box = Box(240.25, 160.5, 480.75, 360.25)
    assert transform.inverse_box(transform.forward_box(box)).almost_equals(box)


def test_synthetic_probe_is_reproducible_and_reports_required_slices():
    samples = make_synthetic_fixture(seed=17)
    first = evaluate_probe(samples)
    second = evaluate_probe(samples)
    assert first == second
    assert first["pixel_size"]["p10_short_side"] > 0
    assert set(first["recall_upper_bound"]) >= {
        "small", "occluded", "late", "heavy", "boss", "gem"
    }
    assert set(first["architectures"]) == {
        "ssdlite320", "ssdlite640_multiscale", "tile_2x2", "coarse_density"
    }
    assert first["latency_ms"]["p95"] >= first["latency_ms"]["median"]


@pytest.mark.parametrize(
    "evidence_update,reason",
    [
        ({"build_ids": ("a", "b")}, "mixed_identity"),
        ({"slice_counts": {"early": 2}}, "missing_required_slices"),
        ({"independent_annotators": ("alice", "alice")}, "independent_annotation"),
    ],
)
def test_verdict_is_fail_closed(evidence_update, reason):
    evidence = GateEvidence.valid_fixture()
    evidence = evidence.replace(**evidence_update)
    verdict = issue_verdict(evidence, load_feasibility_config())
    assert verdict["status"] == "FAIL"
    assert any(reason in item for item in verdict["fail_reasons"])
    assert verdict["downstream"]["allow_04_01"] is False
    assert verdict["downstream"]["allow_long_run_student"] is False


def test_config_and_verdict_wire_reject_unknown_fields(tmp_path):
    config_path = Path(__file__).parents[2] / "configs" / "perception_feasibility_v1.yaml"
    data = json.loads(json.dumps(load_feasibility_config(config_path).to_wire()))
    data["surprise"] = True
    with pytest.raises(ContractValidationError):
        load_feasibility_config(data)

    wire = GateEvidence.valid_fixture().to_wire()
    wire["surprise"] = True
    with pytest.raises(ContractValidationError):
        GateEvidence.from_wire(wire)


def test_action_replay_keeps_proposal_and_measurement_distinct():
    fixture = Path(__file__).parents[2] / "configs" / "golden" / "action_displacement_wasd_v1.json"
    records = replay_action_displacement(fixture)
    assert len(records) == 9
    assert all(isinstance(record, ActionReplayRecord) for record in records)
    assert all(record.proposal_vector is not record.measured_screen_displacement for record in records)
    assert records[0].proposal_vector == (0.0, 1.0)
    assert records[0].measured_screen_displacement == (0.0, -1.0)
    assert all(record.live_input_sent is False for record in records)

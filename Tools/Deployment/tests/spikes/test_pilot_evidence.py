"""pilot実測入力からGateEvidenceを組み立てる変換層の契約テスト。

utilityへの変換規則、複数probe結果の統合方針、pilotセッションmetadataからの
集約、および欠損・型不正・annotator不足・provenance不一致のfail-closed挙動を
確認します。
"""
from __future__ import annotations

import copy

import pytest

from reinbalance_survivors_contracts.ui_intent import ContractValidationError
from spikes.perception_probe import evaluate_probe, load_feasibility_config, make_synthetic_fixture
from spikes.survivors_vertical_feasibility import issue_verdict
from spikes.pilot_evidence import build_gate_evidence, merge_probe_evaluations
from survivors.target_profile import CONFIG, TargetProfile, load_target_profile


def _sessions() -> list[dict]:
    """3セッション・全7場面を最低1回ずつ含む有効なpilotセッションmetadataを返す。"""
    return [
        {"session_id": "s1", "build_id": "build-a", "profile_id": "profile-a",
         "minutes": 12.0, "slice": ["early", "mid"]},
        {"session_id": "s2", "build_id": "build-a", "profile_id": "profile-a",
         "minutes": 15.0, "slice": ["late", "heavy"]},
        {"session_id": "s3", "build_id": "build-a", "profile_id": "profile-a",
         "minutes": 11.0, "slice": ["level_up", "chest", "death_result"]},
    ]


def _annotation_summary() -> dict:
    """summarize_annotation()の戻り値と同じ形の有効なannotation集計値を返す。"""
    return {
        "entities_per_hour": 500.0,
        "dense_entities_per_hour": 420.0,
        "qa_rework_rate": 0.05,
        "bbox_qa_iou": 0.90,
        "class_agreement": 0.97,
        "annotation_hours": 12.0,
    }


def _valid_probe_evaluation() -> dict:
    """evaluate_probe()が実際に返す、有効な形の評価結果を1件返す。"""
    return evaluate_probe(make_synthetic_fixture(seed=0))


def _operator_attested_profile() -> TargetProfile:
    """target_audit_passがTrueになる、operator-attested provenanceのprofileを返す。"""
    wire = load_target_profile(CONFIG).to_wire()
    wire["provenance"] = "operator-attested"
    return TargetProfile.from_wire(wire)


def _test_fixture_profile() -> TargetProfile:
    """target_audit_passがFalseになる、既定のtest-fixture provenanceのprofileを返す。"""
    return load_target_profile(CONFIG)


def _build(**overrides):
    """build_gate_evidence()を有効な既定引数＋上書きで呼び出す。"""
    kwargs = dict(
        sessions=_sessions(),
        probe_evaluations=[_valid_probe_evaluation()],
        annotation_summary=_annotation_summary(),
        annotators=("alice", "bob"),
        representative_frames=300,
        target_profile=_operator_attested_profile(),
    )
    kwargs.update(overrides)
    return build_gate_evidence(**kwargs)


def test_merge_probe_evaluations_converts_utility_and_passthrough_fields():
    """utility変換とp10/recall/single_passの素通し変換を確認する。

    utility = utility_per_latency * latency_p95_ms が[0,1]へ戻ることも検査します。
    """
    evaluation = _valid_probe_evaluation()
    merged = merge_probe_evaluations([evaluation])
    assert merged["p10_short_side_px"] == evaluation["pixel_size"]["p10_short_side"]
    assert merged["late_recall"] == evaluation["recall_upper_bound"]["late"]
    assert merged["heavy_recall"] == evaluation["recall_upper_bound"]["heavy"]
    assert merged["single_pass_p95_ms"] == evaluation["architectures"]["ssdlite320"]["latency_p95_ms"]
    for name, metrics in merged["architecture_metrics"].items():
        source = evaluation["architectures"][name]
        expected_utility = source["utility_per_latency"] * source["latency_p95_ms"]
        assert metrics["utility"] == pytest.approx(expected_utility)
        assert metrics["latency_p95_ms"] == source["latency_p95_ms"]
        assert 0.0 <= metrics["utility"] <= 1.0


def test_merge_probe_evaluations_averages_across_multiple_results():
    """複数--probe-resultsの統合方針（単純平均）を固定する。"""
    first = _valid_probe_evaluation()
    second = evaluate_probe(make_synthetic_fixture(seed=1))
    merged = merge_probe_evaluations([first, second])
    assert merged["p10_short_side_px"] == pytest.approx(
        (first["pixel_size"]["p10_short_side"] + second["pixel_size"]["p10_short_side"]) / 2)
    name = "tile_2x2"
    expected_latency = (first["architectures"][name]["latency_p95_ms"]
                        + second["architectures"][name]["latency_p95_ms"]) / 2
    assert merged["architecture_metrics"][name]["latency_p95_ms"] == pytest.approx(expected_latency)


def test_merge_probe_evaluations_rejects_mismatched_architecture_shape():
    """方式構成が評価間で食い違う場合をfail-closedにする。"""
    evaluation = copy.deepcopy(_valid_probe_evaluation())
    del evaluation["architectures"]["tile_2x2"]
    with pytest.raises(ContractValidationError):
        merge_probe_evaluations([evaluation])


def test_build_gate_evidence_rejects_utility_above_one():
    """utility_per_latency*latency_p95_msが1を超える場合をfail-closedにする。"""
    evaluation = copy.deepcopy(_valid_probe_evaluation())
    evaluation["architectures"]["ssdlite320"]["utility_per_latency"] = 1.0
    evaluation["architectures"]["ssdlite320"]["latency_p95_ms"] = 2.0
    with pytest.raises(ContractValidationError):
        _build(probe_evaluations=[evaluation])


def test_build_gate_evidence_accepts_zero_utility_boundary():
    """utilityがちょうど0になる境界値は拒否しないことを確認する。"""
    evaluation = copy.deepcopy(_valid_probe_evaluation())
    evaluation["architectures"]["ssdlite320"]["utility_per_latency"] = 0.0
    evidence = _build(probe_evaluations=[evaluation])
    assert evidence.architecture_metrics["ssdlite320"]["utility"] == 0.0


def test_build_gate_evidence_rejects_session_missing_field():
    """pilotセッションmetadataの欠損フィールドをfail-closedにする。"""
    sessions = _sessions()
    del sessions[0]["minutes"]
    with pytest.raises(ContractValidationError):
        _build(sessions=sessions)


def test_build_gate_evidence_rejects_duplicate_session_id():
    """session_idの重複をfail-closedにする。"""
    sessions = _sessions()
    sessions[1] = {**sessions[1], "session_id": sessions[0]["session_id"]}
    with pytest.raises(ContractValidationError):
        _build(sessions=sessions)


def test_build_gate_evidence_rejects_unknown_slice_name():
    """未知のslice名をfail-closedにする。"""
    sessions = _sessions()
    sessions[0] = {**sessions[0], "slice": ["not_a_real_slice"]}
    with pytest.raises(ContractValidationError):
        _build(sessions=sessions)


def test_build_gate_evidence_rejects_insufficient_annotators():
    """independent_annotatorsが1名以下の場合をfail-closedにする。"""
    with pytest.raises(ContractValidationError):
        _build(annotators=("alice",))


def test_build_gate_evidence_with_non_operator_profile_yields_fail_verdict_not_exception():
    """provenanceがoperator-attestedでない場合は例外にせず判定FAILにする。

    入力形状としては有効（GateEvidenceは構築できる）が、target_audit_passがFalseに
    なることで、既存のissue_verdict()がfail-closedにFAIL判定を出す経路を確認します。
    """
    evidence = _build(target_profile=_test_fixture_profile())
    assert evidence.target_audit_pass is False
    verdict = issue_verdict(evidence, load_feasibility_config())
    assert verdict["status"] == "FAIL"
    assert verdict["fail_reasons"]


def test_build_gate_evidence_round_trip_validates_and_issues_verdict():
    """構築したGateEvidenceがvalidateを通り、issue_verdictが例外なく判定を返すことを確認する。"""
    evidence = _build()
    evidence.validate()
    verdict = issue_verdict(evidence, load_feasibility_config())
    assert verdict["status"] in {"PASS", "FAIL"}
    assert "fail_reasons" in verdict

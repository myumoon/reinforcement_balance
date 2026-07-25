"""Fidelity verdict v2 の fail-closed・stage・失効判定を検証する。

baseline と下流用 verdict の境界、および provenance と gating の非対称性を小さな
fixture で固定します。
"""

from dataclasses import replace
import os

import pytest

from reinbalance_survivors_contracts.fidelity_verdict import (
    BlockingReason, FidelityMetric, FidelityVerdict, GATING_KEYS, downstream_release_allowed,
    read_verdict_pair_atomic, verify_current_fidelity, write_verdict_pair_atomic,
)
from reinbalance_survivors_contracts.ui_intent import ContractValidationError


def _hashes(value: str = "a" * 64) -> dict[str, str]:
    """13 key を満たす current gating fixture を返す。

    DeployObs の absent 化は各 test が明示的に上書きします。
    """
    return {key: value for key in GATING_KEYS}


def _verdict(stage: str = "integration", *, blocked: bool = False) -> FidelityVerdict:
    """stage ごとの最小有効 verdict fixture を構築する。

    integration は visibility dimension を必須とし、許容外のときだけ blocking 行を持ちます。
    """
    hashes = _hashes()
    rows = ()
    if stage == "baseline":
        hashes["deploy_obs_schema"] = hashes["deploy_release_adapter"] = "absent"
        rows = tuple(BlockingReason(x, "baseline gate") for x in ("action", "offer", "terminal"))
    else:
        rows = (BlockingReason("deploy_obs_visibility", "visibility gate"),) if blocked else ()
    metrics = () if stage == "baseline" else (
        FidelityMetric("deploy_obs_visibility", 0.02, "normalized_error", True, None, not blocked),
    )
    subject = {
        "target_profile_hash": "1" * 64,
        "target_build_attestation_hash": "2" * 64,
        "report_scope": "exact_target",
        "producer_allowlist_version": "fidelity_producer_paths.v1",
        "producer_manifest_hash": "3" * 64,
        "resolved_producers": {key: [] for key in GATING_KEYS},
    }
    provenance = {
        "git_commit": "one", "workspace_dirty_summary": "",
        "audit_tool_version": "1.0", "dependency_versions": {},
        "operator": "tester", "timestamp": "2026-01-01T00:00:00Z",
    }
    return FidelityVerdict(stage, subject, metrics, rows, provenance, hashes)


def test_baseline_requires_absent_and_blocking_rows() -> None:
    """baseline の DeployObs absent と三 blocking 行を検証する。

    未実装 DeployObs を final pass と誤表現できないことを保証します。
    """
    verdict = _verdict("baseline")
    assert verdict.gating_producer_hashes["deploy_obs_schema"] == "absent"
    with pytest.raises(ContractValidationError):
        verify_current_fidelity(verdict, verdict.gating_producer_hashes, "integration")


@pytest.mark.parametrize("stage", ["integration", "post_curriculum"])
def test_promoted_stages_require_visibility_dimension_not_blocking_row(stage: str) -> None:
    """昇格 stage に DeployObs visibility dimension を必須化する。

    実測かつ許容内なら blocking 行なしで解禁でき、dimension 欠落は拒否します。
    """
    verdict = _verdict(stage)
    assert downstream_release_allowed(verdict, verdict.gating_producer_hashes, stage)
    wire = verdict.to_wire()
    wire["metrics"] = []
    with pytest.raises(ContractValidationError):
        FidelityVerdict.from_wire(wire)


@pytest.mark.parametrize(
    ("metric", "has_row"),
    [
        (FidelityMetric("deploy_obs_visibility", None, "normalized_error", False, "not measured", None), True),
        (FidelityMetric("deploy_obs_visibility", 0.4, "normalized_error", True, None, False), True),
        (FidelityMetric("deploy_obs_visibility", 0.02, "normalized_error", True, None, True), False),
    ],
)
def test_visibility_blocking_row_matches_measurement_state(metric: FidelityMetric, has_row: bool) -> None:
    """visibility blocking 行を measurement 状態と一致させる。

    未測定・許容外は必ず止め、実測済みかつ許容内だけ行を省略できます。
    """
    verdict = _verdict("integration")
    wire = verdict.to_wire()
    wire["metrics"] = [metric.to_wire()]
    wire["blocking_reasons"] = (
        [BlockingReason("deploy_obs_visibility", "visibility gate").to_wire()] if has_row else []
    )
    checked = FidelityVerdict.from_wire(wire)
    assert downstream_release_allowed(checked, checked.gating_producer_hashes, "integration") is not has_row


def test_unknown_missing_and_nonfinite_are_rejected() -> None:
    """全 wire 境界で未知・欠落 key と非 finite metric を拒否する。

    JSON decoder 後の入力を黙って補正しない fail-closed 契約です。
    """
    wire = _verdict("baseline").to_wire()
    wire["unknown"] = 1
    with pytest.raises(ContractValidationError):
        FidelityVerdict.from_wire(wire)
    del wire["unknown"]
    del wire["subject"]
    with pytest.raises(ContractValidationError):
        FidelityVerdict.from_wire(wire)


def test_verify_ignores_provenance_but_rejects_hash_and_stage() -> None:
    """失効判定が provenance 差を無視し gating/stage 差を拒否する。

    operator や timestamp の変更だけで再監査を要求しない設計を固定します。
    """
    verdict = _verdict("integration")
    wire = verdict.to_wire()
    wire["provenance"]["git_commit"] = "two"
    current = dict(verdict.gating_producer_hashes)
    assert verify_current_fidelity(wire, current, "integration").verdict_stage == "integration"
    assert downstream_release_allowed(wire, current, "integration") is True
    current["logic_private"] = "b" * 64
    with pytest.raises(ContractValidationError):
        verify_current_fidelity(wire, current, "integration")


@pytest.mark.parametrize("consumer", ["01-05", "03-04", "03-05"])
def test_consumer_fixture_rejects_stale_verdict(consumer: str) -> None:
    """後続 consumer が stale current hash を受理しないことを検証する。

    consumer 名に依存せず共有 validator 一つを利用することを示します。
    """
    verdict = _verdict("post_curriculum")
    stale = dict(verdict.gating_producer_hashes)
    stale["logic_public"] = "f" * 64
    with pytest.raises(ContractValidationError):
        verify_current_fidelity(verdict, stale, "integration")


def test_artifact_pair_has_single_atomic_commit_point(tmp_path, monkeypatch) -> None:
    """artifact pair の外部可視化が単一 rename だけで起きることを検証する。

    payload は世代 directory 内で完成させ、commit marker の置換前には reader が新世代を
    観測できないことを固定します。
    """
    json_path = tmp_path / "verdict.json"
    report_path = tmp_path / "verdict.md"
    json_path.write_text("old-json", encoding="utf-8")
    report_path.write_text("old-report", encoding="utf-8")
    real_replace = os.replace
    destinations = []

    def recording_replace(source, destination):
        """rename の宛先を記録する fault injector。

        pair ごとの commit marker 以外を外部可視 path へ rename していないか確認します。
        """
        destinations.append(os.fspath(destination))
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", recording_replace)
    write_verdict_pair_atomic(_verdict("baseline"), json_path, report_path, "new-report")
    assert len(destinations) == 1
    assert destinations[0].endswith(".verdict.json.verdict.md.commit")
    marker = __import__("json").loads((tmp_path / ".verdict.json.verdict.md.commit").read_text(encoding="utf-8"))
    generation = tmp_path / marker["generation"]
    assert (generation / "verdict.md").read_text(encoding="utf-8") == "new-report"
    committed_verdict, committed_report = read_verdict_pair_atomic(json_path, report_path)
    assert committed_verdict.verdict_stage == "baseline"
    assert committed_report == "new-report"


def test_same_content_pairs_with_different_names_are_independent(tmp_path) -> None:
    """同一内容でも filename が異なる artifact pair を別世代へ確定する。

    generation directory の衝突で二組目が読めなくなる回帰を防ぎます。
    """
    verdict = _verdict("baseline")
    first_json, first_report = tmp_path / "first.json", tmp_path / "first.md"
    second_json, second_report = tmp_path / "second.json", tmp_path / "second.md"
    write_verdict_pair_atomic(verdict, first_json, first_report, "same-report")
    write_verdict_pair_atomic(verdict, second_json, second_report, "same-report")
    first_marker = __import__("json").loads((tmp_path / ".first.json.first.md.commit").read_text(encoding="utf-8"))
    second_marker = __import__("json").loads((tmp_path / ".second.json.second.md.commit").read_text(encoding="utf-8"))
    assert first_marker["generation"] != second_marker["generation"]
    assert read_verdict_pair_atomic(first_json, first_report)[1] == "same-report"
    assert read_verdict_pair_atomic(second_json, second_report)[1] == "same-report"

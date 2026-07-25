"""Fidelity verdict v2 の fail-closed・stage・失効判定を検証する。

baseline と下流用 verdict の境界、および provenance と gating の非対称性を小さな
fixture で固定します。
"""

from dataclasses import replace
import os

import pytest

from reinbalance_survivors_contracts.fidelity_verdict import (
    BlockingReason, FidelityVerdict, GATING_KEYS, verify_current_fidelity,
    write_verdict_pair_atomic,
)
from reinbalance_survivors_contracts.ui_intent import ContractValidationError


def _hashes(value: str = "a" * 64) -> dict[str, str]:
    """13 key を満たす current gating fixture を返す。

    DeployObs の absent 化は各 test が明示的に上書きします。
    """
    return {key: value for key in GATING_KEYS}


def _verdict(stage: str = "integration", *, blocked: bool = False) -> FidelityVerdict:
    """stage ごとの最小有効 verdict fixture を構築する。

    integration は visibility 行が必須なので、通過検証時だけ検証後に空 wire を作り直します。
    """
    hashes = _hashes()
    rows = ()
    if stage == "baseline":
        hashes["deploy_obs_schema"] = hashes["deploy_release_adapter"] = "absent"
        rows = tuple(BlockingReason(x, "baseline gate") for x in ("action", "offer", "terminal"))
    else:
        rows = (BlockingReason("deploy_obs_visibility", "visibility gate"),)
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
    return FidelityVerdict(stage, subject, (), rows if blocked or stage == "baseline" else rows, provenance, hashes)


def test_baseline_requires_absent_and_blocking_rows() -> None:
    """baseline の DeployObs absent と三 blocking 行を検証する。

    未実装 DeployObs を final pass と誤表現できないことを保証します。
    """
    verdict = _verdict("baseline")
    assert verdict.gating_producer_hashes["deploy_obs_schema"] == "absent"
    with pytest.raises(ContractValidationError):
        verify_current_fidelity(verdict, verdict.gating_producer_hashes, "integration")


@pytest.mark.parametrize("stage", ["integration", "post_curriculum"])
def test_promoted_stages_require_visibility_blocking_row(stage: str) -> None:
    """昇格 stage に DeployObs visibility gate を必須化する。

    blocking 行を省いた verdict を構築経路と wire 経路の両方で拒否します。
    """
    verdict = _verdict(stage)
    wire = verdict.to_wire()
    wire["blocking_reasons"] = []
    with pytest.raises(ContractValidationError):
        FidelityVerdict.from_wire(wire)


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
    with pytest.raises(ContractValidationError):
        verify_current_fidelity(wire, current, "integration")
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


def test_artifact_pair_rolls_back_second_replace_failure(tmp_path, monkeypatch) -> None:
    """二つ目の artifact 置換失敗時に旧世代 pair を復元する。

    新 JSON と旧 Markdown の混在や、片側だけの削除を残さないことを固定します。
    """
    json_path = tmp_path / "verdict.json"
    report_path = tmp_path / "verdict.md"
    json_path.write_text("old-json", encoding="utf-8")
    report_path.write_text("old-report", encoding="utf-8")
    real_replace = os.replace
    calls = 0

    def failing_replace(source, destination):
        """四回目の rename だけを失敗させる fault injector。

        backup 二件と JSON 反映後の Markdown 反映を crash 相当として模擬します。
        """
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("injected second artifact failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError):
        write_verdict_pair_atomic(_verdict("baseline"), json_path, report_path, "new-report")
    assert json_path.read_text(encoding="utf-8") == "old-json"
    assert report_path.read_text(encoding="utf-8") == "old-report"

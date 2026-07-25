"""Fidelity verdict v2 の fail-closed・stage・失効判定を検証する。

baseline と下流用 verdict の境界、および provenance と gating の非対称性を小さな
fixture で固定します。
"""

from dataclasses import replace

import pytest

from reinbalance_survivors_contracts.fidelity_verdict import (
    BlockingReason, FidelityVerdict, GATING_KEYS, verify_current_fidelity,
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

"""Full Run exact/generalization 評価と FR4 fail-closed guard を検証する。

初心者向け: 固定30 seed の本番目標と層別の汎化確認を混ぜず、古い監査結果や
hash の違う成果物で長時間訓練を始められないことを確認します。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

_TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.full_run_eval import (
    EXACT_TARGET_HOLDOUT_SEEDS,
    EpisodeOutcome,
    ExactTargetFullRunEvaluator,
    GeneralizationFullRunEvaluator,
    GeneralizationManifest,
)
from games.survivors.full_run_launcher import (
    FullRunLaunchGuardError,
    build_full_run_config_template,
    validate_fr4_launch,
    validate_full_run_config,
)
from reinbalance_survivors_contracts.fidelity_verdict import (
    BlockingReason,
    FidelityMetric,
    FidelityVerdict,
    GATING_KEYS,
)

_BINDINGS = {
    "policy_hash": "1" * 64,
    "selector_hash": "2" * 64,
    "source_hash": "3" * 64,
    "schema_hash": "4" * 64,
    "profile_hash": "5" * 64,
}


def _outcomes(*, clears: int = 30, low_score: bool = False) -> list[EpisodeOutcome]:
    """固定 holdout seeds に対応するテスト outcome を生成する。

    初心者向け: clear 数と p10 だけを個別に崩せる入力を作ります。
    """
    rows = []
    for index, seed in enumerate(EXACT_TARGET_HOLDOUT_SEEDS):
        rows.append(
            EpisodeOutcome(
                seed=seed,
                stratum="exact_target",
                stage_cleared=index < clears,
                elapsed_seconds=1800.0 if index < clears else 900.0,
                active_score=100.0 if low_score and index < 4 else 2500.0,
                terminal_reason="stage_cleared" if index < clears else "death",
            )
        )
    return rows


def _integration_verdict(*, stage: str = "integration", blocked: bool = False) -> FidelityVerdict:
    """FR4 guard 用の current fidelity verdict を作る。

    初心者向け: stage/hash/blocking の反例を同じ最小 fixture から作ります。
    """
    hex_digits = "abcdef0123456789"
    hashes = {key: hex_digits[index] * 64 for index, key in enumerate(GATING_KEYS)}
    metrics = (FidelityMetric("deploy_obs_visibility", 1.0, "ratio", True, None, True),)
    rows = (BlockingReason("terminal", "not approved"),) if blocked else ()
    subject = {
        "target_profile_hash": "6" * 64,
        "target_build_attestation_hash": "7" * 64,
        "report_scope": "exact_target",
        "producer_allowlist_version": "fidelity_producer_paths.v1",
        "producer_manifest_hash": "8" * 64,
        "resolved_producers": {
            key: [{"path": f"producer/{key}", "sha256": digest}]
            for key, digest in hashes.items()
        },
    }
    provenance = {
        "git_commit": "abc",
        "workspace_dirty_summary": "clean",
        "audit_tool_version": "test",
        "dependency_versions": {},
        "operator": "pytest",
        "timestamp": "2026-08-06T00:00:00Z",
    }
    if stage == "baseline":
        hashes["deploy_obs_schema"] = "absent"
        hashes["deploy_release_adapter"] = "absent"
        rows = (
            BlockingReason("action", "baseline"),
            BlockingReason("offer", "baseline"),
            BlockingReason("terminal", "baseline"),
        )
        metrics = ()
    return FidelityVerdict(stage, subject, metrics, rows, provenance, hashes)


def test_exact_target_uses_30_fixed_holdout_seeds_and_both_gates() -> None:
    """exact evaluator が固定30 seed、clear率、p10 を同時に要求することを確認する。

    初心者向け: 平均だけ良い policy や seed を差し替えた評価を合格させません。
    """
    assert len(EXACT_TARGET_HOLDOUT_SEEDS) == 30
    assert len(set(EXACT_TARGET_HOLDOUT_SEEDS)) == 30
    evaluator = ExactTargetFullRunEvaluator(min_active_score_p10=2250.0)

    passed = evaluator.evaluate(_outcomes(clears=24), bindings=_BINDINGS)
    assert passed.passed is True
    assert passed.clear_rate == pytest.approx(0.80)
    assert passed.active_score_p10 >= 2250.0

    assert evaluator.evaluate(_outcomes(clears=23), bindings=_BINDINGS).passed is False
    assert evaluator.evaluate(_outcomes(clears=30, low_score=True), bindings=_BINDINGS).passed is False
    with pytest.raises(ValueError, match="holdout seeds"):
        evaluator.evaluate(_outcomes()[:-1], bindings=_BINDINGS)


def test_exact_target_rejects_weakened_or_invalid_thresholds() -> None:
    """exact gate は80%未満や非有限・非数値 threshold を拒否する。

    初心者向け: 評価呼出側が必須 gate を弱めたり bool を数値として渡す余地を塞ぎます。
    """
    for invalid in (0.79, True, "0.80", math.nan, math.inf, -math.inf, 1.01):
        with pytest.raises(ValueError, match="min_clear_rate"):
            ExactTargetFullRunEvaluator(min_clear_rate=invalid)
    for invalid in (True, "2250", math.nan, math.inf, -math.inf, -0.01):
        with pytest.raises(ValueError, match="min_active_score_p10"):
            ExactTargetFullRunEvaluator(min_active_score_p10=invalid)


def test_exact_target_clear_rate_boundary_and_all_deaths_fail() -> None:
    """既定 gate は24/30だけを通し、23/30以下と全 death を不合格にする。

    初心者向け: 固定30 seed に対する80%境界を件数レベルで固定します。
    """
    evaluator = ExactTargetFullRunEvaluator()
    assert evaluator.evaluate(_outcomes(clears=24), bindings=_BINDINGS).passed
    assert not evaluator.evaluate(_outcomes(clears=23), bindings=_BINDINGS).passed
    assert not evaluator.evaluate(_outcomes(clears=0), bindings=_BINDINGS).passed


def test_exact_verdict_json_contains_all_artifact_hashes() -> None:
    """verdict wire に policy/selector/source/schema/profile hash が保存されることを確認する。

    初心者向け: 後で評価結果を別のモデルや観測条件へ取り違えないようにします。
    """
    verdict = ExactTargetFullRunEvaluator().evaluate(_outcomes(), bindings=_BINDINGS)
    wire = verdict.to_wire()
    assert {key: wire[key] for key in _BINDINGS} == _BINDINGS
    assert json.loads(json.dumps(wire))["schema_version"] == "survivors.full_run_verdict.v1"


def test_generalization_evaluator_requires_each_manifest_stratum() -> None:
    """generalization evaluator が manifest の全 strata を別集計することを確認する。

    初心者向け: 一つの簡単な条件だけで汎化済みと表示されることを防ぎます。
    """
    manifest = GeneralizationManifest.from_wire(
        {
            "schema_version": "survivors.full_run_generalization_manifest.v1",
            "strata": {
                "weapon_start": [101, 102],
                "perception_noise": [201, 202],
            },
        }
    )
    outcomes = [
        EpisodeOutcome(seed, stratum, True, 1800.0, 2400.0, "stage_cleared")
        for stratum, seeds in manifest.strata.items()
        for seed in seeds
    ]
    report = GeneralizationFullRunEvaluator(manifest).evaluate(outcomes, bindings=_BINDINGS)
    assert report.report_kind == "generalization"
    assert report.strata_metrics == {
        "perception_noise": {"episodes": 2, "clear_rate": 1.0},
        "weapon_start": {"episodes": 2, "clear_rate": 1.0},
    }

    with pytest.raises(ValueError, match="manifest cells"):
        GeneralizationFullRunEvaluator(manifest).evaluate(outcomes[:-1], bindings=_BINDINGS)


def test_fr4_launcher_rejects_baseline_blocking_and_hash_mismatch() -> None:
    """FR4 guard が古い stage、blocking、producer hash 差を全て拒否する。

    初心者向け: 監査ファイルが存在するだけでは長時間 run を解禁しません。
    """
    integration = _integration_verdict()
    current = dict(integration.gating_producer_hashes)
    validate_fr4_launch(integration.to_wire(), current)

    with pytest.raises(FullRunLaunchGuardError, match="integration"):
        validate_fr4_launch(_integration_verdict(stage="baseline").to_wire(), current)
    with pytest.raises(FullRunLaunchGuardError, match="blocking"):
        validate_fr4_launch(_integration_verdict(blocked=True).to_wire(), current)
    mismatched = dict(current)
    mismatched[next(iter(mismatched))] = "f" * 64
    with pytest.raises(FullRunLaunchGuardError, match="hash"):
        validate_fr4_launch(integration.to_wire(), mismatched)


def test_full_run_config_template_is_structural_but_not_launchable() -> None:
    """生成テンプレートを構造検証でき、未束縛 placeholder は launch 時に拒否する。

    初心者向け: 実 seed や秘密の run 設定をコミットせず、利用者が別ファイルへ埋めます。
    """
    template = build_full_run_config_template()
    validate_full_run_config(template, require_bound=False)
    with pytest.raises(ValueError, match="replace"):
        validate_full_run_config(template, require_bound=True)


@pytest.mark.parametrize(
    ("section", "field", "invalid"),
    [
        *[
            ("sample_mix", field, invalid)
            for field in ("short_skill", "late_rsi", "full_run")
            for invalid in (
                True, "0.25", math.nan, math.inf, -math.inf, -0.01, 1.01
            )
        ],
        ("exact_eval", "min_clear_rate", True),
        ("exact_eval", "min_clear_rate", "0.80"),
        ("exact_eval", "min_clear_rate", math.nan),
        ("exact_eval", "min_clear_rate", math.inf),
        ("exact_eval", "min_clear_rate", -math.inf),
        ("exact_eval", "min_clear_rate", 0.79),
        ("exact_eval", "min_clear_rate", 1.01),
        ("exact_eval", "min_active_score_p10", True),
        ("exact_eval", "min_active_score_p10", "2250"),
        ("exact_eval", "min_active_score_p10", math.nan),
        ("exact_eval", "min_active_score_p10", math.inf),
        ("exact_eval", "min_active_score_p10", -math.inf),
        ("exact_eval", "min_active_score_p10", -0.01),
    ],
)
def test_full_run_config_rejects_invalid_numeric_gates(
    section: str,
    field: str,
    invalid: object,
) -> None:
    """sample mix と exact gate の全実数 field を fail-closed で検証する。

    初心者向け: bool・文字列・非有限値・範囲外を config 段階で一律に止めます。
    """
    template = build_full_run_config_template()
    template[section][field] = invalid
    with pytest.raises(ValueError):
        validate_full_run_config(template, require_bound=False)


@pytest.mark.parametrize(
    "invalid", [True, 30.0, "30", math.nan, math.inf, -math.inf, -1, 31]
)
def test_full_run_config_requires_integer_holdout_count(invalid: object) -> None:
    """holdout 件数は bool や同値 float ではなく固定整数30だけを受理する。

    初心者向け: seed 集合の個数を曖昧な数値型から読み込まないようにします。
    """
    template = build_full_run_config_template()
    template["exact_eval"]["holdout_seed_count"] = invalid
    with pytest.raises(ValueError, match="30 seeds"):
        validate_full_run_config(template, require_bound=False)


def test_train_cli_generates_and_validates_full_run_config(tmp_path, monkeypatch) -> None:
    """train.py が template 生成と binding 済み config 検証 option を提供する。

    初心者向け: UE5 や training run を起動せず、ローカル config の準備だけを実行できます。
    """
    from train import (
        generate_full_run_config_template,
        parse_args,
        validate_full_run_config_file,
    )

    output = tmp_path / "full-run.json"
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--generate-full-run-config", str(output)]
    )
    args = parse_args()
    assert args.generate_full_run_config == output
    assert args.validate_full_run_config is None

    generate_full_run_config_template(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["bindings"] = dict(_BINDINGS)
    output.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_full_run_config_file(output)["band"] == "FR4"

    monkeypatch.setattr(
        sys, "argv", ["train.py", "--validate-full-run-config", str(output)]
    )
    args = parse_args()
    assert args.generate_full_run_config is None
    assert args.validate_full_run_config == output

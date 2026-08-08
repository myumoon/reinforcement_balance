"""FR4 config template と current-fidelity launch guard を提供する。

初心者向け: 実 run 設定をリポジトリに置かずに雛形を作り、現在の producer hash と
一致する integration verdict が合格している場合だけ FR4 接続点を解禁します。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from games.survivors.modules.task_cell_sampler_module import DEFAULT_FULL_RUN_SAMPLE_MIX
from reinbalance_survivors_contracts.fidelity_verdict import (
    FidelityVerdict,
    verify_current_fidelity,
)
from reinbalance_survivors_contracts.ui_intent import ContractValidationError

FULL_RUN_CONFIG_SCHEMA_VERSION = "survivors.full_run_config.v1"
_HASH_FIELDS = (
    "policy_hash", "selector_hash", "source_hash", "schema_hash", "profile_hash",
)
_PLACEHOLDER = "<replace-with-lowercase-sha256>"


def _finite_config_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """config の有限実数を bool 除外と範囲指定付きで検証する。

    初心者向け: NaN や Infinity は比較をすり抜けるため、全 numeric gate で先に拒否します。
    """
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    checked = float(value)
    if minimum is not None and checked < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and checked > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return checked


class FullRunLaunchGuardError(RuntimeError):
    """FR4 の fail-closed preflight が拒否したことを表す。

    初心者向け: 通常の設定ミスと区別し、監査を更新するまで run を開始させません。
    """


def build_full_run_config_template() -> dict[str, Any]:
    """seed や実 artifact を含まない FR4 config template を返す。

    初心者向け: 利用者は生成先のローカルファイルへ hash と verdict path を記入します。
    """
    return {
        "schema_version": FULL_RUN_CONFIG_SCHEMA_VERSION,
        "band": "FR4",
        "episode_seconds": 1800,
        "sample_mix": dict(DEFAULT_FULL_RUN_SAMPLE_MIX),
        "exact_eval": {"holdout_seed_count": 30, "min_clear_rate": 0.80, "min_active_score_p10": 2250.0},
        "integration_verdict_path": "<replace-with-local-integration-verdict.json>",
        "bindings": {field: _PLACEHOLDER for field in _HASH_FIELDS},
    }


def validate_full_run_config(
    value: Any,
    *,
    require_bound: bool = True,
) -> dict[str, Any]:
    """full-run config を exact-key と固定 target 条件で検証する。

    初心者向け: template 自体は構造確認でき、launch 前には placeholder を必ず拒否します。
    """
    keys = {
        "schema_version", "band", "episode_seconds", "sample_mix", "exact_eval",
        "integration_verdict_path", "bindings",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("full-run config keys mismatch")
    if value["schema_version"] != FULL_RUN_CONFIG_SCHEMA_VERSION or value["band"] != "FR4":
        raise ValueError("full-run config must target FR4 schema v1")
    if type(value["episode_seconds"]) is not int or value["episode_seconds"] != 1800:
        raise ValueError("FR4 episode_seconds must be 1800")
    sample_mix = value["sample_mix"]
    if not isinstance(sample_mix, Mapping) or set(sample_mix) != set(DEFAULT_FULL_RUN_SAMPLE_MIX):
        raise ValueError("full-run sample_mix keys mismatch")
    checked_weights = [
        _finite_config_number(
            weight,
            f"sample_mix.{name}",
            minimum=0.0,
            maximum=1.0,
        )
        for name, weight in sample_mix.items()
    ]
    if not math.isclose(sum(checked_weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("full-run sample_mix weights must sum to 1.0")
    exact = value["exact_eval"]
    if not isinstance(exact, Mapping) or set(exact) != {"holdout_seed_count", "min_clear_rate", "min_active_score_p10"}:
        raise ValueError("full-run exact_eval keys mismatch")
    if type(exact["holdout_seed_count"]) is not int or exact["holdout_seed_count"] != 30:
        raise ValueError("full-run exact target requires exactly 30 seeds")
    _finite_config_number(
        exact["min_clear_rate"],
        "exact_eval.min_clear_rate",
        minimum=0.80,
        maximum=1.0,
    )
    _finite_config_number(
        exact["min_active_score_p10"],
        "exact_eval.min_active_score_p10",
        minimum=0.0,
    )
    if not isinstance(value["integration_verdict_path"], str) or not value["integration_verdict_path"]:
        raise ValueError("integration_verdict_path must be non-empty")
    bindings = value["bindings"]
    if not isinstance(bindings, Mapping) or set(bindings) != set(_HASH_FIELDS):
        raise ValueError("full-run bindings keys mismatch")
    if require_bound:
        for field in _HASH_FIELDS:
            digest = bindings[field]
            if not (isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)):
                raise ValueError(f"replace {field} with a lowercase SHA-256 before launch")
    return dict(value)


def write_full_run_config_template(path: Path) -> None:
    """FR4 config template を新規ファイルとして保存する。

    初心者向け: 既存の実 run 設定を誤って上書きしないよう、出力先の存在を拒否します。
    """
    if path.exists():
        raise FileExistsError(f"full-run config output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_full_run_config_template(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_fr4_launch(
    integration_verdict: FidelityVerdict | Mapping[str, Any],
    current_gating_producer_hashes: Mapping[str, str],
) -> FidelityVerdict:
    """current-hash integration verdict が FR4 を解禁できるか検証する。

    初心者向け: baseline、古い producer hash、blocking 行のどれか一つでもあれば停止します。
    """
    try:
        checked = verify_current_fidelity(
            integration_verdict,
            current_gating_producer_hashes,
            "integration",
        )
    except ContractValidationError as exc:
        message = str(exc)
        if "hash" in message:
            raise FullRunLaunchGuardError(f"FR4 producer hash guard failed: {message}") from exc
        raise FullRunLaunchGuardError(f"FR4 requires a current integration verdict: {message}") from exc
    if checked.blocking_reasons:
        raise FullRunLaunchGuardError("FR4 integration verdict still has blocking reasons")
    return checked

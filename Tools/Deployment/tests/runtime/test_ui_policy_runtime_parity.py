"""02-04 Training evaluator と 05-01 Deployment adapter が同一 package から
byte-identical canonical UiIntent を返すことを、別 subprocess/cwd 実行で検証する。

Training と Deployment を同一プロセスへ直接 import すると、そのプロセス自体が
import 分離契約の反例になる。そのため本テストは reinbalance_survivors_contracts
(Common) だけを import し、Training/Deployment の実行はどちらも別 subprocess に
委ねて、各 worker 自身の sys.modules self-check で cross-import が無いことも確認する。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from reinbalance_survivors_contracts.canonical_json import sha256_hex
from reinbalance_survivors_contracts.item_decision import CandidateFeatures, ItemDecisionFeatures

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TRAINING_ROOT = _REPO_ROOT / "Tools" / "Training"
_DEPLOYMENT_ROOT = _REPO_ROOT / "Tools" / "Deployment"
_TRAINING_WORKER = _TRAINING_ROOT / "tests" / "survivors" / "parity_training_worker.py"
_DEPLOYMENT_WORKER = Path(__file__).resolve().parent / "parity_deployment_worker.py"

_NMAX = 2
_FEATURE_SCHEMA = "context_only_v1"


def _hash(label: str) -> str:
    """テスト固有の識別子ラベルから決定的な 64 桁 hex hash を作る。"""
    return sha256_hex(label.encode("utf-8"))


def _build_case() -> dict:
    """Training/Deployment 両 worker が同一入力から decide できる shared fixture を作る。"""
    candidates = [
        CandidateFeatures(
            kind="item_card",
            item_id="wand",
            new_level=2,
            owned=True,
            is_new=False,
            is_evolve=False,
            is_union=False,
            has_prerequisite=False,
            slot_capacity=1,
        ),
        CandidateFeatures(
            kind="item_card",
            item_id="knife",
            new_level=1,
            owned=False,
            is_new=True,
            is_evolve=False,
            is_union=False,
            has_prerequisite=False,
            slot_capacity=1,
        ),
    ]
    item_context = ItemDecisionFeatures(
        decision_id=_hash("decision"),
        feature_schema=_FEATURE_SCHEMA,
        elapsed_time=125.0,
        level=6,
        hp_ratio=0.7,
        xp_ratio=0.4,
        weapon_slots=(1, 0, 0, 0, 0, 0),
        passive_slots=(0, 0, 0, 0, 0, 0),
        empty_slot_count=11,
        evolution_readiness=0.3,
        choice_count=2,
        card_mask=(True, True),
        fallback_kind="chicken",
        ui_state_validity=0.95,
        ui_state_age=0.2,
        candidates=candidates,
        max_item_cards=_NMAX,
    )
    hash_kwargs = {
        "source_snapshot_hash": item_context.decision_id,
        "source_frame_hash": _hash("frame"),
        "source_content_hash": _hash("content"),
        "ui_state_key": _hash("ui_state_key"),
        "candidate_set_hash": _hash("candidate_set"),
        "inventory_hash": _hash("inventory"),
        "decision_policy_id": "parity-policy-v1",
        "decision_rule_id": "parity-rule-v1",
        "decision_config_hash": _hash("decision_config"),
    }
    ui_presentation = {
        "snapshot_id": hash_kwargs["source_snapshot_hash"],
        "frame_id": hash_kwargs["source_frame_hash"],
        "parser_artifact_hash": _hash("parser_artifact"),
        "screen_state": "level_up_items",
        "candidate_set_hash": hash_kwargs["candidate_set_hash"],
        "inventory_hash": hash_kwargs["inventory_hash"],
        "source_content_hash": hash_kwargs["source_content_hash"],
        "ui_state_key": hash_kwargs["ui_state_key"],
        "candidates": [
            {
                "choice_id": "wand",
                "choice_index": 0,
                "semantic_kind": "item_card",
                "roi": [0.1, 0.1, 0.3, 0.3],
                "validity": True,
                "confidence": 0.9,
            },
            {
                "choice_id": "knife",
                "choice_index": 1,
                "semantic_kind": "item_card",
                "roi": [0.4, 0.1, 0.6, 0.3],
                "validity": True,
                "confidence": 0.85,
            },
        ],
    }
    return {
        "item_context": item_context.to_wire(),
        "hash_kwargs": hash_kwargs,
        "ui_presentation": ui_presentation,
    }


def _run_worker(script: Path, cwd: Path, pythonpath: Path, args: list[str]) -> None:
    """worker script を別 subprocess/cwd/PYTHONPATH で実行し、失敗時は stdout/stderr を添えて落とす。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pythonpath)
    result = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"{script.name} failed (exit {result.returncode})\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_training_and_deployment_return_byte_identical_choose_card_intent(tmp_path: Path) -> None:
    """同じ shared fixture/package から、Training と Deployment が同一 canonical bytes を返す。"""
    case = _build_case()
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(case), encoding="utf-8")

    training_work_dir = tmp_path / "training"
    training_intent = tmp_path / "training_intent.jsonl"
    training_result = tmp_path / "training_result.json"
    _run_worker(
        _TRAINING_WORKER,
        cwd=_TRAINING_ROOT,
        pythonpath=_TRAINING_ROOT,
        args=[
            "--case",
            str(case_path),
            "--work-dir",
            str(training_work_dir),
            "--intent-out",
            str(training_intent),
            "--result-out",
            str(training_result),
        ],
    )
    training_result_payload = json.loads(training_result.read_text(encoding="utf-8"))
    assert training_result_payload["forbidden_imports"] == []

    package_dir = Path(training_result_payload["package_dir"])
    deployment_intent = tmp_path / "deployment_intent.jsonl"
    deployment_result = tmp_path / "deployment_result.json"
    _run_worker(
        _DEPLOYMENT_WORKER,
        cwd=_DEPLOYMENT_ROOT,
        pythonpath=_DEPLOYMENT_ROOT,
        args=[
            "--case",
            str(case_path),
            "--package",
            str(package_dir),
            "--intent-out",
            str(deployment_intent),
            "--result-out",
            str(deployment_result),
        ],
    )
    deployment_result_payload = json.loads(deployment_result.read_text(encoding="utf-8"))
    assert deployment_result_payload["forbidden_imports"] == []

    training_bytes = training_intent.read_bytes()
    deployment_bytes = deployment_intent.read_bytes()
    assert training_bytes == deployment_bytes
    assert training_bytes

"""tests/survivors/test_bootstrap_eval.py

bootstrap_eval の cell spec parser / eval params builder / summary のユニットテスト。
UE5 起動は不要（純粋ロジックのみ）。
"""
from __future__ import annotations

import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import pytest

from games.survivors.bootstrap_eval import (
    BootstrapEvalCell,
    build_eval_params,
    parse_cell_spec,
    summarize_eval_results,
)
from games.survivors.survivors_weapon_curriculum import WeaponType


# ---------------------------------------------------------------------------
# cell spec parser
# ---------------------------------------------------------------------------

def test_parse_cell_spec_known_weapon_and_task_kind():
    cell = parse_cell_spec("king_bible:solo_bootstrap:2")
    assert isinstance(cell, BootstrapEvalCell)
    assert cell.weapon_id == WeaponType.KING_BIBLE
    assert cell.weapon_key == "king_bible"
    assert cell.task_kind == "solo_bootstrap"
    assert cell.enemy_phase_idx == 2


def test_parse_cell_spec_garlic_maintenance():
    cell = parse_cell_spec("garlic:maintenance:2")
    assert cell.weapon_id == WeaponType.GARLIC
    assert cell.task_kind == "maintenance"
    assert cell.enemy_phase_idx == 2


def test_parse_cell_spec_case_insensitive_snake_case():
    """snake_case / 大文字混在でも受け付けること。"""
    cell = parse_cell_spec("KING_BIBLE:Solo_Bootstrap:2")
    assert cell.weapon_id == WeaponType.KING_BIBLE
    assert cell.task_kind == "solo_bootstrap"


def test_parse_cell_spec_unknown_weapon_raises():
    with pytest.raises(ValueError, match="weapon"):
        parse_cell_spec("not_a_weapon:solo_bootstrap:2")


def test_parse_cell_spec_unknown_task_kind_raises():
    with pytest.raises(ValueError, match="task_kind"):
        parse_cell_spec("garlic:not_a_task:2")


def test_parse_cell_spec_out_of_range_phase_raises():
    with pytest.raises(ValueError, match="enemy_phase_idx"):
        parse_cell_spec("garlic:maintenance:999")


def test_parse_cell_spec_bad_format_raises():
    with pytest.raises(ValueError, match="形式"):
        parse_cell_spec("garlic:maintenance")


# ---------------------------------------------------------------------------
# eval params builder
# ---------------------------------------------------------------------------

def test_build_eval_params_includes_phase2_enemy_params():
    """phase2 の enemy params（UE key）が含まれること。"""
    cell = parse_cell_spec("king_bible:solo_bootstrap:2")
    params = build_eval_params(cell)
    # UE key に変換されていること
    assert "MinActiveEnemies" in params
    assert "MaxActiveEnemies" in params
    assert "MaxEnemyTypeId" in params


def test_build_eval_params_includes_target_weapon():
    """対象武器が allowed_weapon_types / initial_weapon_slots に含まれること。"""
    cell = parse_cell_spec("king_bible:solo_bootstrap:2")
    params = build_eval_params(cell)
    assert WeaponType.KING_BIBLE in params["allowed_weapon_types"]
    slot_ids = [s["weapon_id"] for s in params["initial_weapon_slots"]]
    assert WeaponType.KING_BIBLE in slot_ids


def test_build_eval_params_solo_bootstrap_is_target_only():
    """solo_bootstrap は target_only（garlic anchor を含まない）。"""
    cell = parse_cell_spec("king_bible:solo_bootstrap:2")
    params = build_eval_params(cell)
    assert params["allowed_weapon_types"] == [WeaponType.KING_BIBLE]


def test_build_eval_params_maintenance_includes_garlic_anchor_when_unlocked():
    """maintenance は target_plus_anchor_if_unlocked。

    garlic 自身が maintenance の場合、anchor は自分自身なので追加されない。
    king_bible maintenance で garlic がアンロック済み（WU1 stage）なら anchor に garlic が入る。
    """
    cell = parse_cell_spec("king_bible:maintenance:2")
    # king_bible は WU1、その段階では garlic(WU0) がアンロック済み
    params = build_eval_params(cell)
    assert WeaponType.KING_BIBLE in params["allowed_weapon_types"]
    assert WeaponType.GARLIC in params["allowed_weapon_types"]


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _make_episode(active_score: float, ep_length: int, terminated: int):
    return {"active_score": active_score, "ep_length": ep_length, "terminated": terminated}


def test_summarize_eval_results_basic():
    results = [
        _make_episode(100.0, 1200, 0),
        _make_episode(200.0, 1500, 0),
        _make_episode(300.0, 1800, 1),
    ]
    summary = summarize_eval_results(
        "king_bible:solo_bootstrap:2",
        results,
        deterministic=True,
        model_path="/x/best_model.zip",
        global_timestep=5001216,
    )
    assert summary["cell"] == "king_bible:solo_bootstrap:2"
    assert summary["episodes"] == 3
    assert summary["deterministic"] is True
    assert summary["active_score_mean"] == pytest.approx(200.0)
    assert summary["active_score_p50"] == pytest.approx(200.0)
    assert summary["global_timestep"] == 5001216
    assert summary["termination_counts"]["truncated"] == 2
    assert summary["termination_counts"]["terminated"] == 1


def test_summarize_eval_results_empty_does_not_crash():
    """episode 数不足（0件）でも p10 計算が落ちないこと。"""
    summary = summarize_eval_results(
        "garlic:maintenance:2",
        [],
        deterministic=True,
        model_path="/x/best_model.zip",
    )
    assert summary["episodes"] == 0
    assert summary["active_score_p10"] == 0.0
    assert summary["active_score_mean"] == 0.0
    assert summary["episode_length_p10"] == 0.0
    assert summary["short_episode_rate"] == 0.0
    assert summary["termination_counts"] == {}


def test_summarize_short_episode_rate():
    results = [
        _make_episode(10.0, 100, 1),   # short (< 600)
        _make_episode(10.0, 500, 1),   # short
        _make_episode(10.0, 1200, 0),  # not short
    ]
    summary = summarize_eval_results(
        "garlic:maintenance:2", results, deterministic=True, model_path="x",
        short_episode_steps=600,
    )
    assert summary["short_episode_rate"] == pytest.approx(2 / 3)

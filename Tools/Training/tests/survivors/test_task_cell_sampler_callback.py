"""tests/survivors/test_task_cell_sampler_callback.py - smoke tests"""
from __future__ import annotations
import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import pytest
from unittest.mock import MagicMock, patch
from games.survivors.modules.task_cell_sampler_module import TaskCellSamplerStateModule
from games.survivors.modules.weapon_bootstrap_module import WeaponBootstrapStateModule
from games.survivors.modules.weapon_unlock_module import WeaponUnlockStateModule
from games.survivors.survivors_weapon_table import WEAPON_UNLOCK_ORDER
from games.survivors.survivors_weapon_curriculum import WeaponType


def make_mock_hybrid_cb(current_phase=1):
    cb = MagicMock()
    cb.current_phase = current_phase
    return cb


def test_task_cell_sampler_callback_imports():
    """TaskCellSamplerCallback がインポートできること。"""
    from games.survivors.task_cell_sampler_callback import TaskCellSamplerCallback
    assert TaskCellSamplerCallback is not None


def test_active_pending_cell_lifecycle():
    """active/pending のライフサイクルが正しいことを確認する。

    VecEnv auto-reset の 1 episode ラグを補償するため、初期セル cell_0 は
    active と pending の両方に設定され、episode 1 と episode 2 の両方を動かす。

    training_start: active=cell_0, pending=cell_0, params=cell_0送信
    episode 1 終了: active=cell_0→stats更新, active=cell_0(pending昇格), new cell_1→pending
    episode 2 終了: active=cell_0→stats更新, active=cell_1(pending昇格), new cell_2→pending
    episode 3 終了: active=cell_1→stats更新, ...
    """
    from games.survivors.modules.task_cell_sampler_module import TaskCell, TaskCellSamplerStateModule

    tcs = TaskCellSamplerStateModule(min_episodes_per_cell=5)
    tcs.rebuild_candidate_cells("WU0", 0, {0: 100})

    applied_cells = []  # apply() が呼ばれたセルを追跡

    # シミュレーション: training_start（初期セルを active/pending 両方に設定）
    cell_0 = tcs.sample_cell(0)
    active_by_env = {0: cell_0}
    pending_by_env = {0: cell_0}
    applied_cells.append(cell_0)  # params apply（ep1・ep2 を動かす）

    # シミュレーション: episode 1 終了（初回 episode も記録される）
    active_cell = active_by_env[0]  # = cell_0
    assert active_cell == cell_0  # 初回 episode を取りこぼさず stats 更新
    tcs.on_episode_end(active_cell, active_score=400.0, ep_len=200, terminated=False, num_timesteps=100)

    # pending -> active に昇格（pending=cell_0 → active=cell_0）
    active_by_env[0] = pending_by_env.get(0)
    cell_1 = tcs.sample_cell(100)
    pending_by_env[0] = cell_1
    applied_cells.append(cell_1)  # params apply（効くのは ep3）

    # シミュレーション: episode 2 終了（cell_0 の params で動いた episode）
    active_cell = active_by_env[0]  # = cell_0
    assert active_cell == cell_0
    tcs.on_episode_end(active_cell, active_score=420.0, ep_len=200, terminated=False, num_timesteps=200)

    # pending -> active に昇格（pending=cell_1 → active=cell_1）
    active_by_env[0] = pending_by_env.get(0)
    cell_2 = tcs.sample_cell(200)
    pending_by_env[0] = cell_2
    applied_cells.append(cell_2)  # params apply

    # シミュレーション: episode 3 終了（cell_1 の params で動いた episode）
    active_cell = active_by_env[0]  # = cell_1
    assert active_cell == cell_1

    assert len(applied_cells) == 3

    # cell_0 の stats が ep1・ep2 の 2 回更新されていることを確認
    stats_0 = tcs.get_cell_stats(cell_0.first_weapon_id, cell_0.enemy_phase_idx)
    assert stats_0 is not None
    assert stats_0.episode_count == 2
    assert stats_0.active_score_mean == pytest.approx(410.0)


def test_enemy_phase_change_rebuilds_candidates():
    """敵フェーズが変化したとき候補セルが再構築されることを確認する。"""
    from games.survivors.modules.task_cell_sampler_module import TaskCell, TaskCellSamplerStateModule

    tcs = TaskCellSamplerStateModule(enemy_phase_backtrack=1)
    # 初期: phase 0, backtrack=1 → enemy_phase 候補は max(0, 0-1)..0 = [0]
    tcs.rebuild_candidate_cells("WU0", 0, {0: 100, 1: 200})
    assert all(c.enemy_phase_idx == 0 for c in tcs._candidate_cells)

    # 事前に phase 0 のセルで stats を記録しておく
    phase0_cells = list(tcs._candidate_cells)
    assert len(phase0_cells) > 0
    sample_cell = phase0_cells[0]
    tcs.on_episode_end(sample_cell, active_score=500.0, ep_len=100, terminated=True, num_timesteps=0)

    # phase 1 に昇格 → enemy_phase 候補は max(0, 1-1)..1 = [0, 1]
    tcs.rebuild_candidate_cells("WU0", 1, {0: 100, 1: 200})
    phase_idxs = {c.enemy_phase_idx for c in tcs._candidate_cells}
    assert 0 in phase_idxs
    assert 1 in phase_idxs

    # 既存 stats が保持されていること
    stats_phase0 = tcs.get_cell_stats(sample_cell.first_weapon_id, 0)
    assert stats_phase0 is not None  # 再構築後もstatsが保持される
    assert stats_phase0.episode_count == 1
    assert stats_phase0.active_score_mean == pytest.approx(500.0)


def test_bootstrap_stage_advance_resamples_pending_from_new_stage_and_logs_metrics():
    from games.survivors.task_cell_sampler_callback import TaskCellSamplerCallback

    hybrid_cb = make_mock_hybrid_cb(current_phase=2)
    tcs = TaskCellSamplerStateModule(
        min_episodes_per_cell=1,
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
    )
    weapon_unlock = WeaponUnlockStateModule(
        weapon_unlock_min_steps=0,
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
    )
    weapon_bootstrap = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status={"garlic": "maintenance"},
    )
    min_steps = {0: 600, 1: 900, 2: 1200}
    tcs.rebuild_bootstrap_candidate_cells(
        stage_key="WU0",
        max_unlocked_enemy_phase_idx=2,
        min_episode_steps_by_phase=min_steps,
        weapon_bootstrap=weapon_bootstrap,
    )
    old_stage_cell = next(
        cell for cell in tcs._candidate_cells
        if cell.weapon_unlock_stage_key == "WU0"
    )

    wandb_logger = MagicMock()
    wandb_logger.enabled = True
    callback = TaskCellSamplerCallback(
        hybrid_cb=hybrid_cb,
        task_cell_sampler=tcs,
        weapon_unlock=weapon_unlock,
        param_applier=MagicMock(),
        wandb_logger=wandb_logger,
        wandb_log_freq=1,
        weapon_bootstrap=weapon_bootstrap,
        weapon_bootstrap_sample_mix={
            "solo_bootstrap": 1.0,
            "weak_cells": 0.0,
            "maintenance": 0.0,
            "integration": 0.0,
        },
    )
    callback.num_timesteps = 10_000
    callback.locals = {"infos": [{}]}
    callback._score_tracker.process = MagicMock(
        return_value=[(0, 0.0, 1200, 0.0)]
    )
    callback._active_cell_by_env[0] = None
    callback._active_params_by_env[0] = None
    callback._pending_cell_by_env[0] = old_stage_cell
    callback._pending_params_by_env[0] = {}

    assert callback._on_step() is True

    next_cell = callback._pending_cell_by_env[0]
    assert weapon_unlock.current_stage_key == "WU1"
    assert next_cell.weapon_unlock_stage_key == "WU1"
    assert next_cell.first_weapon_id == WeaponType.KING_BIBLE
    assert next_cell.task_kind == "solo_bootstrap"

    logged_metrics = wandb_logger.log.call_args[0][0]
    assert "weapon_bootstrap/garlic/status" in logged_metrics
    assert "weapon_bootstrap/king_bible/status" in logged_metrics


# ---------------------------------------------------------------------------
# 2-episode 擬似 VecEnv: 完了 episode を動かした cell/decision が JSONL に一致する
# ---------------------------------------------------------------------------

def _make_bootstrap_callback(tmp_path, current_phase=2):
    """weapon_bootstrap 有効な TaskCellSamplerCallback を構築する。"""
    from games.survivors.task_cell_sampler_callback import TaskCellSamplerCallback

    hybrid_cb = make_mock_hybrid_cb(current_phase=current_phase)
    tcs = TaskCellSamplerStateModule(
        min_episodes_per_cell=1,
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
    )
    weapon_unlock = WeaponUnlockStateModule(
        weapon_unlock_min_steps=10**9,  # このテスト中はステージ進行させない
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
    )
    weapon_bootstrap = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status={"garlic": "maintenance"},
    )
    callback = TaskCellSamplerCallback(
        hybrid_cb=hybrid_cb,
        task_cell_sampler=tcs,
        weapon_unlock=weapon_unlock,
        param_applier=MagicMock(),
        log_dir=str(tmp_path),
        weapon_bootstrap=weapon_bootstrap,
    )
    return callback, tcs, weapon_bootstrap


def test_two_episode_vecenv_logs_completed_episode_cell_and_lane(tmp_path):
    """擬似 VecEnv を 3 episode 分回し、各 JSONL record が『その episode を実際に
    動かした cell の selected lane』を出すことと、初回 episode も取りこぼさず
    記録されることを検証する。

    VecEnv auto-reset の 1 episode ラグにより:
      - training_start で apply した cell_A が episode 1 と episode 2 を動かす
      - episode 1 done 時に sample した cell_B は episode 3 から効く
    したがって JSONL は [A, A, B, ...] の順で cell/lane を記録するはず。
    """
    import json
    from games.survivors.modules.task_cell_sampler_module import TaskCell, TaskCellSampleDecision

    callback, tcs, weapon_bootstrap = _make_bootstrap_callback(tmp_path)

    # 決定論的な (cell, decision) 列を用意する
    def _mk(lane, weapon_id, phase):
        cell = TaskCell(
            weapon_unlock_stage_key="WU0",
            first_weapon_id=weapon_id,
            enemy_phase_idx=phase,
            task_kind=lane,
            build_policy="target_only",
        )
        decision = TaskCellSampleDecision(
            selected_lane=lane,
            active_lanes={lane: 1},
            lane_probabilities={lane: 1.0},
            selected_task_kind=lane,
            selected_weapon_id=weapon_id,
            selected_enemy_phase_idx=phase,
        )
        return cell, decision

    seq = [
        _mk("solo_bootstrap", WeaponType.GARLIC, 0),   # cell_A: training_start で apply → ep1, ep2 を動かす
        _mk("integration", WeaponType.GARLIC, 2),      # cell_B: ep1 done で sample → ep3 から効く
        _mk("maintenance", WeaponType.GARLIC, 2),      # cell_C
        _mk("solo_bootstrap", WeaponType.GARLIC, 1),   # cell_D
    ]
    calls = {"i": 0}

    def fake_sample(*args, **kwargs):
        cell, decision = seq[min(calls["i"], len(seq) - 1)]
        calls["i"] += 1
        tcs._last_sample_decision = decision
        return cell

    tcs.sample_cell_with_lane_mix = fake_sample  # type: ignore[assignment]

    # training_env は BaseCallback で read-only property（model.get_env() 経由）
    training_env = MagicMock()
    training_env.num_envs = 1
    callback.model = MagicMock()
    callback.model.get_env.return_value = training_env
    callback.num_timesteps = 1000

    # _on_training_start: 初期セル cell_A を active/pending 両方に設定
    callback._on_training_start()
    assert callback._active_cell_by_env[0] is not None
    assert callback._active_cell_by_env[0].task_kind == "solo_bootstrap"
    # active と pending は同一 decision を指す
    assert callback._active_decision_by_env[0] is callback._pending_decision_by_env[0]

    # 3 episode 分 _on_step を回す。各 step で 1 episode 完了させる。
    scores = [111.0, 222.0, 333.0]
    for score in scores:
        callback.num_timesteps += 1000
        callback.locals = {"infos": [{}]}
        callback._score_tracker.process = MagicMock(
            return_value=[(0, score, 1200, 0.0)]
        )
        assert callback._on_step() is True

    lines = (tmp_path / "task_cell_episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    # 初回 episode を取りこぼさず 3 件記録されている
    assert len(lines) == 3
    records = [json.loads(l) for l in lines]

    # ep1, ep2 は cell_A (solo_bootstrap) で動いた → 両方 solo_bootstrap を記録
    assert records[0]["active_score"] == pytest.approx(111.0)
    assert records[0]["selected_bootstrap_lane"] == "solo_bootstrap"
    assert records[0]["bootstrap_lane"] == "solo_bootstrap"

    assert records[1]["active_score"] == pytest.approx(222.0)
    assert records[1]["selected_bootstrap_lane"] == "solo_bootstrap"
    assert records[1]["bootstrap_lane"] == "solo_bootstrap"

    # ep3 は cell_B (integration) で動いた
    assert records[2]["active_score"] == pytest.approx(333.0)
    assert records[2]["selected_bootstrap_lane"] == "integration"
    assert records[2]["bootstrap_lane"] == "integration"

    # 全 record で新旧フィールドが同一 decision/cell を指す
    for rec in records:
        assert rec["selected_bootstrap_lane"] == rec["bootstrap_lane"]
        assert rec["bootstrap_task_kind"] == rec["bootstrap_lane"]


def test_two_episode_vecenv_new_and_legacy_lane_fields_share_single_decision(tmp_path):
    """selected_bootstrap_lane と bootstrap_lane が異なる source から書かれて
    ズレることが無いこと（同一 decision を指すこと）を、lane と task_kind が
    ずれ得るケースで確認する。"""
    import json
    from games.survivors.modules.task_cell_sampler_module import TaskCell, TaskCellSampleDecision

    callback, tcs, weapon_bootstrap = _make_bootstrap_callback(tmp_path)

    # weak_cells lane で maintenance cell が選ばれるケース（lane != task_kind）
    cell = TaskCell(
        weapon_unlock_stage_key="WU0",
        first_weapon_id=WeaponType.GARLIC,
        enemy_phase_idx=2,
        task_kind="maintenance",
        build_policy="target_plus_anchor_if_unlocked",
    )
    decision = TaskCellSampleDecision(
        selected_lane="weak_cells",
        active_lanes={"weak_cells": 2},
        lane_probabilities={"weak_cells": 1.0},
        selected_task_kind="maintenance",
        selected_weapon_id=WeaponType.GARLIC,
        selected_enemy_phase_idx=2,
    )

    def fake_sample(*args, **kwargs):
        tcs._last_sample_decision = decision
        return cell

    tcs.sample_cell_with_lane_mix = fake_sample  # type: ignore[assignment]

    training_env = MagicMock()
    training_env.num_envs = 1
    callback.model = MagicMock()
    callback.model.get_env.return_value = training_env
    callback.num_timesteps = 1000
    callback._on_training_start()

    callback.num_timesteps += 1000
    callback.locals = {"infos": [{}]}
    callback._score_tracker.process = MagicMock(return_value=[(0, 250.0, 1200, 0.0)])
    callback._on_step()

    rec = json.loads(
        (tmp_path / "task_cell_episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    # selected_bootstrap_lane は decision.selected_lane（"weak_cells"）
    assert rec["selected_bootstrap_lane"] == "weak_cells"
    # 既存 bootstrap_lane / bootstrap_task_kind は cell.task_kind（"maintenance"）
    assert rec["bootstrap_lane"] == "maintenance"
    assert rec["bootstrap_task_kind"] == "maintenance"
    # 両者は同一 (cell, decision) ペアから書かれるため候補内訳も一致
    assert rec["bootstrap_lane_candidates"] == {"weak_cells": 2}


# ---------------------------------------------------------------------------
# selected lane logging: JSONL 出力
# ---------------------------------------------------------------------------

def test_write_jsonl_includes_selected_lane_fields(tmp_path):
    """episode JSONL に selected_bootstrap_lane / bootstrap_task_kind が入り、
    既存 bootstrap_lane フィールドも残ること。"""
    import json
    from games.survivors.task_cell_sampler_callback import TaskCellSamplerCallback
    from games.survivors.modules.task_cell_sampler_module import TaskCell, TaskCellSampleDecision

    hybrid_cb = make_mock_hybrid_cb(current_phase=2)
    tcs = TaskCellSamplerStateModule(
        min_episodes_per_cell=1,
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
    )
    weapon_unlock = WeaponUnlockStateModule(
        weapon_unlock_min_steps=0,
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
    )
    weapon_bootstrap = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status={"garlic": "maintenance"},
    )
    callback = TaskCellSamplerCallback(
        hybrid_cb=hybrid_cb,
        task_cell_sampler=tcs,
        weapon_unlock=weapon_unlock,
        param_applier=MagicMock(),
        log_dir=str(tmp_path),
        weapon_bootstrap=weapon_bootstrap,
    )
    callback.num_timesteps = 5000

    cell = TaskCell(
        weapon_unlock_stage_key="WU0",
        first_weapon_id=WeaponType.GARLIC,
        enemy_phase_idx=2,
        task_kind="maintenance",
        build_policy="target_plus_anchor_if_unlocked",
    )
    decision = TaskCellSampleDecision(
        selected_lane="weak_cells",
        active_lanes={"solo_bootstrap": 3, "weak_cells": 2, "maintenance": 1, "integration": 0},
        lane_probabilities={"solo_bootstrap": 0.4118, "weak_cells": 0.3529, "maintenance": 0.2353},
        selected_task_kind="maintenance",
        selected_weapon_id=WeaponType.GARLIC,
        selected_enemy_phase_idx=2,
    )

    callback._write_jsonl(
        env_idx=0, cell=cell, active_score=250.0, ep_len=1200,
        terminated=False, ep_base=300.0, params={}, decision=decision,
    )

    lines = (tmp_path / "task_cell_episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    # 新フィールド
    assert record["selected_bootstrap_lane"] == "weak_cells"
    assert record["bootstrap_task_kind"] == cell.task_kind
    assert record["bootstrap_lane_candidates"]["weak_cells"] == 2
    assert record["bootstrap_lane_probabilities"]["weak_cells"] == pytest.approx(0.3529)
    # 既存 bootstrap_lane は削除せず残す（backward compatible）
    assert "bootstrap_lane" in record
    assert record["bootstrap_lane"] == cell.task_kind


def test_write_jsonl_selected_lane_none_when_no_decision(tmp_path):
    """decision が None の場合でも既存フィールドは残り新フィールドは None になる。"""
    import json
    from games.survivors.task_cell_sampler_callback import TaskCellSamplerCallback
    from games.survivors.modules.task_cell_sampler_module import TaskCell

    hybrid_cb = make_mock_hybrid_cb(current_phase=2)
    tcs = TaskCellSamplerStateModule(min_episodes_per_cell=1, weapon_unlock_order=WEAPON_UNLOCK_ORDER)
    weapon_unlock = WeaponUnlockStateModule(weapon_unlock_min_steps=0, weapon_unlock_order=WEAPON_UNLOCK_ORDER)
    weapon_bootstrap = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status={"garlic": "maintenance"},
    )
    callback = TaskCellSamplerCallback(
        hybrid_cb=hybrid_cb,
        task_cell_sampler=tcs,
        weapon_unlock=weapon_unlock,
        param_applier=MagicMock(),
        log_dir=str(tmp_path),
        weapon_bootstrap=weapon_bootstrap,
    )
    callback.num_timesteps = 5000
    cell = TaskCell(
        weapon_unlock_stage_key="WU0",
        first_weapon_id=WeaponType.GARLIC,
        enemy_phase_idx=2,
        task_kind="maintenance",
    )
    callback._write_jsonl(
        env_idx=0, cell=cell, active_score=250.0, ep_len=1200,
        terminated=False, ep_base=300.0, params={}, decision=None,
    )
    record = json.loads((tmp_path / "task_cell_episode_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["selected_bootstrap_lane"] is None
    assert record["bootstrap_lane_candidates"] is None
    assert record["bootstrap_lane"] == cell.task_kind


# ---------------------------------------------------------------------------
# analyze_bootstrap_lane_mix ヘルパー: parser / aggregation
# ---------------------------------------------------------------------------

def test_analyze_helper_selected_lane_counts_and_ratios():
    from games.survivors.analyze_bootstrap_lane_mix import (
        selected_lane_counts,
        selected_lane_ratios,
    )
    records = [
        {"selected_bootstrap_lane": "weak_cells"},
        {"selected_bootstrap_lane": "weak_cells"},
        {"selected_bootstrap_lane": "solo_bootstrap"},
        {"selected_bootstrap_lane": "fallback"},
    ]
    counts = selected_lane_counts(records)
    assert counts["weak_cells"] == 2
    assert counts["solo_bootstrap"] == 1
    assert counts["fallback"] == 1

    ratios = selected_lane_ratios(records)
    assert ratios["weak_cells"] == pytest.approx(0.5)


def test_analyze_helper_falls_back_to_legacy_bootstrap_lane():
    """新フィールドが無い旧 record は legacy bootstrap_lane を使う。"""
    from games.survivors.analyze_bootstrap_lane_mix import selected_lane_counts
    records = [
        {"bootstrap_lane": "maintenance"},  # 旧 record
        {"selected_bootstrap_lane": "weak_cells"},  # 新 record
    ]
    counts = selected_lane_counts(records)
    assert counts["maintenance"] == 1
    assert counts["weak_cells"] == 1


def test_analyze_helper_lane_task_kind_crosstab():
    from games.survivors.analyze_bootstrap_lane_mix import lane_task_kind_crosstab
    records = [
        {"selected_bootstrap_lane": "weak_cells", "bootstrap_task_kind": "maintenance"},
        {"selected_bootstrap_lane": "weak_cells", "bootstrap_task_kind": "integration"},
        {"selected_bootstrap_lane": "weak_cells", "bootstrap_task_kind": "maintenance"},
    ]
    table = lane_task_kind_crosstab(records)
    assert table["weak_cells"]["maintenance"] == 2
    assert table["weak_cells"]["integration"] == 1


def test_analyze_helper_weapon_phase_lane_counts_and_filter():
    from games.survivors.analyze_bootstrap_lane_mix import (
        filter_records,
        weapon_phase_lane_counts,
        recent_lane_ratios,
    )
    records = [
        {"selected_bootstrap_lane": "solo_bootstrap", "first_weapon_id": 7, "enemy_phase_idx": 2},
        {"selected_bootstrap_lane": "weak_cells", "first_weapon_id": 7, "enemy_phase_idx": 2},
        {"selected_bootstrap_lane": "maintenance", "first_weapon_id": 1, "enemy_phase_idx": 2},
    ]
    wp = weapon_phase_lane_counts(records)
    assert wp[(7, 2)]["solo_bootstrap"] == 1
    assert wp[(7, 2)]["weak_cells"] == 1
    assert wp[(1, 2)]["maintenance"] == 1

    # filter by weapon
    filtered = filter_records(records, weapon_id=7)
    assert len(filtered) == 2

    # recent window
    recent = recent_lane_ratios(records, window_episodes=1)
    assert recent == {"maintenance": 1.0}


def test_analyze_helper_load_records_skips_broken_lines(tmp_path):
    from games.survivors.analyze_bootstrap_lane_mix import load_records, resolve_jsonl_path
    jsonl = tmp_path / "task_cell_episode_metrics.jsonl"
    jsonl.write_text(
        '{"selected_bootstrap_lane": "weak_cells"}\n'
        'THIS IS NOT JSON\n'
        '\n'
        '{"selected_bootstrap_lane": "solo_bootstrap"}\n',
        encoding="utf-8",
    )
    resolved = resolve_jsonl_path(tmp_path)
    assert resolved == jsonl
    records = load_records(resolved)
    assert len(records) == 2

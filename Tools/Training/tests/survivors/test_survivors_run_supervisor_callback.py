from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.modules.weapon_bootstrap_module import WeaponBootstrapStateModule
from games.survivors.modules.weapon_unlock_module import WeaponUnlockStateModule
from games.survivors.run_event_logger import JsonlEventLogger
from games.survivors.survivors_run_supervisor_callback import SurvivorsRunSupervisorCallback
from games.survivors.survivors_weapon_curriculum import WeaponType
from games.survivors.survivors_weapon_table import WEAPON_UNLOCK_ORDER


def _setup(cb: SurvivorsRunSupervisorCallback, step: int):
    model = MagicMock()
    model.get_env.return_value = MagicMock()
    model.logger = MagicMock()
    cb.model = model
    cb.num_timesteps = step
    cb.n_calls = step  # check_freq 判定に使用
    return cb


def make_supervisor_for_unit_tests(
    *,
    item_stage_key: str = "IS0",
    target_stage_key: str = "WU12",
    post_bootstrap_mode: str = "stop",
    initial_status: dict | None = None,
    event_logger: JsonlEventLogger | None = None,
) -> SurvivorsRunSupervisorCallback:
    """ユニットテスト用に SurvivorsRunSupervisorCallback を構築するヘルパー。"""
    bootstrap = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        item_stage_key=item_stage_key,
        initial_status=initial_status,
    )
    unlock = WeaponUnlockStateModule(
        initial_stage_key="WU0",
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
    )
    return SurvivorsRunSupervisorCallback(
        weapon_unlock=unlock,
        weapon_bootstrap=bootstrap,
        target_stage_key=target_stage_key,
        item_stage_key=item_stage_key,
        post_bootstrap_mode=post_bootstrap_mode,
        check_freq=1,
        event_logger=event_logger,
    )


def test_supervisor_export_includes_item_stage_key():
    cb = make_supervisor_for_unit_tests(item_stage_key="IS1")
    state = cb.export_state()
    assert state["item_stage_key"] == "IS1"


def test_supervisor_bootstrap_complete_event_includes_item_stage_key(tmp_path):
    """bootstrap_complete イベント payload に item_stage_key が含まれる。"""
    cb = make_supervisor_for_unit_tests(
        item_stage_key="IS1",
        target_stage_key="WU1",
        initial_status={"garlic": "maintenance", "king_bible": "maintenance"},
        event_logger=JsonlEventLogger(tmp_path / "events.jsonl"),
    )
    _setup(cb, step=100)
    cb._on_step()
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    complete = [e for e in events if e["event"].endswith("bootstrap_complete")]
    assert complete
    assert complete[-1]["payload"]["item_stage_key"] == "IS1"


def test_supervisor_stops_when_bootstrap_complete(tmp_path):
    bootstrap = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status={
            "garlic": "maintenance",
            "king_bible": "maintenance",
            "magic_wand": "maintenance",
        },
    )
    unlock = WeaponUnlockStateModule(initial_stage_key="WU2", weapon_unlock_order=WEAPON_UNLOCK_ORDER)
    cb = SurvivorsRunSupervisorCallback(
        weapon_unlock=unlock,
        weapon_bootstrap=bootstrap,
        target_stage_key="WU2",
        post_bootstrap_mode="stop",
        check_freq=1,
        event_logger=JsonlEventLogger(tmp_path / "events.jsonl"),
    )
    _setup(cb, step=100)

    assert cb._on_step() is False
    state = cb.export_state()
    assert state["bootstrap_complete"] is True
    assert state["exit_reason"] == "bootstrap_complete"

    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event"] == "survivors_supervisor.bootstrap_complete"


def test_supervisor_complete_takes_priority_over_timeout():
    bootstrap = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status={
            "garlic": "maintenance",
            "king_bible": "maintenance",
            "magic_wand": "maintenance",
        },
    )
    unlock = WeaponUnlockStateModule(initial_stage_key="WU2", weapon_unlock_order=WEAPON_UNLOCK_ORDER)
    cb = SurvivorsRunSupervisorCallback(
        weapon_unlock=unlock,
        weapon_bootstrap=bootstrap,
        target_stage_key="WU2",
        post_bootstrap_mode="stop",
        check_freq=1,
        stage_timeout_steps=100,
    )
    _setup(cb, step=101)  # stage_age > timeout だが全武器 maintenance 済み

    assert cb._on_step() is False
    state = cb.export_state()
    assert state["exit_reason"] == "bootstrap_complete", (
        f"完走済みなのに timeout が優先された: {state['exit_reason']}"
    )


def test_supervisor_requests_transition_when_bootstrap_complete_combination_smoke(tmp_path):
    """combination_smoke モードでは training を停止せず遷移フラグのみ立てる。"""
    bootstrap = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status={
            "garlic": "maintenance",
            "king_bible": "maintenance",
            "magic_wand": "maintenance",
        },
    )
    unlock = WeaponUnlockStateModule(initial_stage_key="WU2", weapon_unlock_order=WEAPON_UNLOCK_ORDER)
    cb = SurvivorsRunSupervisorCallback(
        weapon_unlock=unlock,
        weapon_bootstrap=bootstrap,
        target_stage_key="WU2",
        post_bootstrap_mode="combination_smoke",
        check_freq=1,
    )
    _setup(cb, step=100)

    # training は継続する（True）が遷移フラグは立つ
    assert cb._on_step() is True
    assert cb.post_bootstrap_transition_requested is True
    state = cb.export_state()
    assert state["bootstrap_complete"] is True
    assert state["post_bootstrap_transition_requested"] is True
    # 停止しないので exit_reason は None のまま
    assert state["exit_reason"] is None

    # 2 回目の呼び出しでもフラグは維持され、継続する
    assert cb._on_step() is True
    assert cb.post_bootstrap_transition_requested is True


def test_supervisor_requests_transition_when_bootstrap_complete_passive_item_stage(tmp_path):
    """passive_item_stage モードでも training を停止せず遷移フラグのみ立てる。"""
    bootstrap = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status={
            "garlic": "maintenance",
            "king_bible": "maintenance",
            "magic_wand": "maintenance",
        },
    )
    unlock = WeaponUnlockStateModule(initial_stage_key="WU2", weapon_unlock_order=WEAPON_UNLOCK_ORDER)
    cb = SurvivorsRunSupervisorCallback(
        weapon_unlock=unlock,
        weapon_bootstrap=bootstrap,
        target_stage_key="WU2",
        post_bootstrap_mode="passive_item_stage",
        check_freq=1,
        event_logger=JsonlEventLogger(tmp_path / "events.jsonl"),
    )
    _setup(cb, step=100)

    # training は継続する（True）が遷移フラグは立つ
    assert cb._on_step() is True
    assert cb.post_bootstrap_transition_requested is True
    state = cb.export_state()
    assert state["bootstrap_complete"] is True
    assert state["post_bootstrap_transition_requested"] is True
    assert state["exit_reason"] is None

    # passive_item_stage_transition_requested イベントが記録される
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        e.get("event", "").endswith("passive_item_stage_transition_requested")
        for e in events
    ), f"passive_item_stage_transition_requested が記録されていない: {[e.get('event') for e in events]}"

    # 2 回目の呼び出しでもフラグは維持され、継続する
    assert cb._on_step() is True
    assert cb.post_bootstrap_transition_requested is True


def test_supervisor_blocks_on_regression_count_limit():
    bootstrap = WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status={"garlic": "maintenance"},
    )
    bootstrap._states[WeaponType.GARLIC].regression_count = 5
    unlock = WeaponUnlockStateModule(initial_stage_key="WU0", weapon_unlock_order=WEAPON_UNLOCK_ORDER)
    cb = SurvivorsRunSupervisorCallback(
        weapon_unlock=unlock,
        weapon_bootstrap=bootstrap,
        target_stage_key="WU0",
        post_bootstrap_mode="stop",
        max_regression_count=3,
        check_freq=1,
    )
    _setup(cb, step=100)

    assert cb._on_step() is False
    assert cb.export_state()["exit_reason"] == "bootstrap_regression_limit"

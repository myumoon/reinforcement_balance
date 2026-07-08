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
    return cb


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

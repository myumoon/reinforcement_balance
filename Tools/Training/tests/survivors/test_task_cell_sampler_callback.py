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
from games.survivors.modules.weapon_unlock_module import WeaponUnlockStateModule
from games.survivors.survivors_weapon_curriculum import WeaponType


def make_mock_hybrid_cb(current_phase=1):
    cb = MagicMock()
    cb.current_phase = current_phase
    return cb


def test_task_cell_sampler_callback_imports():
    """TaskCellSamplerCallback がインポートできること。"""
    from games.survivors.task_cell_sampler_callback import TaskCellSamplerCallback
    assert TaskCellSamplerCallback is not None

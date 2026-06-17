"""_call_reward_fn と _EurekaMetricsCallback の新規メトリクスのユニットテスト。"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from eureka_loop import _call_reward_fn


def test_call_reward_fn_float():
    fn = lambda obs, prev_obs, base: 0.5
    total, comps = _call_reward_fn(fn, None, None, 0.0)
    assert total == 0.5
    assert comps is None


def test_call_reward_fn_dict_with_total():
    fn = lambda obs, prev_obs, base: {"total": 1.0, "move": 0.6, "kill": 0.4}
    total, comps = _call_reward_fn(fn, None, None, 0.0)
    assert total == 1.0
    assert comps == {"move": 0.6, "kill": 0.4}


def test_call_reward_fn_dict_without_total():
    fn = lambda obs, prev_obs, base: {"move": 0.3, "kill": 0.7}
    total, comps = _call_reward_fn(fn, None, None, 0.0)
    assert abs(total - 1.0) < 1e-6
    assert "move" in comps and "kill" in comps

"""_SharedComponents と _make_reward_fn_wrapper のユニットテスト。"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from eureka_loop import _SharedComponents, _make_reward_fn_wrapper


def test_wrapper_float_passthrough():
    fn = lambda obs, prev_obs, base: 0.5
    shared = _SharedComponents()
    wrapped = _make_reward_fn_wrapper(fn, shared)
    result = wrapped(None, None, 0.0)
    assert result == 0.5
    assert shared.components is None


def test_wrapper_dict_with_total():
    fn = lambda obs, prev_obs, base: {"total": 1.0, "move": 0.6, "kill": 0.4}
    shared = _SharedComponents()
    wrapped = _make_reward_fn_wrapper(fn, shared)
    result = wrapped(None, None, 0.0)
    assert result == 1.0
    assert shared.components == {"move": 0.6, "kill": 0.4}


def test_wrapper_dict_without_total():
    fn = lambda obs, prev_obs, base: {"move": 0.3, "kill": 0.7}
    shared = _SharedComponents()
    wrapped = _make_reward_fn_wrapper(fn, shared)
    result = wrapped(None, None, 0.0)
    assert abs(result - 1.0) < 1e-6
    assert "move" in shared.components and "kill" in shared.components


def test_wrapper_resets_components_on_float():
    shared = _SharedComponents()
    shared.components = {"old": 1.0}
    fn = lambda obs, prev_obs, base: 0.5
    wrapped = _make_reward_fn_wrapper(fn, shared)
    wrapped(None, None, 0.0)
    assert shared.components is None

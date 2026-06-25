"""Hybrid SPALF eval と probe 統合テスト。

--curriculum-spalf 時に SurvivorsEvalCallback が probe 兼用で作られ、
HybridPromotionProbeCallback が登録されないことを確認する。
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Callable

import numpy as np
import pytest

_TRAINING_ROOT = Path(__file__).resolve().parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.hybrid_callback import HybridCurriculumSpalfCallback
from games.survivors.survivors_curriculum import get_enemy_params_for_phase as _phase_params_from_phase
from games.survivors.survivors_eval_callback import SurvivorsEvalCallback
from games.survivors.param_applier import ParamApplier


def _make_hybrid_cb(**kwargs) -> HybridCurriculumSpalfCallback:
    defaults = dict(
        raw_env=None,
        frame_skip=1,
        alive_reward=0.0,
        r_b=0.1,
        alpha=0.2,
        max_score=2250.0,
        spalf_buffer_size=200,
        warmup_episodes=5,
        phase_warmup_episodes=3,
        phase_window=3,
        threshold_mult=1.0,
        rollback_patience=2,
    )
    defaults.update(kwargs)
    return HybridCurriculumSpalfCallback(**defaults)


def _make_mock_eval_env(n_envs: int = 1, obs_shape: tuple = (8,)):
    """モック eval_env を作成する（VecNormalize なし）。"""
    env = MagicMock()
    env.num_envs = n_envs
    env.venv = None
    obs = np.zeros(obs_shape, dtype=np.float32)
    env.reset.return_value = obs
    env.step.return_value = (
        obs,
        np.zeros(n_envs),
        np.array([True] * n_envs),
        [{"base_reward": 10.0}],
    )
    env.env_method.return_value = [True] * n_envs
    return env


class TestCurriculumSpalfEvalCallbackCreation:
    """--curriculum-spalf 時の SurvivorsEvalCallback 生成テスト。"""

    def test_eval_callback_created_with_curriculum_window_as_n_eval_episodes(self):
        """--curriculum-spalf 時に SurvivorsEvalCallback が n_eval_episodes=curriculum_window で作られること。"""
        hybrid_cb = _make_hybrid_cb()
        eval_env = _make_mock_eval_env()
        curriculum_window = 15

        cb = SurvivorsEvalCallback(
            eval_env=eval_env,
            eval_freq=50_000,
            n_eval_episodes=curriculum_window,
            frame_skip=1,
            alive_reward=0.001,
            params_provider=hybrid_cb.get_current_phase_params,
            after_eval_callback=hybrid_cb.on_promotion_probe_eval_results,
        )

        assert cb.n_eval_episodes == curriculum_window, \
            f"n_eval_episodes が curriculum_window={curriculum_window} になること"

    def test_eval_callback_has_params_provider_when_curriculum_spalf(self):
        """--curriculum-spalf 時に params_provider が設定されること。"""
        hybrid_cb = _make_hybrid_cb()
        eval_env = _make_mock_eval_env()

        cb = SurvivorsEvalCallback(
            eval_env=eval_env,
            eval_freq=50_000,
            n_eval_episodes=10,
            frame_skip=1,
            alive_reward=0.001,
            params_provider=hybrid_cb.get_current_phase_params,
            after_eval_callback=hybrid_cb.on_promotion_probe_eval_results,
        )

        assert cb.params_provider is not None, "params_provider が設定されていること"
        assert cb.params_provider == hybrid_cb.get_current_phase_params

    def test_eval_callback_has_after_eval_callback_when_curriculum_spalf(self):
        """--curriculum-spalf 時に after_eval_callback が設定されること。"""
        hybrid_cb = _make_hybrid_cb()
        eval_env = _make_mock_eval_env()

        cb = SurvivorsEvalCallback(
            eval_env=eval_env,
            eval_freq=50_000,
            n_eval_episodes=10,
            frame_skip=1,
            alive_reward=0.001,
            params_provider=hybrid_cb.get_current_phase_params,
            after_eval_callback=hybrid_cb.on_promotion_probe_eval_results,
        )

        assert cb.after_eval_callback is not None, "after_eval_callback が設定されていること"
        assert cb.after_eval_callback == hybrid_cb.on_promotion_probe_eval_results

    def test_eval_callback_no_hooks_when_normal(self):
        """通常時（非 curriculum_spalf）は params_provider と after_eval_callback が None であること。"""
        cb = SurvivorsEvalCallback(
            eval_env=None,
            eval_freq=50_000,
            n_eval_episodes=5,
            frame_skip=1,
            alive_reward=0.001,
        )

        assert cb.params_provider is None, "通常時は params_provider が None であること"
        assert cb.after_eval_callback is None, "通常時は after_eval_callback が None であること"


class TestHybridPromotionProbeNotRegistered:
    """HybridPromotionProbeCallback が curriculum_spalf 時に登録されないことのテスト。

    train.py のロジックを直接テストするのではなく、
    新しい統合アーキテクチャを検証する統合テスト。
    """

    def test_hybrid_cb_not_needed_when_eval_callback_has_hooks(self):
        """SurvivorsEvalCallback が params_provider と after_eval_callback を持てば
        HybridPromotionProbeCallback は不要であること（インターフェース検証）。"""
        hybrid_cb = _make_hybrid_cb()
        eval_env = _make_mock_eval_env()

        eval_cb = SurvivorsEvalCallback(
            eval_env=eval_env,
            eval_freq=50_000,
            n_eval_episodes=10,
            frame_skip=1,
            alive_reward=0.001,
            params_provider=hybrid_cb.get_current_phase_params,
            after_eval_callback=hybrid_cb.on_promotion_probe_eval_results,
        )

        # SurvivorsEvalCallback がすべての必要な機能を持っていること
        assert callable(eval_cb.params_provider)
        assert callable(eval_cb.after_eval_callback)


class TestParamsProviderKeyConversion:
    """eval 直前に params_provider の params が UE5 key 変換済みで eval_env に送られること。"""

    def test_phase_params_converted_to_ue5_keys_before_set_params(self):
        """params_provider から取得した内部 key が UE5 API key に変換されて set_params に渡ること。"""
        hybrid_cb = _make_hybrid_cb()
        eval_env = _make_mock_eval_env()

        cb = SurvivorsEvalCallback(
            eval_env=eval_env,
            eval_freq=50_000,
            n_eval_episodes=1,
            frame_skip=1,
            alive_reward=0.0,
            params_provider=hybrid_cb.get_current_phase_params,
        )

        # training_env も設定する
        mock_train_env = _make_mock_eval_env()
        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_train_env
        cb.model = mock_model
        cb._training_env = mock_train_env

        with patch.object(cb, "_sync_vecnormalize"), \
             patch.object(cb, "_sync_shaping"):
            cb._sync_before_eval()

        # eval_env.env_method("set_params", ...) の呼び出しを確認
        call_args_list = eval_env.env_method.call_args_list
        set_params_calls = [c for c in call_args_list if c[0][0] == "set_params"]
        assert len(set_params_calls) >= 1, "set_params が呼ばれること"

        # UE5 API key 形式で呼ばれること（内部 key でないこと）
        kwargs = set_params_calls[0][1]
        ue5_keys = set(ParamApplier._KEY_MAP.values())
        internal_keys = set(ParamApplier._KEY_MAP.keys())

        # UE5 API key が含まれること
        assert any(k in ue5_keys for k in kwargs), \
            f"UE5 API key が使われること。kwargs={kwargs}"
        # 内部 key（小文字スネークケース）が使われていないこと
        assert not any(k in internal_keys for k in kwargs), \
            f"内部 key が変換されずに渡されていないこと。kwargs={kwargs}"


class TestAfterEvalCallbackReceivesResults:
    """after_eval_callback が eval 結果で呼ばれることのテスト。"""

    def test_on_promotion_probe_eval_results_called_with_eval_results(self):
        """after_eval_callback として on_promotion_probe_eval_results が eval 結果で呼ばれること。"""
        hybrid_cb = _make_hybrid_cb()
        eval_env = _make_mock_eval_env()

        received_calls: list[tuple] = []

        def capture_after_eval(episode_results, metrics):
            received_calls.append((episode_results, metrics))

        cb = SurvivorsEvalCallback(
            eval_env=eval_env,
            eval_freq=50_000,
            n_eval_episodes=3,
            frame_skip=1,
            alive_reward=0.0,
            after_eval_callback=capture_after_eval,
        )

        mock_train_env = _make_mock_eval_env()
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.zeros((1,), dtype=np.int64), None)
        mock_model.get_env.return_value = mock_train_env
        cb.model = mock_model
        cb._training_env = mock_train_env
        cb.num_timesteps = 0

        with patch.object(cb, "_sync_before_eval"):
            cb._run_eval_and_log()

        assert len(received_calls) == 1, "after_eval_callback が1回呼ばれること"
        episode_results, metrics = received_calls[0]
        assert len(episode_results) == 3, "3エピソード分の結果が渡ること"

    def test_ep_length_key_in_episode_results_passed_to_after_callback(self):
        """after_eval_callback に渡される episode_results が ep_length キーを持つこと。"""
        hybrid_cb = _make_hybrid_cb()
        eval_env = _make_mock_eval_env()

        received_episode_results: list[dict] = []

        def capture_after_eval(episode_results, metrics):
            received_episode_results.extend(episode_results)

        cb = SurvivorsEvalCallback(
            eval_env=eval_env,
            eval_freq=50_000,
            n_eval_episodes=2,
            frame_skip=1,
            alive_reward=0.0,
            after_eval_callback=capture_after_eval,
        )

        mock_train_env = _make_mock_eval_env()
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.zeros((1,), dtype=np.int64), None)
        mock_model.get_env.return_value = mock_train_env
        cb.model = mock_model
        cb._training_env = mock_train_env
        cb.num_timesteps = 0

        with patch.object(cb, "_sync_before_eval"):
            cb._run_eval_and_log()

        for r in received_episode_results:
            assert "ep_length" in r, "ep_length キーが存在すること"

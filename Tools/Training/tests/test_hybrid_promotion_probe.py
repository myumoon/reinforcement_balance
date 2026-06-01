"""HybridPromotionProbeCallback のユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

_TRAINING_ROOT = Path(__file__).resolve().parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.hybrid_callback import HybridCurriculumSpalfCallback, _phase_params_from_phase, _CURRICULUM_PHASES
from games.survivors.hybrid_promotion_probe_callback import HybridPromotionProbeCallback


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
    """モック eval_env を作成する（VecNormalize なし）。

    venv=None を明示して _sync_vecnormalize の無限ループを防ぐ。
    """
    env = MagicMock()
    env.num_envs = n_envs
    env.venv = None  # VecNormalize chain を終端させる
    obs = np.zeros(obs_shape, dtype=np.float32)
    env.reset.return_value = obs
    # step: 1 ステップで done にする
    env.step.return_value = (obs, np.zeros(n_envs), np.array([True] * n_envs), [{"base_reward": 10.0}])
    env.env_method.return_value = [True] * n_envs
    return env


def _setup_probe_cb(hybrid_cb, eval_env, probe_freq=1000, n_probe_episodes=3,
                    frame_skip=1, alive_reward=0.0) -> HybridPromotionProbeCallback:
    """テスト用 HybridPromotionProbeCallback をセットアップする。"""
    cb = HybridPromotionProbeCallback(
        hybrid_cb=hybrid_cb,
        eval_env=eval_env,
        probe_freq=probe_freq,
        n_probe_episodes=n_probe_episodes,
        frame_skip=frame_skip,
        alive_reward=alive_reward,
        deterministic=True,
    )
    mock_model = MagicMock()
    mock_model.num_timesteps = probe_freq + 1

    # モデルの predict は (action, None) を返す
    mock_model.predict.return_value = (np.zeros((1,)), None)

    mock_train_env = MagicMock()
    mock_train_env.num_envs = 2
    mock_train_env.venv = None  # VecNormalize chain を終端させる
    mock_model.get_env.return_value = mock_train_env

    cb.model = mock_model
    cb.n_calls = 0
    cb.num_timesteps = probe_freq + 1

    # SB3 BaseCallback が training_env を model.get_env() から取得する
    cb._training_env = mock_train_env

    return cb


class TestHybridPromotionProbeCallback:
    def test_set_params_called_with_phase_params(self):
        """eval env に current phase の固定 params が適用されること（set_params が呼ばれる）。"""
        hybrid_cb = _make_hybrid_cb()
        eval_env = _make_mock_eval_env()
        cb = _setup_probe_cb(hybrid_cb, eval_env, n_probe_episodes=1)

        cb._run_probe()

        # eval_env.env_method("set_params", ...) が呼ばれたか確認
        call_args_list = eval_env.env_method.call_args_list
        set_params_calls = [c for c in call_args_list if c[0][0] == "set_params"]
        assert len(set_params_calls) >= 1, "set_params が呼ばれること"

        # set_params に渡されたキーが UE5 API キー形式であること
        kwargs = set_params_calls[0][1]
        assert "MinActiveEnemies" in kwargs or "MaxActiveEnemies" in kwargs, \
            "UE5 API キー（MinActiveEnemies 等）で set_params が呼ばれること"

    def test_train_env_params_not_changed(self):
        """train env の params は変更されないこと（eval_env の env_method のみ呼ばれる）。"""
        hybrid_cb = _make_hybrid_cb()
        eval_env = _make_mock_eval_env()
        cb = _setup_probe_cb(hybrid_cb, eval_env, n_probe_episodes=1)

        # training_env の env_method は呼ばれないことを確認
        training_env_mock = cb._training_env

        cb._run_probe()

        # training_env.env_method は呼ばれていないこと
        training_env_mock.env_method.assert_not_called()

    def test_probe_results_passed_to_hybrid_cb(self):
        """probe metrics が hybrid_cb.on_promotion_probe_results() に渡ること。"""
        hybrid_cb = _make_hybrid_cb()
        eval_env = _make_mock_eval_env()
        cb = _setup_probe_cb(hybrid_cb, eval_env, n_probe_episodes=3)

        with patch.object(hybrid_cb, "on_promotion_probe_results", return_value=None) as mock_results:
            cb._run_probe()
            mock_results.assert_called_once()
            results_arg = mock_results.call_args[0][0]
            assert len(results_arg) == 3, "3 エピソード分の結果が渡ること"
            for r in results_arg:
                assert "active_score" in r
                assert "ep_len" in r

    def test_deterministic_predict_called(self):
        """deterministic=True で model.predict が deterministic=True で呼ばれること。"""
        hybrid_cb = _make_hybrid_cb()
        eval_env = _make_mock_eval_env()
        cb = _setup_probe_cb(hybrid_cb, eval_env, n_probe_episodes=1)

        cb._run_probe()

        predict_calls = cb.model.predict.call_args_list
        assert len(predict_calls) >= 1
        for c in predict_calls:
            assert c[1].get("deterministic") is True, "deterministic=True で predict が呼ばれること"

    def test_eval_env_training_flag_restored(self):
        """eval_env.training が評価後に元の値に戻ること。"""
        hybrid_cb = _make_hybrid_cb()
        eval_env = _make_mock_eval_env()
        eval_env.training = True  # 初期値を True に設定
        cb = _setup_probe_cb(hybrid_cb, eval_env, n_probe_episodes=1)

        cb._run_probe()

        assert eval_env.training is True, "eval_env.training が元の True に戻ること"

    def test_probe_not_run_if_freq_not_reached(self):
        """probe_freq に達していない場合は _run_probe が呼ばれないこと。"""
        hybrid_cb = _make_hybrid_cb()
        eval_env = _make_mock_eval_env()
        cb = _setup_probe_cb(hybrid_cb, eval_env, probe_freq=10000)

        # num_timesteps を probe_freq より少なく設定
        cb.num_timesteps = 500
        cb._last_probe_step = 0

        with patch.object(cb, "_run_probe") as mock_run:
            cb._on_rollout_end()
            mock_run.assert_not_called()

    def test_probe_run_when_freq_reached(self):
        """probe_freq に達した場合は _run_probe が呼ばれること。"""
        hybrid_cb = _make_hybrid_cb()
        eval_env = _make_mock_eval_env()
        cb = _setup_probe_cb(hybrid_cb, eval_env, probe_freq=1000)

        cb.num_timesteps = 1000
        cb._last_probe_step = 0

        with patch.object(cb, "_run_probe") as mock_run:
            cb._on_rollout_end()
            mock_run.assert_called_once()

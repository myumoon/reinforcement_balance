"""SurvivorsEvalCallback と run_survivors_eval_episodes のユニットテスト。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

_TRAINING_ROOT = Path(__file__).resolve().parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.survivors_eval_callback import (
    SurvivorsEvalCallback,
    run_survivors_eval_episodes,
    _aggregate_eval_results,
)


def _make_mock_env(n_envs: int = 1, obs_shape: tuple = (8,), done_on_first_step: bool = True):
    """モック VecEnv を作成する（VecNormalize なし）。"""
    env = MagicMock()
    env.num_envs = n_envs
    env.venv = None  # VecNormalize chain を終端させる
    obs = np.zeros(obs_shape, dtype=np.float32)
    env.reset.return_value = obs
    if done_on_first_step:
        env.step.return_value = (
            obs,
            np.zeros(n_envs),
            np.array([True] * n_envs),
            [{"base_reward": 10.0}],
        )
    env.env_method.return_value = [True] * n_envs
    return env


def _make_mock_model():
    """モックモデルを作成する。"""
    model = MagicMock()
    model.predict.return_value = (np.zeros((1,), dtype=np.int64), None)
    model.num_timesteps = 0
    return model


def _make_eval_callback(eval_env=None, eval_freq: int = 1000, n_eval_episodes: int = 2,
                        params_provider=None, after_eval_callback=None) -> SurvivorsEvalCallback:
    """テスト用 SurvivorsEvalCallback を作成する。"""
    return SurvivorsEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        frame_skip=1,
        alive_reward=0.0,
        params_provider=params_provider,
        after_eval_callback=after_eval_callback,
        verbose=0,
    )


def _setup_callback(cb: SurvivorsEvalCallback, training_env=None) -> SurvivorsEvalCallback:
    """コールバックのモデルと training_env をセットアップする。"""
    if training_env is None:
        training_env = _make_mock_env()
    mock_model = _make_mock_model()
    mock_model.get_env.return_value = training_env
    cb.model = mock_model
    cb._training_env = training_env
    cb.num_timesteps = 0
    cb.n_calls = 0
    return cb


class TestRunSurvivorsEvalEpisodes:
    """run_survivors_eval_episodes 関数のテスト。"""

    def test_returns_episode_results_and_metrics(self):
        """正常ケース: episode_results と metrics と last_obs を返すこと。"""
        env = _make_mock_env(n_envs=1)
        model = _make_mock_model()

        results, metrics, last_obs = run_survivors_eval_episodes(
            model=model,
            env=env,
            n_eval_episodes=3,
            frame_skip=1,
            alive_reward=0.0,
        )

        assert len(results) == 3
        for r in results:
            assert "ep_length" in r
            assert "active_score" in r
            assert "base_reward" in r
        assert last_obs is not None, "last_obs が返ること"

    def test_episode_results_use_ep_length_key(self):
        """episode_results の各 dict が ep_length キーを持つこと（ep_len ではない）。"""
        env = _make_mock_env(n_envs=1)
        model = _make_mock_model()

        results, _, _last_obs = run_survivors_eval_episodes(
            model=model,
            env=env,
            n_eval_episodes=2,
            frame_skip=1,
            alive_reward=0.0,
        )

        for r in results:
            assert "ep_length" in r, "ep_length キーが存在すること"
            assert "ep_len" not in r, "ep_len キーは存在しないこと（ep_length を使う）"

    def test_env_training_restored_on_normal_exit(self):
        """正常終了時に env.training が元の値に戻ること。"""
        env = _make_mock_env(n_envs=1)
        env.training = True

        run_survivors_eval_episodes(
            model=_make_mock_model(),
            env=env,
            n_eval_episodes=1,
            frame_skip=1,
            alive_reward=0.0,
        )

        assert env.training is True, "eval 後に env.training が元の True に戻ること"

    def test_env_training_restored_on_exception(self):
        """例外発生時でも env.training が try/finally で必ず復元されること。"""
        env = _make_mock_env(n_envs=1)
        env.training = True

        # step が例外を起こすように設定
        env.step.side_effect = RuntimeError("テスト用例外")

        with pytest.raises(RuntimeError):
            run_survivors_eval_episodes(
                model=_make_mock_model(),
                env=env,
                n_eval_episodes=1,
                frame_skip=1,
                alive_reward=0.0,
            )

        assert env.training is True, "例外発生時でも env.training が元の True に戻ること"

    def test_env_training_false_during_eval(self):
        """評価中に env.training が False に設定されること。"""
        training_states_during_eval: list[bool] = []
        env = _make_mock_env(n_envs=1)
        env.training = True
        obs = np.zeros(8, dtype=np.float32)

        original_step = env.step

        def capturing_step(*args, **kwargs):
            training_states_during_eval.append(env.training)
            return (obs, np.zeros(1), np.array([True]), [{"base_reward": 10.0}])

        env.step = capturing_step

        run_survivors_eval_episodes(
            model=_make_mock_model(),
            env=env,
            n_eval_episodes=1,
            frame_skip=1,
            alive_reward=0.0,
        )

        assert all(s is False for s in training_states_during_eval), \
            "評価中は env.training が False であること"


class TestSurvivorsEvalCallbackSyncBeforeEval:
    """_sync_before_eval の動作テスト。"""

    def test_sync_vecnormalize_and_shaping_always_called(self):
        """params_provider の有無にかかわらず _sync_vecnormalize と _sync_shaping が呼ばれること。"""
        eval_env = _make_mock_env()
        cb = _make_eval_callback(eval_env=eval_env)
        _setup_callback(cb)

        with patch.object(cb, "_sync_vecnormalize") as mock_vn, \
             patch.object(cb, "_sync_shaping") as mock_sh, \
             patch.object(cb, "_sync_env_params") as mock_ep:
            cb._sync_before_eval()

        mock_vn.assert_called_once()
        mock_sh.assert_called_once()

    def test_sync_env_params_called_when_no_params_provider(self):
        """params_provider なしの場合、_sync_env_params が呼ばれること。"""
        eval_env = _make_mock_env()
        cb = _make_eval_callback(eval_env=eval_env, params_provider=None)
        _setup_callback(cb)

        with patch.object(cb, "_sync_vecnormalize"), \
             patch.object(cb, "_sync_shaping"), \
             patch.object(cb, "_sync_env_params") as mock_ep:
            cb._sync_before_eval()

        mock_ep.assert_called_once()

    def test_set_params_called_when_params_provider_given(self):
        """params_provider が渡された場合、eval_env.env_method('set_params', ...) が呼ばれること。"""
        eval_env = _make_mock_env()

        def mock_params_provider():
            return {"min_enemies": 5, "max_enemies": 10}

        cb = _make_eval_callback(eval_env=eval_env, params_provider=mock_params_provider)
        _setup_callback(cb)

        with patch.object(cb, "_sync_vecnormalize"), \
             patch.object(cb, "_sync_shaping"), \
             patch.object(cb, "_sync_env_params") as mock_ep:
            cb._sync_before_eval()

        # _sync_env_params は呼ばれないこと
        mock_ep.assert_not_called()

        # eval_env.env_method("set_params", ...) が呼ばれること
        call_args_list = eval_env.env_method.call_args_list
        set_params_calls = [c for c in call_args_list if c[0][0] == "set_params"]
        assert len(set_params_calls) >= 1, "set_params が呼ばれること"

        # UE5 API key に変換されていること
        kwargs = set_params_calls[0][1]
        assert "MinActiveEnemies" in kwargs, "内部 key が UE5 API key に変換されること"
        assert "MaxActiveEnemies" in kwargs, "内部 key が UE5 API key に変換されること"

    def test_sync_vecnormalize_and_shaping_called_even_with_params_provider(self):
        """params_provider がある場合でも _sync_vecnormalize と _sync_shaping が呼ばれること。"""
        eval_env = _make_mock_env()

        def mock_params_provider():
            return {"min_enemies": 5}

        cb = _make_eval_callback(eval_env=eval_env, params_provider=mock_params_provider)
        _setup_callback(cb)

        with patch.object(cb, "_sync_vecnormalize") as mock_vn, \
             patch.object(cb, "_sync_shaping") as mock_sh:
            cb._sync_before_eval()

        mock_vn.assert_called_once()
        mock_sh.assert_called_once()

    def test_no_sync_when_eval_env_is_none(self):
        """eval_env が None の場合、sync 処理が呼ばれないこと。"""
        # eval_env=None の場合は _sync_before_eval は何もしない
        cb = _make_eval_callback(eval_env=None)
        training_env = _make_mock_env()
        _setup_callback(cb, training_env=training_env)

        with patch.object(cb, "_sync_vecnormalize") as mock_vn, \
             patch.object(cb, "_sync_shaping") as mock_sh:
            cb._sync_before_eval()

        # eval_env=None の場合は sync 処理は行わない
        mock_vn.assert_not_called()
        mock_sh.assert_not_called()


class TestSurvivorsEvalCallbackAfterEvalCallback:
    """after_eval_callback の動作テスト。"""

    def test_after_eval_callback_called_with_results_and_metrics(self):
        """after_eval_callback が eval 結果（episode_results, metrics）で呼ばれること。"""
        eval_env = _make_mock_env()
        received_calls: list[tuple] = []

        def after_cb(episode_results, metrics):
            received_calls.append((episode_results, metrics))

        cb = _make_eval_callback(eval_env=eval_env, n_eval_episodes=2,
                                 after_eval_callback=after_cb)
        mock_model = _make_mock_model()
        mock_train_env = _make_mock_env()
        mock_model.get_env.return_value = mock_train_env
        cb.model = mock_model
        cb._training_env = mock_train_env
        cb.num_timesteps = 0

        with patch.object(cb, "_sync_before_eval"):
            cb._run_eval_and_log()

        assert len(received_calls) == 1, "after_eval_callback が1回呼ばれること"
        episode_results, metrics = received_calls[0]
        assert len(episode_results) == 2, "2エピソード分の結果が渡ること"
        assert "ep_length" in episode_results[0], "episode_results に ep_length キーがあること"
        assert "n_episodes" in metrics, "metrics に n_episodes が含まれること"

    def test_after_eval_callback_not_called_when_none(self):
        """after_eval_callback が None の場合はエラーなく動作すること。"""
        eval_env = _make_mock_env()
        cb = _make_eval_callback(eval_env=eval_env, after_eval_callback=None)
        mock_model = _make_mock_model()
        mock_train_env = _make_mock_env()
        mock_model.get_env.return_value = mock_train_env
        cb.model = mock_model
        cb._training_env = mock_train_env
        cb.num_timesteps = 0

        with patch.object(cb, "_sync_before_eval"):
            # 例外なく動作すること
            cb._run_eval_and_log()


class TestTrainingEnvCompatMode:
    """n_envs=1 旧互換パス（eval_env=None）のテスト。"""

    def test_last_obs_set_without_extra_reset(self):
        """旧互換パスで model._last_obs が追加 reset なしに設定されること。

        run_survivors_eval_episodes から返された last_obs を直接使い、
        eval 後に余分な env.reset() を呼ばないことを確認する。
        """
        # eval_env=None で training_env を直接使う旧互換パス
        training_env = _make_mock_env(n_envs=1)
        cb = _make_eval_callback(eval_env=None, n_eval_episodes=1)

        mock_model = _make_mock_model()
        mock_model.get_env.return_value = training_env
        cb.model = mock_model
        cb._training_env = training_env
        cb.num_timesteps = 0

        with patch.object(cb, "_sync_before_eval"):
            cb._run_eval_and_log()

        # model._last_obs が設定されていること
        assert mock_model._last_obs is not None, "model._last_obs が設定されていること"
        # model._last_episode_starts が設定されていること
        assert mock_model._last_episode_starts is not None, "model._last_episode_starts が設定されていること"

        # env.reset() が1回（eval 開始時）しか呼ばれていないこと（追加 reset なし）
        assert training_env.reset.call_count == 1, \
            f"eval 中に env.reset() は1回のみ（eval 開始時）。実際: {training_env.reset.call_count}回"

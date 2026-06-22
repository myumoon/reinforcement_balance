"""SurvivorsEvalCallback と run_survivors_eval_episodes のユニットテスト。"""
from __future__ import annotations

import sys
import threading
import time
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


# ---------------------------------------------------------------------------
# async eval テスト
# ---------------------------------------------------------------------------

def _make_async_callback(eval_env, eval_freq: int = 1000, n_eval_episodes: int = 1,
                         after_eval_callback=None, stale_check_fn=None) -> SurvivorsEvalCallback:
    """async eval 用テストコールバック。"""
    return SurvivorsEvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        frame_skip=1,
        alive_reward=0.0,
        params_provider=None,
        after_eval_callback=after_eval_callback,
        stale_check_fn=stale_check_fn,
        verbose=0,
    )


def _setup_async_callback(cb: SurvivorsEvalCallback, num_timesteps: int = 0) -> SurvivorsEvalCallback:
    """async テスト用のコールバックセットアップ（training_env は使わない）。"""
    training_env = _make_mock_env()
    mock_model = _make_mock_model()
    mock_model.get_env.return_value = training_env
    mock_model.policy = MagicMock()
    cb.model = mock_model
    cb._training_env = training_env
    cb.num_timesteps = num_timesteps
    cb.n_calls = 0
    return cb


class TestAsyncEvalLaunch:
    """ケース1: async eval 起動テスト。thread が起動して訓練が継続できることの確認。"""

    def test_eval_thread_starts_when_eval_env_present(self):
        """eval_env がある場合に _start_eval_async を呼ぶと thread が起動すること。"""
        eval_env = _make_mock_env()
        cb = _make_async_callback(eval_env=eval_env)
        _setup_async_callback(cb, num_timesteps=1000)

        with patch("games.survivors.survivors_eval_callback.run_survivors_eval_episodes") as mock_run:
            mock_run.return_value = ([], {
                "ep_length": 100.0, "active_score": 1.0, "base_reward": 1.0,
                "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
                "gem_pickups": 0.0, "kills": 0.0, "xp_progress": 0.0,
                "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
                "move_speed": 0.0, "stationary": 0.0, "terminated_ratio": 0.0, "n_episodes": 1,
            }, None)
            cb._start_eval_async()

        # thread が起動していること
        assert cb._eval_thread is not None
        assert cb._eval_result_queue is not None

        # thread が完了するまで待つ（最大 5 秒）
        cb._eval_thread.join(timeout=5.0)
        assert not cb._eval_thread.is_alive(), "thread が 5 秒以内に完了すること"

    def test_training_continues_while_eval_thread_runs(self):
        """eval thread が実行中でも _on_rollout_end が早期リターンすること（訓練継続可能）。"""
        eval_env = _make_mock_env()
        cb = _make_async_callback(eval_env=eval_env, eval_freq=1000)
        _setup_async_callback(cb, num_timesteps=1000)

        # eval を一度起動して thread が実行中の状態にする
        barrier = threading.Barrier(2)
        done_event = threading.Event()

        def slow_eval(*args, **kwargs):
            barrier.wait(timeout=5)  # メインスレッドが確認できるまで待つ
            done_event.wait(timeout=5)
            return ([], {
                "ep_length": 100.0, "active_score": 1.0, "base_reward": 1.0,
                "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
                "gem_pickups": 0.0, "kills": 0.0, "xp_progress": 0.0,
                "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
                "move_speed": 0.0, "stationary": 0.0, "terminated_ratio": 0.0, "n_episodes": 1,
            }, None)

        with patch("games.survivors.survivors_eval_callback.run_survivors_eval_episodes", side_effect=slow_eval):
            cb._start_eval_async()

        # eval thread が起動するまで待つ
        barrier.wait(timeout=5)

        # thread が実行中であること
        assert cb._eval_thread is not None
        assert cb._eval_thread.is_alive()

        # eval_freq に達していない場合は _start_eval_async が呼ばれないこと
        cb.num_timesteps = 1500  # 1000 + 500 < eval_freq(1000)
        cb._last_eval_step = 1000  # すでに1回評価済み

        start_async_called = []
        original_start = cb._start_eval_async
        with patch.object(cb, "_start_eval_async", wraps=cb._start_eval_async) as mock_start:
            with patch.object(cb, "_sync_before_eval"):
                cb._on_rollout_end()  # freq に達していない → 新しいスレッドは起動しない

        # eval_freq (1000) に達していないので新しい start_eval_async は呼ばれない
        mock_start.assert_not_called()

        # thread を解放
        done_event.set()
        cb._eval_thread.join(timeout=5)


class TestBackPressure:
    """ケース2: back-pressure テスト。thread が生きているときに join() して次が起動すること。"""

    def test_join_called_on_alive_thread_before_new_eval(self):
        """eval_freq トリガー時に thread が生きていれば join() してから次を起動すること。"""
        eval_env = _make_mock_env()
        cb = _make_async_callback(eval_env=eval_env, eval_freq=1000)
        _setup_async_callback(cb, num_timesteps=0)

        join_called = threading.Event()
        thread_completed = threading.Event()

        class SlowThread(threading.Thread):
            def run(self):
                thread_completed.wait(timeout=5)  # join まで待つ

        # 実行中のダミー thread をセット
        slow_thread = SlowThread(daemon=True)
        slow_thread.start()
        cb._eval_thread = slow_thread

        import queue as _queue
        result_q: _queue.Queue = _queue.Queue()
        result_q.put(([], {
            "ep_length": 100.0, "active_score": 1.0, "base_reward": 1.0,
            "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
            "gem_pickups": 0.0, "kills": 0.0, "xp_progress": 0.0,
            "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
            "move_speed": 0.0, "stationary": 0.0, "terminated_ratio": 0.0, "n_episodes": 1,
        }, None, None))
        cb._eval_result_queue = result_q

        # eval_freq に達した状態を設定
        cb.num_timesteps = 2000
        cb._last_eval_step = 0

        new_start_called = threading.Event()

        def signal_start():
            # thread 解放
            thread_completed.set()

        # _start_eval_async の前に thread が終了するようにする
        original_start_async = cb._start_eval_async

        def patched_start():
            new_start_called.set()

        with patch.object(cb, "_sync_before_eval"):
            with patch.object(cb, "_start_eval_async", side_effect=patched_start):
                with patch.object(cb, "_process_eval_result"):
                    # join が必要になったとき thread を完了させる
                    thread_completed.set()
                    cb._on_rollout_end()

        assert new_start_called.is_set(), "前の thread が join されて新しい eval が起動すること"


class TestAfterEvalCallbackMainThread:
    """ケース3: after_eval_callback が join() 後のメインスレッドで呼ばれることの確認。"""

    def test_after_eval_callback_called_in_main_thread(self):
        """after_eval_callback がメインスレッドで実行されること。"""
        eval_env = _make_mock_env()
        callback_thread_ids: list[int] = []

        def after_cb(episode_results, metrics):
            callback_thread_ids.append(threading.current_thread().ident)

        cb = _make_async_callback(eval_env=eval_env, after_eval_callback=after_cb)
        _setup_async_callback(cb, num_timesteps=1000)

        main_thread_id = threading.current_thread().ident

        with patch("games.survivors.survivors_eval_callback.run_survivors_eval_episodes") as mock_run:
            mock_run.return_value = ([
                {
                    "ep_length": 100, "active_score": 1.0, "base_reward": 1.0,
                    "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
                    "gem_pickups": 0, "kills": 0, "xp_progress": 0.0,
                    "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
                    "move_speed": 0.0, "stationary": 0.0, "terminated": 0,
                    "weapon_acquired": {}, "first_weapon_id": None,
                }
            ], {
                "ep_length": 100.0, "active_score": 1.0, "base_reward": 1.0,
                "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
                "gem_pickups": 0.0, "kills": 0.0, "xp_progress": 0.0,
                "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
                "move_speed": 0.0, "stationary": 0.0, "terminated_ratio": 0.0, "n_episodes": 1,
            }, None)
            cb._start_eval_async()

        # thread が完了するまで待つ
        cb._eval_thread.join(timeout=5)
        assert not cb._eval_thread.is_alive()

        # _try_process_pending_eval_result を呼んで after_eval_callback を実行
        with patch.object(cb, "_log_results"):
            cb._try_process_pending_eval_result()

        assert len(callback_thread_ids) == 1, "after_eval_callback が1回呼ばれること"
        assert callback_thread_ids[0] == main_thread_id, "after_eval_callback がメインスレッドで呼ばれること"


class TestSyncCompatMode:
    """ケース4: n_envs=1 互換（eval_env=None）が sync 動作を維持することの確認。"""

    def test_sync_path_used_when_eval_env_is_none(self):
        """eval_env=None の場合は _run_eval_and_log が呼ばれること（async thread ではない）。"""
        cb = _make_async_callback(eval_env=None, eval_freq=1000)
        training_env = _make_mock_env()
        mock_model = _make_mock_model()
        mock_model.get_env.return_value = training_env
        cb.model = mock_model
        cb._training_env = training_env
        cb.num_timesteps = 1000
        cb.n_calls = 0

        run_eval_called = []

        with patch.object(cb, "_run_eval_and_log", side_effect=lambda: run_eval_called.append(True)):
            with patch.object(cb, "_start_eval_async") as mock_start_async:
                cb._on_rollout_end()

        assert len(run_eval_called) == 1, "_run_eval_and_log が呼ばれること"
        mock_start_async.assert_not_called(), "_start_eval_async は呼ばれないこと"
        assert cb._eval_thread is None, "async thread は起動していないこと"

    def test_sync_path_does_not_modify_eval_thread(self):
        """eval_env=None の sync パスで _eval_thread が None のままであること。"""
        cb = _make_async_callback(eval_env=None, eval_freq=100)
        training_env = _make_mock_env()
        mock_model = _make_mock_model()
        mock_model.get_env.return_value = training_env
        cb.model = mock_model
        cb._training_env = training_env
        cb.num_timesteps = 100

        with patch.object(cb, "_run_eval_and_log"):
            with patch.object(cb, "_sync_before_eval"):
                cb._on_rollout_end()

        assert cb._eval_thread is None


class TestNewPhaseParamsAfterPromotion:
    """ケース5: 昇格後に次の eval が新フェーズ params で起動することの確認。"""

    def test_params_provider_called_at_eval_start(self):
        """_start_eval_async 呼び出し前に _sync_before_eval（params_provider 含む）が実行されること。"""
        eval_env = _make_mock_env()
        phase_at_sync = []

        current_phase = [0]  # mutable container

        def params_provider():
            phase_at_sync.append(current_phase[0])
            return {"min_enemies": 5}

        cb = SurvivorsEvalCallback(
            eval_env=eval_env,
            eval_freq=1000,
            n_eval_episodes=1,
            frame_skip=1,
            alive_reward=0.0,
            params_provider=params_provider,
            verbose=0,
        )
        _setup_async_callback(cb, num_timesteps=1000)
        # env_method の返り値を呼び出しごとに切り替える:
        # get_shaping_weight → [0.5]、get_params → [{"MinActiveEnemies": 3}]
        def env_method_side_effect(method_name, *args, **kwargs):
            if method_name == "get_shaping_weight":
                return [0.5]
            if method_name == "get_params":
                return [{"MinActiveEnemies": 3}]
            return []
        cb._training_env.env_method.side_effect = env_method_side_effect

        with patch.object(cb, "_start_eval_async"):
            cb._on_rollout_end()

        # params_provider が _sync_before_eval の中で呼ばれること
        assert len(phase_at_sync) >= 1, "params_provider が呼ばれること"


class TestStaleResultDiscard:
    """ケース6: stale result 破棄テスト。"""

    def test_after_eval_callback_skipped_when_stale(self):
        """eval 実行中に phase が変わった場合 after_eval_callback がスキップされること。"""
        eval_env = _make_mock_env()
        after_called = []
        log_called = []

        def after_cb(episode_results, metrics):
            after_called.append(True)

        phase_counter = [0]

        def stale_check_fn():
            return phase_counter[0]

        cb = _make_async_callback(
            eval_env=eval_env,
            after_eval_callback=after_cb,
            stale_check_fn=stale_check_fn,
        )
        _setup_async_callback(cb, num_timesteps=2000)

        # stale_snapshot=0 で結果を用意し、処理時には phase=1 になっている
        phase_counter[0] = 1  # phase が変化した状態

        fake_result = ([
            {
                "ep_length": 100, "active_score": 1.0, "base_reward": 1.0,
                "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
                "gem_pickups": 0, "kills": 0, "xp_progress": 0.0,
                "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
                "move_speed": 0.0, "stationary": 0.0, "terminated": 0,
                "weapon_acquired": {}, "first_weapon_id": None,
            }
        ], {
            "ep_length": 100.0, "active_score": 1.0, "base_reward": 1.0,
            "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
            "gem_pickups": 0.0, "kills": 0.0, "xp_progress": 0.0,
            "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
            "move_speed": 0.0, "stationary": 0.0, "terminated_ratio": 0.0, "n_episodes": 1,
        }, 0, None)  # stale_snapshot=0 (古い phase)

        with patch.object(cb, "_log_results") as mock_log:
            cb._process_eval_result(fake_result)

        # after_eval_callback はスキップされること
        assert len(after_called) == 0, "stale result のため after_eval_callback がスキップされること"
        # _log_results（eval metrics）は記録されること
        mock_log.assert_called_once()

    def test_after_eval_callback_called_when_not_stale(self):
        """phase が変化していない場合は after_eval_callback が呼ばれること。"""
        eval_env = _make_mock_env()
        after_called = []

        def after_cb(episode_results, metrics):
            after_called.append(True)

        phase_counter = [0]

        def stale_check_fn():
            return phase_counter[0]

        cb = _make_async_callback(
            eval_env=eval_env,
            after_eval_callback=after_cb,
            stale_check_fn=stale_check_fn,
        )
        _setup_async_callback(cb, num_timesteps=2000)

        # phase は変化していない
        fake_result = ([
            {
                "ep_length": 100, "active_score": 1.0, "base_reward": 1.0,
                "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
                "gem_pickups": 0, "kills": 0, "xp_progress": 0.0,
                "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
                "move_speed": 0.0, "stationary": 0.0, "terminated": 0,
                "weapon_acquired": {}, "first_weapon_id": None,
            }
        ], {
            "ep_length": 100.0, "active_score": 1.0, "base_reward": 1.0,
            "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
            "gem_pickups": 0.0, "kills": 0.0, "xp_progress": 0.0,
            "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
            "move_speed": 0.0, "stationary": 0.0, "terminated_ratio": 0.0, "n_episodes": 1,
        }, 0, None)  # stale_snapshot=0 と現在の phase=0 が一致

        with patch.object(cb, "_log_results"):
            cb._process_eval_result(fake_result)

        assert len(after_called) == 1, "stale でないため after_eval_callback が呼ばれること"


class TestOnTrainingEnd:
    """ケース7: _on_training_end() が eval thread 実行中に呼ばれたとき join して結果処理すること。"""

    def test_on_training_end_joins_alive_thread(self):
        """_on_training_end が実行中の thread を join すること。"""
        eval_env = _make_mock_env()
        cb = _make_async_callback(eval_env=eval_env)
        _setup_async_callback(cb)

        process_result_called = []
        join_called = threading.Event()

        import queue as _queue
        result_q: _queue.Queue = _queue.Queue()
        result_q.put(([], {
            "ep_length": 100.0, "active_score": 1.0, "base_reward": 1.0,
            "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
            "gem_pickups": 0.0, "kills": 0.0, "xp_progress": 0.0,
            "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
            "move_speed": 0.0, "stationary": 0.0, "terminated_ratio": 0.0, "n_episodes": 1,
        }, None, None))
        cb._eval_result_queue = result_q

        # 短時間で終了する thread を作成
        def quick_worker():
            pass

        t = threading.Thread(target=quick_worker, daemon=True)
        t.start()
        t.join()  # すぐ完了させる
        cb._eval_thread = t

        with patch.object(cb, "_process_eval_result", side_effect=lambda r: process_result_called.append(r)):
            cb._on_training_end()

        assert cb._eval_thread is None
        assert cb._eval_result_queue is None
        assert len(process_result_called) == 1, "_on_training_end が結果を処理すること"


class TestWorkerException:
    """ケース8: worker 例外テスト。eval thread 内で例外が起きたとき訓練が継続すること。"""

    def test_worker_exception_does_not_propagate(self):
        """eval worker 内で例外が起きても _process_eval_result がスキップして続行すること。"""
        eval_env = _make_mock_env()
        cb = _make_async_callback(eval_env=eval_env)
        _setup_async_callback(cb, num_timesteps=1000)

        exc = RuntimeError("テスト用 worker 例外")
        fake_result = (None, None, None, exc)

        # 例外なく実行できること
        cb._process_eval_result(fake_result)

        # after_eval_callback は呼ばれないこと（exception path は return する）
        # 訓練が継続できること（例外が再 raise されないこと）

    def test_worker_exception_with_verbose2_prints_traceback(self, capsys):
        """verbose >= 2 の場合は traceback が出力されること。"""
        eval_env = _make_mock_env()
        cb = SurvivorsEvalCallback(
            eval_env=eval_env,
            eval_freq=1000,
            n_eval_episodes=1,
            frame_skip=1,
            alive_reward=0.0,
            verbose=2,
        )
        _setup_async_callback(cb, num_timesteps=1000)

        try:
            raise ValueError("テスト例外")
        except ValueError as exc:
            fake_result = (None, None, None, exc)

        cb._process_eval_result(fake_result)
        captured = capsys.readouterr()
        assert "テスト例外" in captured.out or len(captured.out) > 0


class TestEvalRunningMetrics:
    """ケース9: eval/running 0/1 が eval start/end 時に正しい step でログされること。"""

    def test_eval_running_1_logged_on_async_start(self):
        """async eval 起動成功後に eval/running=1 が現在 step でログされること。"""
        eval_env = _make_mock_env()
        cb = _make_async_callback(eval_env=eval_env, eval_freq=1000)
        _setup_async_callback(cb, num_timesteps=1000)

        mock_wandb = MagicMock()
        mock_wandb.enabled = True
        cb._wandb_logger = mock_wandb

        with patch.object(cb, "_sync_before_eval"):
            with patch.object(cb, "_start_eval_async"):
                cb._on_rollout_end()

        # eval/running=1 がログされること
        logged_calls = mock_wandb.log.call_args_list
        running_1_calls = [c for c in logged_calls if c[0][0].get("eval/running") == 1]
        assert len(running_1_calls) >= 1, "eval/running=1 がログされること"
        # step が現在の num_timesteps であること
        assert running_1_calls[0][1]["step"] == 1000

    def test_eval_running_0_logged_on_result_process(self):
        """結果処理時（_try_process_pending_eval_result）に eval/running=0 がログされること。"""
        eval_env = _make_mock_env()
        cb = _make_async_callback(eval_env=eval_env)
        _setup_async_callback(cb, num_timesteps=5000)

        mock_wandb = MagicMock()
        mock_wandb.enabled = True
        cb._wandb_logger = mock_wandb

        import queue as _queue
        result_q: _queue.Queue = _queue.Queue()
        result_q.put(([
            {
                "ep_length": 100, "active_score": 1.0, "base_reward": 1.0,
                "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
                "gem_pickups": 0, "kills": 0, "xp_progress": 0.0,
                "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
                "move_speed": 0.0, "stationary": 0.0, "terminated": 0,
                "weapon_acquired": {}, "first_weapon_id": None,
            }
        ], {
            "ep_length": 100.0, "active_score": 1.0, "base_reward": 1.0,
            "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
            "gem_pickups": 0.0, "kills": 0.0, "xp_progress": 0.0,
            "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
            "move_speed": 0.0, "stationary": 0.0, "terminated_ratio": 0.0, "n_episodes": 1,
        }, None, None))
        cb._eval_result_queue = result_q

        # 完了済みの thread を作成
        t = threading.Thread(target=lambda: None, daemon=True)
        t.start()
        t.join()
        cb._eval_thread = t

        with patch.object(cb, "_log_results"):
            cb._try_process_pending_eval_result()

        logged_calls = mock_wandb.log.call_args_list
        running_0_calls = [c for c in logged_calls if c[0][0].get("eval/running") == 0]
        assert len(running_0_calls) >= 1, "eval/running=0 がログされること"
        # step が join 完了時（5000）であること
        assert running_0_calls[0][1]["step"] == 5000


class TestBackwardCompatAfterEvalCallback:
    """ケース10: 既存 2 引数 after_eval_callback が動作すること（後方互換）。"""

    def test_two_arg_callback_works(self):
        """step 引数なしの after_eval_callback が正常に呼ばれること。"""
        eval_env = _make_mock_env()
        received_calls: list[tuple] = []

        def after_cb(episode_results, metrics):
            received_calls.append((episode_results, metrics))

        cb = _make_async_callback(eval_env=eval_env, after_eval_callback=after_cb)
        _setup_async_callback(cb, num_timesteps=1000)

        assert not cb._after_eval_callback_supports_step, \
            "2 引数 callback は step 非対応と判定されること"

        fake_result = ([
            {
                "ep_length": 100, "active_score": 1.0, "base_reward": 1.0,
                "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
                "gem_pickups": 0, "kills": 0, "xp_progress": 0.0,
                "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
                "move_speed": 0.0, "stationary": 0.0, "terminated": 0,
                "weapon_acquired": {}, "first_weapon_id": None,
            }
        ], {
            "ep_length": 100.0, "active_score": 1.0, "base_reward": 1.0,
            "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
            "gem_pickups": 0.0, "kills": 0.0, "xp_progress": 0.0,
            "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
            "move_speed": 0.0, "stationary": 0.0, "terminated_ratio": 0.0, "n_episodes": 1,
        }, None, None)

        with patch.object(cb, "_log_results"):
            cb._process_eval_result(fake_result)

        assert len(received_calls) == 1, "2 引数 callback が呼ばれること"
        episode_results, metrics = received_calls[0]
        assert isinstance(episode_results, list)
        assert isinstance(metrics, dict)

    def test_three_arg_callback_with_step_works(self):
        """step 引数ありの after_eval_callback が step 付きで呼ばれること。"""
        eval_env = _make_mock_env()
        received_calls: list[tuple] = []

        def after_cb(episode_results, metrics, step=None):
            received_calls.append((episode_results, metrics, step))

        cb = _make_async_callback(eval_env=eval_env, after_eval_callback=after_cb)
        _setup_async_callback(cb, num_timesteps=3000)

        assert cb._after_eval_callback_supports_step, \
            "step 引数ありの callback は step 対応と判定されること"

        fake_result = ([
            {
                "ep_length": 100, "active_score": 1.0, "base_reward": 1.0,
                "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
                "gem_pickups": 0, "kills": 0, "xp_progress": 0.0,
                "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
                "move_speed": 0.0, "stationary": 0.0, "terminated": 0,
                "weapon_acquired": {}, "first_weapon_id": None,
            }
        ], {
            "ep_length": 100.0, "active_score": 1.0, "base_reward": 1.0,
            "shaped_reward": 0.0, "hp": 1.0, "hp_min": 1.0, "damage_taken": 0.0,
            "gem_pickups": 0.0, "kills": 0.0, "xp_progress": 0.0,
            "gem_dist": 0.0, "enemy_dist": 0.0, "contact": 0.0,
            "move_speed": 0.0, "stationary": 0.0, "terminated_ratio": 0.0, "n_episodes": 1,
        }, None, None)

        with patch.object(cb, "_log_results"):
            cb._process_eval_result(fake_result)

        assert len(received_calls) == 1, "step 引数ありの callback が呼ばれること"
        _, _, step = received_calls[0]
        assert step == 3000, "step が現在の num_timesteps (3000) であること"

"""PerfLoggingCallback のユニットテスト。"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import math
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# SB3が使えない環境ではスキップ
try:
    from stable_baselines3.common.callbacks import BaseCallback
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

if _SB3_AVAILABLE:
    from common.perf_callback import PerfLoggingCallback


pytestmark = pytest.mark.skipif(not _SB3_AVAILABLE, reason="stable_baselines3 not available")


# ---------------------------------------------------------------------------
# ヘルパー: モックモデル/環境を用いたコールバック起動
# ---------------------------------------------------------------------------

def _make_callback(log_dir: Path | None = None, window_size: int = 5) -> PerfLoggingCallback:
    cb = PerfLoggingCallback(log_dir=log_dir, window_size=window_size)
    model = MagicMock()
    model.num_timesteps = 0
    model.n_steps = 128
    model.n_envs = 2  # rollout_size = 256
    cb.init_callback(model)
    return cb


def _simulate_iteration(cb: PerfLoggingCallback, timestep_delta: int = 256, elapsed_rollout: float = 1.0, elapsed_train: float = 0.5):
    """1イテレーション分のフックを手動で発火させる。"""
    # rollout_start → rollout_end → (次の) rollout_start
    with patch("time.perf_counter", side_effect=[
        0.0,               # _on_rollout_start: t0
        elapsed_rollout,   # _on_rollout_end: t1
        elapsed_rollout + elapsed_train,  # 次の _on_rollout_start: t2
    ]):
        cb._on_rollout_start()
        cb.model.num_timesteps += timestep_delta // 2  # rollout_end 前
        cb._on_rollout_end()
        cb.model.num_timesteps += timestep_delta // 2  # 次rollout_start 前
        cb._on_rollout_start()  # ← ここでメトリクスが記録される


# ---------------------------------------------------------------------------
# テスト
# ---------------------------------------------------------------------------

def test_initial_state():
    cb = _make_callback()
    assert cb._rollout_t0 is None
    assert cb._rollout_end_t is None
    assert len(cb._iter_fps_deque) == 0


def test_rollout_start_sets_t0():
    cb = _make_callback()
    t = time.perf_counter()
    with patch("time.perf_counter", return_value=t):
        cb._on_rollout_start()
    assert cb._rollout_t0 == t


def test_rollout_end_sets_t1():
    cb = _make_callback()
    with patch("time.perf_counter", return_value=1.0):
        cb._on_rollout_start()
    with patch("time.perf_counter", return_value=3.0):
        cb._on_rollout_end()
    assert cb._rollout_end_t == 3.0


def test_metrics_logged_on_second_rollout_start():
    """2回目の _on_rollout_start でメトリクスが生成されることを確認。"""
    cb = _make_callback(window_size=3)
    cb._session_start_steps = 0
    cb._session_start_t = 0.0

    recorded = {}

    def fake_log(metrics, step):
        recorded.update(metrics)

    cb._wandb_log = fake_log

    with patch("time.perf_counter", side_effect=[0.0, 1.0, 1.5]):
        cb.model.num_timesteps = 0
        cb._on_rollout_start()
        cb.model.num_timesteps = 256
        cb._on_rollout_end()
        cb.model.num_timesteps = 256
        cb._on_rollout_start()

    assert "perf/rollout_fps" in recorded
    assert "perf/train_fps" in recorded
    assert "perf/iter_fps" in recorded
    assert "perf/rollout_time_s" in recorded
    assert "perf/train_time_s" in recorded
    assert "perf/session_fps" in recorded


def test_fps_values_are_positive():
    cb = _make_callback(window_size=3)
    cb._session_start_steps = 0
    cb._session_start_t = 0.0

    recorded = {}

    def fake_log(metrics, step):
        recorded.update(metrics)

    cb._wandb_log = fake_log

    with patch("time.perf_counter", side_effect=[0.0, 2.0, 3.0]):
        cb.model.num_timesteps = 0
        cb._on_rollout_start()
        cb.model.num_timesteps = 256
        cb._on_rollout_end()
        cb._on_rollout_start()

    # rollout_fps = 256 steps / 2.0s = 128
    assert abs(recorded["perf/rollout_fps"] - 128.0) < 1.0
    # train_fps = 256 steps / 1.0s = 256
    assert abs(recorded["perf/train_fps"] - 256.0) < 1.0
    # iter_fps = 256 steps / 3.0s = ~85.3
    assert abs(recorded["perf/iter_fps"] - 256.0 / 3.0) < 1.0


def test_moving_average_window():
    """window_size=2 のとき iter_fps が移動平均になることを確認。"""
    cb = _make_callback(window_size=2)
    cb._session_start_steps = 0
    cb._session_start_t = 0.0

    fps_values = []

    def fake_log(metrics, step):
        if "perf/iter_fps" in metrics:
            fps_values.append(metrics["perf/iter_fps"])

    cb._wandb_log = fake_log

    # イテレーション1: 256 steps / 2.0s = 128 fps
    with patch("time.perf_counter", side_effect=[0.0, 1.0, 2.0]):
        cb.model.num_timesteps = 0
        cb._on_rollout_start()
        cb.model.num_timesteps = 256
        cb._on_rollout_end()
        cb._on_rollout_start()

    # イテレーション2: iter_time = (t2 - t0) = (6.0 - 2.0) = 4.0s → 256 / 4.0 = 64 fps
    # 移動平均 window=[128, 64] → average=96
    # t0=2.0 (前イテレーションの t2), t1=4.0, t2=6.0 (単調増加を保証)
    with patch("time.perf_counter", side_effect=[4.0, 6.0]):
        cb.model.num_timesteps = 256
        cb._on_rollout_end()
        cb._on_rollout_start()

    assert len(fps_values) == 2
    # 2回目: window=[128, 64] → average=96
    assert abs(fps_values[1] - 96.0) < 1.0


def test_jsonl_written(tmp_path):
    """JSONL ファイルにメトリクスが追記されることを確認。"""
    cb = _make_callback(log_dir=tmp_path, window_size=1)
    cb._session_start_steps = 0
    cb._session_start_t = 0.0

    with patch("time.perf_counter", side_effect=[0.0, 1.0, 1.5]):
        cb.model.num_timesteps = 0
        cb._on_rollout_start()
        cb.model.num_timesteps = 256
        cb._on_rollout_end()
        cb._on_rollout_start()

    jsonl_path = tmp_path / "perf_log.jsonl"
    assert jsonl_path.exists(), "JSONL ファイルが生成されていない"

    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) >= 1

    data = json.loads(lines[0])
    assert "perf/iter_fps" in data
    assert "timestep" in data


def test_jsonl_append_mode(tmp_path):
    """JSONL が追記モードで書き込まれ、複数イテレーション分が記録されることを確認。"""
    cb = _make_callback(log_dir=tmp_path, window_size=1)
    cb._session_start_steps = 0
    cb._session_start_t = 0.0

    # イテレーション1
    with patch("time.perf_counter", side_effect=[0.0, 1.0, 1.5]):
        cb.model.num_timesteps = 0
        cb._on_rollout_start()
        cb.model.num_timesteps = 256
        cb._on_rollout_end()
        cb._on_rollout_start()

    # イテレーション2
    with patch("time.perf_counter", side_effect=[0.0, 1.0, 1.5]):
        cb.model.num_timesteps = 256
        cb._on_rollout_end()
        cb._on_rollout_start()

    jsonl_path = tmp_path / "perf_log.jsonl"
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) >= 2


def test_session_fps_excludes_pre_session_steps():
    """resume 時: セッション開始前のステップを除外して session_fps を計算することを確認。"""
    cb = _make_callback(window_size=1)

    # resume 相当: セッション開始時点では 1000 steps 済み
    cb._session_start_steps = 1000
    cb._session_start_t = 0.0

    recorded = {}

    def fake_log(metrics, step):
        recorded.update(metrics)

    cb._wandb_log = fake_log

    # 256 steps 追加して合計 1256
    with patch("time.perf_counter", side_effect=[0.0, 1.0, 2.0]):
        cb.model.num_timesteps = 1000
        cb._on_rollout_start()
        cb.model.num_timesteps = 1256
        cb._on_rollout_end()
        cb._on_rollout_start()

    # session_fps = (1256 - 1000) / 2.0 = 128
    assert abs(recorded["perf/session_fps"] - 128.0) < 1.0


def test_no_crash_without_wandb(tmp_path):
    """W&B 未インストール環境でもクラッシュしないことを確認。"""
    cb = _make_callback(log_dir=tmp_path, window_size=1)
    cb._session_start_steps = 0
    cb._session_start_t = 0.0

    # W&B のログメソッドをモックで None 相当に
    original_log = cb._wandb_log
    cb._wandb_log = None  # type: ignore[assignment]

    with patch("time.perf_counter", side_effect=[0.0, 1.0, 1.5]):
        cb.model.num_timesteps = 0
        try:
            cb._on_rollout_start()
            cb.model.num_timesteps = 256
            cb._on_rollout_end()
            cb._on_rollout_start()
        except Exception as e:
            pytest.fail(f"W&B なし環境でクラッシュ: {e}")

    # JSONL は書き込まれているはず
    assert (tmp_path / "perf_log.jsonl").exists()


def test_training_start_captures_session_baseline():
    """_on_training_start でセッション開始時のステップ数と時刻を記録することを確認。"""
    cb = _make_callback(window_size=1)
    cb.model.num_timesteps = 500

    t_start = 12345.0
    with patch("time.perf_counter", return_value=t_start):
        cb._on_training_start()

    assert cb._session_start_steps == 500
    assert cb._session_start_t == t_start

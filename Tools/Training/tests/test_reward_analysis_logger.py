"""RewardAnalysisLogger のユニットテスト。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tempfile
from pathlib import Path
from common.reward_analysis_logger import RewardAnalysisLogger, RewardAnalysisCheckpointCallback, SURVIVORS_OBS_SCHEMA


def _make_logger():
    return RewardAnalysisLogger(obs_schema=SURVIVORS_OBS_SCHEMA, game="survivors", version="v10")


def test_on_step_accumulates():
    lg = _make_logger()
    for _ in range(100):
        lg.on_step(0.1, 1.0)
    assert len(lg._shaped_buf) == 100


def test_on_episode_end():
    lg = _make_logger()
    for _ in range(10):
        lg.on_step(0.1, 1.0)
    lg.on_episode_end()
    assert len(lg.ep_shaped_totals) == 1
    assert abs(lg.ep_shaped_totals[0] - 1.0) < 1e-6


def test_to_markdown_runs():
    lg = _make_logger()
    obs = np.zeros(740)
    for i in range(200):
        lg.on_step(0.05 if i % 3 == 0 else 0.0, 1.0, obs)
        if i % 20 == 19:
            lg.on_episode_end()
    md = lg.to_markdown()
    assert "報酬解析レポート" in md
    assert "shaped_reward 分布" in md
    assert "エピソード内時系列" in md


def test_save_creates_files():
    lg = _make_logger()
    for _ in range(50):
        lg.on_step(0.1, 1.0)
    lg.on_episode_end()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = lg.save(Path(tmpdir))
        assert path.exists()
        assert (Path(tmpdir) / "reward_analysis.json").exists()


def test_multi_env_episode_isolation():
    """複数 env のエピソードが混線しないことを確認する。"""
    lg = _make_logger()
    # env 0: 5ステップのエピソード (shaped=1.0)
    for _ in range(5):
        lg.on_step(1.0, 2.0, env_idx=0)
    # env 1: 3ステップ途中（done まだ来ない）
    for _ in range(3):
        lg.on_step(0.5, 1.0, env_idx=1)
    # env 0 のみエピソード終了
    lg.on_episode_end(env_idx=0)
    # エピソード完了は env 0 の 1 件のみ
    assert len(lg.ep_shaped_totals) == 1
    assert abs(lg.ep_shaped_totals[0] - 5.0) < 1e-6  # env 0 の合計
    # env 1 のデータは残っている（まだ done していない）
    assert 1 in lg._ep_shaped_per_env
    assert len(lg._ep_shaped_per_env[1]) == 3


def test_checkpoint_callback_saves_files():
    """RewardAnalysisCheckpointCallback がチェックポイントごとにファイルを保存することを確認する。"""
    lg = _make_logger()
    for i in range(500):
        lg.on_step(0.1 if i % 2 == 0 else -0.05, 1.0)
        if i % 50 == 49:
            lg.on_episode_end()

    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir)
        cb = RewardAnalysisCheckpointCallback(logger=lg, log_dir=log_dir, save_freq=1000, run_name="test-run")

        # save_freq=1000 で step=500 のとき保存しないことを確認
        cb._last_save = 0
        cb.num_timesteps = 500
        cb._on_step()
        assert not (log_dir / "running_reward_analysis_500.md").exists()

        # step=1000 で保存することを確認
        cb.num_timesteps = 1000
        cb._on_step()
        assert (log_dir / "running_reward_analysis_1000.md").exists()
        assert (log_dir / "running_reward_analysis_1000.json").exists()

        # step=2000 で再度保存することを確認（データはリセットされていない）
        cb.num_timesteps = 2000
        cb._on_step()
        assert (log_dir / "running_reward_analysis_2000.md").exists()

        # 各チェックポイントが独立したファイルとして存在することを確認
        import json as _json
        data_1000 = _json.loads((log_dir / "running_reward_analysis_1000.json").read_text())
        data_2000 = _json.loads((log_dir / "running_reward_analysis_2000.json").read_text())
        # step=2000 の時点でも同じ累積データ（lg のバッファは変わっていない）
        assert data_1000["n_steps"] == data_2000["n_steps"]


def test_checkpoint_callback_accumulates_not_reset():
    """チェックポイント保存でデータがリセットされないことを確認する。"""
    lg = _make_logger()
    for _ in range(100):
        lg.on_step(0.1, 1.0)
    lg.on_episode_end()
    n_steps_before = len(lg._shaped_buf)

    with tempfile.TemporaryDirectory() as tmpdir:
        cb = RewardAnalysisCheckpointCallback(logger=lg, log_dir=Path(tmpdir), save_freq=1000)
        cb._last_save = 0
        cb.num_timesteps = 1000
        cb._on_step()

    # 保存後もバッファが維持されていることを確認
    assert len(lg._shaped_buf) == n_steps_before

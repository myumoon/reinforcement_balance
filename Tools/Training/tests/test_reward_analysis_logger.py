"""RewardAnalysisLogger のユニットテスト。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tempfile
from pathlib import Path
from common.reward_analysis_logger import RewardAnalysisLogger, SURVIVORS_OBS_SCHEMA


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

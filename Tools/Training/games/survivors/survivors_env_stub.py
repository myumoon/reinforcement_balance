"""Survivors ゲームのスタブ環境（UE5 接続なし dry-run 用）。"""

import numpy as np
import gymnasium as gym

_OBS_DIM = 279  # 183 + 96 (方向別密度/最近傍距離 16方向×6セグメント)
_NUM_ACTIONS = 9

# UE5 GetObsSchema() と一致するオフセット・スキーマ（dry-run 時の extractor 用）
_OBS_SCHEMA = [
    {"name": "player_pos",               "dim": 2},
    {"name": "player_vel",               "dim": 2},
    {"name": "wall_rays",                "dim": 8},
    {"name": "player_hp",                "dim": 1},
    {"name": "weapon_slots",             "dim": 6},
    {"name": "enemy_count",              "dim": 1},
    {"name": "elapsed_time",             "dim": 1},
    {"name": "xp_progress",              "dim": 1},
    {"name": "player_level",             "dim": 1},
    {"name": "gem_rel_pos",              "dim": 40},
    {"name": "enemy_rel_pos",            "dim": 40},
    {"name": "enemy_vel",                "dim": 40},
    {"name": "enemy_type",               "dim": 20},
    {"name": "enemy_hp",                 "dim": 20},
    {"name": "enemy_nearest_dist_16dir", "dim": 16},
    {"name": "enemy_density_near_16dir", "dim": 16},
    {"name": "enemy_density_mid_16dir",  "dim": 16},
    {"name": "gem_nearest_dist_16dir",   "dim": 16},
    {"name": "gem_density_near_16dir",   "dim": 16},
    {"name": "gem_density_mid_16dir",    "dim": 16},
]

_OBS_OFFSETS: dict[str, int] = {}
_offset = 0
for _seg in _OBS_SCHEMA:
    _OBS_OFFSETS[_seg["name"]] = _offset
    _offset += _seg["dim"]


class DummySurvivorsEnv(gym.Env):
    """ランダム観測を返すスタブ。SB3 の訓練ループ疎通確認用。"""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(_NUM_ACTIONS)
        self._step_count = 0
        self._offsets: dict[str, int] = _OBS_OFFSETS
        self._obs_schema: list[dict] = _OBS_SCHEMA

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        obs[_OBS_OFFSETS["player_hp"]] = 1.0  # player_hp = 1.0 (満タン)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        obs[_OBS_OFFSETS["player_hp"]] = max(0.0, 1.0 - self._step_count / 1000.0)
        reward = 0.005  # AliveReward
        done = self._step_count >= 500
        return obs, reward, done, False, {"base_reward": reward, "shaped_reward": 0.0}

    def set_params(self, **kwargs) -> bool:
        return True

    def close(self):
        super().close()

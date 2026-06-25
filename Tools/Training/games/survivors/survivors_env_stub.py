"""Survivors ゲームのスタブ環境（UE5 接続なし dry-run 用）。"""

import numpy as np
import gymnasium as gym

# obs schema v795_projectiles_stride9 に対応（SurvivorsGameLogic.cpp GetObsSchema() と一致させる）
# MaxWeaponSlots=6, MaxPassiveSlots=6, MaxEnemyObs=32, MaxRedGemObs=10,
# MaxGreenGemObs=12, MaxBlueGemObs=12, MaxProjectileObs=32, ProjectileObsStride=9,
# MaxFloorPickupObs=8, MaxSpecialPickupObs=3, MaxDestructibleObs=10,
# EnemyDensityDirCount=16, GemDensityDirCount=16
_OBS_SCHEMA = [
    {"name": "player_pos",                  "dim": 2},
    {"name": "player_vel",                  "dim": 2},
    {"name": "wall_rays",                   "dim": 8},
    {"name": "player_hp",                   "dim": 1},
    {"name": "shield_active",               "dim": 1},
    {"name": "shield_timer_norm",           "dim": 1},
    {"name": "revival_remaining_norm",      "dim": 1},
    {"name": "armor_flat_norm",             "dim": 1},
    {"name": "regen_per_sec_norm",          "dim": 1},
    {"name": "passive_effect_summary",      "dim": 5},
    {"name": "weapon_slots",                "dim": 18},   # (type_norm, level_norm, cooldown_norm) × 6
    {"name": "passive_slots",               "dim": 12},   # (type_norm, level_norm) × 6
    {"name": "enemy_count",                 "dim": 1},
    {"name": "elapsed_time",                "dim": 1},
    {"name": "xp_progress",                 "dim": 1},
    {"name": "player_level",                "dim": 1},
    {"name": "stage_id_norm",               "dim": 1},
    {"name": "red_gem_rel_pos",             "dim": 20},   # MaxRedGemObs(10) × 2
    {"name": "green_gem_rel_pos",           "dim": 24},   # MaxGreenGemObs(12) × 2
    {"name": "blue_gem_rel_pos",            "dim": 24},   # MaxBlueGemObs(12) × 2
    {"name": "gem_pickup_radius",           "dim": 1},
    {"name": "enemy_rel_pos",               "dim": 64},   # MaxEnemyObs(32) × 2
    {"name": "enemy_vel",                   "dim": 64},   # MaxEnemyObs(32) × 2
    {"name": "enemy_type",                  "dim": 32},   # MaxEnemyObs(32)
    {"name": "enemy_hp",                    "dim": 32},   # MaxEnemyObs(32)
    {"name": "enemy_frozen",                "dim": 32},   # MaxEnemyObs(32)
    {"name": "enemy_nearest_dist_16dir",    "dim": 16},   # EnemyDensityDirCount(16)
    {"name": "enemy_density_near_16dir",    "dim": 16},   # EnemyDensityDirCount(16)
    {"name": "enemy_density_mid_16dir",     "dim": 16},   # EnemyDensityDirCount(16)
    {"name": "gem_density_all_16dir",       "dim": 48},   # GemDensityDirCount(16) × 3
    {"name": "red_green_gem_density_16dir", "dim": 48},   # GemDensityDirCount(16) × 3
    {"name": "projectiles",                 "dim": 288},  # MaxProjectileObs(32) × ProjectileObsStride(9)
    {"name": "floor_pickups",               "dim": 24},   # MaxFloorPickupObs(8) × 3
    {"name": "special_pickups",             "dim": 9},    # MaxSpecialPickupObs(3) × 3
    {"name": "destructibles",               "dim": 20},   # MaxDestructibleObs(10) × 2
    {"name": "weapon_attack_range_norm",    "dim": 6},    # MaxWeaponSlots(6)
    {"name": "weapon_is_directional",       "dim": 6},    # MaxWeaponSlots(6)
    {"name": "weapon_category_onehot",      "dim": 42},   # MaxWeaponSlots(6) × 7カテゴリ
]

_OBS_DIM = sum(seg["dim"] for seg in _OBS_SCHEMA)  # = 890 (794 - 192 + 288)

_NUM_ACTIONS = 9

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

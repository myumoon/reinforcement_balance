"""rule_policy() のユニットテスト。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.survivors.survivors_bc import (
    rule_policy,
    set_rule_policy_seed,
    _select_best_action9,
    _get_center_action9,
)
from games.survivors.survivors_env_stub import _OBS_OFFSETS, _OBS_DIM


# ──────────────────────────────────────────────
# ヘルパー
# ──────────────────────────────────────────────

def _make_obs(**kwargs) -> np.ndarray:
    """デフォルト値（全0、ただし wall_rays=1.0, enemy_nd=1.0）の obs を作成し kwargs で上書き。"""
    obs = np.zeros(_OBS_DIM, dtype=np.float32)
    obs[_OBS_OFFSETS["wall_rays"]:_OBS_OFFSETS["wall_rays"] + 8] = 1.0
    obs[_OBS_OFFSETS["enemy_nearest_dist_16dir"]:_OBS_OFFSETS["enemy_nearest_dist_16dir"] + 16] = 1.0
    for key, val in kwargs.items():
        off = _OBS_OFFSETS[key]
        arr = np.asarray(val, dtype=np.float32)
        obs[off:off + arr.size] = arr
    return obs


def _run_n(obs, n=100, seed=42) -> list[int]:
    rng = np.random.default_rng(seed)
    return [rule_policy(obs, _OBS_OFFSETS, rng=rng) for _ in range(n)]


# ──────────────────────────────────────────────
# テストクラス
# ──────────────────────────────────────────────

class TestGetCenterAction9(unittest.TestCase):

    def test_east(self):
        """プレイヤーが東 → 中心は West (action 6)。"""
        self.assertEqual(_get_center_action9(0.5, 0.0), 6)

    def test_north(self):
        """プレイヤーが北 → 中心は South (action 4)。"""
        self.assertEqual(_get_center_action9(0.0, 0.5), 4)

    def test_west(self):
        """プレイヤーが西 → 中心は East (action 2)。"""
        self.assertEqual(_get_center_action9(-0.5, 0.0), 2)

    def test_south(self):
        """プレイヤーが南 → 中心は North (action 0)。"""
        self.assertEqual(_get_center_action9(0.0, -0.5), 0)

    def test_near_center_returns_none(self):
        """中心付近は None。"""
        self.assertIsNone(_get_center_action9(0.0, 0.0))
        self.assertIsNone(_get_center_action9(0.1, 0.1))

    def test_northeast_quadrant(self):
        """北東にいれば中心は SW (action 5)。"""
        self.assertEqual(_get_center_action9(0.5, 0.5), 5)


class TestSelectBestAction9(unittest.TestCase):

    def test_prefers_high_score(self):
        """スコアが高い dir16 方向の action9 を選ぶ。dir16=7,8 → East(2)。"""
        scores = np.zeros(16)
        scores[7] = 1.0
        self.assertEqual(_select_best_action9(scores, set()), 2)

    def test_blocked_is_excluded(self):
        """blocked の action9 は選ばない。"""
        scores = np.zeros(16)
        scores[7] = 1.0  # East(2)
        result = _select_best_action9(scores, blocked={2})
        self.assertNotEqual(result, 2)

    def test_all_blocked_returns_none(self):
        """全 action9 がブロックされている場合は None。"""
        self.assertIsNone(_select_best_action9(np.ones(16), blocked=set(range(9))))

    def test_center_preference_breaks_tie(self):
        """同点時、center_a9=East(2) に最も近い候補を返す。"""
        scores = np.ones(16)
        result = _select_best_action9(scores, blocked=set(), center_a9=2)
        self.assertEqual(result, 2)

    def test_tie_not_always_dir0(self):
        """全方向同点のとき dir16=0 対応の West(6) に固定されない。"""
        rng = np.random.default_rng(0)
        results = {_select_best_action9(np.ones(16), blocked=set(), rng=rng) for _ in range(50)}
        self.assertGreater(len(results), 1, f"Expected diverse results, got {results}")


class TestRulePolicyWallBlocking(unittest.TestCase):

    def test_north_wall_blocks_north_ne_nw(self):
        """北壁が近い場合、North(0)/NE(1)/NW(7) を選ばない。"""
        wall = np.ones(8, dtype=np.float32)
        wall[2] = 0.05   # wall_rays[2] = N が近い
        obs = _make_obs(wall_rays=wall)
        results = set(_run_n(obs))
        self.assertNotIn(0, results, "North(0) should be blocked")
        self.assertNotIn(1, results, "NE(1) should be blocked")
        self.assertNotIn(7, results, "NW(7) should be blocked")

    def test_east_wall_blocks_east_ne_se(self):
        """東壁が近い場合、East(2)/NE(1)/SE(3) を選ばない。"""
        wall = np.ones(8, dtype=np.float32)
        wall[0] = 0.05   # wall_rays[0] = E が近い
        obs = _make_obs(wall_rays=wall)
        results = set(_run_n(obs))
        self.assertNotIn(2, results, "East(2) should be blocked")
        self.assertNotIn(1, results, "NE(1) should be blocked")
        self.assertNotIn(3, results, "SE(3) should be blocked")


class TestRulePolicyGem(unittest.TestCase):

    def test_gem_east_safe_east_chooses_east(self):
        """Gem が東に多く、安全も東なら East(2) を選ぶ。"""
        gem_near = np.zeros(16, dtype=np.float32)
        gem_near[7] = 1.0   # dir16=7 → East
        gem_near[8] = 1.0   # dir16=8 → East
        obs = _make_obs(gem_density_near_16dir=gem_near)
        action = rule_policy(obs, _OBS_OFFSETS)
        self.assertEqual(action, 2, f"Expected East(2), got {action}")

    def test_gem_east_blocked_by_nearby_enemy(self):
        """東に Gem があっても東に敵が接触距離なら East(2) を選ばない。"""
        gem_near = np.zeros(16, dtype=np.float32)
        gem_near[7] = 1.0
        gem_near[8] = 1.0
        enemy_nd = np.ones(16, dtype=np.float32)
        enemy_nd[7] = 0.05   # dir16=7 (East) に敵が接触距離
        enemy_nd[8] = 0.05
        obs = _make_obs(gem_density_near_16dir=gem_near, enemy_nearest_dist_16dir=enemy_nd)
        action = rule_policy(obs, _OBS_OFFSETS)
        self.assertNotEqual(action, 2, f"East(2) should be avoided due to contact enemy")


class TestRulePolicyEnemyRetreat(unittest.TestCase):

    def test_contact_north_retreats_south(self):
        """最近傍敵が北（dir16=11,12）にいれば South(4) へ退避する。"""
        enemy_nd = np.ones(16, dtype=np.float32)
        enemy_nd[11] = 0.05   # dir16=11 → North に接触距離
        enemy_nd[12] = 0.05
        obs = _make_obs(enemy_nearest_dist_16dir=enemy_nd)
        action = rule_policy(obs, _OBS_OFFSETS)
        self.assertEqual(action, 4, f"Expected South(4), got {action}")


class TestRulePolicyTieBreaking(unittest.TestCase):

    def test_all_zero_not_fixed(self):
        """全密度0・敵なし・中心位置の場合、固定方向に偏らない（同点ランダム）。"""
        obs = _make_obs()   # wall_rays=1, enemy_nd=1, それ以外0
        results = set(_run_n(obs, n=200, seed=0))
        self.assertGreater(len(results), 1, f"Expected diverse results, got {results}")

    def test_uniform_gem_not_fixed_to_west(self):
        """全方向 gem が同じ場合、dir16=0 対応の West(6) に固定されない。"""
        gem_near = np.ones(16, dtype=np.float32) * 0.5
        obs = _make_obs(gem_density_near_16dir=gem_near)
        results = set(_run_n(obs, n=200, seed=1))
        self.assertGreater(len(results), 1, f"Expected diverse results, got {results}")
        if 6 in results:
            self.assertGreater(len(results - {6}), 0, "Only West(6) — np.argmax bug suspected")


if __name__ == "__main__":
    unittest.main()

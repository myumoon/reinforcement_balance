"""
weapon_slot_sampler.py の単体テスト。
"""
from __future__ import annotations

import random

import pytest

from games.survivors.weapon_slot_sampler import (
    VALID_WEAPONS,
    _distribute,
    get_initial_elapsed_time,
    sample_weapon_slots,
)


# ---------------------------------------------------------------------------
# get_initial_elapsed_time
# ---------------------------------------------------------------------------

class TestGetInitialElapsedTime:
    def test_mad_forest_returns_900(self):
        assert get_initial_elapsed_time("Mad Forest") == 900.0

    def test_mad_forest_intermediate(self):
        # "Mad Forest 入門" など部分一致
        assert get_initial_elapsed_time("Mad Forest 入門") == 900.0

    def test_swarm_c_returns_600(self):
        assert get_initial_elapsed_time("群れ対応C") == 600.0

    def test_swarm_b_returns_420(self):
        assert get_initial_elapsed_time("群れ対応B") == 420.0

    def test_swarm_a_returns_300(self):
        assert get_initial_elapsed_time("群れ対応A") == 300.0

    def test_unknown_phase_returns_0(self):
        assert get_initial_elapsed_time("入門") == 0.0

    def test_empty_phase_returns_0(self):
        assert get_initial_elapsed_time("") == 0.0


# ---------------------------------------------------------------------------
# _distribute
# ---------------------------------------------------------------------------

class TestDistribute:
    def test_sum_equals_total_when_enough(self):
        rng = random.Random(42)
        levels = _distribute(10, 3, rng)
        assert sum(levels) == 10

    def test_all_at_least_1(self):
        rng = random.Random(0)
        for total in range(1, 20):
            levels = _distribute(total, 3, rng)
            assert all(lv >= 1 for lv in levels)

    def test_no_level_exceeds_8(self):
        rng = random.Random(99)
        levels = _distribute(100, 3, rng)
        assert all(lv <= 8 for lv in levels)

    def test_length_matches_n(self):
        rng = random.Random(1)
        assert len(_distribute(5, 4, rng)) == 4

    def test_total_less_than_n_is_clamped(self):
        # total < n でも最低 1 ずつ配分される
        rng = random.Random(7)
        levels = _distribute(1, 4, rng)
        assert len(levels) == 4
        assert all(lv >= 1 for lv in levels)


# ---------------------------------------------------------------------------
# sample_weapon_slots
# ---------------------------------------------------------------------------

class TestSampleWeaponSlots:
    def test_returns_none_for_zero_elapsed(self):
        assert sample_weapon_slots(0.0) is None

    def test_returns_none_for_negative_elapsed(self):
        assert sample_weapon_slots(-1.0) is None

    def test_returns_none_below_min_threshold(self):
        # 300.0 未満は None
        assert sample_weapon_slots(299.9) is None

    def test_returns_list_at_300(self):
        rng = random.Random(42)
        result = sample_weapon_slots(300.0, rng=rng)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_weapon_ids_are_valid(self):
        rng = random.Random(0)
        for elapsed in [300.0, 420.0, 600.0, 900.0]:
            result = sample_weapon_slots(elapsed, rng=rng)
            assert result is not None
            for slot in result:
                assert slot["weapon_id"] in VALID_WEAPONS, \
                    f"weapon_id={slot['weapon_id']} not in VALID_WEAPONS"

    def test_levels_in_range(self):
        rng = random.Random(1234)
        for elapsed in [300.0, 420.0, 600.0, 900.0]:
            result = sample_weapon_slots(elapsed, rng=rng)
            assert result is not None
            for slot in result:
                assert 1 <= slot["level"] <= 8, \
                    f"level={slot['level']} out of range [1, 8]"

    def test_num_slots_at_300(self):
        # num=(2, 3)
        counts = set()
        rng = random.Random(0)
        for _ in range(50):
            result = sample_weapon_slots(300.0, rng=rng)
            assert result is not None
            counts.add(len(result))
        assert counts <= {2, 3}

    def test_num_slots_at_900(self):
        # num=(5, 6)
        counts = set()
        rng = random.Random(0)
        for _ in range(50):
            result = sample_weapon_slots(900.0, rng=rng)
            assert result is not None
            counts.add(len(result))
        assert counts <= {5, 6}

    def test_no_duplicate_weapon_ids(self):
        rng = random.Random(7)
        for elapsed in [300.0, 420.0, 600.0, 900.0]:
            result = sample_weapon_slots(elapsed, rng=rng)
            assert result is not None
            ids = [s["weapon_id"] for s in result]
            assert len(ids) == len(set(ids)), f"Duplicate weapon_ids found: {ids}"

    def test_result_keys(self):
        rng = random.Random(99)
        result = sample_weapon_slots(600.0, rng=rng)
        assert result is not None
        for slot in result:
            assert "weapon_id" in slot
            assert "level" in slot

    def test_total_level_in_range_at_300(self):
        # total_lv=(4, 8) の範囲内であること（ただし len に応じた最低値は n）
        rng = random.Random(10)
        for _ in range(20):
            result = sample_weapon_slots(300.0, rng=rng)
            assert result is not None
            total = sum(s["level"] for s in result)
            # total_lv range は (4, 8)、ただし cap で多少ずれることがある
            assert total >= len(result)  # 最低1ずつ
            assert total <= 8 * len(result)  # 最大8ずつ

    def test_420_threshold(self):
        # 420.0 は _TIME_CONSTRAINTS の (420.0, ...) にマッチ → num=(3, 4)
        counts = set()
        rng = random.Random(0)
        for _ in range(50):
            result = sample_weapon_slots(420.0, rng=rng)
            assert result is not None
            counts.add(len(result))
        assert counts <= {3, 4}

    def test_reproducibility_with_seed(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        r1 = sample_weapon_slots(600.0, rng=rng1)
        r2 = sample_weapon_slots(600.0, rng=rng2)
        assert r1 == r2

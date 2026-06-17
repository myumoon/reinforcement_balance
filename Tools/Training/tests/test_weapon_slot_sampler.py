"""
weapon_slot_sampler.py の単体テスト。
"""
from __future__ import annotations

import random

import pytest

from games.survivors.weapon_slot_sampler import (
    _distribute,
    get_initial_elapsed_time,
    sample_weapon_slots,
)
from games.survivors.survivors_vs_spec import (
    WEAPON_VALID_FOR_RSI,
    WEAPON_MAX_LEVEL,
    PASSIVE_VALID_FOR_RSI,
    PASSIVE_MAX_LEVEL,
    EVOLUTION_TABLE,
    PassiveItemType,
)


# ---------------------------------------------------------------------------
# get_initial_elapsed_time
# ---------------------------------------------------------------------------

class TestGetInitialElapsedTime:
    def test_mad_forest_returns_900(self):
        assert get_initial_elapsed_time("Mad Forest") == 900.0

    def test_mad_forest_intermediate_returns_900(self):
        assert get_initial_elapsed_time("Mad Forest 中級") == 900.0

    def test_mad_forest_beginner_returns_600(self):
        assert get_initial_elapsed_time("Mad Forest 入門") == 600.0

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
        levels = _distribute(10, 3, WEAPON_MAX_LEVEL, rng)
        assert sum(levels) == 10

    def test_all_at_least_1(self):
        rng = random.Random(0)
        for total in range(1, 20):
            levels = _distribute(total, 3, WEAPON_MAX_LEVEL, rng)
            assert all(lv >= 1 for lv in levels)

    def test_no_level_exceeds_max_per(self):
        rng = random.Random(99)
        levels = _distribute(100, 3, WEAPON_MAX_LEVEL, rng)
        assert all(lv <= WEAPON_MAX_LEVEL for lv in levels)

    def test_length_matches_n(self):
        rng = random.Random(1)
        assert len(_distribute(5, 4, WEAPON_MAX_LEVEL, rng)) == 4

    def test_total_less_than_n_is_clamped(self):
        # total < n でも最低 1 ずつ配分される
        rng = random.Random(7)
        levels = _distribute(1, 4, WEAPON_MAX_LEVEL, rng)
        assert len(levels) == 4
        assert all(lv >= 1 for lv in levels)


# ---------------------------------------------------------------------------
# sample_weapon_slots — 新シグネチャ（phase_name: str）
# ---------------------------------------------------------------------------

class TestSampleWeaponSlots:
    def test_returns_none_for_unknown_phase(self):
        assert sample_weapon_slots("入門") is None

    def test_returns_none_for_empty_phase(self):
        assert sample_weapon_slots("") is None

    def test_returns_none_for_phase_without_rsi(self):
        # RSI 不要フェーズ（elapsed_time=0.0）
        assert sample_weapon_slots("通常序盤") is None

    def test_returns_dict_for_swarm_a(self):
        rng = random.Random(42)
        result = sample_weapon_slots("群れ対応A", rng=rng)
        assert result is not None
        assert isinstance(result, dict)

    def test_result_keys(self):
        rng = random.Random(99)
        result = sample_weapon_slots("群れ対応B", rng=rng)
        assert result is not None
        assert "initial_elapsed_time"  in result
        assert "initial_weapon_slots"  in result
        assert "initial_passive_slots" in result
        assert "MaxEpisodeTime"        in result

    def test_initial_elapsed_time_matches_phase(self):
        rng = random.Random(0)
        result = sample_weapon_slots("群れ対応A", rng=rng)
        assert result is not None
        assert result["initial_elapsed_time"] == 300.0

    def test_max_episode_time_is_elapsed_plus_300(self):
        rng = random.Random(0)
        result = sample_weapon_slots("群れ対応B", rng=rng)
        assert result is not None
        assert result["MaxEpisodeTime"] == result["initial_elapsed_time"] + 300.0

    def test_weapon_ids_are_valid(self):
        rng = random.Random(0)
        for phase in ["群れ対応A", "群れ対応B", "群れ対応C", "Mad Forest 入門", "Mad Forest"]:
            result = sample_weapon_slots(phase, rng=rng)
            assert result is not None, f"phase={phase} returned None"
            for slot in result["initial_weapon_slots"]:
                wid = slot["weapon_id"]
                # 通常武器は WEAPON_VALID_FOR_RSI に含まれる
                # 進化後武器（allow_evolved=True のフェーズ）は EVOLUTION_TABLE に含まれる
                evolved_ids = {e["evolved"] for e in EVOLUTION_TABLE}
                assert wid in WEAPON_VALID_FOR_RSI or wid in evolved_ids, \
                    f"phase={phase}: weapon_id={wid} not valid"

    def test_weapon_levels_in_range(self):
        rng = random.Random(1234)
        for phase in ["群れ対応A", "群れ対応B", "群れ対応C", "Mad Forest"]:
            result = sample_weapon_slots(phase, rng=rng)
            assert result is not None
            for slot in result["initial_weapon_slots"]:
                assert 1 <= slot["level"] <= WEAPON_MAX_LEVEL, \
                    f"phase={phase}: level={slot['level']} out of range"

    def test_passive_ids_are_valid(self):
        rng = random.Random(7)
        for phase in ["群れ対応B", "群れ対応C", "Mad Forest 入門", "Mad Forest"]:
            result = sample_weapon_slots(phase, rng=rng)
            assert result is not None
            for slot in result["initial_passive_slots"]:
                pid = slot["passive_id"]
                assert pid in PASSIVE_VALID_FOR_RSI, \
                    f"phase={phase}: passive_id={pid} not in PASSIVE_VALID_FOR_RSI"

    def test_passive_levels_in_range(self):
        rng = random.Random(321)
        for phase in ["群れ対応B", "群れ対応C", "Mad Forest"]:
            result = sample_weapon_slots(phase, rng=rng)
            assert result is not None
            for slot in result["initial_passive_slots"]:
                pid = slot["passive_id"]
                max_lv = PASSIVE_MAX_LEVEL[pid]
                assert 1 <= slot["level"] <= max_lv, \
                    f"phase={phase}: passive_id={pid} level={slot['level']} out of [1, {max_lv}]"

    def test_no_duplicate_weapon_ids(self):
        rng = random.Random(7)
        for phase in ["群れ対応A", "群れ対応B", "群れ対応C", "Mad Forest 入門", "Mad Forest"]:
            result = sample_weapon_slots(phase, rng=rng)
            assert result is not None
            ids = [s["weapon_id"] for s in result["initial_weapon_slots"]]
            assert len(ids) == len(set(ids)), f"phase={phase}: Duplicate weapon_ids found: {ids}"

    def test_no_duplicate_passive_ids(self):
        rng = random.Random(55)
        for phase in ["群れ対応C", "Mad Forest 入門", "Mad Forest"]:
            result = sample_weapon_slots(phase, rng=rng)
            assert result is not None
            ids = [s["passive_id"] for s in result["initial_passive_slots"]]
            assert len(ids) == len(set(ids)), f"phase={phase}: Duplicate passive_ids found: {ids}"

    def test_swarm_a_has_no_passives_sometimes(self):
        # 群れ対応A の passive_num=(0, 2) → 0 になりうる
        found_zero = False
        rng = random.Random(0)
        for _ in range(100):
            result = sample_weapon_slots("群れ対応A", rng=rng)
            assert result is not None
            if len(result["initial_passive_slots"]) == 0:
                found_zero = True
                break
        assert found_zero, "群れ対応A: passive_slots が 0 になるケースが確認できなかった"

    def test_reproducibility_with_seed(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        r1 = sample_weapon_slots("群れ対応C", rng=rng1)
        r2 = sample_weapon_slots("群れ対応C", rng=rng2)
        assert r1 == r2

    def test_num_weapon_slots_in_range_swarm_a(self):
        # weapon_num=(2, 3)
        counts: set[int] = set()
        rng = random.Random(0)
        for _ in range(100):
            result = sample_weapon_slots("群れ対応A", rng=rng)
            assert result is not None
            counts.add(len(result["initial_weapon_slots"]))
        assert counts <= {2, 3}

    def test_num_weapon_slots_in_range_mad_forest(self):
        # weapon_num=(5, 6); allow_evolved=True で 1 スロットが進化後武器になると -1 されうる
        counts: set[int] = set()
        rng = random.Random(0)
        for _ in range(100):
            result = sample_weapon_slots("Mad Forest", rng=rng)
            assert result is not None
            counts.add(len(result["initial_weapon_slots"]))
        # 進化後武器が選ばれると num_w -= 1 されるため 4〜6 になりうる
        assert all(c >= 4 for c in counts)


# ---------------------------------------------------------------------------
# survivors_vs_spec.py 整合性テスト
# ---------------------------------------------------------------------------

class TestSurvivorsVsSpec:
    def test_passive_max_level_keys_are_unique(self):
        assert len(PASSIVE_MAX_LEVEL) == len(set(PASSIVE_MAX_LEVEL.keys()))

    def test_passive_valid_for_rsi_excludes_none(self):
        assert PassiveItemType.NONE not in PASSIVE_VALID_FOR_RSI

    def test_passive_valid_for_rsi_excludes_zero_max_level(self):
        for pid in PASSIVE_VALID_FOR_RSI:
            assert PASSIVE_MAX_LEVEL[pid] > 0, f"passive_id={pid} has max_level=0 but is in PASSIVE_VALID_FOR_RSI"

    def test_evolution_table_has_unique_base_weapons(self):
        bases = [e["base"] for e in EVOLUTION_TABLE]
        assert len(bases) == len(set(bases)), "EVOLUTION_TABLE has duplicate base weapons"

    def test_evolution_table_evolved_passives_exist_in_passive_max_level(self):
        for entry in EVOLUTION_TABLE:
            pid = entry["passive"]
            assert pid in PASSIVE_MAX_LEVEL, f"passive_id={pid} from EVOLUTION_TABLE not in PASSIVE_MAX_LEVEL"

    def test_weapon_valid_for_rsi_has_no_duplicates(self):
        assert len(WEAPON_VALID_FOR_RSI) == len(set(WEAPON_VALID_FOR_RSI))

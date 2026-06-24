"""武器露出ガードのユニットテスト。

テストシナリオ:
1. ガード中は敵Phase昇格が発生しない
2. ガード中でも1段目の通常rollbackは許可される
3. ガード中の2段目通常rollbackはブロックされる
4. emergency ep_len条件では2段目以降もrollbackされる
5. guard_steps と新規武器episode数を満たすとガードが解除される
6. export/import後もガード状態が復元される
"""
from __future__ import annotations

import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import pytest

from games.survivors.state_modules import CurriculumStateModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_curriculum(window: int = 5, phase_idx: int = 4) -> CurriculumStateModule:
    c = CurriculumStateModule(window=window, threshold_mult=1.0, rollback_patience=1)
    c._phase_idx = phase_idx
    return c


def _fill_scores(c: CurriculumStateModule, scores: list[float]) -> None:
    from games.survivors.survivors_curriculum import PHASES
    phase = PHASES[c._phase_idx]
    for s in scores:
        c.on_episode_end(s, ep_len=2400)


def _trigger_guard(c: CurriculumStateModule, new_ids: list[int] = None, start_step: int = 100_000) -> None:
    c.start_weapon_exposure_guard(new_ids or [7], num_timesteps=start_step)


# ---------------------------------------------------------------------------
# 1. ガード中は昇格しない
# ---------------------------------------------------------------------------

class TestGuardBlocksPromotion:

    def test_promotion_blocked_during_guard(self):
        c = _make_curriculum(window=3, phase_idx=3)
        c.configure_weapon_exposure_guard(guard_steps=100_000, min_new_weapon_episodes=10)
        _trigger_guard(c, start_step=50_000)

        # スコアを高めに保って昇格条件を満たす
        from games.survivors.survivors_curriculum import PHASES
        phase = PHASES[3]
        threshold = phase.threshold
        assert threshold is not None
        high_score = threshold * 2.0
        for _ in range(3):
            c.on_episode_end(high_score, ep_len=3000)

        event = c.check_phase_transition(allow_promotion=True, num_timesteps=60_000)
        assert event != "advance", "ガード中は昇格が発生しないこと"

    def test_promotion_allowed_after_guard_ends(self):
        c = _make_curriculum(window=3, phase_idx=3)
        c.configure_weapon_exposure_guard(
            guard_steps=100_000,
            min_new_weapon_episodes=2,
        )
        _trigger_guard(c, new_ids=[7], start_step=0)

        # new weapon episode を満たす
        for _ in range(2):
            c.on_weapon_exposure_episode_end(7)

        from games.survivors.survivors_curriculum import PHASES
        phase = PHASES[3]
        high_score = phase.threshold * 2.0 if phase.threshold else 9999.0
        for _ in range(3):
            c.on_episode_end(high_score, ep_len=3000)

        # guard_steps 経過後にチェック
        event = c.check_phase_transition(allow_promotion=True, num_timesteps=200_000)
        assert event == "advance", "ガード終了後は昇格できること"


# ---------------------------------------------------------------------------
# 2. ガード中でも1段目のrollbackは許可される
# ---------------------------------------------------------------------------

class TestGuardAllowsFirstRollback:

    def test_first_rollback_allowed(self):
        c = _make_curriculum(window=3, phase_idx=4)
        c.configure_weapon_exposure_guard(guard_steps=100_000, min_new_weapon_episodes=20, max_normal_rollbacks=1)
        _trigger_guard(c, start_step=50_000)

        # 低スコアを積んでrollback判定を発生させる
        from games.survivors.survivors_curriculum import PHASES
        phase = PHASES[4]
        threshold = phase.threshold or 1000.0
        low_score = threshold * phase.rollback_score_ratio * 0.5
        for _ in range(3):
            c.on_episode_end(low_score, ep_len=1000)

        event = c.check_phase_transition(allow_promotion=False, num_timesteps=60_000)
        assert event == "rollback", "ガード中1段目のrollbackは許可されること"
        assert c._weapon_exposure_guard_normal_rollbacks == 1


# ---------------------------------------------------------------------------
# 3. ガード中の2段目rollbackはブロックされる
# ---------------------------------------------------------------------------

class TestGuardBlocksSecondRollback:

    def test_second_rollback_blocked(self):
        c = _make_curriculum(window=3, phase_idx=5)
        c.configure_weapon_exposure_guard(
            guard_steps=500_000,
            min_new_weapon_episodes=20,
            max_normal_rollbacks=1,
            emergency_ep_len_ratio=0.1,  # 緊急条件を厳しくして通常rollbackのみ検証
        )
        _trigger_guard(c, start_step=50_000)

        # 1段目のrollbackを発生させる
        from games.survivors.survivors_curriculum import PHASES
        phase5 = PHASES[5]
        threshold = phase5.threshold or 1000.0
        low_score = threshold * phase5.rollback_score_ratio * 0.5
        # phase_idx=5 でrollback → phase_idx=4 へ
        for _ in range(3):
            c.on_episode_end(low_score, ep_len=1000)
        event1 = c.check_phase_transition(allow_promotion=False, num_timesteps=60_000)
        assert event1 == "rollback"
        assert c._phase_idx == 4
        assert c._weapon_exposure_guard_normal_rollbacks == 1

        # 2段目: phase_idx=4 で低スコアを積む
        phase4 = PHASES[4]
        threshold4 = phase4.threshold or 1000.0
        low4 = threshold4 * phase4.rollback_score_ratio * 0.5
        for _ in range(3):
            c.on_episode_end(low4, ep_len=1000)
        event2 = c.check_phase_transition(allow_promotion=False, num_timesteps=70_000)
        assert event2 is None, "ガード中2段目rollbackはブロックされること"
        assert c._phase_idx == 4, "フェーズは変わっていないこと"


# ---------------------------------------------------------------------------
# 4. 緊急rollbackは2段目以降も許可される
# ---------------------------------------------------------------------------

class TestEmergencyRollback:

    def test_emergency_rollback_bypasses_guard(self):
        c = _make_curriculum(window=3, phase_idx=5)
        c.configure_weapon_exposure_guard(
            guard_steps=500_000,
            min_new_weapon_episodes=20,
            max_normal_rollbacks=1,
            emergency_ep_len_ratio=0.25,
        )
        _trigger_guard(c, start_step=50_000)

        # 1段目のrollbackを発生させる（ノーマルスコア失敗、ep_len普通）
        from games.survivors.survivors_curriculum import PHASES
        phase5 = PHASES[5]
        threshold = phase5.threshold or 1000.0
        low_score = threshold * phase5.rollback_score_ratio * 0.5
        for _ in range(3):
            c.on_episode_end(low_score, ep_len=1500)
        c.check_phase_transition(allow_promotion=False, num_timesteps=60_000)
        assert c._phase_idx == 4

        # 2段目: 緊急条件（ep_len が min_episode_steps * 0.25 未満）
        phase4 = PHASES[4]
        emergency_ep_len = int(phase4.min_episode_steps * 0.24)  # 0.25 未満
        phase4_threshold = phase4.threshold or 1000.0
        low4 = phase4_threshold * phase4.rollback_score_ratio * 0.5
        for _ in range(3):
            c.on_episode_end(low4, ep_len=emergency_ep_len)
        event2 = c.check_phase_transition(allow_promotion=False, num_timesteps=70_000)
        assert event2 == "rollback", "緊急rollbackはガード中でも発生すること"
        assert c._phase_idx == 3


# ---------------------------------------------------------------------------
# 5. ガード終了条件
# ---------------------------------------------------------------------------

class TestGuardEndsWhenConditionsMet:

    def test_guard_ends_after_steps_and_episodes(self):
        c = _make_curriculum(window=3, phase_idx=4)
        c.configure_weapon_exposure_guard(guard_steps=100_000, min_new_weapon_episodes=3)
        _trigger_guard(c, new_ids=[7], start_step=0)

        assert c._weapon_exposure_guard_active

        # エピソード数を満たす
        for _ in range(3):
            c.on_weapon_exposure_episode_end(7)

        # steps まだ不足（50_000 < 100_000）
        from games.survivors.survivors_curriculum import PHASES
        phase = PHASES[4]
        high_score = (phase.threshold or 1000.0) * 2.0
        for _ in range(3):
            c.on_episode_end(high_score, ep_len=3000)
        c.check_phase_transition(allow_promotion=True, num_timesteps=50_000)
        assert c._weapon_exposure_guard_active, "steps 不足でガードは継続すること"

        # steps 満たす
        for _ in range(3):
            c.on_episode_end(high_score, ep_len=3000)
        c.check_phase_transition(allow_promotion=True, num_timesteps=110_000)
        assert not c._weapon_exposure_guard_active, "条件を満たしたのでガード終了すること"

    def test_guard_does_not_end_without_enough_episodes(self):
        c = _make_curriculum(window=3, phase_idx=4)
        c.configure_weapon_exposure_guard(guard_steps=100_000, min_new_weapon_episodes=5)
        _trigger_guard(c, new_ids=[7], start_step=0)

        # episode 数が足りない（1 < 5）
        c.on_weapon_exposure_episode_end(7)

        from games.survivors.survivors_curriculum import PHASES
        phase = PHASES[4]
        high_score = (phase.threshold or 1000.0) * 2.0
        for _ in range(3):
            c.on_episode_end(high_score, ep_len=3000)
        c.check_phase_transition(allow_promotion=True, num_timesteps=200_000)
        assert c._weapon_exposure_guard_active, "episode 数不足ではガードが終了しないこと"


# ---------------------------------------------------------------------------
# 6. export/import 後もガード状態が復元される
# ---------------------------------------------------------------------------

class TestExportImportPreservesGuard:

    def test_export_includes_guard_state(self):
        c = _make_curriculum(phase_idx=4)
        _trigger_guard(c, new_ids=[7], start_step=12345)
        c._weapon_exposure_guard_normal_rollbacks = 1
        c._weapon_exposure_guard_episode_counts[7] = 3

        state = c.export_state()
        guard = state["weapon_exposure_guard"]
        assert guard["active"] is True
        assert guard["new_weapon_ids"] == [7]
        assert guard["start_step"] == 12345
        assert guard["normal_rollbacks"] == 1
        assert guard["episode_counts"] == {"7": 3}

    def test_import_restores_guard_state(self):
        c = _make_curriculum(phase_idx=4)
        _trigger_guard(c, new_ids=[7], start_step=12345)
        c._weapon_exposure_guard_normal_rollbacks = 1
        c._weapon_exposure_guard_episode_counts[7] = 3
        state = c.export_state()

        c2 = _make_curriculum(phase_idx=4)
        c2.import_state(state)
        assert c2._weapon_exposure_guard_active is True
        assert c2._weapon_exposure_guard_new_weapon_ids == [7]
        assert c2._weapon_exposure_guard_start_step == 12345
        assert c2._weapon_exposure_guard_normal_rollbacks == 1
        assert c2._weapon_exposure_guard_episode_counts == {7: 3}

    def test_guard_continues_after_import(self):
        c = _make_curriculum(window=3, phase_idx=4)
        c.configure_weapon_exposure_guard(guard_steps=200_000, min_new_weapon_episodes=10)
        _trigger_guard(c, new_ids=[7], start_step=50_000)
        state = c.export_state()

        c2 = _make_curriculum(window=3, phase_idx=4)
        c2.configure_weapon_exposure_guard(guard_steps=200_000, min_new_weapon_episodes=10)
        c2.import_state(state)

        assert c2._weapon_exposure_guard_active
        # ガードが機能していることを確認（昇格がブロックされる）
        from games.survivors.survivors_curriculum import PHASES
        phase = PHASES[4]
        high_score = (phase.threshold or 1000.0) * 2.0
        for _ in range(3):
            c2.on_episode_end(high_score, ep_len=3000)
        event = c2.check_phase_transition(allow_promotion=True, num_timesteps=100_000)
        assert event != "advance", "import 後もガードが機能すること"


# ---------------------------------------------------------------------------
# 7. probe 経由の昇格もガード中はブロックされる（P1修正の検証）
# ---------------------------------------------------------------------------

class TestGuardBlocksProbePromotion:

    def test_check_promotion_transition_blocked_during_guard(self):
        c = _make_curriculum(window=3, phase_idx=3)
        c.configure_weapon_exposure_guard(guard_steps=200_000, min_new_weapon_episodes=20)
        _trigger_guard(c, new_ids=[7], start_step=0)

        # probe スコアを昇格条件を満たすレベルに積む
        from games.survivors.survivors_curriculum import PHASES
        phase = PHASES[3]
        threshold = phase.threshold
        assert threshold is not None
        high = threshold * 2.0
        for _ in range(3):
            c.on_promotion_probe_episode_end(high, ep_len=3000)
            c._promotion_scores.append(high)
            c._promotion_episode_lengths.append(3000)

        event = c.check_promotion_transition(promotion_source="probe", num_timesteps=50_000)
        assert event is None, "ガード中はprobe経由の昇格もブロックされること"
        assert c._phase_idx == 3

    def test_check_promotion_transition_allowed_after_guard_ends(self):
        c = _make_curriculum(window=3, phase_idx=3)
        c.configure_weapon_exposure_guard(guard_steps=100_000, min_new_weapon_episodes=2)
        _trigger_guard(c, new_ids=[7], start_step=0)

        # ガード終了条件を満たす
        for _ in range(2):
            c.on_weapon_exposure_episode_end(7)

        from games.survivors.survivors_curriculum import PHASES
        phase = PHASES[3]
        high = phase.threshold * 2.0
        for _ in range(3):
            c._promotion_scores.append(high)
            c._promotion_episode_lengths.append(3000)

        # guard_steps 経過後（ガードが終了する）
        event = c.check_promotion_transition(promotion_source="probe", num_timesteps=200_000)
        assert event == "advance", "ガード終了後はprobe昇格できること"


# ---------------------------------------------------------------------------
# 8. W2以降の複数新規武器で全武器が露出されないとガード終了しない（P2修正の検証）
# ---------------------------------------------------------------------------

class TestMultiWeaponGuard:

    def test_guard_does_not_end_if_only_one_weapon_exposed(self):
        """W1→W2で FIRE_WAND(8) と LIGHTNING_RING(11) の2武器が追加された場合、
        FIRE_WAND だけ 20 episode 達成してもガードが終了しないこと。"""
        c = _make_curriculum(window=3, phase_idx=6)
        c.configure_weapon_exposure_guard(guard_steps=100_000, min_new_weapon_episodes=5)
        _trigger_guard(c, new_ids=[8, 11], start_step=0)

        # FIRE_WAND(8) だけ 5 episode 達成
        for _ in range(5):
            c.on_weapon_exposure_episode_end(8)

        from games.survivors.survivors_curriculum import PHASES
        phase = PHASES[6]
        high = (phase.threshold or 1000.0) * 2.0
        for _ in range(3):
            c.on_episode_end(high, ep_len=3000)

        c.check_phase_transition(allow_promotion=True, num_timesteps=200_000)
        assert c._weapon_exposure_guard_active, \
            "LIGHTNING_RING が未露出のままガードが終了してはいけない"

    def test_guard_ends_when_all_weapons_exposed(self):
        """全新規武器が min 以上露出されてからガードが終了すること。"""
        c = _make_curriculum(window=3, phase_idx=6)
        c.configure_weapon_exposure_guard(guard_steps=100_000, min_new_weapon_episodes=3)
        _trigger_guard(c, new_ids=[8, 11], start_step=0)

        # 両武器とも 3 episode 達成
        for _ in range(3):
            c.on_weapon_exposure_episode_end(8)
            c.on_weapon_exposure_episode_end(11)

        from games.survivors.survivors_curriculum import PHASES
        phase = PHASES[6]
        high = (phase.threshold or 1000.0) * 2.0
        for _ in range(3):
            c.on_episode_end(high, ep_len=3000)

        c.check_phase_transition(allow_promotion=True, num_timesteps=200_000)
        assert not c._weapon_exposure_guard_active, \
            "全武器が露出されたのでガードが終了すること"

    def test_episode_counts_tracked_per_weapon(self):
        """各武器のカウントが個別に記録されること。"""
        c = _make_curriculum(phase_idx=4)
        _trigger_guard(c, new_ids=[7, 8], start_step=0)

        c.on_weapon_exposure_episode_end(7)
        c.on_weapon_exposure_episode_end(7)
        c.on_weapon_exposure_episode_end(8)

        assert c._weapon_exposure_guard_episode_counts[7] == 2
        assert c._weapon_exposure_guard_episode_counts[8] == 1

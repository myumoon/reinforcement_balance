"""state_modules のユニットテスト。"""
import sys
from pathlib import Path
import numpy as np
import pytest

_TRAINING_ROOT = Path(__file__).resolve().parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.state_modules import EpisodeScoreTracker, SpalfStateModule, CurriculumStateModule, _PHASE0_PARAMS


class TestEpisodeScoreTracker:
    def test_basic_single_env(self):
        """基本的な per-env 蓄積と active_score 計算（4-tuple 仕様）。"""
        tracker = EpisodeScoreTracker(frame_skip=4, alive_reward=0.001)
        tracker.reset(1)
        infos = [{"base_reward": 10.0, "episode": {"l": 100, "r": 10.0}}]
        results = tracker.process(infos)
        assert len(results) == 1
        env_idx, active_score, ep_len, ep_base = results[0]
        assert env_idx == 0
        assert ep_len == 100
        alive_total = 0.001 * 4 * 100
        expected = max(0.0, 10.0 - alive_total)
        assert abs(active_score - expected) < 1e-6
        assert abs(ep_base - 10.0) < 1e-6

    def test_n_envs(self):
        """n_envs=2 での独立蓄積（4-tuple 仕様）。"""
        tracker = EpisodeScoreTracker(frame_skip=1, alive_reward=0.0)
        tracker.reset(3)
        infos = [
            {"base_reward": 5.0, "episode": {"l": 50, "r": 5.0}},
            {"base_reward": 3.0},  # episode not done
            {"base_reward": 8.0, "episode": {"l": 80, "r": 8.0}},
        ]
        results = tracker.process(infos)
        assert len(results) == 2
        assert results[0][0] == 0
        assert results[1][0] == 2
        assert abs(results[0][1] - 5.0) < 1e-6
        assert abs(results[1][1] - 8.0) < 1e-6

    def test_accumulation_across_steps(self):
        tracker = EpisodeScoreTracker(frame_skip=1, alive_reward=0.0)
        tracker.reset(1)
        tracker.process([{"base_reward": 3.0}])
        results = tracker.process([{"base_reward": 7.0, "episode": {"l": 2, "r": 10.0}}])
        assert len(results) == 1
        assert abs(results[0][1] - 10.0) < 1e-6


class TestSpalfStateModuleRoundtrip:
    def test_export_import_empty(self):
        m = SpalfStateModule(r_b=0.1, alpha=0.2, max_score=2250.0,
                              buffer_size=200, warmup_episodes=50)
        state = m.export_state()
        m2 = SpalfStateModule(r_b=0.1, alpha=0.2, max_score=2250.0,
                               buffer_size=200, warmup_episodes=50)
        m2.import_state(state)
        assert m2._total_episodes == 0
        assert m2._use_spalf_mode is True

    def test_export_import_with_data(self):
        m = SpalfStateModule(r_b=0.1, alpha=0.2, max_score=2250.0,
                              buffer_size=200, warmup_episodes=50)
        vec = np.full(8, 0.5, dtype=np.float32)
        m._reward_history.append((vec, 0.3))
        m._alp_buffer.append((vec, 0.1))
        m._recent_reward_buffer.append(0.3)
        m._total_episodes = 10
        state = m.export_state()
        m2 = SpalfStateModule(r_b=0.1, alpha=0.2, max_score=2250.0,
                               buffer_size=200, warmup_episodes=50)
        m2.import_state(state)
        assert m2._total_episodes == 10
        assert len(m2._reward_history) == 1
        assert len(m2._alp_buffer) == 1

    def test_params_to_vec_and_back(self):
        m = SpalfStateModule(r_b=0.1, alpha=0.2, max_score=2250.0,
                              buffer_size=200, warmup_episodes=50)
        vec = m.params_to_vec(_PHASE0_PARAMS)
        assert vec.shape == (8,)
        assert np.all(vec >= 0.0) and np.all(vec <= 1.0)
        params_back = m.vec_to_params(vec)
        assert params_back["min_enemies"] == _PHASE0_PARAMS["min_enemies"]


class TestCurriculumStateModule:
    def _make(self, window=3):
        return CurriculumStateModule(window=window, threshold_mult=1.0, rollback_patience=2)

    def test_initial_state(self):
        m = self._make()
        assert m.current_phase == 0
        assert not m.is_final_phase

    def test_advance(self):
        """十分なスコアが続いた場合にフェーズが昇格すること。"""
        m = self._make(window=3)
        phase0 = m._PHASES[0]
        threshold = (phase0.threshold or float("inf")) * m.threshold_mult
        high_score = threshold * 2.0
        for _ in range(3):
            m.on_episode_end(high_score, ep_len=3000)
        result = m.check_phase_transition()
        assert result == "advance"
        assert m.current_phase == 1

    def test_rollback(self):
        """スコアが低い状態が rollback_patience 回続いた場合にフェーズが降格すること。"""
        m = self._make(window=3)
        # まず phase 1 に昇格させる
        phase0 = m._PHASES[0]
        high_score = (phase0.threshold or 1.0) * 2.0
        for _ in range(3):
            m.on_episode_end(high_score, ep_len=3000)
        m.check_phase_transition()
        assert m.current_phase == 1
        # phase 1 の rollback_score_floor を下回るスコアを入れる
        phase1 = m._PHASES[1]
        threshold1 = (phase1.threshold or 1.0) * m.threshold_mult
        floor = threshold1 * phase1.rollback_score_ratio
        low_score = floor * 0.5
        for _ in range(3):
            m.on_episode_end(low_score, ep_len=100)
        m.check_phase_transition()  # bad window 1
        for _ in range(3):
            m.on_episode_end(low_score, ep_len=100)
        result = m.check_phase_transition()  # bad window 2 → rollback
        assert result == "rollback"
        assert m.current_phase == 0

    def test_export_import_roundtrip(self):
        """export_state / import_state が状態を正確に復元すること。"""
        m = self._make(window=5)
        for i in range(3):
            m.on_episode_end(float(i * 100), ep_len=1000)
        state = m.export_state()
        m2 = self._make(window=5)
        m2.import_state(state)
        assert m2.current_phase == m.current_phase
        assert m2._rollback_bad_windows == m._rollback_bad_windows
        assert len(m2._scores) == len(m._scores)

    def test_export_keys(self):
        """export_state が必要なキーをすべて含むこと（resume 互換性）。"""
        m = self._make()
        state = m.export_state()
        required_keys = {
            "phase_idx", "phase_name", "scores_window", "rollback_bad_windows",
            "episode_scores", "episode_lengths", "steps_in_phase", "episodes_in_phase",
        }
        for key in required_keys:
            assert key in state, f"missing key: {key}"


class TestCurriculumStateModuleProbe:
    """probe window と check_promotion_transition のテスト。"""

    def _make(self, window=3):
        return CurriculumStateModule(window=window, threshold_mult=1.0, rollback_patience=2)

    def _get_high_score(self, m: CurriculumStateModule) -> float:
        phase = m._PHASES[m.current_phase]
        threshold = (phase.threshold or float("inf")) * m.threshold_mult
        return threshold * 2.0

    def test_probe_scores_only_advance(self):
        """train scores が閾値未達でも probe scores が達成なら advance すること。"""
        m = self._make(window=3)
        high_score = self._get_high_score(m)
        # train には低スコアのみ入れる（昇格しない）
        for _ in range(3):
            m.on_episode_end(1.0, ep_len=100)
        assert m.check_phase_transition() is None

        # probe window に高スコアを入れる
        for _ in range(3):
            m.on_promotion_probe_episode_end(high_score, ep_len=3000)
        result = m.check_promotion_transition(promotion_source="probe")
        assert result == "advance"
        assert m.current_phase == 1

    def test_allow_promotion_false_does_not_advance(self):
        """allow_promotion=False では train scores が達成しても advance しないこと。"""
        m = self._make(window=3)
        high_score = self._get_high_score(m)
        for _ in range(3):
            m.on_episode_end(high_score, ep_len=3000)
        result = m.check_phase_transition(allow_promotion=False, promotion_source="train")
        assert result is None
        assert m.current_phase == 0

    def test_rollback_from_train_scores(self):
        """rollback は train scores で動くこと（probe scores は関係しない）。

        probe に高スコアを入れても _should_rollback の判定は train scores (self._scores) に基づく。
        _rollback_phase を直接呼んで clear_promotion_probe_window が実行されることを確認する。
        """
        m = self._make(window=3)
        # phase 1 に手動で設定
        m._phase_idx = 1

        # probe に高スコアを入れる（rollback には影響しないこと）
        m.on_promotion_probe_episode_end(9999.0, ep_len=3000)
        m.on_promotion_probe_episode_end(9999.0, ep_len=3000)
        assert len(m._promotion_scores) == 2

        # _rollback_phase を呼ぶと probe window がクリアされること
        m._rollback_phase(mean=0.0, threshold=100.0, mean_len=100.0, reason="low_score")
        assert m.current_phase == 0
        assert len(m._promotion_scores) == 0

    def test_probe_window_cleared_on_advance(self):
        """phase advance 後に probe window がクリアされること。"""
        m = self._make(window=3)
        high_score = self._get_high_score(m)
        for _ in range(3):
            m.on_promotion_probe_episode_end(high_score, ep_len=3000)
        m.check_promotion_transition()
        assert len(m._promotion_scores) == 0

    def test_probe_window_cleared_on_rollback(self):
        """phase rollback 後に probe window がクリアされること。

        _rollback_phase を直接呼んで clear_promotion_probe_window が実行されることを確認する。
        """
        m = self._make(window=3)
        # probe window にデータを入れる
        m.on_promotion_probe_episode_end(100.0, ep_len=500)
        m.on_promotion_probe_episode_end(200.0, ep_len=600)
        assert len(m._promotion_scores) == 2

        # _rollback_phase を直接呼んで clear_promotion_probe_window が呼ばれることを確認
        # （rollback 発生シナリオを直接テスト）
        m._phase_idx = 1  # phase 1 に強制設定
        m._rollback_phase(mean=0.0, threshold=100.0, mean_len=100.0, reason="test")
        assert m.current_phase == 0
        # rollback 後は probe window がクリアされること
        assert len(m._promotion_scores) == 0

    def test_export_import_with_probe_buffer(self):
        """export/import で probe buffer が復元されること。"""
        m = self._make(window=3)
        m.on_promotion_probe_episode_end(100.0, ep_len=500, base_reward=100.0)
        m.on_promotion_probe_episode_end(200.0, ep_len=600)

        state = m.export_state()
        assert "promotion_scores" in state
        assert len(state["promotion_scores"]) == 2

        m2 = self._make(window=3)
        m2.import_state(state)
        assert len(m2._promotion_scores) == 2
        assert abs(m2._promotion_scores[0] - 100.0) < 1e-6
        assert len(m2._promotion_episode_lengths) == 2
        assert m2._promotion_episode_lengths[0] == 500

    def test_check_promotion_transition_insufficient_window(self):
        """probe window が window 数未満のとき None を返すこと。"""
        m = self._make(window=3)
        high_score = self._get_high_score(m)
        # window=3 に対して 2 エピソードしか入れない
        for _ in range(2):
            m.on_promotion_probe_episode_end(high_score, ep_len=3000)
        result = m.check_promotion_transition()
        assert result is None
        assert m.current_phase == 0


class TestSpalfStateModuleHybridMethods:
    """SpalfStateModule に追加した Hybrid 用メソッドのテスト。"""

    def _make(self) -> SpalfStateModule:
        return SpalfStateModule(r_b=0.1, alpha=0.2, max_score=2250.0,
                                buffer_size=200, warmup_episodes=50)

    def test_reset_buffers_for_phase_change_preserves_recent_reward(self):
        """reset_buffers_for_phase_change は _recent_reward_buffer を維持すること。"""
        m = self._make()
        m._recent_reward_buffer.append(0.5)
        m._recent_reward_buffer.append(0.3)
        m._total_episodes = 10
        m._use_spalf_mode = False
        m._reward_history.append((np.zeros(8), 0.3))
        m._alp_buffer.append((np.zeros(8), 0.1))

        m.reset_buffers_for_phase_change()

        # ALP バッファはクリアされること
        assert len(m._reward_history) == 0
        assert len(m._alp_buffer) == 0
        assert m._gmm is None
        # 習熟度情報は維持されること
        assert len(m._recent_reward_buffer) == 2
        assert m._total_episodes == 10
        assert m._use_spalf_mode is False

    def test_set_phase_warmup_triggers_warmup(self):
        """set_phase_warmup 後の N エピソードは warmup として扱われること。"""
        m = self._make()
        # warmup_episodes を超えるほどエピソードを消費
        m._total_episodes = 100
        vec = np.zeros(8, dtype=np.float32)

        m.set_phase_warmup(3)

        # 3 エピソードは per-phase warmup
        assert m.on_episode_end(0, 1000.0, vec) is True
        assert m._phase_warmup_remaining == 2
        assert m.on_episode_end(0, 1000.0, vec) is True
        assert m._phase_warmup_remaining == 1
        assert m.on_episode_end(0, 1000.0, vec) is True
        assert m._phase_warmup_remaining == 0
        # 4 回目は warmup 終了（global warmup_episodes も超えているので False）
        result = m.on_episode_end(0, 1000.0, vec)
        assert result is False

    def test_phase_warmup_remaining_initialized_to_zero(self):
        """初期状態では _phase_warmup_remaining が 0 であること。"""
        m = self._make()
        assert m._phase_warmup_remaining == 0

    def test_params_to_vec_zero_width_bounds(self):
        """ゼロ幅 bounds では params_to_vec が 0.5 を返すこと。"""
        m = self._make()
        from games.survivors.survivors_difficulty import PARAM_BOUNDS
        # min_enemies だけゼロ幅に設定
        bounds = dict(PARAM_BOUNDS)
        bounds["min_enemies"] = (20.0, 20.0)  # lo == hi
        m.set_bounds(bounds)
        params = dict(_PHASE0_PARAMS)
        params["min_enemies"] = 20
        vec = m.params_to_vec(params)
        # ゼロ幅キーは 0.5 になること
        from games.survivors.state_modules import _PARAM_KEYS
        idx = _PARAM_KEYS.index("min_enemies")
        assert abs(vec[idx] - 0.5) < 1e-6

    def test_vec_to_params_zero_width_bounds(self):
        """ゼロ幅 bounds では vec_to_params が lo 固定値を返すこと。"""
        m = self._make()
        from games.survivors.survivors_difficulty import PARAM_BOUNDS
        bounds = dict(PARAM_BOUNDS)
        bounds["enemy_hp_scale"] = (1.5, 1.5)  # lo == hi
        m.set_bounds(bounds)
        vec = np.full(8, 0.7, dtype=np.float32)
        params = m.vec_to_params(vec)
        # ゼロ幅キーは lo 固定
        assert abs(params["enemy_hp_scale"] - 1.5) < 1e-6

    def test_set_bounds_affects_normalization(self):
        """set_bounds で正規化が変わること。"""
        m = self._make()
        from games.survivors.survivors_difficulty import PARAM_BOUNDS
        default_bounds = dict(PARAM_BOUNDS)
        m.set_bounds(default_bounds)
        params_mid = {"min_enemies": 22, "max_enemies": 78, "speed_mult": 1.0,
                      "spawn_rate_mult": 2.5, "max_enemy_type_id": 5,
                      "enemy_hp_scale": 1.25, "enemy_damage_scale": 1.25, "time_scaling": False}
        vec_default = m.params_to_vec(params_mid)

        narrow_bounds = dict(PARAM_BOUNDS)
        narrow_bounds["min_enemies"] = (4.0, 44.0)  # 幅を変える
        m.set_bounds(narrow_bounds)
        vec_narrow = m.params_to_vec(params_mid)

        # 幅が変わったので min_enemies の正規化値が変わるはず
        from games.survivors.state_modules import _PARAM_KEYS
        idx = _PARAM_KEYS.index("min_enemies")
        assert abs(vec_default[idx] - vec_narrow[idx]) > 1e-3

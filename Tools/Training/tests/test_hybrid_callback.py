"""HybridCurriculumSpalfCallback の統合テスト。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_TRAINING_ROOT = Path(__file__).resolve().parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.hybrid_callback import (
    HybridCurriculumSpalfCallback,
    _phase_bounds,
    _FINAL_PHASE_EXTENDED_BOUNDS,
    _CURRICULUM_PHASES,
)
from games.survivors.survivors_curriculum import get_enemy_params_for_phase as _phase_params_from_phase
from games.survivors.survivors_difficulty import PARAM_BOUNDS
from games.survivors.modules.state_modules import _PARAM_KEYS


def _make_callback(**kwargs) -> HybridCurriculumSpalfCallback:
    """テスト用 HybridCurriculumSpalfCallback を生成する（raw_env=None）。"""
    defaults = dict(
        raw_env=None,
        frame_skip=1,
        alive_reward=0.0,
        r_b=0.1,
        alpha=0.2,
        max_score=2250.0,
        spalf_buffer_size=200,
        warmup_episodes=5,
        phase_warmup_episodes=3,
        phase_window=3,
        threshold_mult=1.0,
        rollback_patience=2,
    )
    defaults.update(kwargs)
    return HybridCurriculumSpalfCallback(**defaults)


class TestPhaseBounds:
    def test_initial_bounds_is_phase0_range(self):
        """phase_idx=0 の bounds が PHASES[0]〜PHASES[1] の範囲になること。"""
        bounds = _phase_bounds(0)
        phase0 = _CURRICULUM_PHASES[0]
        phase1 = _CURRICULUM_PHASES[1]
        # min_enemies の範囲は [min(phase0, phase1), max(phase0, phase1)]
        expected_lo = min(float(phase0.min_enemies), float(phase1.min_enemies))
        expected_hi = max(float(phase0.min_enemies), float(phase1.min_enemies))
        assert abs(bounds["min_enemies"][0] - expected_lo) < 1e-6
        assert abs(bounds["min_enemies"][1] - expected_hi) < 1e-6

    def test_final_phase_without_completion_is_fixed(self):
        """最終フェーズ completion 前の bounds は lo == hi（固定値）であること。"""
        last_idx = len(_CURRICULUM_PHASES) - 1
        bounds = _phase_bounds(last_idx, completion_ready=False)
        for key in _PARAM_KEYS:
            lo, hi = bounds[key]
            assert abs(lo - hi) < 1e-6, f"key={key}: lo={lo}, hi={hi} は固定値（lo==hi）であるべき"

    def test_final_phase_with_completion_uses_extended_bounds(self):
        """最終フェーズ completion 後の bounds の下限が PARAM_BOUNDS max に一致すること。"""
        last_idx = len(_CURRICULUM_PHASES) - 1
        bounds = _phase_bounds(last_idx, completion_ready=True)
        for key in _PARAM_KEYS:
            expected_lo = PARAM_BOUNDS[key][1]
            assert abs(bounds[key][0] - expected_lo) < 1e-6, (
                f"key={key}: bounds[0]={bounds[key][0]}, expected_lo={expected_lo}"
            )

    def test_phase_bounds_monotonicity(self):
        """すべての非最終フェーズで lo <= hi が保証されること（enemy_damage_scale の一時下降を含む）。"""
        for i in range(len(_CURRICULUM_PHASES) - 1):
            bounds = _phase_bounds(i)
            for key in _PARAM_KEYS:
                lo, hi = bounds[key]
                assert lo <= hi, f"phase_idx={i}, key={key}: lo={lo} > hi={hi}"


class TestHybridCallbackInit:
    def test_extended_bounds_enabled_is_false_initially(self):
        """初期状態で _extended_bounds_enabled が False であること。"""
        cb = _make_callback()
        assert cb._extended_bounds_enabled is False

    def test_pending_resume_state_is_false_initially(self):
        """初期状態で _pending_resume_state が False であること。"""
        cb = _make_callback()
        assert cb._pending_resume_state is False

    def test_spalf_initial_bounds_is_phase0(self):
        """SPALF モジュールの初期 bounds がフェーズ 0 の範囲になること。"""
        cb = _make_callback()
        expected = _phase_bounds(0)
        for key in _PARAM_KEYS:
            assert cb._spalf._active_bounds[key] == expected[key], f"key={key}"


class TestHybridCallbackOnStep:
    """_on_step の動作を検証するテスト（training_env をモックで代替）。"""

    def _setup_callback_with_mock_env(self, n_envs: int = 1, **kwargs):
        """モック training_env を持つコールバックをセットアップする。

        BaseCallback.training_env は model.get_env() を返す property のため、
        model をモックして training_env を間接的に設定する。
        """
        cb = _make_callback(**kwargs)
        # training_env を間接的にモック（model.get_env() 経由）
        mock_env = MagicMock()
        mock_env.num_envs = n_envs
        mock_env.env_method.return_value = [True] * n_envs
        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_env
        mock_model.num_timesteps = 0
        cb.model = mock_model
        # ParamApplier の training_env を設定
        cb._param_applier.set_training_env(mock_env)
        # _on_training_start に相当する初期化
        cb._score_tracker.reset(n_envs)
        initial_params = _phase_params_from_phase(0)
        initial_vec = cb._spalf.params_to_vec(initial_params)
        cb._ep_start_param_vec_per_env = [initial_vec.copy() for _ in range(n_envs)]
        cb._spalf._current_params = initial_params
        cb._spalf._current_param_vec = initial_vec
        return cb

    def _make_episode_info(self, base_reward: float, ep_len: int, ep_r: float = 0.0) -> dict:
        return {"base_reward": base_reward, "episode": {"l": ep_len, "r": ep_r}}

    def test_phase_advance_resets_spalf_buffers(self):
        """フェーズ昇格時に SPALF の ALP バッファがリセットされること。"""
        cb = self._setup_callback_with_mock_env(n_envs=1,
                                                warmup_episodes=0, phase_warmup_episodes=0)
        # ALP バッファに何か入れる
        cb._spalf._reward_history.append((np.zeros(8), 0.3))
        cb._spalf._alp_buffer.append((np.zeros(8), 0.1))

        # フェーズ 0 を十分に高いスコアで昇格させる
        phase0 = _CURRICULUM_PHASES[0]
        threshold = (phase0.threshold or 1.0) * cb._curriculum.threshold_mult
        high_score = threshold * 2.0

        for _ in range(3):
            cb._curriculum.on_episode_end(high_score, 3000)
        cb._curriculum.check_phase_transition()  # advance
        # 昇格後、_on_phase_changed を呼ぶ
        cb._on_phase_changed("advance")

        # ALP バッファがリセットされること
        assert len(cb._spalf._reward_history) == 0
        assert len(cb._spalf._alp_buffer) == 0

    def test_phase_advance_updates_bounds(self):
        """フェーズ昇格時に SPALF の bounds が新フェーズの範囲に更新されること。"""
        cb = self._setup_callback_with_mock_env(n_envs=1,
                                                warmup_episodes=0, phase_warmup_episodes=0)
        initial_bounds = dict(cb._spalf._active_bounds)

        phase0 = _CURRICULUM_PHASES[0]
        threshold = (phase0.threshold or 1.0) * cb._curriculum.threshold_mult
        high_score = threshold * 2.0
        for _ in range(3):
            cb._curriculum.on_episode_end(high_score, 3000)
        cb._curriculum.check_phase_transition()
        cb._on_phase_changed("advance")

        new_bounds = cb._spalf._active_bounds
        expected = _phase_bounds(1)
        assert new_bounds == expected, "フェーズ昇格後の bounds がフェーズ 1 の範囲になること"

    def test_rollback_decrements_phase(self):
        """rollback 時にフェーズが戻ること。"""
        cb = self._setup_callback_with_mock_env(n_envs=1,
                                                warmup_episodes=0, phase_warmup_episodes=0,
                                                rollback_patience=2)
        # まず phase 1 へ昇格
        phase0 = _CURRICULUM_PHASES[0]
        high_score = (phase0.threshold or 1.0) * 2.0
        for _ in range(3):
            cb._curriculum.on_episode_end(high_score, 3000)
        cb._curriculum.check_phase_transition()
        assert cb._curriculum.current_phase == 1

        # phase 1 で低スコアを入れてロールバック
        phase1 = _CURRICULUM_PHASES[1]
        threshold1 = (phase1.threshold or 1.0) * cb._curriculum.threshold_mult
        floor = threshold1 * phase1.rollback_score_ratio
        low_score = floor * 0.5
        for _ in range(3):
            cb._curriculum.on_episode_end(low_score, 100)
        cb._curriculum.check_phase_transition()  # bad window 1
        for _ in range(3):
            cb._curriculum.on_episode_end(low_score, 100)
        result = cb._curriculum.check_phase_transition()  # rollback

        if result == "rollback":
            cb._on_phase_changed("rollback")
            assert cb._curriculum.current_phase == 0

    def test_sampled_params_within_bounds(self):
        """サンプリングしたパラメータが現在の bounds 内に収まること。"""
        cb = self._setup_callback_with_mock_env(n_envs=1, warmup_episodes=0)
        # ALP バッファが少なくてランダムサンプリングになる
        for _ in range(20):
            params, vec = cb._spalf.sample_next_params()
            bounds = cb._spalf._active_bounds
            for key in _PARAM_KEYS:
                val = params.get(key)
                if val is None:
                    continue
                lo, hi = bounds[key]
                if isinstance(val, bool):
                    fval = 1.0 if val else 0.0
                elif isinstance(val, int):
                    fval = float(val)
                else:
                    fval = float(val)
                # ゼロ幅 bounds は lo 固定なので lo <= hi の範囲で許容
                if hi > lo:
                    assert fval >= lo - 1e-3, f"key={key}: val={fval} < lo={lo}"
                    assert fval <= hi + 1e-3, f"key={key}: val={fval} > hi={hi}"


class TestHybridCallbackExportImport:
    def test_export_import_roundtrip(self):
        """export_state / import_state が状態を正確に復元すること。"""
        cb1 = _make_callback()
        cb1._curriculum._phase_idx = 2
        cb1._spalf._total_episodes = 30
        cb1._extended_bounds_enabled = False

        state = cb1.export_state()

        cb2 = _make_callback()
        cb2.import_state(state)
        assert cb2._curriculum.current_phase == 2
        assert cb2._spalf._total_episodes == 30
        assert cb2._extended_bounds_enabled is False
        assert cb2._pending_resume_state is True  # resume フラグが立つこと

    def test_export_state_is_flat_dict(self):
        """export_state が flat dict を返すこと（phase_idx がトップレベルにあること）。"""
        cb = _make_callback()
        state = cb.export_state()
        assert "phase_idx" in state, "phase_idx がトップレベルにあること（flat dict）"
        assert "spalf_state" in state, "spalf_state キーが含まれること"
        assert "extended_bounds_enabled" in state, "extended_bounds_enabled キーが含まれること"

    def test_import_state_restores_bounds(self):
        """import_state 後の bounds がフェーズに対応した範囲になること。"""
        cb1 = _make_callback()
        cb1._curriculum._phase_idx = 2
        state = cb1.export_state()

        cb2 = _make_callback()
        cb2.import_state(state)
        expected = _phase_bounds(2)
        assert cb2._spalf._active_bounds == expected

    def test_import_extended_bounds_state(self):
        """extended_bounds_enabled=True の状態を import すると拡張 bounds が設定されること。"""
        last_idx = len(_CURRICULUM_PHASES) - 1
        cb1 = _make_callback()
        cb1._curriculum._phase_idx = last_idx
        cb1._extended_bounds_enabled = True
        state = cb1.export_state()

        cb2 = _make_callback()
        cb2.import_state(state)
        # 拡張 bounds が設定されること
        expected = _phase_bounds(last_idx, completion_ready=True)
        assert cb2._spalf._active_bounds == expected


class TestHybridCallbackFinalPhase:
    def test_final_phase_uses_extended_bounds_after_completion(self):
        """最終フェーズ completion 後の bounds の下限が PARAM_BOUNDS max に一致すること。"""
        last_idx = len(_CURRICULUM_PHASES) - 1
        bounds_extended = _phase_bounds(last_idx, completion_ready=True)
        for key in _PARAM_KEYS:
            assert abs(bounds_extended[key][0] - PARAM_BOUNDS[key][1]) < 1e-6, (
                f"key={key}: extended lower bound should be PARAM_BOUNDS max={PARAM_BOUNDS[key][1]}"
            )

    def test_is_final_phase_property(self):
        """is_final_phase プロパティが正しく動作すること。"""
        cb = _make_callback()
        assert cb.is_final_phase is False
        cb._curriculum._phase_idx = len(_CURRICULUM_PHASES) - 1
        assert cb.is_final_phase is True


class TestHybridCallbackWandbMetrics:
    def test_wandb_metrics_merged_from_modules(self):
        """get_wandb_metrics が全モジュールのメトリクスをマージすること。"""
        cb = _make_callback()
        metrics = {}
        for m in cb._state_modules:
            metrics.update(m.get_wandb_metrics())

        # SPALF メトリクスが含まれること
        assert "spalf/mode" in metrics
        assert "spalf/buffer_size" in metrics
        # Curriculum メトリクスが含まれること
        assert "curriculum/phase_idx" in metrics


class TestPhaseBoundsEdgeCases:
    def test_phase_bounds_for_all_non_final_phases(self):
        """全非最終フェーズで _phase_bounds が正しい形式を返すこと。"""
        for i in range(len(_CURRICULUM_PHASES) - 1):
            bounds = _phase_bounds(i)
            assert set(bounds.keys()) == set(_PARAM_KEYS), f"phase_idx={i}: keys mismatch"
            for key in _PARAM_KEYS:
                lo, hi = bounds[key]
                assert lo <= hi, f"phase_idx={i}, key={key}: lo={lo} > hi={hi}"

    def test_phase_params_from_phase_returns_all_keys(self):
        """_phase_params_from_phase がすべての PARAM_KEYS を返すこと。"""
        for i in range(len(_CURRICULUM_PHASES)):
            params = _phase_params_from_phase(i)
            for key in _PARAM_KEYS:
                assert key in params, f"phase_idx={i}: missing key={key}"


class TestHybridProbePromotion:
    """HybridCurriculumSpalfCallback の probe 昇格 API テスト。"""

    def _setup_callback_with_mock_env(self, n_envs: int = 2, **kwargs):
        cb = _make_callback(**kwargs)
        mock_env = MagicMock()
        mock_env.num_envs = n_envs
        mock_env.env_method.return_value = [True] * n_envs
        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_env
        mock_model.num_timesteps = 0
        cb.model = mock_model
        cb._param_applier.set_training_env(mock_env)
        cb._score_tracker.reset(n_envs)
        initial_params = _phase_params_from_phase(0)
        initial_vec = cb._spalf.params_to_vec(initial_params)
        cb._ep_start_param_vec_per_env = [initial_vec.copy() for _ in range(n_envs)]
        cb._spalf._current_params = initial_params
        cb._spalf._current_param_vec = initial_vec
        return cb

    def test_on_step_does_not_advance_train_only(self):
        """allow_promotion=False の _on_step では train episode だけで advance しないこと。"""
        cb = self._setup_callback_with_mock_env(n_envs=2, warmup_episodes=0, phase_warmup_episodes=0)
        phase0 = _CURRICULUM_PHASES[0]
        threshold = (phase0.threshold or 1.0) * cb._curriculum.threshold_mult
        high_score = threshold * 2.0

        # window 分の高スコアを train window に入れる
        for _ in range(3):
            cb._curriculum.on_episode_end(high_score, ep_len=3000)

        # _on_step の check_phase_transition(allow_promotion=False) を直接呼び出して検証
        event = cb._curriculum.check_phase_transition(allow_promotion=False, promotion_source="train")
        assert event is None, "allow_promotion=False では train scores だけで advance しないこと"
        assert cb._curriculum.current_phase == 0

    def test_on_promotion_probe_results_advances_phase(self):
        """on_promotion_probe_results で昇格条件達成時に _on_phase_changed が呼ばれること。"""
        cb = self._setup_callback_with_mock_env(n_envs=2, warmup_episodes=0, phase_warmup_episodes=0)
        phase0 = _CURRICULUM_PHASES[0]
        threshold = (phase0.threshold or 1.0) * cb._curriculum.threshold_mult
        high_score = threshold * 2.0

        results = [{"active_score": high_score, "ep_len": 3000, "base_reward": high_score}
                   for _ in range(3)]
        event = cb.on_promotion_probe_results(results)
        assert event == "advance"
        assert cb._curriculum.current_phase == 1

    def test_probe_advance_updates_spalf_bounds(self):
        """probe advance 後に SPALF bounds が新 phase 用に更新されること。"""
        cb = self._setup_callback_with_mock_env(n_envs=2, warmup_episodes=0, phase_warmup_episodes=0)
        phase0 = _CURRICULUM_PHASES[0]
        threshold = (phase0.threshold or 1.0) * cb._curriculum.threshold_mult
        high_score = threshold * 2.0

        results = [{"active_score": high_score, "ep_len": 3000, "base_reward": high_score}
                   for _ in range(3)]
        cb.on_promotion_probe_results(results)

        from games.survivors.hybrid_callback import _phase_bounds
        expected = _phase_bounds(1)
        assert cb._spalf._active_bounds == expected

    def test_get_current_phase_params_returns_next_phase(self):
        """get_current_phase_params が次フェーズ（phase 1）の params を返すこと。

        仕様変更: probe eval は昇格後の環境を事前検証するため、
        現フェーズ(0)ではなく次フェーズ(1)のパラメータを返す。
        """
        cb = _make_callback()
        # 初期状態は phase 0 なので次フェーズ (phase 1) のパラメータを返す
        params = cb.get_current_phase_params()
        phase1_params = _phase_params_from_phase(1)
        assert params == phase1_params

    def test_get_current_phase_params_returns_current_at_final_phase(self):
        """最終フェーズでは次フェーズが存在しないため現フェーズの params を返すこと。"""
        cb = _make_callback()
        last_idx = len(_CURRICULUM_PHASES) - 1
        cb._curriculum._phase_idx = last_idx
        params = cb.get_current_phase_params()
        final_params = _phase_params_from_phase(last_idx)
        assert params == final_params


class TestOnPromotionProbeEvalResults:
    """on_promotion_probe_eval_results のテスト。"""

    def _setup_callback_with_mock_env(self, n_envs: int = 2, **kwargs):
        cb = _make_callback(**kwargs)
        mock_env = MagicMock()
        mock_env.num_envs = n_envs
        mock_env.env_method.return_value = [True] * n_envs
        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_env
        mock_model.num_timesteps = 0
        cb.model = mock_model
        cb._param_applier.set_training_env(mock_env)
        cb._score_tracker.reset(n_envs)
        initial_params = _phase_params_from_phase(0)
        initial_vec = cb._spalf.params_to_vec(initial_params)
        cb._ep_start_param_vec_per_env = [initial_vec.copy() for _ in range(n_envs)]
        cb._spalf._current_params = initial_params
        cb._spalf._current_param_vec = initial_vec
        return cb

    def test_ep_length_converted_to_ep_len_for_on_promotion_probe_results(self):
        """ep_length が ep_len に変換されて on_promotion_probe_results に渡ること。"""
        from unittest.mock import patch
        cb = _make_callback()

        episode_results = [
            {"ep_length": 100, "active_score": 5.0, "base_reward": 5.0},
            {"ep_length": 200, "active_score": 8.0, "base_reward": 8.0},
        ]
        metrics = {"active_score": 6.5, "ep_length": 150.0, "n_episodes": 2}

        received: list[list[dict]] = []

        with patch.object(cb, "on_promotion_probe_results", side_effect=lambda r, **kw: received.append(r)) as mock_probe, \
             patch.object(cb, "_log_promotion_probe_metrics"):
            cb.on_promotion_probe_eval_results(episode_results, metrics)

        mock_probe.assert_called_once()
        passed_results = received[0]
        assert len(passed_results) == 2

        # ep_length → ep_len 変換が行われていること
        for r in passed_results:
            assert "ep_len" in r, "ep_len キーが存在すること"
            assert r["ep_len"] in (100, 200), "ep_length の値が ep_len に変換されること"

        # ep_length も引き続き存在すること（元の結果を汚染しないこと）
        assert passed_results[0]["ep_length"] == 100
        assert passed_results[1]["ep_length"] == 200

    def test_log_promotion_probe_metrics_called_with_metrics(self):
        """_log_promotion_probe_metrics が metrics・pre_diag・event で呼ばれること。"""
        from unittest.mock import patch
        cb = _make_callback()

        episode_results = [
            {"ep_length": 100, "active_score": 5.0, "base_reward": 5.0},
        ]
        metrics = {"active_score": 5.0, "ep_length": 100.0, "n_episodes": 1}

        with patch.object(cb, "on_promotion_probe_results", return_value=None) as mock_probe, \
             patch.object(cb, "_log_promotion_probe_metrics") as mock_log:
            cb.on_promotion_probe_eval_results(episode_results, metrics)

        mock_log.assert_called_once()
        call_kwargs = mock_log.call_args
        # 第1引数が metrics であること
        assert call_kwargs[0][0] == metrics
        # pre_diag / event キーワード引数が渡されること
        assert "pre_diag" in call_kwargs[1]
        assert "event" in call_kwargs[1]


class TestLogPromotionProbeMetrics:
    """_log_promotion_probe_metrics の W&B ログテスト。"""

    def _make_pre_diag(self) -> dict:
        """テスト用の probe diagnostics を返す。"""
        return {
            "threshold": 10.0,
            "promotion_ready": False,
            "n_probe_episodes": 5,
        }

    def test_curriculum_probe_metrics_logged_to_wandb(self):
        """curriculum/* probe 関連メトリクスが W&B に記録されること。"""
        from common.wandb_logger import WandbLogger

        wandb_logger = MagicMock(spec=WandbLogger)
        wandb_logger.enabled = True

        cb = _make_callback(wandb_logger=wandb_logger)
        # num_timesteps を設定
        mock_model = MagicMock()
        mock_model.num_timesteps = 1000
        cb.model = mock_model

        metrics = {
            "active_score": 7.5,
            "ep_length": 150.0,
            "n_episodes": 5,
        }
        pre_diag = self._make_pre_diag()

        cb._log_promotion_probe_metrics(metrics, pre_diag=pre_diag, event=None)

        wandb_logger.log.assert_called_once()
        logged_metrics = wandb_logger.log.call_args[0][0]

        # curriculum/* probe キーが含まれること
        assert "curriculum/probe_threshold" in logged_metrics
        assert "curriculum/probe_promotion_ready" in logged_metrics
        assert "curriculum/probe_event" in logged_metrics

    def test_log_not_called_when_wandb_logger_none(self):
        """wandb_logger が None の場合は何も記録しないこと。"""
        cb = _make_callback(wandb_logger=None)
        mock_model = MagicMock()
        mock_model.num_timesteps = 1000
        cb.model = mock_model

        metrics = {"active_score": 5.0, "ep_length": 100.0, "n_episodes": 3}
        pre_diag = self._make_pre_diag()

        # 例外なく動作すること
        cb._log_promotion_probe_metrics(metrics, pre_diag=pre_diag, event=None)

    def test_log_not_called_when_wandb_disabled(self):
        """wandb_logger が disabled の場合は何も記録しないこと。"""
        from common.wandb_logger import WandbLogger
        wandb_logger = MagicMock(spec=WandbLogger)
        wandb_logger.enabled = False

        cb = _make_callback(wandb_logger=wandb_logger)
        mock_model = MagicMock()
        mock_model.num_timesteps = 1000
        cb.model = mock_model

        metrics = {"active_score": 5.0, "ep_length": 100.0, "n_episodes": 3}
        pre_diag = self._make_pre_diag()

        cb._log_promotion_probe_metrics(metrics, pre_diag=pre_diag, event=None)

        wandb_logger.log.assert_not_called()


class TestProgressMetricsKeyConsistency:
    """CurriculumCallback と HybridCurriculumSpalfCallback の W&B メトリクスキーが一致することを保証する。"""

    def test_curriculum_and_hybrid_progress_metric_keys_match(self):
        """--curriculum と --curriculum-spalf の get_wandb_progress_metrics() キーセットが一致すること。

        両 callback は CurriculumStateModule.get_wandb_progress_metrics() に委譲しているため、
        エピソードを流さずキーセットだけ比較すれば追従漏れを検出できる。
        """
        from games.survivors.survivors_curriculum import CurriculumCallback

        curriculum_cb = CurriculumCallback(
            raw_env=None,
            frame_skip=1,
            window=20,
            threshold_mult=5.0,
            alive_reward=0.001,
        )
        hybrid_cb = _make_callback(phase_window=20, threshold_mult=5.0)

        # num_timesteps は SB3 BaseCallback.__init__ で 0 に初期化される
        # エピソードデータ無しでも get_wandb_progress_metrics() は全キーを返す
        curriculum_metrics = curriculum_cb.get_wandb_progress_metrics()
        hybrid_metrics = hybrid_cb.get_wandb_progress_metrics()

        assert set(hybrid_metrics.keys()) == set(curriculum_metrics.keys()), (
            f"キー差分: curriculum only={set(curriculum_metrics) - set(hybrid_metrics)}, "
            f"hybrid only={set(hybrid_metrics) - set(curriculum_metrics)}"
        )

"""HybridCurriculumSpalfCallback: カリキュラム × SPALF ハイブリッドコールバック。

SpalfCallback を継承せず BaseCallback を直接継承し、
SpalfStateModule・CurriculumStateModule・EpisodeScoreTracker を組み合わせて実装する。
これにより SpalfCallback / CurriculumCallback の変更が自動的に Hybrid にも反映される。
"""
from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from games.survivors.state_modules import (
    BaseStateModule,
    EpisodeScoreTracker,
    SpalfStateModule,
    CurriculumStateModule,
)
from games.survivors.param_applier import ParamApplier
from games.survivors.survivors_difficulty import PARAM_BOUNDS
from common.wandb_logger import WandbLogger

# PHASES をインポート（_phase_bounds / _phase_params_from_phase で使用）
from games.survivors.survivors_curriculum import PHASES as _CURRICULUM_PHASES

_PARAM_KEYS: list[str] = list(PARAM_BOUNDS.keys())

# 最終フェーズ専用拡張範囲: 上限値のみ定義。下限は PARAM_BOUNDS max から自動導出（同期のため）。
_FINAL_PHASE_EXTENDED_UPPER: dict[str, float] = {
    "min_enemies":        80.0,
    "max_enemies":        300.0,
    "speed_mult":         1.2,    # 上限固定（PARAM_BOUNDS max と同値）
    "spawn_rate_mult":    6.0,
    "max_enemy_type_id":  10.0,   # 上限固定
    "enemy_hp_scale":     4.0,
    "enemy_damage_scale": 4.0,
    "time_scaling":       1.0,    # True 固定
}

# PARAM_BOUNDS.max を下限として動的導出 → PARAM_BOUNDS 変更時に自動同期
_FINAL_PHASE_EXTENDED_BOUNDS: dict[str, tuple[float, float]] = {
    key: (PARAM_BOUNDS[key][1], _FINAL_PHASE_EXTENDED_UPPER[key])
    for key in _PARAM_KEYS
}


def _phase_bounds(phase_idx: int, completion_ready: bool = False) -> dict[str, tuple[float, float]]:
    """フェーズインデックスから Hybrid 専用 SPALF 探索範囲を返す。

    - 通常フェーズ: [PHASES[i].param, PHASES[i+1].param]（単調性保証のため min/max swap あり）
    - 最終フェーズ（completion 前）: PHASES[-1] の固定値（拡張せず安定学習を優先）
    - 最終フェーズ（completion 後）: _FINAL_PHASE_EXTENDED_BOUNDS（下限=PARAM_BOUNDS max）

    completion 前に拡張 bounds を適用すると、まだ Mad Forest を安定クリアできていない段階で
    さらに難しい環境が大量に試され、学習が停滞する可能性がある。
    """
    if phase_idx < len(_CURRICULUM_PHASES) - 1:
        phase = _CURRICULUM_PHASES[phase_idx]
        next_phase = _CURRICULUM_PHASES[phase_idx + 1]
        bounds: dict[str, tuple[float, float]] = {}
        for key in _PARAM_KEYS:
            lo_global, hi_global = PARAM_BOUNDS[key]
            cur_raw = getattr(phase, key)
            nxt_raw = getattr(next_phase, key)
            cur_val = float(1.0 if cur_raw else 0.0) if isinstance(cur_raw, bool) else float(cur_raw)
            nxt_val = float(1.0 if nxt_raw else 0.0) if isinstance(nxt_raw, bool) else float(nxt_raw)
            cur_c = max(lo_global, min(hi_global, cur_val))
            nxt_c = max(lo_global, min(hi_global, nxt_val))
            bounds[key] = (min(cur_c, nxt_c), max(cur_c, nxt_c))
        return bounds
    elif completion_ready:
        # completion 後: Mad Forest 超えの拡張 bounds に切り替え
        return dict(_FINAL_PHASE_EXTENDED_BOUNDS)
    else:
        # completion 前: PHASES[-1] 固定値（探索幅ゼロ = 固定難易度で安定化を優先）
        # PHASES[-1] から固定値 bounds を生成（lo == hi = fixed value）
        phase = _CURRICULUM_PHASES[-1]
        fixed: dict[str, tuple[float, float]] = {}
        for key in _PARAM_KEYS:
            raw = getattr(phase, key)
            val = float(1.0 if raw else 0.0) if isinstance(raw, bool) else float(raw)
            fixed[key] = (val, val)  # lo == hi → ゼロ幅 bounds = 固定値
        return fixed


def _phase_params_from_phase(phase_idx: int) -> dict:
    """_Phase から SPALF params dict を生成するヘルパー（bool/int 変換を統一）。"""
    phase = _CURRICULUM_PHASES[phase_idx]
    return {
        key: (
            bool(getattr(phase, key)) if key == "time_scaling"
            else int(getattr(phase, key)) if key in ("min_enemies", "max_enemies", "max_enemy_type_id")
            else float(getattr(phase, key))
        )
        for key in _PARAM_KEYS
    }


class HybridCurriculumSpalfCallback(BaseCallback):
    """カリキュラムフェーズで探索範囲を制限した SPALF コールバック。

    SpalfCallback を継承せず BaseCallback を直接継承し、
    SpalfStateModule・CurriculumStateModule・EpisodeScoreTracker を組み合わせて実装する。
    これにより SpalfCallback / CurriculumCallback の変更が自動的に Hybrid にも反映される。
    """

    def __init__(
        self,
        raw_env,
        frame_skip: int = 4,
        alive_reward: float = 0.001,
        # SPALF args
        r_b: float = 0.1,
        alpha: float = 0.2,
        max_score: float = 15000.0,
        spalf_buffer_size: int = 200,
        warmup_episodes: int = 50,          # 初期 warmup（フェーズ 0）
        phase_warmup_episodes: int = 10,    # フェーズ切り替え後の per-phase warmup
        spalf_status_path=None,
        # Curriculum args
        phase_window: int = 20,
        threshold_mult: float = 5.0,
        rollback_patience: int = 3,
        curriculum_status_path=None,
        # Infrastructure
        wandb_logger: WandbLogger | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        # UE5 パラメータ送信モジュール
        self._param_applier = ParamApplier(raw_env)

        # 共通スコア計算モジュール（frame_skip / alive_reward を1箇所で管理）
        self._score_tracker = EpisodeScoreTracker(frame_skip, alive_reward)

        # W&B ログ
        self._wandb_logger = wandb_logger
        if wandb_logger:
            wandb_logger.add_metric_prefix("hybrid/")
            wandb_logger.add_metric_prefix("spalf/")
            wandb_logger.add_metric_prefix("curriculum/")
            wandb_logger.add_metric_prefix("survivors/")

        # SPALF モジュール（初期 bounds はフェーズ 0 に設定）
        self._phase_warmup_episodes = phase_warmup_episodes
        self._spalf = SpalfStateModule(
            r_b=r_b, alpha=alpha, max_score=max_score,
            buffer_size=spalf_buffer_size, warmup_episodes=warmup_episodes,
            status_path=spalf_status_path,
            initial_bounds=_phase_bounds(0),
        )

        # Curriculum モジュール
        self._curriculum = CurriculumStateModule(
            window=phase_window, threshold_mult=threshold_mult,
            rollback_patience=rollback_patience, status_path=curriculum_status_path,
        )

        # ポリモーフィズムで扱うモジュールリスト（W&B ログ・状態保存に使用）
        self._state_modules: list[BaseStateModule] = [self._spalf, self._curriculum]

        # per-env の SPALF パラメータ開始 vec
        self._ep_start_param_vec_per_env: list[np.ndarray] = []

        # 最終フェーズ completion 後の拡張 bounds 適用済みフラグ
        # （_on_step で一度だけ set_bounds を呼ぶ。phase change に頼らず completion_ready になった時点で切り替え）
        self._extended_bounds_enabled: bool = False

        # resume 処理用フラグ（import_state 後に _on_training_start で使用）
        self._pending_resume_state: bool = False

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs if self.training_env is not None else 1
        self._param_applier.set_training_env(self.training_env)
        self._score_tracker.reset(n)

        # import_state 後の resume: _ep_start_param_vec_per_env を current_param_vec から再構築
        if self._pending_resume_state:
            self._pending_resume_state = False
            # bounds は import_state で設定済み。current_param_vec から per-env vec を初期化。
            self._ep_start_param_vec_per_env = [
                self._spalf._current_param_vec.copy() for _ in range(n)
            ]
            # resume 時も復元した current_params を UE5 env へ適用する
            if self._spalf._current_params:
                self._param_applier.apply(self._spalf._current_params)
            phase_name = _CURRICULUM_PHASES[self._curriculum.current_phase].name
            self._log_phase_event(f"Resume（Phase {self._curriculum.current_phase} {phase_name}）")
            return

        # 新規開始: フェーズ 0 のパラメータを適用
        self._spalf.set_bounds(_phase_bounds(0))
        initial_params = _phase_params_from_phase(0)
        self._param_applier.apply(initial_params)
        initial_vec = self._spalf.params_to_vec(initial_params)
        self._ep_start_param_vec_per_env = [initial_vec.copy() for _ in range(n)]
        self._spalf._current_params = initial_params
        self._spalf._current_param_vec = initial_vec
        self._log_phase_event("初期設定")

    def _on_step(self) -> bool:
        episode_results = self._score_tracker.process(self.locals["infos"])

        ep_active_scores: list[float] = []
        ep_score_norms: list[float] = []
        ep_has_warmup: bool = False

        # 最終フェーズ completion 後の拡張 bounds 切り替え（phase change に依存しないため _on_step で検出）
        if (not self._extended_bounds_enabled
                and self._curriculum.is_final_phase
                and self._curriculum.get_completion_diagnostics().get("complete", False)):
            self._extended_bounds_enabled = True
            self._spalf.set_bounds(_phase_bounds(self._curriculum.current_phase, completion_ready=True))
            self._spalf.reset_buffers_for_phase_change()
            # per-env 開始ベクトルを新しい拡張 bounds での current_param_vec に同期する
            extended_vec = self._spalf.params_to_vec(self._spalf._current_params)
            self._spalf._current_param_vec = extended_vec
            n = len(self._ep_start_param_vec_per_env)
            if n > 0:
                self._ep_start_param_vec_per_env = [extended_vec.copy() for _ in range(n)]
            self._log_phase_event("completion_ready → 拡張 bounds へ切り替え")

        infos = self.locals["infos"]
        # EpisodeScoreTracker.process() は 4-tuple (env_idx, active_score, ep_len, ep_base) を返す
        for env_idx, active_score, ep_len, ep_base in episode_results:
            # Curriculum モジュールへ通知（診断情報を既存 CurriculumCallback と同等に渡す）
            info = infos[env_idx] if env_idx < len(infos) else {}
            alive_r = self._score_tracker.alive_reward * self._score_tracker.frame_skip * ep_len
            is_truncated = bool(info.get("TimeLimit.truncated", False))
            self._curriculum.on_episode_end(
                active_score, ep_len,
                base_reward=ep_base,
                alive_reward=alive_r,
                terminated=not is_truncated,
            )
            # SPALF モジュールへ通知（ALP 計算・バッファ更新）
            ep_param_vec = self._ep_start_param_vec_per_env[env_idx]
            is_warmup = self._spalf.on_episode_end(env_idx, active_score, ep_param_vec)
            score_norm = active_score / self._spalf._max_score
            ep_active_scores.append(active_score)
            ep_score_norms.append(score_norm)
            if is_warmup:
                ep_has_warmup = True
                # warmup 中もランダムサンプリングして次パラメータを適用する
                new_params, new_vec = self._spalf.sample_next_params()
                self._apply_params(new_params, env_idx=env_idx)
                self._ep_start_param_vec_per_env[env_idx] = new_vec.copy()
                self._spalf._current_params = new_params
                self._spalf._current_param_vec = new_vec
                continue
            # non-warmup: 次パラメータをサンプリングして適用
            new_params, new_vec = self._spalf.sample_next_params()
            self._apply_params(new_params, env_idx=env_idx)
            self._ep_start_param_vec_per_env[env_idx] = new_vec.copy()
            # W&B ログ・export_state・resume の整合性のため current_params/vec を更新する
            self._spalf._current_params = new_params
            self._spalf._current_param_vec = new_vec

        # W&B ログ
        if ep_active_scores:
            self._log_wandb_per_step(ep_active_scores, ep_score_norms, ep_has_warmup)

        # フェーズ遷移判定（エピソード終了があったステップのみ）
        # allow_promotion=False: train episode での昇格は発生させない（probe path に委譲）
        if episode_results:
            event = self._curriculum.check_phase_transition(allow_promotion=False, promotion_source="train")
            if event in ("rollback",):
                self._on_phase_changed(event)

        self._curriculum._steps_in_phase += 1
        return True

    def _on_phase_changed(self, event: str) -> None:
        """フェーズ変化時の処理。"""
        # SPALF バッファリセット（習熟度 _recent_reward_buffer / _total_episodes は維持）
        self._spalf.reset_buffers_for_phase_change()
        # 最終フェーズ completion 済みなら拡張 bounds へ
        completion_ready = (
            self._curriculum.is_final_phase
            and self._curriculum.get_completion_diagnostics().get("complete", False)
        )
        # CurriculumStateModule.current_phase は int（phase_idx）を返す
        new_bounds = _phase_bounds(self._curriculum.current_phase, completion_ready=completion_ready)
        self._spalf.set_bounds(new_bounds)
        # per-phase warmup（set_phase_warmup を使用）
        self._spalf.set_phase_warmup(self._phase_warmup_episodes)
        # フェーズのパラメータを適用
        phase_params = _phase_params_from_phase(self._curriculum.current_phase)
        self._apply_params(phase_params)
        self._spalf._current_params = phase_params
        phase_vec = self._spalf.params_to_vec(phase_params)
        self._spalf._current_param_vec = phase_vec
        # フェーズ変更後の per-env 開始ベクトルを新しいフェーズパラメータで同期する
        n = len(self._ep_start_param_vec_per_env)
        if n > 0:
            self._ep_start_param_vec_per_env = [phase_vec.copy() for _ in range(n)]
        # current_phase は int → _CURRICULUM_PHASES[...].name でフェーズ名を取得
        phase_name = _CURRICULUM_PHASES[self._curriculum.current_phase].name
        self._log_phase_event(f"{event}: Phase {self._curriculum.current_phase} {phase_name}")

    def _log_wandb_per_step(self, active_scores: list[float], score_norms: list[float], has_warmup: bool) -> None:
        if not self._wandb_logger or not self._wandb_logger.enabled:
            return
        metrics: dict = {}
        # ポリモーフィズムで全モジュールのメトリクスをマージ
        for m in self._state_modules:
            metrics.update(m.get_wandb_metrics())
        if has_warmup:
            metrics["spalf/mode"] = 0
        metrics.update({
            "survivors/active_score":     sum(active_scores) / len(active_scores),
            "survivors/score_normalized": sum(score_norms) / len(score_norms),
        })
        self._wandb_logger.log(metrics, step=self.num_timesteps)

    def _apply_params(self, params: dict, env_idx=None) -> None:
        """ParamApplier に委譲（SpalfCallback と同じロジックを重複しない）。"""
        self._param_applier.apply(params, env_idx=env_idx)

    def _log_phase_event(self, event: str) -> None:
        phase = _CURRICULUM_PHASES[self._curriculum.current_phase]
        b = self._spalf._active_bounds
        print(
            f"[HybridSPALF] {event}: {phase.name} | "
            f"bounds enemies [{b['min_enemies'][0]:.0f},{b['min_enemies'][1]:.0f}] "
            f"hp [{b['enemy_hp_scale'][0]:.2f},{b['enemy_hp_scale'][1]:.2f}]"
        )

    # ---- CurriculumCallback 互換 API ----

    @property
    def current_phase(self) -> int:
        return self._curriculum.current_phase

    @property
    def is_final_phase(self) -> bool:
        return self._curriculum.is_final_phase

    def configure_completion(self, window, min_episodes, min_score_ratio, min_ep_len_ratio) -> None:
        self._curriculum.configure_completion(window, min_episodes, min_score_ratio, min_ep_len_ratio)

    def get_completion_diagnostics(self) -> dict:
        return self._curriculum.get_completion_diagnostics()

    def get_diagnostics(self) -> dict:
        return self._curriculum.get_diagnostics(self.num_timesteps)

    def _compute_blocker_category(self, window_dict: dict, phase_threshold_is_none: bool) -> int:
        return self._curriculum._compute_blocker_category(window_dict, phase_threshold_is_none)

    def get_wandb_progress_metrics(self) -> dict:
        return self._curriculum.get_wandb_progress_metrics(self.num_timesteps)

    # ---- 状態保存・復元 ----

    def export_state(self) -> dict:
        """resume 用状態を flat dict で返す。

        "curriculum" キーが flat dict（既存 CurriculumCallback と同形式）になるよう、
        CurriculumStateModule の export_state() を flat に展開する。
        train.py の consumer（_save_training_status / L1523 episode_scores チェック等）は
        flat dict を前提としており、nested にすると既存ステータス解析が壊れる。
        SPALF 状態と extended_bounds フラグは別キー "spalf_state" / "extended_bounds_enabled" に保存。
        """
        curriculum_flat = self._curriculum.export_state()  # {"phase_idx": ..., "rollback_bad_windows": ..., ...}
        return {
            **curriculum_flat,
            "spalf_state": self._spalf.export_state(),
            "extended_bounds_enabled": self._extended_bounds_enabled,
        }

    def import_state(self, state: dict) -> None:
        """export_state() の出力から状態を復元する。"""
        # 1. Curriculum の flat dict からフェーズを先に復元（bounds 決定に必要）
        self._curriculum.import_state(state)
        # 2. extended_bounds フラグを復元してから bounds を設定
        self._extended_bounds_enabled = state.get("extended_bounds_enabled", False)
        completion_ready = self._extended_bounds_enabled and self._curriculum.is_final_phase
        self._spalf.set_bounds(_phase_bounds(self._curriculum.current_phase, completion_ready=completion_ready))
        # 3. SPALF 状態を復元
        self._spalf.import_state(state.get("spalf_state", {}))
        # 4. _ep_start_param_vec_per_env は _on_training_start で n_envs が確定してから初期化する
        self._pending_resume_state = True  # _on_training_start で current_param_vec から再構築する

    def rollback_one_phase(self, reason: str = "weapon_phase_advanced") -> None:
        """武器フェーズ昇格に伴うカリキュラム1フェーズ強制降格。

        CurriculumStateModule.rollback_one_phase() でフェーズ状態を更新した後、
        SPALF の探索範囲と敵難易度パラメータを新しいフェーズに合わせて更新する。

        Note:
            v09 ではこの関数は WeaponPhaseAutoStateModule からは呼ばれない。
            代わりに on_weapon_phase_advance() がスコアウィンドウのみリセットする。
            backward compatibility のため残している。
        """
        self._curriculum.rollback_one_phase(reason)
        self._on_phase_changed("forced_rollback")

    def on_weapon_phase_advance(self) -> None:
        """武器フェーズ昇格時: カリキュラムフェーズを維持して評価状態をリセット。

        v09 での forced_rollback 廃止に対応。武器フェーズが上がっても
        カリキュラムの難易度設定はそのままとし、スコアウィンドウ・rollback カウンタ・
        probe window をクリアして新しい武器セットで再評価を開始する。
        """
        self._curriculum.reset_evaluation_window()
        print(
            f"[HybridCurriculum] 武器フェーズ昇格による評価状態リセット "
            f"(phase={self._curriculum.current_phase} を維持)"
        )

    def _save_status(self) -> None:
        """全モジュールのステータスを保存する。"""
        if hasattr(self._spalf, 'save_status'):
            self._spalf.save_status()
        if hasattr(self._curriculum, 'save_status'):
            self._curriculum.save_status()

    # ---- Probe 昇格 API ----

    def get_current_phase_name(self) -> str:
        """現在のカリキュラムフェーズ名を返す（RSI 用）。"""
        return _CURRICULUM_PHASES[self._curriculum.current_phase].name

    def get_current_phase_params(self) -> dict:
        """probe eval で eval_env に適用する params dict を返す。

        次フェーズの lo パラメータ（難易度下限）で probe を実施することで、
        昇格後の環境に対応できるかを事前に検証する。
        最終フェーズでは次フェーズが存在しないため現フェーズのパラメータを返す。
        """
        current = self._curriculum.current_phase
        next_phase = current + 1
        if next_phase >= len(_CURRICULUM_PHASES):
            return _phase_params_from_phase(current)
        return _phase_params_from_phase(next_phase)

    def on_promotion_probe_results(self, episode_results: list[dict]) -> str | None:
        """probe episode 結果を受け取り昇格判定を行う。

        Args:
            episode_results: {"active_score": float, "ep_len": int, "base_reward": float} のリスト

        Returns:
            "advance" | None
        """
        for result in episode_results:
            self._curriculum.on_promotion_probe_episode_end(
                active_score=result["active_score"],
                ep_len=result["ep_len"],
                base_reward=result.get("base_reward", 0.0),
            )
        event = self._curriculum.check_promotion_transition(promotion_source="probe")
        if event == "advance":
            self._on_phase_changed("advance")
        return event

    def on_promotion_probe_eval_results(self, episode_results: list[dict], metrics: dict, step: int | None = None) -> None:
        """SurvivorsEvalCallback の after_eval_callback として呼ばれるエントリポイント。

        SurvivorsEvalCallback は ep_length キーを使う。
        on_promotion_probe_results() は ep_len を読むため、ここで変換する。

        Args:
            episode_results: SurvivorsEvalCallback が返す episode_results（ep_length キー）
            metrics:         SurvivorsEvalCallback が返す aggregate_metrics
            step:            eval 完了時の timestep（async eval 時は join 後の timestep）。
                             None の場合は self.num_timesteps を使用する。
        """
        # ep_length → ep_len 変換
        normalized = [
            {**r, "ep_len": r["ep_length"]} if "ep_length" in r else r
            for r in episode_results
        ]
        # advance 前に diagnostics を保存する。
        # on_promotion_probe_results() で advance が起きると current_phase が変わるため、
        # ログは昇格を発生させた phase の値で記録しなければならない。
        pre_diag = self._curriculum.get_promotion_probe_diagnostics()
        event = self.on_promotion_probe_results(normalized)
        self._log_promotion_probe_metrics(metrics, pre_diag=pre_diag, event=event, step=step)

    def _log_promotion_probe_metrics(self, metrics: dict, *, pre_diag: dict, event: str | None, step: int | None = None) -> None:
        """probe eval の昇格判定関連メトリクスを curriculum/* として W&B に記録する。

        active_score_mean / ep_length_mean / n_episodes / phase_idx は eval/* および
        curriculum/* に既出のため重複を避けて省略する。
        昇格判定固有の 3 項目のみ curriculum/* に追記する。

        Args:
            metrics:  SurvivorsEvalCallback が返す aggregate_metrics
            pre_diag: on_promotion_probe_results() 呼び出し前に取得した probe diagnostics
            event:    "advance" / None（on_promotion_probe_results() の戻り値）
            step:     ログに使用する timestep。None の場合は self.num_timesteps を使用する。
        """
        if not self._wandb_logger or not self._wandb_logger.enabled:
            return

        log_step = step if step is not None else self.num_timesteps
        wandb_metrics = {
            "curriculum/probe_threshold":        pre_diag.get("threshold"),
            "curriculum/probe_promotion_ready":  int(bool(pre_diag.get("promotion_ready"))),
            "curriculum/probe_event":            int(event == "advance"),
        }
        self._wandb_logger.log(wandb_metrics, step=log_step)

"""BootstrapGateEvalCallback: Bootstrap Lane の deterministic eval を定期実行するコールバック。

SurvivorsEvalCallback とは独立した BaseCallback サブクラス。
probe_freq ごとに全対象武器（solo_bootstrap / integration / maintenance）の
deterministic eval を実行し、WeaponBootstrapStateModule.set_deterministic_result() に
結果を注入する。
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from games.survivors.bootstrap_eval import (
    build_eval_params,
    parse_cell_spec,
    summarize_eval_results,
)
from games.survivors.survivors_eval_callback import run_survivors_eval_episodes
from games.survivors.survivors_weapon_table import WeaponEntry

if TYPE_CHECKING:
    from games.survivors.modules.weapon_bootstrap_module import WeaponBootstrapStateModule

try:
    import wandb  # noqa: F401
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class BootstrapGateEvalCallback(BaseCallback):
    """Bootstrap Lane のゲート判定用 deterministic eval コールバック。

    probe_freq ステップごとに solo_bootstrap / integration / maintenance ステータスの
    全武器に対して deterministic eval を実行し、WeaponBootstrapStateModule へ結果を注入する。

    Args:
        weapon_bootstrap_module: 武器状態管理モジュール。
        eval_env:               評価専用 VecEnv。
        weapon_unlock_order:    武器アンロック順序リスト。
        probe_freq:             eval 実行間隔（timesteps）。
        n_probe_episodes:       各武器の eval エピソード数。
        alive_reward:           alive_reward 値（active_score 計算に使用）。
        frame_skip:             フレームスキップ数。
        enemy_phase_idx:        eval 時の敵フェーズインデックス。
        item_stage_key:         武器プールの ItemSystem ステージキー。
        stage_key_provider:     現在の weapon unlock stage key を返す callable。
                                integration の build_policy 判定に使用する。
                                None の場合は各武器の unlock_stage_key をフォールバックとして使う。
        wandb_logger:           W&B ロガー。None の場合はログなし。
        verbose:                詳細出力レベル。
    """

    # eval 対象ステータス
    _EVAL_STATUSES = frozenset({"solo_bootstrap", "integration", "maintenance"})

    def __init__(
        self,
        weapon_bootstrap_module: "WeaponBootstrapStateModule",
        eval_env,
        weapon_unlock_order: list[WeaponEntry],
        probe_freq: int = 50_000,
        n_probe_episodes: int = 10,
        alive_reward: float = 0.001,
        frame_skip: int = 2,
        enemy_phase_idx: int = 2,
        item_stage_key: str = "IS0",
        stage_key_provider: "Callable[[], str] | None" = None,
        short_episode_steps: int = 600,
        wandb_logger=None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self._weapon_bootstrap_module = weapon_bootstrap_module
        self._eval_env = eval_env
        self._weapon_unlock_order = weapon_unlock_order
        self._probe_freq = probe_freq
        self._n_probe_episodes = n_probe_episodes
        self._alive_reward = alive_reward
        self._frame_skip = frame_skip
        self._enemy_phase_idx = enemy_phase_idx
        self._item_stage_key = item_stage_key
        self._stage_key_provider = stage_key_provider  # 必ず代入（代入漏れで AttributeError 防止）
        self._short_episode_steps = short_episode_steps
        self._wandb_logger = wandb_logger
        self._last_probe_step: int = 0
        self._id_to_key: dict[int, str] = {e.weapon_id: e.key for e in weapon_unlock_order}
        self._id_to_entry: dict[int, WeaponEntry] = {e.weapon_id: e for e in weapon_unlock_order}
        # ゲート閾値（W&B ログ用に保持）- status 別の閾値
        self._solo_target_p10: float = weapon_bootstrap_module._solo_bootstrap_target_p10
        self._integration_target_p10: float = weapon_bootstrap_module._integration_target_p10

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_probe_step >= self._probe_freq:
            self._last_probe_step = self.num_timesteps
            self._run_probe()
        return True

    def _sync_vecnormalize(self) -> None:
        """訓練側 VecNormalize の obs_rms/ret_rms を eval_env へコピーする。"""
        train_vecnorm: VecNormalize | None = None
        cur = self.training_env
        while cur is not None:
            if isinstance(cur, VecNormalize):
                train_vecnorm = cur
                break
            cur = getattr(cur, "venv", None)

        eval_vecnorm: VecNormalize | None = None
        cur = self._eval_env
        while cur is not None:
            if isinstance(cur, VecNormalize):
                eval_vecnorm = cur
                break
            cur = getattr(cur, "venv", None)

        if train_vecnorm is not None and eval_vecnorm is not None:
            eval_vecnorm.obs_rms = copy.deepcopy(train_vecnorm.obs_rms)
            eval_vecnorm.ret_rms = copy.deepcopy(train_vecnorm.ret_rms)

    def _run_probe(self) -> None:
        """全対象武器に deterministic eval を実行する。"""
        # 対象ステータスの武器を全て収集
        targets: list[tuple[str, int, str]] = []  # (weapon_key, weapon_id, status)
        for status in self._EVAL_STATUSES:
            for state in self._weapon_bootstrap_module.get_weapons_by_status(status):
                weapon_key = self._id_to_key.get(state.weapon_id)
                if weapon_key is None:
                    print(f"[BootstrapGateEval][WARN] unknown weapon_id={state.weapon_id}, skip")
                    continue
                targets.append((weapon_key, state.weapon_id, status))

        if not targets:
            return

        self._sync_vecnormalize()

        was_training = getattr(self._eval_env, "training", None)
        try:
            if was_training is not None:
                self._eval_env.training = False

            for weapon_key, weapon_id, status in targets:
                try:
                    self._eval_weapon(weapon_key, weapon_id, status)
                except Exception as e:
                    # 武器単位で握って訓練継続。この cycle は deterministic_p10 を更新しない
                    print(
                        f"[BootstrapGateEval][WARN] weapon={weapon_key}({weapon_id})"
                        f"[{status}] failed: {e}"
                    )
        finally:
            if was_training is not None:
                self._eval_env.training = was_training

    def _eval_weapon(self, weapon_key: str, weapon_id: int, status: str) -> None:
        """単一武器の deterministic eval を実行して結果を注入する。

        cell_spec の task_kind は武器の現在のステータスに合わせる。
        integration の build_policy は weapon_bootstrap_module.get_eval_build_policy() で
        task_cell_sampler と同じロジックで決定する。

        BootstrapEvalCell が含まれないため、entry.unlock_stage_key から取得する。
        self._id_to_entry は __init__ 時に構築済みなので KeyError は起きない。
        train.py の weapon_unlock_order と合わせて entry.unlock_stage_key を利用する。
        """
        entry = self._id_to_entry.get(weapon_id)
        if entry is None:
            raise ValueError(
                f"[BootstrapGateEval] weapon_id={weapon_id} が _id_to_entry に存在しません。"
                f"weapon_unlock_order に含まれているか確認してください。"
            )

        # stage_key_provider があればそれを使い、なければ entry.unlock_stage_key をフォールバックとして使う
        current_stage_key = (
            (self._stage_key_provider() if self._stage_key_provider else None)
            or entry.unlock_stage_key
        )

        cell_spec = f"{weapon_key}:{status}:{self._enemy_phase_idx}"
        cell = parse_cell_spec(cell_spec, weapon_unlock_order=self._weapon_unlock_order)

        # integration の build_policy は weapon_bootstrap_module.get_eval_build_policy() で決定
        # （garlic maintenance 済みかどうかで target_only / target_plus_anchor_if_unlocked を切り替える）
        build_policy_override = self._weapon_bootstrap_module.get_eval_build_policy(
            status,
            current_stage_key=current_stage_key,
        )

        ue_params = build_eval_params(
            cell,
            stage_key=current_stage_key,
            item_stage_key=self._item_stage_key,
            weapon_unlock_order=self._weapon_unlock_order,
            build_policy_override=build_policy_override,
        )

        # set_params の戻り値をリストで返すため、全 env で成功したか確認する
        set_params_results = self._eval_env.env_method("set_params", **ue_params)
        if not all(set_params_results):
            failed_indices = [i for i, ok in enumerate(set_params_results) if not ok]
            raise RuntimeError(
                f"[BootstrapGateEval] set_params が失敗しました（env indices: {failed_indices}）。"
                f"前の武器の条件で eval が走るのを防ぐため、この武器の eval をスキップします。"
            )

        ep_results, _, _ = run_survivors_eval_episodes(
            model=self.model,
            env=self._eval_env,
            n_eval_episodes=self._n_probe_episodes,
            deterministic=True,
            frame_skip=self._frame_skip,
            alive_reward=self._alive_reward,
        )

        summary = summarize_eval_results(
            short_episode_steps=self._short_episode_steps,
            cell_spec=cell_spec,
            episode_results=ep_results,
            deterministic=True,
            model_path="",
            global_timestep=self.num_timesteps,
        )
        p10 = summary["active_score_p10"]

        self._weapon_bootstrap_module.set_deterministic_result(
            weapon_id=weapon_id,
            task_kind=status,  # どの task_kind の eval かを記録
            enemy_phase_idx=self._enemy_phase_idx,
            p10=p10,
            episode_length_p10=summary["episode_length_p10"],
            short_episode_rate=summary["short_episode_rate"],
            num_timesteps=self.num_timesteps,
        )

        self._log_wandb(weapon_key, status, summary, len(ep_results))

        threshold = (
            self._solo_target_p10 if status == "solo_bootstrap"
            else self._integration_target_p10 if status == "integration"
            else -1
        )
        threshold_str = f"{threshold:.1f}" if threshold >= 0 else "(regression-based)"
        print(
            f"[BootstrapGateEval] weapon={weapon_key}[{status}], det_p10={p10:.1f}, "
            f"ep_len_p10={summary['episode_length_p10']:.0f}, "
            f"threshold={threshold_str}"
        )

    def _log_wandb(
        self,
        weapon_key: str,
        status: str,
        summary: dict,
        n_episodes: int,
    ) -> None:
        """W&B へ eval 結果をログする。"""
        if not (self._wandb_logger and self._wandb_logger.enabled):
            return

        prefix = f"bootstrap_gate/{weapon_key}/{status}"

        # score_passed は status 別の閾値で判定
        # solo_bootstrap → solo 閾値、integration → integration 閾値、maintenance → -1（N/A）
        if status == "solo_bootstrap":
            score_passed = float(summary["active_score_p10"] >= self._solo_target_p10)
        elif status == "integration":
            score_passed = float(summary["active_score_p10"] >= self._integration_target_p10)
        else:
            # maintenance はスコアベース。-1 は N/A を意味する
            score_passed = -1.0

        self._wandb_logger.log(
            {
                f"{prefix}/deterministic_p10": summary["active_score_p10"],
                f"{prefix}/deterministic_ep_len_p10": summary["episode_length_p10"],
                f"{prefix}/deterministic_short_rate": summary["short_episode_rate"],
                f"{prefix}/score_passed": score_passed,
                f"{prefix}/n_episodes": n_episodes,
            },
            step=self.num_timesteps,
        )

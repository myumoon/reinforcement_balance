"""BootstrapGateEvalCallback: Bootstrap Lane の deterministic eval を定期実行するコールバック。

SurvivorsEvalCallback とは独立した BaseCallback サブクラス。
probe_freq ごとに全対象武器（solo_bootstrap / integration / maintenance）の
deterministic eval を実行し、WeaponBootstrapStateModule.set_deterministic_result() に
結果を注入する。

async_eval=True（デフォルト）の場合、eval はバックグラウンドスレッドで実行され、
訓練ループをブロックしない。結果は次の _on_step() 周期でメインスレッドに取り込まれ、
WeaponBootstrapStateModule へ注入される（module 更新はメインスレッドで実行される）。
"""
from __future__ import annotations

import copy
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

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


@dataclass(frozen=True)
class BootstrapGateEvalTarget:
    """1 武器分の eval を実行するために必要な情報（プラン）。

    メインスレッドで構築し、非同期 worker はこれと policy snapshot だけを使って eval する。
    module への参照は持たず、スレッド間で共有可能な immutable なプランに保つ。
    """

    weapon_key: str
    weapon_id: int
    status: str
    cell_spec: str
    ue_params: dict
    snapshot_step: int
    snapshot_stage_key: str | None
    build_policy: str | None = None


@dataclass
class BootstrapGateEvalResult:
    """1 武器分の eval 結果。error があれば apply 時に破棄される。"""

    target: BootstrapGateEvalTarget
    summary: dict | None
    n_episodes: int
    error: Exception | None = None


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
        short_episode_steps:    short episode 判定のステップ数。
        async_eval:             True の場合 eval をバックグラウンドスレッドで実行する。
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
        async_eval: bool = True,
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
        self._async_eval = async_eval
        self._wandb_logger = wandb_logger
        self._last_probe_step: int = 0
        self._id_to_key: dict[int, str] = {e.weapon_id: e.key for e in weapon_unlock_order}
        self._id_to_entry: dict[int, WeaponEntry] = {e.weapon_id: e for e in weapon_unlock_order}
        # ゲート閾値（W&B ログ用に保持）- status 別の閾値
        self._solo_target_p10: float = weapon_bootstrap_module._solo_bootstrap_target_p10
        self._integration_target_p10: float = weapon_bootstrap_module._integration_target_p10

        # 非同期 eval 用の内部状態
        self._probe_thread: threading.Thread | None = None
        self._probe_result_queue: queue.Queue | None = None
        self._probe_started_at: float | None = None
        self._probe_snapshot_step: int | None = None

    # ------------------------------------------------------------------
    # BaseCallback フック
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        # 走行中の非同期 eval があれば定期的に取り込む
        if self.n_calls % 64 == 0:
            self._try_process_pending_probe_result()

        if self.num_timesteps - self._last_probe_step >= self._probe_freq:
            self._last_probe_step = self.num_timesteps
            if self._async_eval:
                self._start_probe_async_or_skip()
            else:
                self._run_probe()
        return True

    def _on_training_end(self) -> None:
        """訓練終了時、走行中の非同期 eval があれば join して結果を取り込む。"""
        if self._probe_thread is not None:
            self._probe_thread.join()
        self._try_process_pending_probe_result()

    # ------------------------------------------------------------------
    # VecNormalize 同期
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # ターゲット収集 / 構築 (plan)
    # ------------------------------------------------------------------

    def _collect_targets(self) -> list[tuple[str, int, str]]:
        """対象ステータスの武器を (weapon_key, weapon_id, status) の raw タプルで列挙する。"""
        targets: list[tuple[str, int, str]] = []
        for status in self._EVAL_STATUSES:
            for state in self._weapon_bootstrap_module.get_weapons_by_status(status):
                weapon_key = self._id_to_key.get(state.weapon_id)
                if weapon_key is None:
                    print(f"[BootstrapGateEval][WARN] unknown weapon_id={state.weapon_id}, skip")
                    continue
                targets.append((weapon_key, state.weapon_id, status))
        return targets

    def _build_target(
        self,
        weapon_key: str,
        weapon_id: int,
        status: str,
        snapshot_step: int,
    ) -> BootstrapGateEvalTarget:
        """1 武器分の eval プランを構築する（メインスレッドで実行）。

        cell_spec の task_kind は武器の現在のステータスに合わせる。
        integration の build_policy は weapon_bootstrap_module.get_eval_build_policy() で
        task_cell_sampler と同じロジックで決定する。

        BootstrapEvalCell が含まれないため、entry.unlock_stage_key から取得する。
        self._id_to_entry は __init__ 時に構築済みなので KeyError は起きない。
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

        return BootstrapGateEvalTarget(
            weapon_key=weapon_key,
            weapon_id=weapon_id,
            status=status,
            cell_spec=cell_spec,
            ue_params=ue_params,
            snapshot_step=snapshot_step,
            snapshot_stage_key=current_stage_key,
            build_policy=build_policy_override,
        )

    # ------------------------------------------------------------------
    # eval 実行 (worker / sync 共通)
    # ------------------------------------------------------------------

    def _eval_target(
        self,
        target: BootstrapGateEvalTarget,
        model_snapshot,
        eval_env=None,
    ) -> BootstrapGateEvalResult:
        """1 武器分の deterministic eval を実行する。

        set_params 失敗 / eval 例外はともに error フィールドに入れて返す（raise しない）。
        これにより、非同期 worker が 1 武器の失敗で全体クラッシュしないようにする。
        """
        env = eval_env if eval_env is not None else self._eval_env
        try:
            # set_params の戻り値をリストで返すため、全 env で成功したか確認する
            set_params_results = env.env_method("set_params", **target.ue_params)
            if not all(set_params_results):
                failed_indices = [i for i, ok in enumerate(set_params_results) if not ok]
                raise RuntimeError(
                    f"[BootstrapGateEval] set_params が失敗しました（env indices: {failed_indices}）。"
                    f"前の武器の条件で eval が走るのを防ぐため、この武器の eval をスキップします。"
                )

            ep_results, _, _ = run_survivors_eval_episodes(
                model=model_snapshot,
                env=env,
                n_eval_episodes=self._n_probe_episodes,
                deterministic=True,
                frame_skip=self._frame_skip,
                alive_reward=self._alive_reward,
            )

            summary = summarize_eval_results(
                short_episode_steps=self._short_episode_steps,
                cell_spec=target.cell_spec,
                episode_results=ep_results,
                deterministic=True,
                model_path="",
                global_timestep=target.snapshot_step,
            )
            return BootstrapGateEvalResult(
                target=target,
                summary=summary,
                n_episodes=len(ep_results),
            )
        except Exception as e:  # noqa: BLE001 - target 単位で握って worker を継続させる
            return BootstrapGateEvalResult(
                target=target,
                summary=None,
                n_episodes=0,
                error=e,
            )

    # ------------------------------------------------------------------
    # 結果注入 (apply, メインスレッド専用)
    # ------------------------------------------------------------------

    def _apply_result(self, result: BootstrapGateEvalResult, log_step: int) -> None:
        """eval 結果を WeaponBootstrapStateModule へ注入する（メインスレッドで実行）。

        非同期 eval では snapshot 撮影から apply までの間に武器の状態が変化しうるため、
        stale guard で古い結果を破棄する。
        """
        target = result.target
        if result.error is not None:
            print(
                f"[BootstrapGateEval][WARN] weapon={target.weapon_key}({target.weapon_id})"
                f"[{target.status}] failed: {result.error}"
            )
            return

        # stale guard: 現在の状態が eval 時点と変わっていたら破棄する
        state = self._weapon_bootstrap_module._states.get(target.weapon_id)
        if state is None:
            print(
                f"[BootstrapGateEval][WARN] weapon={target.weapon_key}: current state なし、"
                f"stale result を破棄"
            )
            return
        if state.status != target.status:
            print(
                f"[BootstrapGateEval][WARN] weapon={target.weapon_key}: status 変化 "
                f"({target.status} -> {state.status})、stale result を破棄"
            )
            return
        if self._stage_key_provider is not None:
            current_stage_key = self._stage_key_provider()
            if current_stage_key != target.snapshot_stage_key:
                print(
                    f"[BootstrapGateEval][WARN] weapon={target.weapon_key}: stage_key 変化 "
                    f"({target.snapshot_stage_key} -> {current_stage_key})、stale result を破棄"
                )
                return

        # build_policy stale guard:
        # integration eval の build_policy は garlic の bootstrap 状態にも依存するため、
        # snapshot 撮影時と apply 時で build_policy が変化していたら（例: garlic が
        # maintenance に到達して target_only → target_plus_anchor_if_unlocked に切り替わった）
        # 古い条件で走った deterministic 結果を注入せず破棄する。
        if target.build_policy is not None:
            current_build_policy = self._weapon_bootstrap_module.get_eval_build_policy(
                target.status,
                current_stage_key=target.snapshot_stage_key,
            )
            if current_build_policy != target.build_policy:
                print(
                    f"[BootstrapGateEval][WARN] weapon={target.weapon_key}: build_policy 変化 "
                    f"({target.build_policy} -> {current_build_policy})、stale result を破棄"
                )
                return

        summary = result.summary
        p10 = summary["active_score_p10"]

        self._weapon_bootstrap_module.set_deterministic_result(
            weapon_id=target.weapon_id,
            task_kind=target.status,  # どの task_kind の eval かを記録
            enemy_phase_idx=self._enemy_phase_idx,
            p10=p10,
            episode_length_p10=summary["episode_length_p10"],
            short_episode_rate=summary["short_episode_rate"],
            num_timesteps=target.snapshot_step,
        )

        self._log_wandb(
            target.weapon_key, target.status, summary, result.n_episodes, step=log_step
        )

        threshold = (
            self._solo_target_p10 if target.status == "solo_bootstrap"
            else self._integration_target_p10 if target.status == "integration"
            else -1
        )
        threshold_str = f"{threshold:.1f}" if threshold >= 0 else "(regression-based)"
        print(
            f"[BootstrapGateEval] weapon={target.weapon_key}[{target.status}], det_p10={p10:.1f}, "
            f"ep_len_p10={summary['episode_length_p10']:.0f}, "
            f"threshold={threshold_str}"
        )

    # ------------------------------------------------------------------
    # 同期 probe（async_eval=False）
    # ------------------------------------------------------------------

    def _run_probe(self) -> None:
        """全対象武器に deterministic eval を同期実行する。"""
        raw_targets = self._collect_targets()
        if not raw_targets:
            return

        self._sync_vecnormalize()

        was_training = getattr(self._eval_env, "training", None)
        try:
            if was_training is not None:
                self._eval_env.training = False

            for weapon_key, weapon_id, status in raw_targets:
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
        """単一武器の deterministic eval を実行して結果を注入する（thin wrapper）。"""
        target = self._build_target(
            weapon_key, weapon_id, status, snapshot_step=self.num_timesteps
        )
        result = self._eval_target(target, model_snapshot=self.model)
        if result.error is not None:
            raise result.error
        self._apply_result(result, log_step=self.num_timesteps)

    # ------------------------------------------------------------------
    # 非同期 probe（async_eval=True）
    # ------------------------------------------------------------------

    def _start_probe_async_or_skip(self) -> None:
        """非同期 eval worker を起動する。既に走行中なら skip する。

        起動前に、前回の完了済み thread の結果を必ず取り込む。これを怠ると
        result_queue / thread を上書きしてしまい、前回の deterministic result が
        失われる（_on_step の 64 周期ポーリングでまだ取り込まれていない場合に発生しうる）。
        """
        # 前回の完了済み結果を先に処理（queue/thread 上書きによる result 消失を防ぐ）
        self._try_process_pending_probe_result()

        if self._probe_thread is not None and self._probe_thread.is_alive():
            self._log_wandb_scalar("bootstrap_gate/async_skipped_in_flight", 1)
            print("[BootstrapGateEval] 前回の非同期 eval が走行中のため今周期は skip")
            return

        raw_targets = self._collect_targets()
        if not raw_targets:
            return

        # snapshot をメインスレッドで撮る
        self._sync_vecnormalize()
        snapshot_step = self.num_timesteps
        policy_snapshot = copy.deepcopy(self.model.policy)

        built_targets: list[BootstrapGateEvalTarget] = []
        for weapon_key, weapon_id, status in raw_targets:
            try:
                built_targets.append(
                    self._build_target(weapon_key, weapon_id, status, snapshot_step=snapshot_step)
                )
            except Exception as e:  # noqa: BLE001 - target 構築失敗は skip して他を続行
                print(
                    f"[BootstrapGateEval][WARN] weapon={weapon_key}({weapon_id})"
                    f"[{status}] target 構築失敗: {e}"
                )

        if not built_targets:
            return

        result_queue: queue.Queue = queue.Queue()
        eval_env = self._eval_env

        # worker が使う model は policy snapshot をラップした軽量オブジェクト。
        # run_survivors_eval_episodes は model.predict を呼ぶので policy を持てば足りる。
        class _PolicyModel:
            def __init__(self, policy):
                self.policy = policy

            def predict(self, *args, **kwargs):
                return self.policy.predict(*args, **kwargs)

        model_snapshot = _PolicyModel(policy_snapshot)

        def _worker():
            results: list[BootstrapGateEvalResult] = []
            was_training = getattr(eval_env, "training", None)
            try:
                if was_training is not None:
                    eval_env.training = False
                for target in built_targets:
                    results.append(
                        self._eval_target(target, model_snapshot=model_snapshot, eval_env=eval_env)
                    )
            except Exception as e:  # noqa: BLE001 - worker crash でも result を必ず put する
                print(f"[BootstrapGateEval][WARN] 非同期 eval worker が例外で終了: {e}")
            finally:
                if was_training is not None:
                    eval_env.training = was_training
                # crash 時も含め必ず put（受信側が queue 空で待たないように）
                result_queue.put(results)

        thread = threading.Thread(target=_worker, name="bootstrap-gate-eval", daemon=True)
        self._probe_result_queue = result_queue
        self._probe_thread = thread
        self._probe_started_at = time.time()
        self._probe_snapshot_step = snapshot_step
        thread.start()
        self._log_wandb_scalar("bootstrap_gate/async_running", 1)

    def _try_process_pending_probe_result(self) -> None:
        """走行中の非同期 eval が完了していれば結果を取り込む。"""
        if self._probe_thread is None:
            return
        if self._probe_thread.is_alive():
            return

        results: list[BootstrapGateEvalResult] = []
        if self._probe_result_queue is not None:
            try:
                results = self._probe_result_queue.get_nowait()
            except queue.Empty:
                results = []

        log_step = self.num_timesteps
        for result in results:
            self._apply_result(result, log_step=log_step)

        # メトリクス（duration / age / errors）
        if self._probe_started_at is not None:
            duration = time.time() - self._probe_started_at
            self._log_wandb_scalar("bootstrap_gate/async_duration_sec", duration)
        if self._probe_snapshot_step is not None:
            age = self.num_timesteps - self._probe_snapshot_step
            self._log_wandb_scalar("bootstrap_gate/async_result_age_steps", age)
        n_errors = sum(1 for r in results if r.error is not None)
        self._log_wandb_scalar("bootstrap_gate/async_errors", n_errors)

        # 内部状態をリセット
        self._probe_thread = None
        self._probe_result_queue = None
        self._probe_started_at = None
        self._probe_snapshot_step = None
        self._log_wandb_scalar("bootstrap_gate/async_running", 0)

    # ------------------------------------------------------------------
    # W&B ロギング
    # ------------------------------------------------------------------

    def _log_wandb_scalar(self, key: str, value) -> None:
        """単一スカラーを W&B へログする。logger が無効ならなにもしない。"""
        if not (self._wandb_logger and self._wandb_logger.enabled):
            return
        self._wandb_logger.log({key: value}, step=self.num_timesteps)

    def _log_wandb(
        self,
        weapon_key: str,
        status: str,
        summary: dict,
        n_episodes: int,
        step: int | None = None,
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
            step=step if step is not None else self.num_timesteps,
        )

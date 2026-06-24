"""BaseEvalCallback - ゲーム非依存の評価コールバック基底クラス。"""

from __future__ import annotations

import copy
import queue
import threading
from abc import abstractmethod

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

class BaseEvalCallback(BaseCallback):
    """ゲーム非依存の評価ロジック基底クラス。

    共通処理:
    - eval 頻度管理
    - VecNormalize の obs_rms / ret_rms を eval_env へコピー
    - eval_env がある場合の async eval（background thread）

    ゲーム固有処理（サブクラスで実装）:
    - _sync_before_eval()
    - _start_eval_async()
    - _process_eval_result(result)
    - _run_eval_and_log()  （eval_env=None の sync 互換パス用）
    """

    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int,
                 wandb_logger=None, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = max(1, eval_freq)
        self.n_eval_episodes = max(1, n_eval_episodes)
        self._wandb_logger = wandb_logger
        self._last_eval_step: int = 0
        self._eval_thread: threading.Thread | None = None
        self._eval_result_queue: queue.Queue | None = None

    def _on_training_start(self) -> None:
        """訓練開始時に eval/running=0 を初期ログし、W&B チャートの左端を 0 に固定する。"""
        if self._wandb_logger and self._wandb_logger.enabled:
            self._wandb_logger.log({"eval/running": 0}, step=self.num_timesteps)

    def _on_step(self) -> bool:
        # eval thread が存在する間だけ 64 step ごとにポーリングし、
        # eval 終了から eval/running=0 ログまでの遅れを最大 64 step に抑える。
        if self._eval_thread is not None and self.n_calls % 64 == 0:
            self._try_process_pending_eval_result()
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
        cur = self.eval_env
        while cur is not None:
            if isinstance(cur, VecNormalize):
                eval_vecnorm = cur
                break
            cur = getattr(cur, "venv", None)
        if train_vecnorm is not None and eval_vecnorm is not None:
            eval_vecnorm.obs_rms = copy.deepcopy(train_vecnorm.obs_rms)
            eval_vecnorm.ret_rms = copy.deepcopy(train_vecnorm.ret_rms)

    def _sync_before_eval(self) -> None:
        """評価前の同期処理。サブクラスでオーバーライドして shaping/params 同期等を追加する。"""
        self._sync_vecnormalize()

    @abstractmethod
    def _run_eval_and_log(self) -> None:
        """評価を実行してログを記録する。eval_env=None の sync 互換パスで使用する。サブクラスで実装する。"""

    def _start_eval_async(self) -> None:
        """background thread で eval を起動する。サブクラスで実装する。"""
        raise NotImplementedError

    def _process_eval_result(self, result) -> None:
        """_start_eval_async が queue に積んだ result を処理する。サブクラスで実装する。"""
        raise NotImplementedError

    def _try_process_pending_eval_result(self) -> None:
        """完了済みの eval thread から結果を取り出して処理する（non-blocking）。"""
        if self._eval_thread is None:
            return
        if self._eval_thread.is_alive():
            return
        # thread が終了している
        if self._eval_result_queue is None or self._eval_result_queue.empty():
            # 異常系: thread が結果を put せずに死亡（daemon 強制終了など）
            print("[Eval] WARN: eval thread が結果を返さずに終了しました")
            if self._wandb_logger and self._wandb_logger.enabled:
                self._wandb_logger.log({"eval/running": 0, "eval/aborted": 1}, step=self.num_timesteps)
            self._eval_thread = None
            self._eval_result_queue = None
            return
        result = self._eval_result_queue.get_nowait()
        self._eval_thread = None
        self._eval_result_queue = None
        # eval/running=0 を現在 step でログ
        if self._wandb_logger and self._wandb_logger.enabled:
            self._wandb_logger.log({"eval/running": 0}, step=self.num_timesteps)
        self._process_eval_result(result)

    def _on_training_end(self) -> None:
        """SB3 の正常終了時に呼ばれる（例外終了では呼ばれない場合あり）。"""
        if self._eval_thread is not None and self._eval_thread.is_alive():
            self._eval_thread.join()
        self._try_process_pending_eval_result()
        self._eval_thread = None
        self._eval_result_queue = None

    def _on_rollout_end(self) -> None:
        # eval_env=None は sync パス（旧来の動作を維持）
        if self.eval_env is None:
            if self.num_timesteps - self._last_eval_step >= self.eval_freq:
                self._last_eval_step = self.num_timesteps
                self._sync_before_eval()
                self._run_eval_and_log()
            return

        # 1. 前の eval が完了していれば結果を処理（non-blocking）
        self._try_process_pending_eval_result()

        # 2. 新しい eval を起動するか判定
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return

        # 3. 前の thread が残っていれば完了を待つ（back-pressure）
        if self._eval_thread is not None and self._eval_thread.is_alive():
            self._eval_thread.join()
            self._try_process_pending_eval_result()

        self._last_eval_step = self.num_timesteps
        # 4. sync / thread 起動。失敗時は eval/running=1 を出さない（W&B に 1 が残らないよう）
        try:
            self._sync_before_eval()
            self._start_eval_async()
            # スレッド起動成功後に eval/running=1 をログ
            if self._wandb_logger and self._wandb_logger.enabled:
                self._wandb_logger.log({"eval/running": 1}, step=self.num_timesteps)
        except Exception:
            if self._wandb_logger and self._wandb_logger.enabled:
                self._wandb_logger.log({"eval/running": 0, "eval/error": 1}, step=self.num_timesteps)
            raise

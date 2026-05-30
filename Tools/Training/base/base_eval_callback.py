"""BaseEvalCallback - ゲーム非依存の評価コールバック基底クラス。"""

from __future__ import annotations

import copy
from abc import abstractmethod

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

class BaseEvalCallback(BaseCallback):
    """ゲーム非依存の評価ロジック基底クラス。

    共通処理:
    - eval 頻度管理
    - VecNormalize の obs_rms / ret_rms を eval_env へコピー

    ゲーム固有処理（サブクラスで実装）:
    - _sync_before_eval()
    """

    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int,
                 wandb_logger=None, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = max(1, eval_freq)
        self.n_eval_episodes = max(1, n_eval_episodes)
        self._wandb_logger = wandb_logger
        self._last_eval_step: int = 0

    def _on_step(self) -> bool:
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
        """評価を実行してログを記録する。サブクラスで実装する。"""

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return
        self._last_eval_step = self.num_timesteps
        self._sync_before_eval()
        self._run_eval_and_log()

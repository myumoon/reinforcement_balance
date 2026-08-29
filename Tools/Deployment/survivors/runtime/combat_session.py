"""RecurrentPPO combat session: GRU 隠れ状態を episode 単位で管理する。

perception snapshot から観測 tensor を受け取り、9 方向 action index を返す。
OS input には触れず、decision の種別だけを caller へ返す。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch as th

from .artifact_bundle import _CombatGRU


class StaleSnapshotError(ValueError):
    """観測が stale または無効で inference を進められない場合の例外。

    invalid/unknown snapshot に対して no_op/stop を選ぶよう caller に通知する。
    """


class CombatSession:
    """エピソード単位の GRU 隠れ状態を保持する combat 推論セッション。

    VecNormalize は含まない — student は正規化済みスコアで訓練済みと仮定する。
    エピソード間で reset_episode() を必ず呼ぶ。
    """

    def __init__(self, model: _CombatGRU) -> None:
        """model と hidden state を初期化する。

        eval mode と CPU-only を確認し、推論以外の用途を排除する。
        """
        if not isinstance(model, _CombatGRU):
            raise ValueError("model must be _CombatGRU")
        self._model = model
        self._model.eval()
        self._hidden: th.Tensor = th.zeros(1, model.hidden_dim, dtype=th.float32)
        self._episode_started = False

    @property
    def hidden_dim(self) -> int:
        """GRU 隠れ状態の次元を返す。"""
        return self._model.hidden_dim

    @property
    def observation_dim(self) -> int:
        """期待する観測 vector の次元を返す。"""
        return self._model.observation_dim

    @property
    def action_dim(self) -> int:
        """action 空間の大きさを返す。"""
        return self._model.action_dim

    def reset_episode(self) -> None:
        """エピソード境界で GRU 隠れ状態をゼロにリセットする。

        death / run 終了 / 新 session 開始時に必ず呼ぶ。
        呼ばなければ前 episode の記憶が次 episode に混入する。
        """
        self._hidden = th.zeros(1, self._model.hidden_dim, dtype=th.float32)
        self._episode_started = False

    def decide(self, obs_vector: np.ndarray, *, episode_start: bool = False) -> int:
        """観測 vector から action index (0-8) を返す。

        episode_start=True の場合は隠れ状態をリセットしてから推論する。
        観測が finite でなければ StaleSnapshotError を送出する。
        """
        if episode_start:
            self.reset_episode()
        self._episode_started = True

        obs = np.asarray(obs_vector, dtype=np.float32)
        if obs.ndim != 1 or obs.shape[0] != self._model.observation_dim:
            raise StaleSnapshotError(
                f"obs shape {obs.shape} != ({self._model.observation_dim},)"
            )
        if not np.all(np.isfinite(obs)):
            raise StaleSnapshotError("combat observation contains non-finite values")

        obs_tensor = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
        with th.no_grad():
            logit, new_hidden = self._model.step(obs_tensor, self._hidden)
        if not bool(th.isfinite(logit).all()):
            raise StaleSnapshotError("combat model produced non-finite logits")

        self._hidden = new_hidden.detach()
        action_index = int(logit.argmax(dim=-1).item())
        if not (0 <= action_index < self._model.action_dim):
            raise StaleSnapshotError("combat model returned out-of-range action index")
        return action_index

    def hidden_state_copy(self) -> np.ndarray:
        """現在の GRU 隠れ状態のコピーを numpy 配列で返す。

        shadow / telemetry 目的で使う。状態本体は変更しない。
        """
        return self._hidden.detach().numpy().copy()

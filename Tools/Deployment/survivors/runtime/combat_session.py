"""検証済み combat actor の recurrent state を episode 単位で管理する。

やさしい説明: 03-05 package から復元済みの GRU actor をそのまま一段ずつ進め、
OS input には触れず共有 action index と確信度だけを返します。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch as th

from reinbalance_survivors_contracts.target_action import ActionSemantics

from .artifact_bundle import CombatPolicy


class StaleSnapshotError(ValueError):
    """combat 推論へ渡せない observation または model 出力を表す。

    やさしい説明: caller はこの例外を movement ではなく安全な decision へ変換します。
    """


@dataclass(frozen=True)
class CombatDecision:
    """combat actor が選んだ action index と確信度を保持する。

    やさしい説明: 実際の key 入力へ変換せず、共有 action map 上の番号だけを返します。
    """

    action_index: int
    confidence: float


class CombatSession:
    """episode ごとの recurrent actor state と episode_start を保持する。

    やさしい説明: state と開始 flag を常に一緒に更新し、別 run の記憶を混ぜません。
    """

    def __init__(
        self,
        combat_policy: CombatPolicy,
        *,
        action_semantics: Sequence[str] | None = None,
    ) -> None:
        """検証済み policy と一意な action semantics を記録する。

        やさしい説明: package loader を迂回した model や曖昧な action map は起動時に拒否します。
        """
        if not isinstance(combat_policy, CombatPolicy):
            raise ValueError("combat_policy must be CombatPolicy")
        semantics = tuple(
            action_semantics
            if action_semantics is not None
            else ActionSemantics.default_v1().actions
        )
        if len(semantics) != combat_policy.action_dim or len(set(semantics)) != len(semantics):
            raise ValueError("action_semantics must map every action index uniquely")
        if not all(isinstance(value, str) and value for value in semantics):
            raise ValueError("action_semantics must contain non-empty strings")
        self._policy = combat_policy
        self._action_semantics = semantics
        self._recurrent_state: th.Tensor | None = None
        self._episode_start = True

    @property
    def observation_dim(self) -> int:
        """combat observation の固定次元を返す。

        やさしい説明: deploy schema の三平面を連結した長さと一致します。
        """
        return self._policy.observation_dim

    @property
    def action_dim(self) -> int:
        """共有 action map の要素数を返す。

        やさしい説明: 現契約では 8 方向と idle の合計 9 です。
        """
        return self._policy.action_dim

    @property
    def recurrent_state_shape(self) -> tuple[int, int, int]:
        """外部検査用 state shape `[1, 1, hidden]` を返す。

        やさしい説明: layer・batch・hidden の軸順を固定します。
        """
        return (1, 1, self._policy.hidden_dim)

    @property
    def episode_start_pending(self) -> bool:
        """次の推論が episode 先頭かを返す。

        やさしい説明: state が空のときだけ True になり、両者は常に同期します。
        """
        return self._episode_start

    def semantic_for(self, action_index: int) -> str:
        """action index に対応する共有 semantic を返す。

        やさしい説明: 範囲外番号を Python の負 index として誤解せず拒否します。
        """
        if type(action_index) is not int or not 0 <= action_index < self.action_dim:
            raise ValueError("action_index out of range")
        return self._action_semantics[action_index]

    def reset_episode(self, reason: str = "episode_reset") -> None:
        """recurrent state を破棄し episode_start を同時に立てる。

        やさしい説明: death・result・unknown gap・new run の全経路がこの一か所を呼びます。
        """
        if not isinstance(reason, str) or not reason:
            raise ValueError("reset reason must be a non-empty string")
        self._recurrent_state = None
        self._episode_start = True

    def restore_recurrent_state(
        self, state: np.ndarray | None, *, episode_start: bool
    ) -> None:
        """timeout 前の recurrent state と episode flag を一対で復元する。

        やさしい説明: 破棄した推論の記憶だけが残らないよう、runtime の rollback を原子的にします。
        """
        if (state is None) != episode_start:
            raise ValueError("state and episode_start must be restored together")
        if state is None:
            self._recurrent_state = None
            self._episode_start = True
            return
        restored = np.asarray(state)
        if (
            restored.dtype != np.float32
            or restored.shape != self.recurrent_state_shape
            or not np.all(np.isfinite(restored))
        ):
            raise ValueError("invalid recurrent state checkpoint")
        self._recurrent_state = th.from_numpy(restored[0].copy())
        self._episode_start = False

    def decide(self, obs_vector: np.ndarray, *, episode_start: bool = False) -> CombatDecision:
        """一つの observation から決定的な action を返す。

        やさしい説明: 成功した推論だけが state と flag を更新し、失敗時 state は保存しません。
        """
        if episode_start:
            self.reset_episode("explicit_episode_start")
        obs = np.asarray(obs_vector)
        if obs.dtype != np.float32 or obs.ndim != 1 or obs.shape != (self.observation_dim,):
            raise StaleSnapshotError(
                f"combat observation must be float32 shape ({self.observation_dim},)"
            )
        if not np.all(np.isfinite(obs)):
            raise StaleSnapshotError("combat observation contains non-finite values")

        hidden = (
            self._policy.initial_hidden_state()
            if self._recurrent_state is None
            else self._recurrent_state
        )
        try:
            with th.inference_mode():
                logits, _, next_state = self._policy.model.step(
                    th.from_numpy(obs).reshape(1, -1), hidden
                )
                probabilities = th.softmax(logits, dim=1)
        except Exception as exc:
            raise StaleSnapshotError(f"combat recurrent inference failed: {exc}") from exc

        if (
            logits.shape != (1, self.action_dim)
            or next_state.shape != (1, self._policy.hidden_dim)
            or not bool(th.isfinite(logits).all())
            or not bool(th.isfinite(next_state).all())
        ):
            raise StaleSnapshotError("combat policy returned invalid recurrent output")
        action_index = int(th.argmax(logits, dim=1).item())
        confidence = float(probabilities[0, action_index].item())
        if not 0 <= action_index < self.action_dim or not np.isfinite(confidence):
            raise StaleSnapshotError("combat policy returned invalid action")

        self._recurrent_state = next_state.detach().clone()
        self._episode_start = False
        return CombatDecision(action_index=action_index, confidence=confidence)

    def recurrent_state_copy(self) -> np.ndarray | None:
        """現在の state を `[1, 1, hidden]` numpy copy で返す。

        やさしい説明: telemetry/test が内部 tensor を書き換えられないようコピーします。
        """
        if self._recurrent_state is None:
            return None
        return self._recurrent_state.detach().cpu().numpy()[None, ...].copy()

    def lstm_state_copy(self) -> np.ndarray | None:
        """旧 inspection 名から同じ recurrent state copy を返す。

        やさしい説明: runtime 利用側の段階移行中も state の意味は一つに保ちます。
        """
        return self.recurrent_state_copy()

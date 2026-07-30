"""PPO/RecurrentPPO の critic で Survivors choice candidate を評価する。

raw observation を保存済み VecNormalize 統計で正規化し、全 candidate を同じ critic
hidden_in から独立に一回の batch として score する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from games.survivors.recurrent_policy_session import (
    CriticContext,
    RecurrentPolicySession,
)
from games.survivors.value_source_loader import (
    LoadedValueSource,
    load_value_source,
)

TIE_EPSILON = 1e-5


@dataclass(frozen=True, slots=True)
class CandidateValue:
    """一候補の normalized / unscaled return value を保持する。

    ordering は normalized value だけで決め、unscaled 値は reward RMS による診断へ限定する。
    """

    choice_id: str
    value_normalized_return: float
    value_unscaled_return: float


class ValueScorer:
    """検証済み source policy の value head を公開する。

    model loading・normalization・context validation を score 前に一つの境界で強制する。
    """

    def __init__(self, source: LoadedValueSource) -> None:
        """検証済み LoadedValueSource を保持する。

        path や descriptor を再解釈せず、全 score が同じ model/VecNormalize instance を使う。
        """
        self.source = source

    @classmethod
    def load(
        cls,
        manifest_path: Path,
        device: str = "cpu",
    ) -> "ValueScorer":
        """immutable source manifest から scorer を構築する。

        artifact と policy schema の mismatch は loader 例外として ranking 生成前に伝播する。
        """
        return cls(load_value_source(manifest_path, device=device))

    def new_session(self) -> RecurrentPolicySession:
        """同じ policy に結ばれた独立 recurrent session を作る。

        candidate preview は session state を触らず、movement/selected commit だけをこの型へ委譲する。
        """
        return RecurrentPolicySession(self.source)

    def _resolve_burn_in(self, context: CriticContext) -> CriticContext:
        """ordered movement sequence から pi/vf state を再構成する。

        pending/base obs は sequence 外の hash binding に留め、burn-in 各 obs を exactly once 進める。
        """
        if context.context_mode != "burn_in":
            return context
        context.validate_integrity()
        raw_sequence = context.burn_in_raw_obs
        episode_starts = context.burn_in_episode_starts
        if (
            raw_sequence is None
            or episode_starts is None
            or raw_sequence.ndim != 2
            or raw_sequence.shape[1] != self.source.observation_dim
            or episode_starts.shape != (raw_sequence.shape[0],)
        ):
            raise ValueError("burn-in observation/episode-start sequence is invalid")
        session = self.new_session()
        for raw_obs, episode_start in zip(raw_sequence, episode_starts):
            session.advance_movement(
                raw_obs,
                episode_start=bool(episode_start),
            )
        pi, vf = session.states
        return context.with_updates(pi=pi, vf=vf)

    def _validate_raw_batch(self, raw_obs_batch: np.ndarray) -> np.ndarray:
        """candidate batch の shape・dtype・finite を検証する。

        scalar/sequence の暗黙 reshape をせず、raw 二次元 batch だけを normalization へ渡す。
        """
        try:
            batch = np.asarray(raw_obs_batch, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("raw observation batch must be numeric") from exc
        expected = self.source.observation_dim
        if batch.ndim != 2 or batch.shape[1] != expected or batch.shape[0] <= 0:
            raise ValueError(
                f"raw observation batch must have shape (N, {expected})"
            )
        if not np.all(np.isfinite(batch)):
            raise ValueError("raw observation batch must contain only finite values")
        return batch

    def score(
        self,
        raw_obs_batch: np.ndarray,
        context: CriticContext,
        *,
        choice_ids: Sequence[str] | None = None,
        allow_zero_state_smoke: bool = False,
    ) -> list[CandidateValue]:
        """candidate value を direct policy.predict_values と同じ経路で計算する。

        recurrent state は candidate 数だけ batch 軸へ複製し、hidden_out を次候補へ持ち回らない。
        """
        import torch as th

        batch = self._validate_raw_batch(raw_obs_batch)
        context = self._resolve_burn_in(context)
        context.validate_for_policy(
            dict(self.source.policy_state_schema),
            allow_zero_state_smoke=allow_zero_state_smoke,
        )
        if choice_ids is None:
            normalized_ids = [str(index) for index in range(len(batch))]
        else:
            normalized_ids = list(choice_ids)
            if (
                len(normalized_ids) != len(batch)
                or any(not isinstance(value, str) or not value for value in normalized_ids)
                or len(set(normalized_ids)) != len(normalized_ids)
            ):
                raise ValueError(
                    "choice_ids must be unique non-empty strings matching batch size"
                )
        normalized = self.source.normalize_raw_obs(batch)
        obs_tensor, _ = self.source.policy.obs_to_tensor(normalized)
        with th.no_grad():
            if self.source.algorithm == "RecurrentPPO":
                vf = context.vf
                repeated = tuple(
                    np.repeat(component, len(batch), axis=1)
                    for component in vf
                )
                state = tuple(
                    th.as_tensor(
                        component,
                        dtype=th.float32,
                        device=self.source.policy.device,
                    )
                    for component in repeated
                )
                starts = th.full(
                    (len(batch),),
                    float(context.episode_start),
                    dtype=th.float32,
                    device=self.source.policy.device,
                )
                values = self.source.policy.predict_values(
                    obs_tensor,
                    state,
                    starts,
                )
            else:
                values = self.source.policy.predict_values(obs_tensor)
        numbers = values.detach().cpu().numpy().reshape(-1)
        if len(numbers) != len(batch) or not np.all(np.isfinite(numbers)):
            raise ValueError("policy returned invalid candidate values")
        return [
            CandidateValue(
                choice_id=choice_id,
                value_normalized_return=float(value),
                value_unscaled_return=self.source.unscale_value(float(value)),
            )
            for choice_id, value in zip(normalized_ids, numbers)
        ]


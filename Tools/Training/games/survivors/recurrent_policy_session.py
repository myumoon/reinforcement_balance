"""Movement policy の actor/critic recurrent state を一つの session で管理する。

level-up pending・preview・retry では state を凍結し、selected post-choice observation
だけを同じ decision ID で exactly once commit する。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np

from reinbalance_survivors_contracts.canonical_json import canonical_hash


class RecurrentSessionError(ValueError):
    """Recurrent transition または context binding が不正な場合の例外。

    retry を新 timestep として進めるなどの回復不能な state divergence を formal session で止める。
    """


def observation_hash(raw_obs: np.ndarray) -> str:
    """raw observation の dtype・shape・値を canonical hash へ変換する。

    Python 独自 JSON/hashing を持たず、共有 canonical_json.py の内容 ID だけを利用する。
    """
    array = np.asarray(raw_obs, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise RecurrentSessionError("observation must contain only finite values")
    return canonical_hash(
        {
            "dtype": "float32",
            "shape": list(array.shape),
            "values": array.tolist(),
        }
    )


def _array_payload(value: np.ndarray | None) -> Any:
    """state ndarray を canonical JSON native 値へ変換する。

    欠落は null として残し、actor/critic の片側欠落も context seal が区別できるようにする。
    """
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    return {
        "dtype": "float32",
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def recurrent_state_hash(
    pi: tuple[np.ndarray, np.ndarray] | None,
    vf: tuple[np.ndarray, np.ndarray] | None,
) -> str:
    """RNNStates.pi/vf の四配列を一つの state hash に束縛する。

    actor state だけの hash を作らず、critic sibling の欠落・置換を必ず内容差として検出する。
    """
    return canonical_hash(
        {
            "pi_h": _array_payload(None if pi is None else pi[0]),
            "pi_c": _array_payload(None if pi is None else pi[1]),
            "vf_h": _array_payload(None if vf is None else vf[0]),
            "vf_c": _array_payload(None if vf is None else vf[1]),
        }
    )


def _context_payload(context: "CriticContext") -> dict[str, Any]:
    """CriticContext の全 identity field を canonical payload にする。

    state schema・phase・step・pending hash・burn-in sequence を同じ seal で一括検証する。
    """
    return {
        "environment_step": context.environment_step,
        "decision_id": context.decision_id,
        "phase": context.phase,
        "pending_obs_hash": context.pending_obs_hash,
        "episode_start": context.episode_start,
        "pi_h": _array_payload(context.pi_h),
        "pi_c": _array_payload(context.pi_c),
        "vf_h": _array_payload(context.vf_h),
        "vf_c": _array_payload(context.vf_c),
        "policy_state_schema_hash": context.policy_state_schema_hash,
        "context_mode": context.context_mode,
        "burn_in_raw_obs": _array_payload(context.burn_in_raw_obs),
        "burn_in_episode_starts": _array_payload(
            context.burn_in_episode_starts
        ),
    }


@dataclass(frozen=True, slots=True)
class CriticContext:
    """decision 時点の policy-bound critic context を immutable field で表す。

    ndarray の後編集も context_sha256 再計算で検出し、NPZ 内の state substitution を拒否する。
    """

    environment_step: int
    decision_id: str
    phase: Literal["pending_pre_commit"]
    pending_obs_hash: str
    episode_start: bool
    pi_h: np.ndarray | None
    pi_c: np.ndarray | None
    vf_h: np.ndarray | None
    vf_c: np.ndarray | None
    policy_state_schema_hash: str
    context_mode: Literal["captured", "burn_in", "zero_state_smoke"]
    burn_in_raw_obs: np.ndarray | None
    burn_in_episode_starts: np.ndarray | None
    context_sha256: str

    @property
    def pi(self) -> tuple[np.ndarray, np.ndarray] | None:
        """actor h/c を tuple として返す。

        片側だけ存在する破損 context は validation へ渡し、暗黙補完しない。
        """
        if self.pi_h is None or self.pi_c is None:
            return None
        return self.pi_h, self.pi_c

    @property
    def vf(self) -> tuple[np.ndarray, np.ndarray] | None:
        """critic h/c を tuple として返す。

        ``model.predict`` の actor state から推測せず、明示保存された vf sibling だけを使う。
        """
        if self.vf_h is None or self.vf_c is None:
            return None
        return self.vf_h, self.vf_c

    @classmethod
    def _create(
        cls,
        *,
        environment_step: int,
        decision_id: str,
        pending_obs_hash: str,
        episode_start: bool,
        pi: tuple[np.ndarray, np.ndarray] | None,
        vf: tuple[np.ndarray, np.ndarray] | None,
        policy_state_schema_hash: str,
        context_mode: Literal["captured", "burn_in", "zero_state_smoke"],
        burn_in_raw_obs: np.ndarray | None = None,
        burn_in_episode_starts: np.ndarray | None = None,
        phase: Literal["pending_pre_commit"] = "pending_pre_commit",
    ) -> "CriticContext":
        """全 context mode の field を copy して seal を計算する。

        captured・burn-in・zero smoke が別 hash 実装を持たないよう、唯一の constructor に集約する。
        """
        context = cls(
            environment_step=environment_step,
            decision_id=decision_id,
            phase=phase,
            pending_obs_hash=pending_obs_hash,
            episode_start=episode_start,
            pi_h=None if pi is None else np.asarray(pi[0], dtype=np.float32).copy(),
            pi_c=None if pi is None else np.asarray(pi[1], dtype=np.float32).copy(),
            vf_h=None if vf is None else np.asarray(vf[0], dtype=np.float32).copy(),
            vf_c=None if vf is None else np.asarray(vf[1], dtype=np.float32).copy(),
            policy_state_schema_hash=policy_state_schema_hash,
            context_mode=context_mode,
            burn_in_raw_obs=(
                None
                if burn_in_raw_obs is None
                else np.asarray(burn_in_raw_obs, dtype=np.float32).copy()
            ),
            burn_in_episode_starts=(
                None
                if burn_in_episode_starts is None
                else np.asarray(burn_in_episode_starts, dtype=np.bool_).copy()
            ),
            context_sha256="",
        )
        object.__setattr__(
            context,
            "context_sha256",
            canonical_hash(_context_payload(context)),
        )
        return context

    @classmethod
    def captured(
        cls,
        *,
        environment_step: int,
        decision_id: str,
        pending_obs: np.ndarray,
        episode_start: bool,
        pi: tuple[np.ndarray, np.ndarray] | None,
        vf: tuple[np.ndarray, np.ndarray] | None,
        policy_state_schema_hash: str,
    ) -> "CriticContext":
        """policy.forward が保持した RNNStates.pi/vf を captured context にする。

        pending observation は policy へ入力せず、その raw hash だけを decision binding に保存する。
        """
        return cls._create(
            environment_step=environment_step,
            decision_id=decision_id,
            pending_obs_hash=observation_hash(pending_obs),
            episode_start=episode_start,
            pi=pi,
            vf=vf,
            policy_state_schema_hash=policy_state_schema_hash,
            context_mode="captured",
        )

    @classmethod
    def burn_in(
        cls,
        *,
        environment_step: int,
        decision_id: str,
        pending_obs: np.ndarray,
        episode_start: bool,
        raw_obs_sequence: np.ndarray,
        episode_starts: np.ndarray,
        policy_state_schema_hash: str,
    ) -> "CriticContext":
        """ordered movement observation から state を再構成する context を作る。

        raw sequence と episode-start flag を seal し、pending/preview observation を burn-in に混ぜない。
        """
        return cls._create(
            environment_step=environment_step,
            decision_id=decision_id,
            pending_obs_hash=observation_hash(pending_obs),
            episode_start=episode_start,
            pi=None,
            vf=None,
            policy_state_schema_hash=policy_state_schema_hash,
            context_mode="burn_in",
            burn_in_raw_obs=raw_obs_sequence,
            burn_in_episode_starts=episode_starts,
        )

    @classmethod
    def zero_state(
        cls,
        *,
        environment_step: int,
        decision_id: str,
        pending_obs: np.ndarray,
        episode_start: bool,
        policy_state_schema: dict[str, Any],
    ) -> "CriticContext":
        """診断専用 zero state context を policy schema shape で構築する。

        formal captured と区別する mode を seal し、出力側が training label ready にできないようにする。
        """
        if policy_state_schema["algorithm"] == "RecurrentPPO":
            shape = (
                policy_state_schema["n_lstm_layers"],
                1,
                policy_state_schema["lstm_hidden_size"],
            )
            pi = (
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
            )
            vf = (pi[0].copy(), pi[1].copy())
        else:
            pi = vf = None
        return cls._create(
            environment_step=environment_step,
            decision_id=decision_id,
            pending_obs_hash=observation_hash(pending_obs),
            episode_start=episode_start,
            pi=pi,
            vf=vf,
            policy_state_schema_hash=policy_state_schema[
                "policy_state_schema_hash"
            ],
            context_mode="zero_state_smoke",
        )

    def with_updates(self, **updates: Any) -> "CriticContext":
        """テスト・検証用に指定 field を変えた新しい sealed context を返す。

        元配列は共有せず、変更後も同じ constructor と canonical hash 規則を通す。
        """
        known = {field.name for field in fields(self)}
        unknown = set(updates) - known - {"pi", "vf"}
        if unknown:
            raise RecurrentSessionError(
                f"unknown context update fields: {sorted(unknown)}"
            )
        pi = updates.pop("pi", self.pi)
        vf = updates.pop("vf", self.vf)
        values = {
            "environment_step": self.environment_step,
            "decision_id": self.decision_id,
            "pending_obs_hash": self.pending_obs_hash,
            "episode_start": self.episode_start,
            "policy_state_schema_hash": self.policy_state_schema_hash,
            "context_mode": self.context_mode,
            "burn_in_raw_obs": self.burn_in_raw_obs,
            "burn_in_episode_starts": self.burn_in_episode_starts,
            "phase": self.phase,
        }
        values.update(updates)
        return self._create(pi=pi, vf=vf, **values)

    def validate_integrity(self) -> None:
        """保存済み context seal を現在の全 field から再計算する。

        actor/critic state や phase の後編集を formal score の前に検出する。
        """
        actual = canonical_hash(_context_payload(self))
        if actual != self.context_sha256:
            raise RecurrentSessionError("critic context integrity seal mismatch")

    def validate_for_policy(
        self,
        policy_state_schema: dict[str, Any],
        *,
        allow_zero_state_smoke: bool,
    ) -> None:
        """context phase・schema・state shape・finite を policy へ束縛する。

        PPO と recurrent、pi と vf、h と c の全 sibling を同じ入口で fail-closed に検証する。
        """
        self.validate_integrity()
        if self.phase != "pending_pre_commit":
            raise RecurrentSessionError("context phase must be pending_pre_commit")
        if self.policy_state_schema_hash != policy_state_schema[
            "policy_state_schema_hash"
        ]:
            raise RecurrentSessionError("policy state schema hash mismatch")
        if self.context_mode == "zero_state_smoke" and not allow_zero_state_smoke:
            raise RecurrentSessionError(
                "zero-state context is not a formal ranking context"
            )
        if policy_state_schema["algorithm"] == "PPO":
            if self.pi is not None or self.vf is not None:
                raise RecurrentSessionError("PPO context must not contain RNN state")
            return
        expected_shape = (
            policy_state_schema["n_lstm_layers"],
            1,
            policy_state_schema["lstm_hidden_size"],
        )
        for name, pair in (("pi", self.pi), ("vf", self.vf)):
            if pair is None:
                raise RecurrentSessionError(
                    f"recurrent formal context requires policy-bound {name} state"
                )
            for component, array in zip(("h", "c"), pair):
                if array.shape != expected_shape:
                    raise RecurrentSessionError(
                        f"{name}_{component} shape mismatch: "
                        f"expected={expected_shape}, actual={array.shape}"
                    )
                if not np.all(np.isfinite(array)):
                    raise RecurrentSessionError(
                        f"{name}_{component} must contain only finite values"
                    )
        if (
            policy_state_schema["shared_lstm"]
            or not policy_state_schema["enable_critic_lstm"]
        ) and (
            not np.array_equal(self.pi[0], self.vf[0])
            or not np.array_equal(self.pi[1], self.vf[1])
        ):
            raise RecurrentSessionError(
                "shared/disabled critic requires vf state bound to policy forward output"
            )
        if (
            not policy_state_schema["shared_lstm"]
            and policy_state_schema["enable_critic_lstm"]
            and (
                np.any(self.pi[0] != 0.0)
                or np.any(self.pi[1] != 0.0)
                or np.any(self.vf[0] != 0.0)
                or np.any(self.vf[1] != 0.0)
            )
            and np.array_equal(self.pi[0], self.vf[0])
            and np.array_equal(self.pi[1], self.vf[1])
        ):
            raise RecurrentSessionError(
                "separate critic context cannot reuse actor-only predict state as vf"
            )


@dataclass(frozen=True, slots=True)
class MovementStep:
    """一回の movement policy forward 結果を表す。

    action と state hash を同じ transition 証跡として返し、preview 結果とは混在させない。
    """

    movement_action: int
    value_normalized_return: float
    state_before_hash: str
    state_after_hash: str


@dataclass(frozen=True, slots=True)
class RecurrentCommit:
    """selected post-choice observation の exactly-once commit 証跡。

    decision・step・choice・obs・前後 state・movement action を一つの immutable record に束縛する。
    """

    environment_step: int
    decision_id: str
    selected_choice_id: str
    selected_post_obs_hash: str
    state_before_hash: str
    state_after_hash: str
    movement_action: int
    commit_count: Literal[1]


@dataclass(slots=True)
class _PendingDecision:
    """session 内部の pending decision と commit 状態を保持する。

    外部へ mutable record を公開せず、retry と commit の比較対象を開始時点で固定する。
    """

    environment_step: int
    decision_id: str
    pending_obs_hash: str
    state_before_hash: str
    context: CriticContext
    commit: RecurrentCommit | None = None


class RecurrentPolicySession:
    """movement observation だけで actor/critic state を進める session。

    level-up lifecycle を state machine にし、pending retry と selected commit の順序違反を拒否する。
    """

    def __init__(self, source: Any) -> None:
        """検証済み source の policy schema から初期 state を作る。

        recurrent は pi/vf 四配列を零初期化し、PPO は state を持たない同じ lifecycle を使う。
        """
        self.source = source
        schema = source.policy_state_schema
        if source.algorithm == "RecurrentPPO":
            shape = (
                schema["n_lstm_layers"],
                1,
                schema["lstm_hidden_size"],
            )
            self._pi = (
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
            )
            self._vf = (self._pi[0].copy(), self._pi[1].copy())
        else:
            self._pi = None
            self._vf = None
        self._pending: _PendingDecision | None = None
        self._completed_decisions: set[str] = set()

    @property
    def state_hash(self) -> str:
        """現在の pi/vf state を canonical hash で返す。

        preview 前後の不変性と commit 前後の一回遷移を同じ比較値で監査できる。
        """
        return recurrent_state_hash(self._pi, self._vf)

    @property
    def states(
        self,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray] | None,
        tuple[np.ndarray, np.ndarray] | None,
    ]:
        """現在 state の defensive copy を返す。

        呼び出し側が session 内部配列を編集して policy history を改変できないようにする。
        """
        copy_pair = lambda pair: (
            None if pair is None else (pair[0].copy(), pair[1].copy())
        )
        return copy_pair(self._pi), copy_pair(self._vf)

    @property
    def pending_context(self) -> CriticContext | None:
        """現在 pending 中の immutable critic context を返す。

        collector の HTTP retry/resume が private state を直接参照せず、begin 時に発行された
        seal 済み context と同一 object だけを再利用できるようにする。
        """
        return None if self._pending is None else self._pending.context

    def _advance_policy(
        self,
        raw_obs: np.ndarray,
        *,
        episode_start: bool,
    ) -> MovementStep:
        """一つの movement observation を policy.forward へ渡す。

        RecurrentPPO は RNNStates.pi/vf を同時に更新し、actor-only predict state は使用しない。
        """
        import torch as th

        batch = np.asarray(raw_obs, dtype=np.float32)
        if batch.shape != (self.source.observation_dim,):
            raise RecurrentSessionError(
                f"movement observation shape must be ({self.source.observation_dim},)"
            )
        normalized = self.source.normalize_raw_obs(batch[None, :])
        obs_tensor, _ = self.source.policy.obs_to_tensor(normalized)
        before = self.state_hash
        with th.no_grad():
            if self.source.algorithm == "RecurrentPPO":
                from sb3_contrib.common.recurrent.type_aliases import RNNStates

                device = self.source.policy.device
                states = RNNStates(
                    tuple(
                        th.as_tensor(item, dtype=th.float32, device=device)
                        for item in self._pi
                    ),
                    tuple(
                        th.as_tensor(item, dtype=th.float32, device=device)
                        for item in self._vf
                    ),
                )
                starts = th.as_tensor(
                    [episode_start],
                    dtype=th.float32,
                    device=device,
                )
                actions, values, _, next_states = self.source.policy.forward(
                    obs_tensor,
                    states,
                    starts,
                    deterministic=True,
                )
                self._pi = tuple(
                    item.detach().cpu().numpy().astype(np.float32, copy=True)
                    for item in next_states.pi
                )
                self._vf = tuple(
                    item.detach().cpu().numpy().astype(np.float32, copy=True)
                    for item in next_states.vf
                )
            else:
                actions, values, _ = self.source.policy.forward(
                    obs_tensor,
                    deterministic=True,
                )
        action = int(actions.detach().cpu().numpy().reshape(-1)[0])
        value = float(values.detach().cpu().numpy().reshape(-1)[0])
        return MovementStep(
            movement_action=action,
            value_normalized_return=value,
            state_before_hash=before,
            state_after_hash=self.state_hash,
        )

    def advance_movement(
        self,
        raw_obs: np.ndarray,
        *,
        episode_start: bool,
    ) -> MovementStep:
        """通常 movement tick の observation で state を一回進める。

        level-up pending 中の外部 advance は retry 二重消費になるため拒否する。
        """
        if self._pending is not None:
            raise RecurrentSessionError(
                "movement state cannot advance while level-up is pending"
            )
        return self._advance_policy(raw_obs, episode_start=episode_start)

    def begin_level_up(
        self,
        *,
        environment_step: int,
        decision_id: str,
        pending_obs: np.ndarray,
        episode_start: bool,
    ) -> CriticContext:
        """pending/base obs を消費せず pre-commit context を capture する。

        decision 開始時の state hash と raw pending hash を固定し、preview 回数に依存させない。
        """
        if self._pending is not None:
            raise RecurrentSessionError("another level-up decision is already pending")
        if (
            type(environment_step) is not int
            or environment_step < 0
            or not isinstance(decision_id, str)
            or not decision_id
        ):
            raise RecurrentSessionError("invalid environment step or decision id")
        if decision_id in self._completed_decisions:
            raise RecurrentSessionError("decision id was already finalized")
        pi, vf = self.states
        context = CriticContext.captured(
            environment_step=environment_step,
            decision_id=decision_id,
            pending_obs=pending_obs,
            episode_start=episode_start,
            pi=pi,
            vf=vf,
            policy_state_schema_hash=self.source.policy_state_schema[
                "policy_state_schema_hash"
            ],
        )
        self._pending = _PendingDecision(
            environment_step=environment_step,
            decision_id=decision_id,
            pending_obs_hash=context.pending_obs_hash,
            state_before_hash=self.state_hash,
            context=context,
        )
        return context

    def validate_context(self, context: CriticContext) -> None:
        """score context を現在 pending decision と state へ照合する。

        begin_level_up() が発行した context seal（context_sha256）と完全一致するものだけを受理し、
        episode_start や context_mode を含む全フィールドの事後変更を拒否する。
        seal チェックを validate_for_policy() より先に実施し、変更検出を優先する。
        """
        if self._pending is None:
            raise RecurrentSessionError("no level-up decision is pending")
        expected = self._pending
        if context.context_sha256 != expected.context.context_sha256:
            raise RecurrentSessionError("context seal mismatch: context does not match the one issued by begin_level_up")
        context.validate_for_policy(
            self.source.policy_state_schema,
            allow_zero_state_smoke=False,
        )
        if self.state_hash != expected.state_before_hash:
            raise RecurrentSessionError("pending retry advanced recurrent state")

    def validate_pending_retry(
        self,
        *,
        environment_step: int,
        decision_id: str,
        pending_obs: np.ndarray,
    ) -> None:
        """retry の step・decision・raw obs を開始時点へ照合する。

        validation は policy.forward を呼ばず、何回実行しても state hash を変えない。
        """
        if self._pending is None:
            raise RecurrentSessionError("no level-up decision is pending")
        expected = self._pending
        if environment_step != expected.environment_step:
            raise RecurrentSessionError("pending retry environment step mismatch")
        if decision_id != expected.decision_id:
            raise RecurrentSessionError("pending retry decision id mismatch")
        if observation_hash(pending_obs) != expected.pending_obs_hash:
            raise RecurrentSessionError("pending retry observation hash mismatch")
        if self.state_hash != expected.state_before_hash:
            raise RecurrentSessionError("pending retry advanced recurrent state")

    def commit_selected(
        self,
        *,
        environment_step: int,
        decision_id: str,
        selected_choice_id: str,
        selected_post_obs: np.ndarray,
    ) -> RecurrentCommit:
        """selected post-choice observation を exactly once movement commit する。

        production apply 後の実 observation だけで pi/vf を一回進め、重複 decision を拒否する。
        """
        if decision_id in self._completed_decisions:
            raise RecurrentSessionError("decision was already finalized")
        if self._pending is None:
            raise RecurrentSessionError("no level-up decision is pending")
        pending = self._pending
        if pending.commit is not None:
            raise RecurrentSessionError("duplicate selected observation commit")
        if environment_step != pending.environment_step:
            raise RecurrentSessionError("selected commit environment step mismatch")
        if decision_id != pending.decision_id:
            raise RecurrentSessionError("selected commit decision id mismatch")
        if not isinstance(selected_choice_id, str) or not selected_choice_id:
            raise RecurrentSessionError("selected choice id must be non-empty")
        if self.state_hash != pending.state_before_hash:
            raise RecurrentSessionError("state changed before selected commit")
        transition = self._advance_policy(
            selected_post_obs,
            episode_start=pending.context.episode_start,
        )
        commit = RecurrentCommit(
            environment_step=environment_step,
            decision_id=decision_id,
            selected_choice_id=selected_choice_id,
            selected_post_obs_hash=observation_hash(selected_post_obs),
            state_before_hash=transition.state_before_hash,
            state_after_hash=transition.state_after_hash,
            movement_action=transition.movement_action,
            commit_count=1,
        )
        pending.commit = commit
        return commit

    def finalize_level_up(self) -> RecurrentCommit:
        """exactly one commit 済み decision だけを閉じる。

        commit 0 回を成功扱いせず、二回目は commit API 側で拒否した同じ record を確定する。
        """
        if self._pending is None:
            raise RecurrentSessionError("no level-up decision is pending")
        if self._pending.commit is None:
            raise RecurrentSessionError(
                "selected observation was not committed exactly once"
            )
        commit = self._pending.commit
        self._completed_decisions.add(commit.decision_id)
        self._pending = None
        return commit


def write_critic_context(path: Path, context: CriticContext) -> None:
    """CriticContext を pickle-free NPZ へ保存する。

    optional array の有無を明示 flag にし、reader が object dtype や暗黙 None を受理しないようにする。
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "schema_version": np.asarray("survivors.critic_context.v1"),
        "environment_step": np.asarray(context.environment_step, dtype=np.int64),
        "decision_id": np.asarray(context.decision_id),
        "phase": np.asarray(context.phase),
        "pending_obs_hash": np.asarray(context.pending_obs_hash),
        "episode_start": np.asarray(context.episode_start, dtype=np.bool_),
        "policy_state_schema_hash": np.asarray(
            context.policy_state_schema_hash
        ),
        "context_mode": np.asarray(context.context_mode),
        "context_sha256": np.asarray(context.context_sha256),
    }
    for name in (
        "pi_h",
        "pi_c",
        "vf_h",
        "vf_c",
        "burn_in_raw_obs",
        "burn_in_episode_starts",
    ):
        value = getattr(context, name)
        arrays[f"has_{name}"] = np.asarray(value is not None, dtype=np.bool_)
        arrays[name] = (
            np.asarray([], dtype=np.float32)
            if value is None
            else np.asarray(value)
        )
    with destination.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def load_critic_context(path: Path) -> CriticContext:
    """pickle-free NPZ から CriticContext を読み seal を検証する。

    未知・不足 key と scalar 型を fail-closed にし、全 state sibling を同じ規則で復元する。
    """
    base_keys = {
        "schema_version",
        "environment_step",
        "decision_id",
        "phase",
        "pending_obs_hash",
        "episode_start",
        "policy_state_schema_hash",
        "context_mode",
        "context_sha256",
    }
    array_names = {
        "pi_h",
        "pi_c",
        "vf_h",
        "vf_c",
        "burn_in_raw_obs",
        "burn_in_episode_starts",
    }
    expected_keys = base_keys | array_names | {
        f"has_{name}" for name in array_names
    }
    try:
        with np.load(Path(path), allow_pickle=False) as data:
            if set(data.files) != expected_keys:
                raise RecurrentSessionError("critic context NPZ fields mismatch")
            if data["schema_version"].item() != "survivors.critic_context.v1":
                raise RecurrentSessionError(
                    "unsupported critic context NPZ schema"
                )
            loaded = {
                name: (
                    np.asarray(data[name]).copy()
                    if bool(data[f"has_{name}"].item())
                    else None
                )
                for name in array_names
            }
            context = CriticContext(
                environment_step=int(data["environment_step"].item()),
                decision_id=str(data["decision_id"].item()),
                phase=str(data["phase"].item()),
                pending_obs_hash=str(data["pending_obs_hash"].item()),
                episode_start=bool(data["episode_start"].item()),
                pi_h=loaded["pi_h"],
                pi_c=loaded["pi_c"],
                vf_h=loaded["vf_h"],
                vf_c=loaded["vf_c"],
                policy_state_schema_hash=str(
                    data["policy_state_schema_hash"].item()
                ),
                context_mode=str(data["context_mode"].item()),
                burn_in_raw_obs=loaded["burn_in_raw_obs"],
                burn_in_episode_starts=loaded["burn_in_episode_starts"],
                context_sha256=str(data["context_sha256"].item()),
            )
    except (OSError, ValueError, TypeError) as exc:
        raise RecurrentSessionError("critic context NPZ is invalid") from exc
    context.validate_integrity()
    return context

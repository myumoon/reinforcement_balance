"""Survivors recurrent choice trace を一 transaction で収集する。

current-hash integration fidelity verdict を親 gate とし、decision context を freeze したまま
preview/score/behavior apply を行う。selected post-choice observation だけを session へ一回
commit し、retry/resume は同じ external decision と canonical record ID へ収束させる。
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reinbalance_survivors_contracts.canonical_json import (
    canonical_json_bytes,
)
from reinbalance_survivors_contracts.fidelity_verdict import (
    FidelityVerdict,
    downstream_release_allowed,
)

from games.survivors.recurrent_policy_session import (
    RecurrentCommit,
    RecurrentPolicySession,
    RecurrentSessionError,
    observation_hash,
)
from games.survivors.value_choice_dataset import (
    DatasetWriter,
    record_id_for,
)
from games.survivors.value_scorer import CandidateValue, ValueScorer

DEFAULT_BEHAVIOR_POLICY = "epsilon_source_scorer"
DEFAULT_EPSILON = 0.20
JOURNAL_SCHEMA_VERSION = "survivors.value_choice_collector_journal.v1"


class CollectorError(ValueError):
    """formal gate または choice transaction が不正な場合の例外。

    HTTP timeout は同じ ID の内部 retry 対象とし、binding/schema/state 違反だけを caller が
    episode 失敗として扱える一つの例外境界へ変換する。
    """


@dataclass(frozen=True, slots=True)
class CollectorDecisionResult:
    """一 decision の behavior selection と recurrent commit 証跡。

    teacher best は dataset row の別 field に保持し、この結果の selected choice を正解
    label として公開しない。
    """

    record_id: str
    selected_choice_id: str
    propensity: float
    teacher_best_choice_id: str
    commit: RecurrentCommit


def epsilon_source_scorer_propensities(
    choice_ids: Sequence[str],
    teacher_best_choice_id: str,
    *,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, float]:
    """epsilon-source-scorer の各候補 propensity を返す。

    best candidate は ``1-epsilon + epsilon/K``、他候補は ``epsilon/K`` とし、候補数に
    応じた実 behavior probability を選択結果とは独立に記録する。
    """

    choices = list(choice_ids)
    if (
        len(choices) < 2
        or any(not isinstance(choice, str) or not choice for choice in choices)
        or len(set(choices)) != len(choices)
    ):
        raise CollectorError("choice_ids must contain at least two unique choices")
    if teacher_best_choice_id not in choices:
        raise CollectorError("teacher best choice is unknown")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or not 0.0 <= float(epsilon) <= 1.0
    ):
        raise CollectorError("epsilon must be finite and in [0, 1]")
    explore = float(epsilon) / len(choices)
    return {
        choice: (
            1.0 - float(epsilon) + explore
            if choice == teacher_best_choice_id
            else explore
        )
        for choice in choices
    }


def _atomic_json_write(path: Path, value: Any) -> None:
    """collector journal を canonical finite JSON として atomic publish する。

    response ack と selected behavior を process interruption 前に保存し、途中までの JSON を
    resume reader が completed transaction と誤認しないようにする。
    """

    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_observation(value: Any, expected_shape: tuple[int, ...]) -> np.ndarray:
    """HTTP observation を fixed shape の finite float32 copy にする。

    preview と apply ack の両経路へ同じ validation を適用し、NaN や暗黙 reshape を
    recurrent commit または NPZ writer へ渡さない。
    """

    try:
        observation = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise CollectorError("choice observation must be numeric") from exc
    if observation.shape != expected_shape:
        raise CollectorError(
            "choice observation shape mismatch: "
            f"expected={expected_shape}, actual={observation.shape}"
        )
    if not np.all(np.isfinite(observation)):
        raise CollectorError("choice observation must contain only finite values")
    return observation.copy()


class ChoiceTraceCollector:
    """一つの recurrent session と dataset writer を結ぶ formal collector。

    behavior policy は既定で epsilon-source-scorer/0.20 とし、fidelity/source gate を
    constructor で通過するまで preview、UE5 apply、dataset append を開始しない。
    """

    def __init__(
        self,
        *,
        env: Any,
        scorer: ValueScorer,
        session: RecurrentPolicySession,
        writer: DatasetWriter,
        source_identity_sha256: str,
        fidelity_verdict: FidelityVerdict | Mapping[str, Any] | None,
        current_gating_producer_hashes: Mapping[str, str] | None,
        epsilon: float = DEFAULT_EPSILON,
        seed: int = 0,
        journal_path: Path | None = None,
        max_http_retries: int = 2,
        shard_size: int | None = None,
    ) -> None:
        """formal parent と全 injected collaborator を起動前に検証する。

        baseline/stale/missing/blocked verdict、別 source の scorer/writer、負 retry 数を
        sibling 経路ごと fail-closed で拒否する。
        """

        if fidelity_verdict is None or current_gating_producer_hashes is None:
            raise CollectorError(
                "formal collector requires a current integration fidelity verdict"
            )
        try:
            allowed = downstream_release_allowed(
                fidelity_verdict,
                current_gating_producer_hashes,
                "integration",
            )
            checked_verdict = FidelityVerdict.from_wire(
                fidelity_verdict.to_wire()
                if isinstance(fidelity_verdict, FidelityVerdict)
                else fidelity_verdict
            )
        except (TypeError, ValueError) as exc:
            raise CollectorError(f"fidelity verdict gate failed: {exc}") from exc
        if not allowed:
            raise CollectorError("fidelity verdict contains blocking reasons")
        descriptor_identity = scorer.source.descriptor["identity_sha256"]
        if source_identity_sha256 != descriptor_identity:
            raise CollectorError("collector source identity does not match scorer")
        if (
            writer.manifest["source_identity_sha256"]
            != source_identity_sha256
        ):
            raise CollectorError("collector source identity does not match dataset")
        if session.source is not scorer.source:
            raise CollectorError("recurrent session is not bound to scorer source")
        if type(seed) is not int:
            raise CollectorError("collector seed must be an integer")
        if type(max_http_retries) is not int or max_http_retries < 0:
            raise CollectorError("max_http_retries must be non-negative")
        if shard_size is not None and (
            type(shard_size) is not int or shard_size <= 0
        ):
            raise CollectorError("shard_size must be positive")
        epsilon_source_scorer_propensities(
            ["validation-a", "validation-b"],
            "validation-a",
            epsilon=epsilon,
        )
        self.env = env
        self.scorer = scorer
        self.session = session
        self.writer = writer
        self.source_identity_sha256 = source_identity_sha256
        self.fidelity_verdict = checked_verdict
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.max_http_retries = max_http_retries
        self.shard_size = shard_size
        self.journal_path = Path(
            journal_path or writer.root / "collector-journal.json"
        )
        self.artifact_identity = {
            "source_identity_sha256": source_identity_sha256,
            "fidelity_verdict_sha256": checked_verdict.identity_hash,
            "model_sha256": scorer.source.descriptor["artifacts"]["model"][
                "sha256"
            ],
            "vecnormalize_sha256": scorer.source.descriptor["artifacts"][
                "vecnormalize"
            ]["sha256"],
            "observation_schema_sha256": scorer.source.descriptor[
                "observation_schema"
            ]["sha256"],
            "policy_state_schema_sha256": scorer.source.policy_state_schema[
                "policy_state_schema_hash"
            ],
        }
        self._journal = self._load_journal()
        self._runtime_results: dict[str, CollectorDecisionResult] = {}
        self._runtime_acknowledged: set[str] = set()
        self._active_record_id: str | None = None
        self._episode_active = False
        self._replay_events: list[dict[str, Any]] = [
            {"kind": "artifact_identity", **self.artifact_identity}
        ]

    def _load_journal(self) -> dict[str, Any]:
        """既存 collector journal を strict top-level schema で読む。

        unknown version/field や非 canonical JSON を空 resume state として無視せず、formal
        collection の開始を停止する。
        """

        if not self.journal_path.exists():
            journal = {
                "schema_version": JOURNAL_SCHEMA_VERSION,
                "source_identity_sha256": self.source_identity_sha256,
                "transactions": {},
            }
            _atomic_json_write(self.journal_path, journal)
            return journal
        try:
            raw = json.loads(self.journal_path.read_bytes())
            canonical_json_bytes(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise CollectorError(f"collector journal could not be read: {exc}") from exc
        if (
            not isinstance(raw, Mapping)
            or set(raw)
            != {
                "schema_version",
                "source_identity_sha256",
                "transactions",
            }
            or raw["schema_version"] != JOURNAL_SCHEMA_VERSION
            or raw["source_identity_sha256"] != self.source_identity_sha256
            or not isinstance(raw["transactions"], Mapping)
        ):
            raise CollectorError("collector journal schema/source mismatch")
        return json.loads(canonical_json_bytes(raw))

    def _save_journal(self) -> None:
        """現在の transaction map を canonical/atomic journal へ保存する。

        behavior selection、ack、completed の各 state transition 直後に呼び、resume が新しい
        random choice や record ID を発行しないようにする。
        """

        _atomic_json_write(self.journal_path, self._journal)

    def _event(self, kind: str, **payload: Any) -> None:
        """replay event を finite canonical JSON として ordered log へ追加する。

        reset/action/preview/choice request/ack/external decision/artifact identity が同じ list の
        時系列を共有し、個別ログ間の順序推測を不要にする。
        """

        event = {"kind": kind, **payload}
        canonical_json_bytes(event)
        self._replay_events.append(json.loads(canonical_json_bytes(event)))
        if self._active_record_id is not None:
            transaction = self._journal["transactions"].get(
                self._active_record_id
            )
            if isinstance(transaction, dict):
                transaction["replay_events"] = list(self._replay_events)
                self._save_journal()

    def _preview(
        self,
        decision_id: str,
        choice_ids: list[str],
    ) -> Any:
        """no-commit preview を同じ decision ID で timeout retry する。

        各 request と成功 ack を replay event に残し、retry 回数にかかわらず recurrent
        session には observation を一度も入力しない。
        """

        for attempt in range(self.max_http_retries + 1):
            self._event(
                "choice_preview_request",
                decision_id=decision_id,
                attempt=attempt,
            )
            try:
                preview = self.env.preview_level_up(
                    decision_id,
                    choice_ids,
                )
                self._event(
                    "choice_preview_ack",
                    decision_id=decision_id,
                    attempt=attempt,
                )
                return preview
            except (TimeoutError, ConnectionError):
                if attempt >= self.max_http_retries:
                    raise
        raise AssertionError("unreachable preview retry")

    def _apply_choice(
        self,
        decision_id: str,
        selected_choice_id: str,
    ) -> tuple[np.ndarray, Mapping[str, Any]]:
        """同じ decision/choice を exactly-once endpoint へ timeout retry する。

        request ごとの ID を replay event へ保存し、ack が失われても別 record/choice を
        選び直さない。
        """

        for attempt in range(self.max_http_retries + 1):
            self._event(
                "choice_request",
                decision_id=decision_id,
                choice_id=selected_choice_id,
                attempt=attempt,
            )
            try:
                observation, info = self.env.choose_level_up(
                    decision_id,
                    selected_choice_id,
                )
                self._event(
                    "choice_ack",
                    decision_id=decision_id,
                    choice_id=selected_choice_id,
                    attempt=attempt,
                )
                return observation, info
            except (TimeoutError, ConnectionError):
                if attempt >= self.max_http_retries:
                    raise
        raise AssertionError("unreachable choice retry")

    def _completed_result(
        self,
        record_id: str,
        transaction: Mapping[str, Any],
    ) -> CollectorDecisionResult:
        """journal の completed transaction を immutable result に復元する。

        process resume で UE5 apply/session commit/writer append を繰り返さず、保存済み
        exactly-once commit evidence だけを返す。
        """

        result = transaction.get("result")
        if not isinstance(result, Mapping) or set(result) != {
            "selected_choice_id",
            "propensity",
            "teacher_best_choice_id",
            "commit",
        }:
            raise CollectorError("completed journal result is malformed")
        commit_wire = result["commit"]
        if not isinstance(commit_wire, Mapping) or set(commit_wire) != {
            "environment_step",
            "decision_id",
            "selected_choice_id",
            "selected_post_obs_hash",
            "state_before_hash",
            "state_after_hash",
            "movement_action",
            "commit_count",
        }:
            raise CollectorError("completed journal commit is malformed")
        return CollectorDecisionResult(
            record_id=record_id,
            selected_choice_id=result["selected_choice_id"],
            propensity=float(result["propensity"]),
            teacher_best_choice_id=result["teacher_best_choice_id"],
            commit=RecurrentCommit(**dict(commit_wire)),
        )

    def collect_decision(
        self,
        *,
        episode_logical_id: str,
        environment_step: int,
        decision_id: str,
        pending_obs: np.ndarray,
        episode_start: bool,
        choice_ids: Sequence[str],
    ) -> CollectorDecisionResult:
        """decision→preview→score→behavior→apply→commit を一 transaction で実行する。

        preview/retry 中は actor/vf context を freeze し、ack 済み selected observation を
        exactly once commit してから behavior/teacher/replay を dataset へ追加する。
        """

        choices = list(choice_ids)
        if (
            len(choices) < 2
            or any(not isinstance(choice, str) or not choice for choice in choices)
            or len(set(choices)) != len(choices)
        ):
            raise CollectorError(
                "choice_ids must contain at least two unique choices"
            )
        expected_shape = (self.scorer.source.observation_dim,)
        pending = _finite_observation(pending_obs, expected_shape)
        record_id = record_id_for(
            self.source_identity_sha256,
            episode_logical_id,
            decision_id,
        )
        existing = self._journal["transactions"].get(record_id)
        if record_id in self._runtime_results:
            return self._runtime_results[record_id]
        if existing is not None and (
            not isinstance(existing, Mapping)
            or existing.get("episode_logical_id") != episode_logical_id
            or existing.get("decision_id") != decision_id
            or existing.get("environment_step") != environment_step
            or existing.get("pending_obs_sha256") != observation_hash(pending)
            or existing.get("choice_ids") != choices
        ):
            raise CollectorError("resume transaction binding mismatch")
        if self.writer.active_shard_id is None:
            self.writer.start_shard()
        self._active_record_id = record_id
        if isinstance(existing, Mapping):
            stored_events = existing.get("replay_events")
            if not isinstance(stored_events, list) or not stored_events:
                raise CollectorError("resume replay events are missing")
            canonical_json_bytes(stored_events)
            self._replay_events = json.loads(canonical_json_bytes(stored_events))
        try:
            context = self.session.begin_level_up(
                environment_step=environment_step,
                decision_id=decision_id,
                pending_obs=pending,
                episode_start=episode_start,
            )
        except RecurrentSessionError as exc:
            try:
                self.session.validate_pending_retry(
                    environment_step=environment_step,
                    decision_id=decision_id,
                    pending_obs=pending,
                )
            except RecurrentSessionError:
                raise exc
            if existing is None or "context" not in existing:
                raise CollectorError("pending session has no matching resume context") from exc
            context = self.session.pending_context
            if context is None:
                raise CollectorError("pending retry context disappeared") from exc
        self.session.validate_context(context)
        if existing is None:
            self._event(
                "external_decision",
                episode_logical_id=episode_logical_id,
                environment_step=environment_step,
                decision_id=decision_id,
                choice_ids=choices,
            )
            existing = {
                "status": "started",
                "episode_logical_id": episode_logical_id,
                "environment_step": environment_step,
                "decision_id": decision_id,
                "pending_obs_sha256": observation_hash(pending),
                "choice_ids": choices,
                "context": {
                    "sha256": context.context_sha256,
                    "state_before_sha256": self.session.state_hash,
                },
                "replay_events": list(self._replay_events),
            }
            self._journal["transactions"][record_id] = existing
            self._save_journal()
        elif existing["context"]["sha256"] != context.context_sha256:
            raise CollectorError("resume critic context mismatch")

        preview = self._preview(decision_id, choices)
        expected_schema_hash = (
            self.scorer.source.descriptor["observation_schema"][
                "reported_hash"
            ]
            or self.scorer.source.descriptor["observation_schema"]["sha256"]
        )
        if (
            preview.decision_id != decision_id
            or preview.obs_schema_hash != expected_schema_hash
            or set(preview.by_choice_id) != set(choices)
        ):
            raise CollectorError(
                "preview decision, schema, or choice binding mismatch"
            )
        base_obs = _finite_observation(preview.base_obs, expected_shape)
        if observation_hash(base_obs) != observation_hash(pending):
            raise CollectorError("preview base observation mismatch")
        candidate_observations = np.asarray(
            [
                _finite_observation(
                    preview.by_choice_id[choice].projected_obs,
                    expected_shape,
                )
                for choice in choices
            ],
            dtype=np.float32,
        )
        self.session.validate_context(context)
        values = self.scorer.score(
            candidate_observations,
            context,
            choice_ids=choices,
        )
        ordered_values = sorted(
            enumerate(values),
            key=lambda item: item[1].value_normalized_return,
            reverse=True,
        )
        teacher_best = ordered_values[0][1].choice_id
        propensities = epsilon_source_scorer_propensities(
            choices,
            teacher_best,
            epsilon=self.epsilon,
        )
        selected = existing.get("selected_choice_id")
        if selected is None:
            if float(self.rng.random()) < self.epsilon:
                selected = choices[int(self.rng.integers(0, len(choices)))]
            else:
                selected = teacher_best
            existing["status"] = "selected"
            existing["selected_choice_id"] = selected
            existing["propensity"] = propensities[selected]
            existing["teacher_best_choice_id"] = teacher_best
            self._save_journal()
        if selected not in choices:
            raise CollectorError("journal selected choice is unknown")
        if existing.get("teacher_best_choice_id") != teacher_best:
            raise CollectorError("resume teacher ranking changed")
        if float(existing.get("propensity")) != propensities[selected]:
            raise CollectorError("resume behavior propensity changed")

        selected_observation: np.ndarray
        if record_id in self._runtime_acknowledged:
            selected_observation = _finite_observation(
                existing.get("selected_post_obs"),
                expected_shape,
            )
        else:
            applied_obs, info = self._apply_choice(decision_id, selected)
            selected_observation = _finite_observation(
                applied_obs,
                expected_shape,
            )
            if (
                not isinstance(info, Mapping)
                or info.get("level_up_pending") is not False
                or not isinstance(info.get("level_up_choices"), list)
                or info["level_up_choices"]
            ):
                raise CollectorError("choice acknowledgement info is malformed")
            expected_selected = candidate_observations[choices.index(selected)]
            if observation_hash(selected_observation) != observation_hash(
                expected_selected
            ):
                raise CollectorError(
                    "applied observation does not match selected preview"
                )
            existing["status"] = "acknowledged"
            existing["selected_post_obs"] = selected_observation.tolist()
            self._save_journal()
            self._runtime_acknowledged.add(record_id)

        commit = self.session.commit_selected(
            environment_step=environment_step,
            decision_id=decision_id,
            selected_choice_id=selected,
            selected_post_obs=selected_observation,
        )
        ordered_candidates: list[CandidateValue] = [
            candidate for _, candidate in ordered_values
        ]
        metadata = {
            "episode_logical_id": episode_logical_id,
            "decision_id": decision_id,
            "environment_step": environment_step,
            "candidate_choice_ids": choices,
            "behavior": {
                "policy": DEFAULT_BEHAVIOR_POLICY,
                "epsilon": self.epsilon,
                "selected_choice_id": selected,
                "propensity": propensities[selected],
            },
            "teacher_label": {
                "policy": "source_scorer",
                "best_choice_id": teacher_best,
                "ordered_choice_ids": [
                    candidate.choice_id for candidate in ordered_candidates
                ],
                "normalized_returns": [
                    candidate.value_normalized_return
                    for candidate in ordered_candidates
                ],
                "unscaled_returns": [
                    candidate.value_unscaled_return
                    for candidate in ordered_candidates
                ],
            },
            "context": {
                "sha256": context.context_sha256,
                "mode": context.context_mode,
                "pending_obs_sha256": context.pending_obs_hash,
                "state_before_sha256": commit.state_before_hash,
                "state_after_sha256": commit.state_after_hash,
            },
            "replay_events": list(self._replay_events),
            "artifact_identity": dict(self.artifact_identity),
        }
        arrays: dict[str, np.ndarray] = {
            "pending_obs": pending,
            "candidate_obs": candidate_observations,
            "selected_post_obs": selected_observation,
            "movement_actions": np.asarray(
                [
                    event["action"]
                    for event in self._replay_events
                    if event["kind"] == "action"
                ],
                dtype=np.int32,
            ),
        }
        for name in ("pi_h", "pi_c", "vf_h", "vf_c"):
            value = getattr(context, name)
            if value is not None:
                arrays[name] = np.asarray(value, dtype=np.float32).copy()
        record_was_committed = self.writer.contains_record(
            record_id,
            committed_only=True,
        )
        appended_id = (
            record_id
            if self.writer.contains_record(record_id)
            else self.writer.append(metadata, arrays)
        )
        if appended_id != record_id:
            raise CollectorError("dataset writer returned unexpected record ID")
        finalized = self.session.finalize_level_up()
        if finalized != commit or commit.commit_count != 1:
            raise CollectorError("recurrent decision did not finalize exactly once")
        commit_wire = {
            "environment_step": commit.environment_step,
            "decision_id": commit.decision_id,
            "selected_choice_id": commit.selected_choice_id,
            "selected_post_obs_hash": commit.selected_post_obs_hash,
            "state_before_hash": commit.state_before_hash,
            "state_after_hash": commit.state_after_hash,
            "movement_action": commit.movement_action,
            "commit_count": commit.commit_count,
        }
        existing["status"] = "appended"
        existing["result"] = {
            "selected_choice_id": selected,
            "propensity": propensities[selected],
            "teacher_best_choice_id": teacher_best,
            "commit": commit_wire,
        }
        if record_was_committed:
            existing["status"] = "completed"
        elif (
            self.shard_size is not None
            and not self._episode_active
            and self.writer.active_row_count >= self.shard_size
        ):
            self.writer.commit_shard()
            existing["status"] = "completed"
        self._save_journal()
        result = CollectorDecisionResult(
            record_id=record_id,
            selected_choice_id=selected,
            propensity=propensities[selected],
            teacher_best_choice_id=teacher_best,
            commit=commit,
        )
        self._runtime_results[record_id] = result
        self._active_record_id = None
        return result

    def collect_episode(
        self,
        *,
        seed: int,
        episode_logical_id: str,
        max_environment_steps: int | None = None,
    ) -> int:
        """reset から terminal まで movement と external choice を収集する。

        pending 応答後は ``/step`` を呼ばず choice transaction を完了し、その commit が返す
        次 movement action だけを使用する。戻り値は dataset へ追加した decision 数。
        """

        if type(seed) is not int:
            raise CollectorError("episode seed must be an integer")
        if (
            max_environment_steps is not None
            and (
                type(max_environment_steps) is not int
                or max_environment_steps <= 0
            )
        ):
            raise CollectorError("max_environment_steps must be positive")
        reset_result = self.env.reset(seed=seed)
        if (
            not isinstance(reset_result, tuple)
            or len(reset_result) != 2
            or not isinstance(reset_result[1], Mapping)
        ):
            raise CollectorError("environment reset contract is malformed")
        observation = _finite_observation(
            reset_result[0],
            (self.scorer.source.observation_dim,),
        )
        self._replay_events = [
            {"kind": "artifact_identity", **self.artifact_identity}
        ]
        self._event(
            "reset",
            seed=seed,
            episode_logical_id=episode_logical_id,
        )
        transition = self.session.advance_movement(
            observation,
            episode_start=True,
        )
        action = transition.movement_action
        environment_step = 0
        decision_count = 0
        episode_record_ids: list[str] = []
        while True:
            if (
                max_environment_steps is not None
                and environment_step >= max_environment_steps
            ):
                break
            self._event(
                "action",
                environment_step=environment_step,
                action=int(action),
            )
            step_result = self.env.step(int(action))
            if not isinstance(step_result, tuple) or len(step_result) != 5:
                raise CollectorError("environment step contract is malformed")
            raw_obs, _, done, truncated, info = step_result
            if not isinstance(info, Mapping):
                raise CollectorError("environment step info must be an object")
            observation = _finite_observation(
                raw_obs,
                (self.scorer.source.observation_dim,),
            )
            environment_step += 1
            if done or truncated:
                break
            if info.get("level_up_pending") is True:
                raw_choices = info.get("level_up_choices")
                decision_id = info.get("level_up_decision_id")
                if (
                    not isinstance(raw_choices, list)
                    or not isinstance(decision_id, str)
                    or not decision_id
                ):
                    raise CollectorError("pending choice info is malformed")
                choices = []
                for choice in raw_choices:
                    if not isinstance(choice, Mapping):
                        raise CollectorError("pending choice must be an object")
                    choice_id = choice.get("choice_id")
                    if not isinstance(choice_id, str) or not choice_id:
                        raise CollectorError("pending choice ID is invalid")
                    choices.append(choice_id)
                self._episode_active = True
                try:
                    result = self.collect_decision(
                        episode_logical_id=episode_logical_id,
                        environment_step=environment_step,
                        decision_id=decision_id,
                        pending_obs=observation,
                        episode_start=False,
                        choice_ids=choices,
                    )
                finally:
                    self._episode_active = False
                decision_count += 1
                episode_record_ids.append(result.record_id)
                action = result.commit.movement_action
                continue
            transition = self.session.advance_movement(
                observation,
                episode_start=False,
            )
            action = transition.movement_action
        if episode_record_ids:
            self.writer.finalize_replay_trace(
                episode_record_ids,
                self._replay_events,
            )
            for record_id in episode_record_ids:
                transaction = self._journal["transactions"].get(record_id)
                if isinstance(transaction, dict):
                    transaction["replay_events"] = list(self._replay_events)
            self._save_journal()
            if (
                self.shard_size is not None
                and self.writer.active_row_count >= self.shard_size
            ):
                self.writer.commit_shard()
        return decision_count


__all__ = [
    "DEFAULT_BEHAVIOR_POLICY",
    "DEFAULT_EPSILON",
    "ChoiceTraceCollector",
    "CollectorDecisionResult",
    "CollectorError",
    "epsilon_source_scorer_propensities",
]

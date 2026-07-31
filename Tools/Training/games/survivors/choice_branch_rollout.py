"""Survivors choice の semantic replay と paired branch 割当を定義する。

post-decision RNG は source/episode/decision/replication/schema だけへ束縛し、候補順・
worker 数・thread scheduling を stream identity へ混入させない。
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reinbalance_survivors_contracts.canonical_json import canonical_hash

BRANCH_RNG_SCHEMA_VERSION = "survivors.branch_rng.v1"
HORIZON_OUTCOME_SCHEMA_VERSION = "survivors.branch_horizon_outcome.v1"
POLICY_RNG_DOMAIN = "survivors.validation.policy_rng.v1"
OUTCOME_RNG_DOMAIN = "survivors.validation.outcome_rng.v1"

_TRACE_REQUIRED_FIELDS = frozenset(
    {
        "source_identity",
        "episode_logical_id",
        "decision_id",
        "reset_options",
        "task",
        "curriculum",
        "movement_actions",
        "prior_choices",
        "prior_acks",
        "base_obs_sha256",
        "choice_ids",
    }
)
_INVALID_STATUSES = {
    "replay_mismatch": "replay-mismatch",
    "early_death_before_decision": "early-death-before-decision",
    "duplicate_choice": "duplicate-choice",
    "timeout": "timeout",
    "branch_count_missing": "branch-count-missing",
    "recurrent_state_leakage": "recurrent-state-leakage",
}
_ACCEPTED_BRANCH_STATUSES = frozenset({"ok", "early_death"})
_HORIZON_OUTCOME_FIELDS = frozenset(
    {
        "schema_version",
        "decision_id",
        "candidate_id",
        "horizon",
        "replication_key",
        "stream_sha256",
        "policy_rng_mode",
        "status",
        "failure_reason",
        "censored",
        "terminal",
        "stage_cleared_or_survived_horizon",
        "survival_seconds",
        "level_gain",
        "gem_gain",
        "kill_gain",
        "provenance",
    }
)


class BranchRolloutError(ValueError):
    """semantic replay または branch RNG 契約を満たせない場合の例外。

    不完全な trace や stream 衝突を outcome 欠損として黙認せず、quarantine 前に識別する。
    """


@dataclass(frozen=True)
class BranchAssignment:
    """一候補・一 replication の validation-only RNG 割当。

    candidate ID は監査用に保持するが、replication key と stream の導出入力には含めない。
    """

    candidate_id: str
    replication_index: int
    replication_key: str
    stream_sha256: str
    stream_seed: int
    policy_stream_sha256: str
    policy_stream_seed: int
    worker_slot: int
    branch_rng_schema_version: str = BRANCH_RNG_SCHEMA_VERSION

    def to_wire(self) -> dict[str, Any]:
        """canonical JSON へ保存可能な assignment object を返す。

        dataclass の内部表現を wire field へ一方向に変換する。
        """
        return {
            "candidate_id": self.candidate_id,
            "replication_index": self.replication_index,
            "replication_key": self.replication_key,
            "stream_sha256": self.stream_sha256,
            "stream_seed": self.stream_seed,
            "policy_stream_sha256": self.policy_stream_sha256,
            "policy_stream_seed": self.policy_stream_seed,
            "worker_slot": self.worker_slot,
            "branch_rng_schema_version": self.branch_rng_schema_version,
        }


def _require_sha256(value: Any, label: str) -> str:
    """小文字 64 桁 SHA-256 を検証して返す。

    source・observation・stream の全 identity sibling に同じ形式制約を適用する。
    """
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BranchRolloutError(f"{label} must be lowercase sha256")
    return value


def _require_non_empty_string(value: Any, label: str) -> str:
    """空でない文字列を検証して返す。

    logical ID を bool/number へ暗黙変換せず、replay identity の曖昧さを防ぐ。
    """
    if not isinstance(value, str) or not value:
        raise BranchRolloutError(f"{label} must be a non-empty string")
    return value


def validate_complete_semantic_trace(trace: Any) -> None:
    """complete semantic trace の必須 field と exactly-once choice を検証する。

    reset options・task/curriculum・movement・prior choice/ack のいずれも省略を許さない。
    """
    if not isinstance(trace, Mapping):
        raise BranchRolloutError("trace must be an object")
    missing = _TRACE_REQUIRED_FIELDS - set(trace)
    if missing:
        raise BranchRolloutError(f"missing trace field: {sorted(missing)[0]}")
    _require_sha256(trace["source_identity"], "source_identity")
    _require_sha256(trace["base_obs_sha256"], "base_obs_sha256")
    _require_non_empty_string(trace["episode_logical_id"], "episode_logical_id")
    _require_non_empty_string(trace["decision_id"], "decision_id")
    for name in ("reset_options", "task", "curriculum"):
        if not isinstance(trace[name], Mapping):
            raise BranchRolloutError(f"{name} must be an object")
    if not isinstance(trace["movement_actions"], list):
        raise BranchRolloutError("movement_actions must be a list")
    if any(type(action) is not int or action < 0 for action in trace["movement_actions"]):
        raise BranchRolloutError("movement_actions contains an invalid action")
    choice_ids = trace["choice_ids"]
    if (
        not isinstance(choice_ids, list)
        or len(choice_ids) < 2
        or any(not isinstance(item, str) or not item for item in choice_ids)
        or len(set(choice_ids)) != len(choice_ids)
    ):
        raise BranchRolloutError("choice_ids must contain unique choices")

    prior_choices = trace["prior_choices"]
    prior_acks = trace["prior_acks"]
    if not isinstance(prior_choices, list) or not isinstance(prior_acks, list):
        raise BranchRolloutError("prior_choices and prior_acks must be lists")
    seen_decisions: set[str] = set()
    normalized_choices: list[tuple[str, str]] = []
    normalized_acks: list[tuple[str, str]] = []
    for label, rows, destination in (
        ("prior_choices", prior_choices, normalized_choices),
        ("prior_acks", prior_acks, normalized_acks),
    ):
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise BranchRolloutError(f"{label}[{index}] must be an object")
            decision_id = _require_non_empty_string(
                row.get("decision_id"), f"{label}[{index}].decision_id"
            )
            choice_id = _require_non_empty_string(
                row.get("choice_id"), f"{label}[{index}].choice_id"
            )
            destination.append((decision_id, choice_id))
            if label == "prior_choices":
                if decision_id in seen_decisions:
                    raise BranchRolloutError("duplicate-choice in semantic trace")
                seen_decisions.add(decision_id)
    if normalized_choices != normalized_acks:
        raise BranchRolloutError("replay-mismatch between choices and acknowledgements")


def derive_replication_key(
    source_identity: str,
    episode_logical_id: str,
    decision_id: str,
    replication_index: int,
    branch_rng_schema_version: str = BRANCH_RNG_SCHEMA_VERSION,
) -> str:
    """仕様の五要素から candidate 非依存 replication key を導出する。

    SHA-256 実装は共有 canonical JSON 契約だけを利用し、独自 JSON hash を作らない。
    """
    _require_sha256(source_identity, "source_identity")
    _require_non_empty_string(episode_logical_id, "episode_logical_id")
    _require_non_empty_string(decision_id, "decision_id")
    if type(replication_index) is not int or replication_index < 0:
        raise BranchRolloutError("replication_index must be a non-negative integer")
    if branch_rng_schema_version != BRANCH_RNG_SCHEMA_VERSION:
        raise BranchRolloutError("unsupported branch_rng_schema_version")
    return canonical_hash(
        [
            source_identity,
            episode_logical_id,
            decision_id,
            replication_index,
            branch_rng_schema_version,
        ]
    )


def validate_replay_observation(
    trace: Mapping[str, Any],
    replayed: Mapping[str, Any],
) -> None:
    """replay 後の decision/base observation/choice set identity を照合する。

    complete trace が存在しても producer state が一致しない branch は score せず拒否する。
    """
    validate_complete_semantic_trace(trace)
    if not isinstance(replayed, Mapping):
        raise BranchRolloutError("replayed decision must be an object")
    comparisons = {
        "source_identity": trace["source_identity"],
        "episode_logical_id": trace["episode_logical_id"],
        "decision_id": trace["decision_id"],
        "base_obs_sha256": trace["base_obs_sha256"],
    }
    for field, expected in comparisons.items():
        if replayed.get(field) != expected:
            raise BranchRolloutError(f"replay-mismatch in {field}")
    replayed_choices = replayed.get("choice_ids")
    if (
        not isinstance(replayed_choices, list)
        or set(replayed_choices) != set(trace["choice_ids"])
        or len(replayed_choices) != len(trace["choice_ids"])
    ):
        raise BranchRolloutError("replay-mismatch in choice_ids")


def build_horizon_outcome(
    *,
    decision_id: str,
    candidate_id: str,
    horizon: str,
    replication_key: str,
    stream_sha256: str,
    policy_rng_mode: str,
    status: str,
    failure_reason: str | None,
    censored: bool,
    terminal: bool,
    stage_cleared_or_survived_horizon: bool,
    survival_seconds: float,
    level_gain: float,
    gem_gain: float,
    kill_gain: float,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """一 replication の short/full outcome artifact を構築する。

    failed/censored/observed を schema field で区別し、training reward terms は provenance
    にだけ保存する。
    """
    artifact = {
        "schema_version": HORIZON_OUTCOME_SCHEMA_VERSION,
        "decision_id": decision_id,
        "candidate_id": candidate_id,
        "horizon": horizon,
        "replication_key": replication_key,
        "stream_sha256": stream_sha256,
        "policy_rng_mode": policy_rng_mode,
        "status": status,
        "failure_reason": failure_reason,
        "censored": censored,
        "terminal": terminal,
        "stage_cleared_or_survived_horizon": (
            stage_cleared_or_survived_horizon
        ),
        "survival_seconds": survival_seconds,
        "level_gain": level_gain,
        "gem_gain": gem_gain,
        "kill_gain": kill_gain,
        "provenance": dict(provenance),
    }
    validate_horizon_outcome(artifact)
    return artifact


def validate_horizon_outcome(artifact: Any) -> None:
    """branch horizon outcome の exact schema・status・finite metrics を検証する。

    failed row の偽 utility と observed row の failure reason を両方 fail-closed で拒否する。
    """
    if not isinstance(artifact, Mapping):
        raise BranchRolloutError("horizon outcome must be an object")
    actual = frozenset(artifact)
    if actual != _HORIZON_OUTCOME_FIELDS:
        raise BranchRolloutError(
            "horizon outcome fields mismatch: "
            f"missing={sorted(_HORIZON_OUTCOME_FIELDS - actual)}, "
            f"unknown={sorted(actual - _HORIZON_OUTCOME_FIELDS)}"
        )
    if artifact["schema_version"] != HORIZON_OUTCOME_SCHEMA_VERSION:
        raise BranchRolloutError("unsupported horizon outcome schema")
    _require_non_empty_string(artifact["decision_id"], "decision_id")
    _require_non_empty_string(artifact["candidate_id"], "candidate_id")
    _require_sha256(artifact["replication_key"], "replication_key")
    _require_sha256(artifact["stream_sha256"], "stream_sha256")
    if artifact["horizon"] not in {"short", "full"}:
        raise BranchRolloutError("horizon must be short or full")
    if artifact["policy_rng_mode"] not in {
        "deterministic",
        "replication_key_crn",
    }:
        raise BranchRolloutError("unsupported policy_rng_mode")
    if artifact["status"] not in {"observed", "censored", "failed"}:
        raise BranchRolloutError("unsupported outcome status")
    for field in (
        "censored",
        "terminal",
        "stage_cleared_or_survived_horizon",
    ):
        if type(artifact[field]) is not bool:
            raise BranchRolloutError(f"{field} must be bool")
    if artifact["status"] == "failed":
        if (
            not isinstance(artifact["failure_reason"], str)
            or not artifact["failure_reason"]
        ):
            raise BranchRolloutError("failed outcome requires failure_reason")
    elif artifact["failure_reason"] is not None:
        raise BranchRolloutError("non-failed outcome cannot have failure_reason")
    if artifact["censored"] != (artifact["status"] == "censored"):
        raise BranchRolloutError("censored flag/status mismatch")
    for field in (
        "survival_seconds",
        "level_gain",
        "gem_gain",
        "kill_gain",
    ):
        value = artifact[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise BranchRolloutError(f"{field} must be finite and non-negative")
    horizon_limit = 300.0 if artifact["horizon"] == "short" else 1800.0
    if float(artifact["survival_seconds"]) > horizon_limit:
        raise BranchRolloutError("survival_seconds exceeds horizon")
    if not isinstance(artifact["provenance"], Mapping):
        raise BranchRolloutError("provenance must be an object")
    if artifact["status"] == "failed" and (
        artifact["stage_cleared_or_survived_horizon"]
        or any(
            float(artifact[field]) != 0.0
            for field in (
                "survival_seconds",
                "level_gain",
                "gem_gain",
                "kill_gain",
            )
        )
    ):
        raise BranchRolloutError("failed outcome cannot contain scoreable utility")


def _derive_domain_stream(replication_key: str, domain: str) -> tuple[str, int]:
    """replication key から domain-separated stream hash と UE int32 seed を返す。

    policy と outcome の乱数消費が互いの state を進めないよう domain を分離する。
    """
    _require_sha256(replication_key, "replication_key")
    stream_sha256 = canonical_hash(
        {
            "replication_key": replication_key,
            "domain": domain,
            "branch_rng_schema_version": BRANCH_RNG_SCHEMA_VERSION,
        }
    )
    seed = int(stream_sha256[:8], 16) & 0x7FFFFFFF
    return stream_sha256, seed


def build_branch_assignments(
    trace: Mapping[str, Any],
    candidate_ids: Sequence[str],
    replication_count: int,
    *,
    worker_count: int = 1,
) -> list[BranchAssignment]:
    """全 candidate×replication の stable RNG assignment を構築する。

    execution order と worker slot は scheduling 情報に留め、stream 導出後にだけ割り当てる。
    """
    validate_complete_semantic_trace(trace)
    if (
        isinstance(candidate_ids, (str, bytes))
        or len(candidate_ids) < 2
        or any(not isinstance(item, str) or not item for item in candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise BranchRolloutError("candidate_ids must contain unique choices")
    if set(candidate_ids) != set(trace["choice_ids"]):
        raise BranchRolloutError("replay-mismatch in candidate choice set")
    if type(replication_count) is not int or replication_count <= 0:
        raise BranchRolloutError("replication_count must be positive")
    if type(worker_count) is not int or worker_count <= 0:
        raise BranchRolloutError("worker_count must be positive")

    streams: dict[int, tuple[str, str, int, str, int]] = {}
    for replication_index in range(replication_count):
        key = derive_replication_key(
            trace["source_identity"],
            trace["episode_logical_id"],
            trace["decision_id"],
            replication_index,
        )
        stream_sha256, stream_seed = _derive_domain_stream(key, OUTCOME_RNG_DOMAIN)
        policy_sha256, policy_seed = _derive_domain_stream(key, POLICY_RNG_DOMAIN)
        streams[replication_index] = (
            key,
            stream_sha256,
            stream_seed,
            policy_sha256,
            policy_seed,
        )

    assignments: list[BranchAssignment] = []
    for ordinal, candidate_id in enumerate(candidate_ids):
        for replication_index in range(replication_count):
            key, stream_sha256, stream_seed, policy_sha256, policy_seed = streams[
                replication_index
            ]
            assignments.append(
                BranchAssignment(
                    candidate_id=candidate_id,
                    replication_index=replication_index,
                    replication_key=key,
                    stream_sha256=stream_sha256,
                    stream_seed=stream_seed,
                    policy_stream_sha256=policy_sha256,
                    policy_stream_seed=policy_seed,
                    worker_slot=ordinal % worker_count,
                )
            )
    return assignments


def execute_paired_rollouts(
    trace: Mapping[str, Any],
    candidate_ids: Sequence[str],
    replication_count: int,
    execute_branch: Callable[
        [Mapping[str, Any], str, BranchAssignment], Mapping[str, Any]
    ],
    *,
    worker_count: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """transport 非依存 callback で paired branches を実行し seam verdict を返す。

    callback は replay/timeout/early-death を status として返し、本関数が identity を封印する。
    """
    assignments = build_branch_assignments(
        trace,
        candidate_ids,
        replication_count,
        worker_count=worker_count,
    )
    records: list[dict[str, Any]] = []
    for assignment in assignments:
        try:
            result = dict(
                execute_branch(trace, assignment.candidate_id, assignment)
            )
        except TimeoutError:
            result = {"status": "timeout"}
        except BranchRolloutError as exc:
            message = str(exc)
            status = (
                "recurrent_state_leakage"
                if "recurrent" in message
                else "replay_mismatch"
            )
            result = {"status": status, "error": message}
        sealed = {
            **result,
            **assignment.to_wire(),
            "decision_id": trace["decision_id"],
            "expected_branch_count": len(candidate_ids),
        }
        records.append(sealed)
    accepted, quarantined = quarantine_branch_records(records)
    seam = validate_rng_seam(accepted) if accepted else {
        "passed": False,
        "blocking_reasons": ["no-effective-branches"],
        "zero_outcome_variance": False,
        "effective_replications": 0,
    }
    return accepted, quarantined, seam


def quarantine_branch_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """異常 status・重複 choice・branch count 不足を理由付きで隔離する。

    quarantine row は outcome の有無に関係なく scoring collection から除外する。
    """
    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    seen_rows: set[tuple[Any, ...]] = set()
    for raw in records:
        row = dict(raw)
        status = row.get("status", "ok")
        reason = _INVALID_STATUSES.get(str(status))
        if reason is None and status not in _ACCEPTED_BRANCH_STATUSES:
            reason = "unknown-status"
        identity = (
            row.get("decision_id"),
            row.get("candidate_id"),
            row.get("replication_key"),
        )
        if identity in seen_rows:
            reason = "duplicate-choice"
        seen_rows.add(identity)
        if reason:
            row["quarantine_reason"] = reason
            quarantined.append(row)
        else:
            accepted.append(row)

    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        grouped[(row.get("decision_id"), row.get("replication_key"))].append(row)
    missing_ids: set[int] = set()
    for group in grouped.values():
        expected = {row.get("expected_branch_count") for row in group}
        if len(expected) != 1 or next(iter(expected), None) != len(group):
            missing_ids.update(id(row) for row in group)
    if missing_ids:
        kept: list[dict[str, Any]] = []
        for row in accepted:
            if id(row) in missing_ids:
                row["quarantine_reason"] = "branch-count-missing"
                quarantined.append(row)
            else:
                kept.append(row)
        accepted = kept
    return accepted, quarantined


def effective_replication_count(records: Sequence[Mapping[str, Any]]) -> int:
    """unique decision/stream pair の数を返す。

    replication key が異なっても同じ stream へ衝突した replay は一標本だけに数える。
    """
    identities: set[tuple[Any, Any]] = set()
    for row in records:
        if row.get("status", "ok") not in _ACCEPTED_BRANCH_STATUSES:
            continue
        identities.add(
            (
                row.get("decision_id"),
                row.get("stream_sha256"),
            )
        )
    return len(identities)


def validate_rng_seam(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """stream identity/mapping/draw trace から RNG seam 成立を判定する。

    zero outcome variance は診断値に留め、stream collision や CRN 不一致だけを block する。
    """
    blocking: set[str] = set()
    by_replication: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    by_decision: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    outcomes: list[float] = []
    for row in records:
        if row.get("status", "ok") not in _ACCEPTED_BRANCH_STATUSES:
            blocking.add("quarantined-branch-present")
            continue
        try:
            _require_sha256(row.get("replication_key"), "replication_key")
            _require_sha256(row.get("stream_sha256"), "stream_sha256")
            _require_sha256(row.get("draw_trace_sha256"), "draw_trace_sha256")
        except BranchRolloutError:
            blocking.add("invalid-stream-identity")
            continue
        key = (row.get("decision_id"), row.get("replication_index"))
        by_replication[key].append(row)
        by_decision[row.get("decision_id")].append(row)
        outcome = row.get("outcome_utility")
        if (
            not isinstance(outcome, bool)
            and isinstance(outcome, (int, float))
            and math.isfinite(float(outcome))
        ):
            outcomes.append(float(outcome))

    for group in by_replication.values():
        expected = {row.get("expected_branch_count") for row in group}
        candidate_ids = [row.get("candidate_id") for row in group]
        if (
            len(expected) != 1
            or next(iter(expected), None) != len(group)
            or len(set(candidate_ids)) != len(candidate_ids)
        ):
            blocking.add("branch-count-missing")
        if len({row.get("replication_key") for row in group}) != 1:
            blocking.add("replication-key-mismatch")
        if len({row.get("stream_sha256") for row in group}) != 1:
            blocking.add("candidate-crn-mismatch")
        if len({row.get("draw_trace_sha256") for row in group}) != 1:
            blocking.add("draw-trace-mismatch")
        policy_states = {
            row.get("policy_state_before_sha256")
            for row in group
            if row.get("policy_state_before_sha256") is not None
        }
        if len(policy_states) > 1:
            blocking.add("recurrent-state-leakage")

    for decision_rows in by_decision.values():
        replication_to_key: dict[Any, Any] = {}
        replication_to_stream: dict[Any, Any] = {}
        for row in decision_rows:
            replication = row.get("replication_index")
            replication_to_key.setdefault(replication, row.get("replication_key"))
            replication_to_stream.setdefault(replication, row.get("stream_sha256"))
        if len(set(replication_to_key.values())) != len(replication_to_key):
            blocking.add("replication-key-collision")
        if len(set(replication_to_stream.values())) != len(replication_to_stream):
            blocking.add("stream-collision")

    zero_variance = (
        bool(outcomes)
        and max(outcomes) == min(outcomes)
    )
    return {
        "passed": not blocking and bool(by_replication),
        "blocking_reasons": sorted(blocking),
        "zero_outcome_variance": zero_variance,
        "effective_replications": effective_replication_count(records),
    }

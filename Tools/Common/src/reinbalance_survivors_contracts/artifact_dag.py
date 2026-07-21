"""Artifact identity DAG validation for Survivors release artifacts."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .artifact_identity import ArtifactDescriptor
from .ui_intent import ContractValidationError, ensure

__all__ = [
    "ArtifactDagValidationError",
    "ArtifactDagReport",
    "ALLOWED_PARENT_KINDS",
    "validate_artifact_dag",
]


class ArtifactDagValidationError(ContractValidationError):
    """Raised when artifact descriptors do not form an allowed immutable DAG."""


ALLOWED_PARENT_KINDS: dict[str, frozenset[str]] = {
    "source_descriptor": frozenset(),
    "teacher_validation_verdict": frozenset({"source_descriptor"}),
    "choice_dataset_release": frozenset({"teacher_validation_verdict"}),
    "item_selector_release": frozenset({"choice_dataset_release"}),
    "combat_student_release": frozenset({"choice_dataset_release"}),
    "runtime_bundle": frozenset({"item_selector_release", "combat_student_release"}),
    "replay_shadow_verdict": frozenset({"runtime_bundle"}),
    "canary_campaign": frozenset({"replay_shadow_verdict"}),
    "goal_evidence": frozenset({"canary_campaign", "restore_test_verdict"}),
    "restore_test_verdict": frozenset(
        {"runtime_bundle", "canary_campaign", "goal_evidence"}
    ),
}


_KIND_ORDER = {
    "source_descriptor": 0,
    "teacher_validation_verdict": 1,
    "choice_dataset_release": 2,
    "item_selector_release": 3,
    "combat_student_release": 3,
    "runtime_bundle": 4,
    "replay_shadow_verdict": 5,
    "canary_campaign": 6,
    "restore_test_verdict": 7,
    "goal_evidence": 8,
}


@dataclass(frozen=True)
class ArtifactDagReport:
    node_count: int
    topological_identity_hashes: tuple[str, ...]


def _coerce_descriptor(value: ArtifactDescriptor | Mapping[str, Any]) -> ArtifactDescriptor:
    try:
        if isinstance(value, ArtifactDescriptor):
            return value
        return ArtifactDescriptor.from_wire(value)
    except ContractValidationError as exc:
        raise ArtifactDagValidationError(str(exc)) from exc


def _sort_key(descriptor: ArtifactDescriptor) -> tuple[int, str, str]:
    return (
        _KIND_ORDER.get(descriptor.node_kind, 999),
        descriptor.logical_id,
        descriptor.identity_hash,
    )


def validate_artifact_dag(
    descriptors: Sequence[ArtifactDescriptor | Mapping[str, Any]],
) -> ArtifactDagReport:
    """Validate descriptor identity, parent references, and conceptual DAG edges."""
    nodes = tuple(_coerce_descriptor(value) for value in descriptors)
    identity_to_node: dict[str, ArtifactDescriptor] = {}
    logical_to_hash: dict[str, str] = {}

    for node in nodes:
        if node.node_kind not in ALLOWED_PARENT_KINDS:
            raise ArtifactDagValidationError(f"unsupported node_kind {node.node_kind!r}")
        existing_hash = logical_to_hash.get(node.logical_id)
        if existing_hash is not None and existing_hash != node.identity_hash:
            raise ArtifactDagValidationError(
                f"logical_id {node.logical_id!r} has multiple identity hashes"
            )
        logical_to_hash[node.logical_id] = node.identity_hash
        existing = identity_to_node.get(node.identity_hash)
        if existing is not None and existing.to_wire() != node.to_wire():
            raise ArtifactDagValidationError(
                f"identity_hash {node.identity_hash} has conflicting descriptors"
            )
        identity_to_node[node.identity_hash] = node

    children: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {node.identity_hash: 0 for node in identity_to_node.values()}

    for node in identity_to_node.values():
        allowed_parent_kinds = ALLOWED_PARENT_KINDS[node.node_kind]
        if not allowed_parent_kinds and node.parents:
            raise ArtifactDagValidationError(
                f"{node.node_kind} must not declare parent refs"
            )
        for parent_ref in node.parents:
            if parent_ref.identity_hash == node.identity_hash:
                raise ArtifactDagValidationError(
                    f"{node.logical_id!r} self-references its own identity hash"
                )
            parent = identity_to_node.get(parent_ref.identity_hash)
            if parent is None:
                raise ArtifactDagValidationError(
                    f"{node.logical_id!r} references missing parent {parent_ref.identity_hash}"
                )
            if (
                parent.logical_id != parent_ref.logical_id
                or parent.node_kind != parent_ref.node_kind
            ):
                raise ArtifactDagValidationError(
                    f"{node.logical_id!r} parent ref metadata does not match parent descriptor"
                )
            if parent.node_kind not in allowed_parent_kinds:
                raise ArtifactDagValidationError(
                    f"{node.node_kind} cannot use {parent.node_kind} as a parent"
                )
            children[parent.identity_hash].append(node.identity_hash)
            indegree[node.identity_hash] += 1

    ordered_zero_indegree = sorted(
        (identity_to_node[hash_] for hash_, degree in indegree.items() if degree == 0),
        key=_sort_key,
    )
    queue = deque(node.identity_hash for node in ordered_zero_indegree)
    topological: list[str] = []

    while queue:
        current = queue.popleft()
        topological.append(current)
        next_nodes = sorted(
            (identity_to_node[child] for child in children[current]),
            key=_sort_key,
        )
        for child in next_nodes:
            indegree[child.identity_hash] -= 1
            ensure(indegree[child.identity_hash] >= 0, "invalid DAG indegree state")
            if indegree[child.identity_hash] == 0:
                queue.append(child.identity_hash)

    if len(topological) != len(identity_to_node):
        raise ArtifactDagValidationError("artifact descriptors contain a cycle")

    return ArtifactDagReport(
        node_count=len(identity_to_node),
        topological_identity_hashes=tuple(topological),
    )

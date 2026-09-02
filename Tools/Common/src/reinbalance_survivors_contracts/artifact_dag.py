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
    "validate_formal_runtime_dag",
]


class ArtifactDagValidationError(ContractValidationError):
    """Raised when artifact descriptors do not form an allowed immutable DAG."""


ALLOWED_PARENT_KINDS: dict[str, frozenset[str]] = {
    "source_descriptor": frozenset(),
    "teacher_validation_verdict": frozenset({"source_descriptor"}),
    "choice_dataset_release": frozenset({"teacher_validation_verdict"}),
    "item_selector_release": frozenset({"choice_dataset_release"}),
    "combat_student_release": frozenset({"choice_dataset_release"}),
    # perception calibration は source_descriptor（capture dataset）を親に持つ
    "perception_calibration_profile": frozenset({"source_descriptor"}),
    # perception final verdict は calibration profile を必須親とする
    "perception_final_verdict": frozenset({"perception_calibration_profile"}),
    "runtime_bundle": frozenset(
        {"item_selector_release", "combat_student_release", "perception_final_verdict"}
    ),
    "replay_shadow_verdict": frozenset({"runtime_bundle"}),
    "canary_campaign": frozenset({"replay_shadow_verdict"}),
    "goal_evidence": frozenset({"canary_campaign", "restore_test_verdict"}),
    "restore_test_verdict": frozenset(
        {"runtime_bundle", "canary_campaign", "goal_evidence"}
    ),
}


_REQUIRED_PARENT_KINDS: dict[str, frozenset[str]] = {
    "goal_evidence": frozenset({"canary_campaign", "restore_test_verdict"}),
    "perception_final_verdict": frozenset({"perception_calibration_profile"}),
}

_FORMAL_RUNTIME_PARENT_KINDS = frozenset(
    {"item_selector_release", "combat_student_release", "perception_final_verdict"}
)

# perception_error_fit._HASH_FIELDS と同一の wire-level formal 契約。
# Common consumer は Deployment producer を import できないため、境界側でも exact
# vocabulary を固定し、欠落・追加・自己申告の非 SHA-256 値を fail-closed にする。
_FORMAL_PERCEPTION_SUBJECT_HASH_FIELDS = frozenset({
    "parser_artifact_hash",
    "detector_artifact_hash",
    "model_hash",
    "build_hash",
    "assembler_schema_hash",
    "ui_presentation_schema_hash",
    "ui_presentation_golden_fixture_hash",
    "config_hash",
    "capture_dataset_hash",
    "calibration_profile_hash",
    "threshold_hash",
    "atlas_vocabulary_hash",
    "assembler_impl_hash",
    "roi_resolver_input_hash",
    "benchmark_fit_code_hash",
    "lineage_seal_hash",
})


_KIND_ORDER = {
    "source_descriptor": 0,
    "teacher_validation_verdict": 1,
    "perception_calibration_profile": 1,
    "choice_dataset_release": 2,
    "perception_final_verdict": 2,
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
        parent_node_kinds: set[str] = set()
        if node.node_kind == "source_descriptor":
            if node.parents:
                raise ArtifactDagValidationError(
                    f"{node.node_kind} must not declare parent refs"
                )
        elif not node.parents:
            raise ArtifactDagValidationError(
                f"{node.node_kind} must declare at least one parent ref"
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
            parent_node_kinds.add(parent.node_kind)
            children[parent.identity_hash].append(node.identity_hash)
            indegree[node.identity_hash] += 1
        missing_required = (
            _REQUIRED_PARENT_KINDS.get(node.node_kind, frozenset()) - parent_node_kinds
        )
        if missing_required:
            required = ", ".join(sorted(missing_required))
            raise ArtifactDagValidationError(
                f"{node.node_kind} must declare at least one parent of kind {required}"
            )

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


def validate_formal_runtime_dag(
    descriptors: Sequence[ArtifactDescriptor | Mapping[str, Any]],
) -> ArtifactDagReport:
    """formal runtime bundle の parent と perception verdict 内容を検証する。

    汎用 DAG validator は development bundle の部分 parent を許容する。formal
    publish/restore 境界では本 validator を追加で通し、model 二種と exact final
    verdict、passed/development/subject binding を全て必須にする。
    """
    nodes = tuple(_coerce_descriptor(value) for value in descriptors)
    report = validate_artifact_dag(nodes)
    by_identity = {node.identity_hash: node for node in nodes}
    runtimes = [node for node in nodes if node.node_kind == "runtime_bundle"]
    if not runtimes:
        raise ArtifactDagValidationError("formal runtime DAG requires runtime_bundle")
    for runtime in runtimes:
        parents = [by_identity[parent.identity_hash] for parent in runtime.parents]
        parent_kinds = {parent.node_kind for parent in parents}
        missing = _FORMAL_RUNTIME_PARENT_KINDS - parent_kinds
        if missing:
            raise ArtifactDagValidationError(
                "formal runtime_bundle missing required parent kind(s): "
                + ", ".join(sorted(missing))
            )
        verdicts = [
            parent for parent in parents
            if parent.node_kind == "perception_final_verdict"
        ]
        if len(verdicts) != 1:
            raise ArtifactDagValidationError(
                "formal runtime_bundle requires exactly one perception_final_verdict"
            )
        verdict_metadata = verdicts[0].identity_metadata
        if (
            verdict_metadata.get("passed") is not True
            or verdict_metadata.get("development_only") is not False
        ):
            raise ArtifactDagValidationError(
                "perception_final_verdict must be passed and production-only"
            )
        verdict_subjects = verdict_metadata.get("subject_hashes")
        runtime_subjects = runtime.identity_metadata.get("perception_subject_hashes")
        if (
            not isinstance(verdict_subjects, Mapping)
            or set(verdict_subjects) != _FORMAL_PERCEPTION_SUBJECT_HASH_FIELDS
        ):
            raise ArtifactDagValidationError(
                "perception_final_verdict subject_hashes must exactly match the formal contract"
            )
        if not all(
            type(value) is str
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in verdict_subjects.values()
        ):
            raise ArtifactDagValidationError(
                "perception_final_verdict subject_hashes must be lowercase SHA-256 values"
            )
        if (
            not isinstance(runtime_subjects, Mapping)
            or dict(verdict_subjects) != dict(runtime_subjects)
        ):
            raise ArtifactDagValidationError(
                "runtime perception subject hashes do not match final verdict"
            )
    return report

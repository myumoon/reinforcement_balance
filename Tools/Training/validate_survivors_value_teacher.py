"""paired rollout corpus から Survivors teacher release verdict を生成する CLI。

split freeze、development-only scale/calibration fit、untouched final test 評価を順に行い、
固定 gate 未達を override できない canonical artifacts として保存する。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes
from reinbalance_survivors_contracts.fidelity_verdict import (
    FidelityVerdict,
    downstream_release_allowed,
)

from games.survivors.choice_branch_rollout import (
    effective_replication_count,
    quarantine_branch_records,
    validate_rng_seam,
)
from games.survivors.teacher_reliability import fit_teacher_reliability
from games.survivors.teacher_score_scale import (
    commit_teacher_score_scale,
    fit_teacher_score_scale,
    transform_teacher_scores,
)
from games.survivors.teacher_validation import (
    OutcomeNormalizer,
    evaluate_teacher,
    make_label_release_verdict,
)
from games.survivors.teacher_validation_split import (
    commit_frozen_episode_split,
    freeze_episode_split,
)
from games.survivors.value_source_descriptor import (
    ValueSourceDescriptorError,
    validate_value_source_descriptor,
)


class TeacherPipelineError(ValueError):
    """CLI input lineage または full pipeline 契約違反を表す。

    source mismatch・final fit 混入・current fidelity 欠落を verdict 前に識別する。
    """


def _load_json(path: Path, label: str) -> dict[str, Any]:
    """JSON object を file から読み取り、parse/type error を契約例外へ変換する。

    読み込んだ artifact は後段の各 schema validator で内容 identity まで検証する。
    """
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TeacherPipelineError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TeacherPipelineError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    """小文字 64 桁 SHA-256 を検証して返す。

    source・teacher・fidelity identity の binding field 全てへ同じ検証を使う。
    """
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TeacherPipelineError(f"{label} must be lowercase sha256")
    return value


def _write_canonical(path: Path, value: Mapping[str, Any]) -> None:
    """canonical JSON を同一 directory の一時 file 経由で atomic replace する。

    中断時に半端な verdict を残さず、同じ論理値は byte-identical に保存する。
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    encoded = canonical_json_bytes(dict(value)) + b"\n"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o644,
    )
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("short canonical artifact write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)


def _extract_identity(
    artifact: Mapping[str, Any],
    candidates: Sequence[str],
    label: str,
) -> str:
    """artifact の許可 field から一意な content identity を取得する。

    field 不在時に artifact 全体を勝手に再hashせず、producer の binding を必須にする。
    """
    values = [artifact.get(field) for field in candidates if artifact.get(field)]
    if len(values) != 1:
        raise TeacherPipelineError(f"{label} must expose exactly one identity field")
    return _require_sha256(values[0], f"{label} identity")


def _partition_decisions(
    decisions: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """decision rows を frozen episode assignment へ束縛して複製する。

    corpus が自己申告した partition は信用せず、freeze artifact の値で上書きする。
    """
    assignments = split["episode_partitions"]
    partitions = {
        "development_train": [],
        "development_validation": [],
        "final_test": [],
    }
    seen_decisions: set[str] = set()
    for row in decisions:
        if not isinstance(row, Mapping):
            raise TeacherPipelineError("decision must be an object")
        decision_id = row.get("decision_id")
        episode_id = row.get("episode_id")
        if (
            not isinstance(decision_id, str)
            or not decision_id
            or decision_id in seen_decisions
        ):
            raise TeacherPipelineError("decision IDs must be unique")
        seen_decisions.add(decision_id)
        if episode_id not in assignments:
            raise TeacherPipelineError("decision references an unknown episode")
        partition = assignments[episode_id]
        copied = copy.deepcopy(dict(row))
        copied["partition"] = partition
        partitions[partition].append(copied)
    return partitions


def _score_difference_refs(
    decisions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """development decisions の全 candidate pair から score-difference refs を作る。

    candidate IDs を lineage ref に含めるが、scale fit へ final partition を渡さない。
    """
    refs: list[dict[str, Any]] = []
    for decision in decisions:
        partition = decision["partition"]
        if partition not in {
            "development_train",
            "development_validation",
        }:
            raise TeacherPipelineError("final decision reached score scale fit")
        candidates = decision.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise TeacherPipelineError("decision requires at least two candidates")
        for left, right in combinations(candidates, 2):
            left_score = left.get("teacher_score")
            right_score = right.get("teacher_score")
            if (
                isinstance(left_score, bool)
                or not isinstance(left_score, (int, float))
                or isinstance(right_score, bool)
                or not isinstance(right_score, (int, float))
            ):
                raise TeacherPipelineError("teacher scores must be numeric")
            refs.append(
                {
                    "ref_id": (
                        f"{decision['decision_id']}|{left.get('choice_id')}"
                        f"|{right.get('choice_id')}"
                    ),
                    "partition": partition,
                    "difference": float(left_score) - float(right_score),
                }
            )
    return refs


def _apply_teacher_scale(
    decisions: Sequence[Mapping[str, Any]],
    score_scale: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """各 decision の raw teacher scores を committed scale の z 値へ変換する。

    raw score は provenance として保持し、validation metrics は teacher_score_z を優先する。
    """
    transformed_decisions: list[dict[str, Any]] = []
    for decision in decisions:
        copied = copy.deepcopy(dict(decision))
        candidates = copied["candidates"]
        transformed = transform_teacher_scores(
            [candidate.get("teacher_score") for candidate in candidates],
            score_scale,
        )
        for candidate, score_z in zip(candidates, transformed):
            candidate["teacher_score_z"] = score_z
        transformed_decisions.append(copied)
    return transformed_decisions


def _reliability_rows(
    decisions: Sequence[Mapping[str, Any]],
    *,
    teacher_type: str,
    teacher_identity: str,
    score_scale_identity: str,
    normalizers: Mapping[str, OutcomeNormalizer],
) -> list[dict[str, Any]]:
    """scaled development decisions を pairwise residual rows へ展開する。

    short/full outcome refs と別々の normalization identity を全 pair に保存する。
    """
    rows: list[dict[str, Any]] = []
    for decision in decisions:
        if decision["partition"] not in {
            "development_train",
            "development_validation",
        }:
            raise TeacherPipelineError("final decision reached reliability fit")
        pairs: list[dict[str, Any]] = []
        candidates = decision["candidates"]
        for left, right in combinations(candidates, 2):
            outcome_margins: dict[str, float] = {}
            outcome_refs: dict[str, str] = {}
            for horizon in ("short", "full"):
                left_outcome = left[horizon]
                right_outcome = right[horizon]
                outcome_margins[horizon] = (
                    normalizers[horizon].utility(left_outcome)
                    - normalizers[horizon].utility(right_outcome)
                )
                outcome_refs[horizon] = str(
                    left_outcome.get(
                        "outcome_ref",
                        f"{decision['decision_id']}|{left['choice_id']}|{horizon}",
                    )
                ) + "::" + str(
                    right_outcome.get(
                        "outcome_ref",
                        f"{decision['decision_id']}|{right['choice_id']}|{horizon}",
                    )
                )
            pairs.append(
                {
                    "candidate_i": left["choice_id"],
                    "candidate_j": right["choice_id"],
                    "teacher_margin_z": (
                        float(left["teacher_score_z"])
                        - float(right["teacher_score_z"])
                    ),
                    "outcome_margin_z": outcome_margins,
                    "outcome_refs": outcome_refs,
                }
            )
        rows.append(
            {
                "decision_id": decision["decision_id"],
                "episode_id": decision["episode_id"],
                "seed_cluster_id": decision.get(
                    "seed_cluster_id", decision["episode_id"]
                ),
                "partition": decision["partition"],
                "teacher_type": teacher_type,
                "teacher_identity": teacher_identity,
                "choice_kind": decision["choice_kind"],
                "elapsed_seconds": decision["elapsed_seconds"],
                "teacher_score_scale_id": score_scale_identity,
                "outcome_scale_ids": {
                    horizon: normalizers[horizon].normalization_identity
                    for horizon in ("short", "full")
                },
                "pairs": pairs,
            }
        )
    return rows


def _expected_branch_lineage(
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, str]]]:
    """corpus decisions から candidate・short/full outcome lineage を列挙する。

    明示 outcome_ref がない旧 corpus も reliability と同じ決定的 fallback ref へ束縛する。
    """
    expected: dict[str, dict[str, dict[str, str]]] = {}
    for decision in decisions:
        decision_id = decision.get("decision_id")
        candidates = decision.get("candidates")
        if (
            not isinstance(decision_id, str)
            or not decision_id
            or decision_id in expected
            or not isinstance(candidates, list)
            or len(candidates) < 2
        ):
            raise TeacherPipelineError("branch lineage decisions are invalid")
        candidate_refs: dict[str, dict[str, str]] = {}
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise TeacherPipelineError("branch lineage candidate is invalid")
            choice_id = candidate.get("choice_id")
            if (
                not isinstance(choice_id, str)
                or not choice_id
                or choice_id in candidate_refs
            ):
                raise TeacherPipelineError("branch lineage choice IDs must be unique")
            horizon_refs: dict[str, str] = {}
            for horizon in ("short", "full"):
                outcome = candidate.get(horizon)
                if not isinstance(outcome, Mapping):
                    raise TeacherPipelineError(
                        "branch lineage outcomes are required"
                    )
                outcome_ref = outcome.get(
                    "outcome_ref",
                    f"{decision_id}|{choice_id}|{horizon}",
                )
                if not isinstance(outcome_ref, str) or not outcome_ref:
                    raise TeacherPipelineError(
                        "branch lineage outcome_ref must be non-empty"
                    )
                horizon_refs[horizon] = outcome_ref
            candidate_refs[choice_id] = horizon_refs
        expected[decision_id] = candidate_refs
    return expected


def _validate_branch_evidence(
    decisions: Sequence[Mapping[str, Any]],
    branch_records: Any,
) -> dict[str, Any]:
    """RNG seam evidence の非空性・coverage・outcome lineage を fail-closed 検証する。

    欠落・全 quarantine・部分 decision/candidate/replication は blocker として verdict へ渡す。
    """
    blocking: set[str] = set()
    empty_seam = {
        "passed": False,
        "blocking_reasons": ["no-effective-branches"],
        "zero_outcome_variance": False,
        "effective_replications": 0,
    }
    if not isinstance(branch_records, list) or not branch_records:
        return {
            "accepted_branch_count": 0,
            "quarantine_count": 0,
            "effective_replications": 0,
            "seam": empty_seam,
            "blocking_reasons": ["missing-branch-records"],
        }
    if any(not isinstance(row, Mapping) for row in branch_records):
        return {
            "accepted_branch_count": 0,
            "quarantine_count": len(branch_records),
            "effective_replications": 0,
            "seam": empty_seam,
            "blocking_reasons": ["malformed-branch-record"],
        }

    expected = _expected_branch_lineage(decisions)
    accepted, quarantined = quarantine_branch_records(branch_records)
    seam = validate_rng_seam(accepted) if accepted else empty_seam
    blocking.update(seam["blocking_reasons"])
    effective = effective_replication_count(accepted)
    if effective == 0:
        blocking.add("no-effective-branches")

    covered: dict[tuple[str, int], set[str]] = {}
    covered_decisions: set[str] = set()
    for row in accepted:
        decision_id = row.get("decision_id")
        candidate_id = row.get("candidate_id")
        replication_index = row.get("replication_index")
        if (
            decision_id not in expected
            or candidate_id not in expected.get(decision_id, {})
        ):
            blocking.add("decision-candidate-lineage-mismatch")
            continue
        if (
            isinstance(replication_index, bool)
            or not isinstance(replication_index, int)
            or replication_index < 0
        ):
            blocking.add("replication-coverage-incomplete")
            continue
        outcome_refs = row.get("outcome_refs")
        if (
            not isinstance(outcome_refs, Mapping)
            or set(outcome_refs) != {"short", "full"}
            or dict(outcome_refs) != expected[decision_id][candidate_id]
        ):
            blocking.add("outcome-lineage-mismatch")
        covered.setdefault((decision_id, replication_index), set()).add(
            candidate_id
        )
        covered_decisions.add(decision_id)

    if covered_decisions != set(expected):
        blocking.add("decision-coverage-incomplete")
    for decision_id, candidate_refs in expected.items():
        replication_groups = [
            candidate_ids
            for (group_decision_id, _), candidate_ids in covered.items()
            if group_decision_id == decision_id
        ]
        if (
            not replication_groups
            or any(group != set(candidate_refs) for group in replication_groups)
        ):
            blocking.add("candidate-replication-coverage-incomplete")

    return {
        "accepted_branch_count": len(accepted),
        "quarantine_count": len(quarantined),
        "effective_replications": effective,
        "seam": seam,
        "blocking_reasons": sorted(blocking),
    }


def run_validation_pipeline(
    *,
    corpus: Mapping[str, Any],
    source_descriptor: Mapping[str, Any],
    integration_fidelity_verdict: Mapping[str, Any],
    split_seed: str = "phase6-paired-rollout-teacher-validation",
    bootstrap_seed: int = 20260718,
    bootstrap_resamples: int = 2000,
) -> dict[str, dict[str, Any]]:
    """入力 artifacts から split/scale/reliability/report/verdict を一方向生成する。

    final decisions は development method/calibration 完了後にだけ開き、gate は固定値を使う。
    """
    if (
        corpus.get("schema_version")
        != "survivors.paired_rollout_corpus.v1"
    ):
        raise TeacherPipelineError("unsupported paired rollout corpus")
    try:
        validate_value_source_descriptor(source_descriptor)
    except (TypeError, ValueSourceDescriptorError) as exc:
        raise TeacherPipelineError(f"source descriptor is invalid: {exc}") from exc
    if source_descriptor.get("ready_for_probe") is not True:
        raise TeacherPipelineError("source descriptor is not probe-ready")
    source_identity = _extract_identity(
        source_descriptor, ("identity_sha256",), "source descriptor"
    )
    if corpus.get("source_identity") != source_identity:
        raise TeacherPipelineError("corpus/source descriptor identity mismatch")
    teacher_type = corpus.get("teacher_type")
    if teacher_type not in {"raw_critic", "overlay", "source_critic"}:
        raise TeacherPipelineError("unsupported teacher_type")
    teacher_identity = _require_sha256(
        corpus.get("teacher_identity"), "teacher_identity"
    )
    current_producer_hashes = corpus.get("current_gating_producer_hashes")
    if not isinstance(current_producer_hashes, Mapping):
        raise TeacherPipelineError(
            "corpus must bind current_gating_producer_hashes"
        )
    try:
        checked_fidelity = FidelityVerdict.from_wire(
            integration_fidelity_verdict
        )
        fidelity_passed = downstream_release_allowed(
            checked_fidelity,
            current_producer_hashes,
            "integration",
        )
    except (TypeError, ValueError) as exc:
        raise TeacherPipelineError(
            f"integration fidelity verdict is not current: {exc}"
        ) from exc
    fidelity_identity = checked_fidelity.identity_hash

    decisions = corpus.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise TeacherPipelineError("corpus decisions are required")
    episode_ids = sorted(
        {
            decision.get("episode_id")
            for decision in decisions
            if isinstance(decision, Mapping)
            and isinstance(decision.get("episode_id"), str)
            and decision.get("episode_id")
        }
    )
    if len(episode_ids) < 3:
        raise TeacherPipelineError("at least three source episodes are required")
    split = freeze_episode_split(episode_ids, seed=split_seed)
    partitions = _partition_decisions(decisions, split)
    if not all(partitions.values()):
        raise TeacherPipelineError("all frozen partitions require decisions")

    normalizers = {
        horizon: OutcomeNormalizer.fit(
            partitions["development_train"],
            horizon,
            partition_id=split["split_identity"],
        )
        for horizon in ("short", "full")
    }
    development = (
        partitions["development_train"]
        + partitions["development_validation"]
    )
    score_scale = fit_teacher_score_scale(
        teacher_type=teacher_type,
        teacher_identity=teacher_identity,
        development_fit_partition_id=split["split_identity"],
        score_difference_refs=_score_difference_refs(development),
    )
    scaled_development = _apply_teacher_scale(development, score_scale)
    reliability = fit_teacher_reliability(
        _reliability_rows(
            scaled_development,
            teacher_type=teacher_type,
            teacher_identity=teacher_identity,
            score_scale_identity=score_scale["scale_identity"],
            normalizers=normalizers,
        ),
        development_split_id=split["split_identity"],
        integration_fidelity_identity=fidelity_identity,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
    )

    release_blockers: list[str] = []
    if not fidelity_passed:
        release_blockers.append("current-integration-fidelity-not-pass")
    release_blockers.extend(
        f"reliability:{item['slice_key']}"
        for item in reliability["slices"]
        if item["release_blocker"]
    )
    branch_evidence = _validate_branch_evidence(
        decisions,
        corpus.get("branch_records"),
    )
    quarantine_count = branch_evidence["quarantine_count"]
    release_blockers.extend(
        f"rng-evidence:{reason}"
        for reason in branch_evidence["blocking_reasons"]
    )

    scaled_final = _apply_teacher_scale(partitions["final_test"], score_scale)
    report = evaluate_teacher(
        scaled_final,
        normalizers,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        quarantine_count=quarantine_count,
    )
    report["release_blockers"] = sorted(set(release_blockers))
    report["final_test_split_identity"] = split["split_identity"]
    report["rng_evidence"] = branch_evidence
    verdict = make_label_release_verdict(
        report=report,
        source_descriptor_identity=source_identity,
        split_identity=split["split_identity"],
        reliability_identity=reliability["calibration_identity"],
        score_scale_identity=score_scale["scale_identity"],
        integration_fidelity_identity=fidelity_identity,
    )
    if verdict["subject"]["source_descriptor_identity"] != source_identity:
        raise TeacherPipelineError("verdict changed source descriptor identity")
    return {
        "split": split,
        "score_scale": score_scale,
        "reliability": reliability,
        "report": report,
        "verdict": verdict,
    }


def _build_parser() -> argparse.ArgumentParser:
    """manual gate override を持たない CLI parser を構築する。

    bootstrap 設定は CI 再現性だけを制御し、release threshold は変更できない。
    """
    parser = argparse.ArgumentParser(
        description="Validate a Survivors value teacher with paired rollouts."
    )
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--source-descriptor", type=Path, required=True)
    parser.add_argument(
        "--integration-fidelity-verdict", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--verdict-output",
        type=Path,
        default=None,
        help="Optional verdict path; defaults under --output-dir.",
    )
    parser.add_argument(
        "--split-seed",
        default="phase6-paired-rollout-teacher-validation",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 引数を読み full pipeline artifacts と label verdict を保存する。

    gate FAIL は正常な検証結果として exit 2、契約/IO error は例外として exit 1 にする。
    """
    args = _build_parser().parse_args(argv)
    artifacts = run_validation_pipeline(
        corpus=_load_json(args.corpus, "paired rollout corpus"),
        source_descriptor=_load_json(
            args.source_descriptor, "source descriptor"
        ),
        integration_fidelity_verdict=_load_json(
            args.integration_fidelity_verdict,
            "integration fidelity verdict",
        ),
        split_seed=args.split_seed,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    output_dir = Path(args.output_dir)
    commit_frozen_episode_split(
        output_dir / "teacher_validation_split.json",
        artifacts["split"],
    )
    commit_teacher_score_scale(
        output_dir / "teacher_score_scale.json",
        artifacts["score_scale"],
    )
    _write_canonical(
        output_dir / "teacher_reliability.json",
        artifacts["reliability"],
    )
    _write_canonical(
        output_dir / "teacher_validation_report.json",
        artifacts["report"],
    )
    verdict_path = (
        Path(args.verdict_output)
        if args.verdict_output is not None
        else output_dir / "survivors_label_release_verdict.json"
    )
    _write_canonical(verdict_path, artifacts["verdict"])
    return 0 if artifacts["verdict"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Survivors ItemSelector 教師 dataset の split・soft target・release 契約を実装する。

episode/prefix group split を create-once で封印し、01-06/01-07 の teacher artifact を
再 fit せず読み取る。derived shard は development 行だけを source/label lineage 付きで公開する。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
)

from games.survivors.teacher_reliability import (
    validate_reliability_calibration,
)
from games.survivors.teacher_score_scale import (
    TIE_EPSILON_Z,
    assert_calibration_scale_binding,
    load_teacher_score_scale,
    transform_teacher_scores,
    validate_teacher_score_scale,
)

SPLIT_SCHEMA_VERSION = "survivors.item_selector_split.v1"
DATASET_SCHEMA_VERSION = "survivors.item_selector_dataset.v1"
SOFT_TARGET_SCHEMA_VERSION = "survivors.item_soft_target.v1"
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
DEVELOPMENT_SPLITS = frozenset({"train", "validation"})
ALL_SPLITS = DEVELOPMENT_SPLITS | {"test"}
TEMPERATURE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "seed",
        "ratios",
        "group_assignments",
        "group_members",
        "frozen",
        "test_read_policy",
    }
)
_TEST_FORBIDDEN_PURPOSES = frozenset(
    {
        "ablation",
        "ceiling",
        "feature_ablation",
        "feature_selection",
        "learning_curve",
        "temperature_selection",
    }
)


class SplitSealedError(ValueError):
    """split overlap、binding、test partition sealing 違反を表す。

    caller が空結果として続行せず dataset/解析を fail-closed に停止するための例外。
    """


class TeacherArtifactError(ValueError):
    """score scale と reliability artifact の binding 違反を表す。

    substitution・再 fit・release blocker を soft target 計算前に停止する。
    """


class DatasetReleaseError(ValueError):
    """source identity、label verdict、DAG、derived write の違反を表す。

    validation 完了前に出力 directory を公開しない transaction 境界として使う。
    """


def _is_sha256(value: Any) -> bool:
    """小文字 64 桁の SHA-256 identity だけを受理する。

    source・teacher・artifact sibling で共通の形式判定を使う。
    """
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _group_id(row: Mapping[str, Any]) -> str:
    """decision row から episode/prefix group identity を返す。

    prefix がある場合は episode より粗い単位を優先し、decision 単位再分割を防ぐ。
    """
    prefix = row.get("prefix_group_id")
    episode = row.get("episode_id")
    if not isinstance(episode, str) or not episode:
        raise SplitSealedError("episode_id must be non-empty")
    if prefix is not None:
        if not isinstance(prefix, str) or not prefix:
            raise SplitSealedError("prefix_group_id must be non-empty")
        return f"prefix:{prefix}"
    return f"episode:{episode}"


def _validated_split_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """split 入力行の型、decision identity、episode identity を検証する。

    retry duplicate を独立 decision として split 集計へ入れない。
    """
    if isinstance(rows, (str, bytes)) or not rows:
        raise SplitSealedError("split rows are required")
    result: list[Mapping[str, Any]] = []
    seen_decisions: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SplitSealedError(f"rows[{index}] must be an object")
        decision_id = row.get("decision_id")
        episode_id = row.get("episode_id")
        if (
            not isinstance(decision_id, str)
            or not decision_id
            or decision_id in seen_decisions
        ):
            raise SplitSealedError("decision_id values must be unique")
        if not isinstance(episode_id, str) or not episode_id:
            raise SplitSealedError("episode_id must be non-empty")
        seen_decisions.add(decision_id)
        _group_id(row)
        result.append(row)
    return result


def _manifest_identity_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """manifest ID 自身を除く immutable split payload を返す。

    assignment、member、policy、ratio の全 semantic field を同じ identity へ束縛する。
    """
    return {
        key: value[key]
        for key in sorted(_MANIFEST_FIELDS - {"manifest_id"})
    }


def _validate_formal_label_verdict(
    verdict: Mapping[str, Any],
    *,
    source_identity: str,
) -> str:
    """01-06/01-07 formal label release verdict の PASS と parent binding を検証する。

    status 文字列だけでなく gate checks、blocking slices、canonical verdict identity まで照合する。
    """
    fields = {
        "schema_version",
        "subject",
        "status",
        "gates",
        "report_identity",
        "verdict_identity",
    }
    if set(verdict) != fields:
        raise DatasetReleaseError("formal label verdict fields mismatch")
    if verdict["schema_version"] != "survivors.label_release_verdict.v1":
        raise DatasetReleaseError("unsupported label verdict schema_version")
    subject = verdict["subject"]
    subject_fields = {
        "source_descriptor_identity",
        "split_identity",
        "reliability_identity",
        "score_scale_identity",
        "integration_fidelity_identity",
    }
    if not isinstance(subject, Mapping) or set(subject) != subject_fields:
        raise DatasetReleaseError("label verdict subject fields mismatch")
    if subject["source_descriptor_identity"] != source_identity:
        raise DatasetReleaseError("label verdict source identity mismatch")
    if any(not _is_sha256(subject[field]) for field in subject_fields):
        raise DatasetReleaseError("label verdict parent identity is invalid")
    gates = verdict["gates"]
    if (
        verdict["status"] != "PASS"
        or not isinstance(gates, Mapping)
        or gates.get("passed") is not True
        or not isinstance(gates.get("checks"), Mapping)
        or not gates["checks"]
        or any(value is not True for value in gates["checks"].values())
    ):
        raise DatasetReleaseError("label verdict is not an approved PASS")
    if not _is_sha256(verdict["report_identity"]):
        raise DatasetReleaseError("label verdict report identity is invalid")
    payload = {
        key: verdict[key]
        for key in (
            "schema_version",
            "subject",
            "status",
            "gates",
            "report_identity",
        )
    }
    if verdict["verdict_identity"] != canonical_hash(payload):
        raise DatasetReleaseError("label verdict identity mismatch")
    return verdict["verdict_identity"]


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """episode/prefix group の 70/15/15 create-once assignment を保持する。

    development reader は train/validation だけを返し、test は専用 ``read_test_rows`` から
    sealed evaluation purpose でのみ開ける。
    """

    seed: str
    group_assignments: Mapping[str, str]
    group_members: Mapping[str, tuple[str, ...]]
    manifest_id: str
    ratios: Mapping[str, float] = field(default_factory=lambda: dict(SPLIT_RATIOS))
    frozen: bool = True
    test_read_policy: str = "read_test_rows_only_after_method_lock"
    schema_version: str = SPLIT_SCHEMA_VERSION
    _access_audit: list[dict[str, Any]] = field(
        default_factory=list, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        """manifest schema、全 partition coverage、canonical identity を検証する。

        assignment の改変と incomplete split を load/constructor の両経路で拒否する。
        """
        if self.schema_version != SPLIT_SCHEMA_VERSION:
            raise SplitSealedError("unsupported split manifest schema_version")
        if not isinstance(self.seed, str) or not self.seed:
            raise SplitSealedError("split seed must be non-empty")
        if self.frozen is not True:
            raise SplitSealedError("split manifest must be frozen")
        if self.test_read_policy != "read_test_rows_only_after_method_lock":
            raise SplitSealedError("test read policy changed")
        if dict(self.ratios) != SPLIT_RATIOS:
            raise SplitSealedError("split ratios must be exactly 70/15/15")
        assignments = dict(self.group_assignments)
        if (
            len(assignments) < 3
            or set(assignments.values()) != ALL_SPLITS
            or any(not isinstance(key, str) or not key for key in assignments)
        ):
            raise SplitSealedError("group assignments are incomplete")
        members = dict(self.group_members)
        if set(members) != set(assignments):
            raise SplitSealedError("group member keys must match assignments")
        seen_episodes: set[str] = set()
        for group, episodes in members.items():
            valid_group_name = (
                group.startswith("prefix:") and len(group) > len("prefix:")
            ) or (
                group.startswith("episode:") and len(group) > len("episode:")
            )
            if (
                not valid_group_name
                or not isinstance(episodes, (tuple, list))
                or not episodes
                or any(not isinstance(item, str) or not item for item in episodes)
                or len(set(episodes)) != len(episodes)
            ):
                raise SplitSealedError(f"group {group} members are invalid")
            if group.startswith("episode:") and tuple(episodes) != (
                group.removeprefix("episode:"),
            ):
                raise SplitSealedError("episode group must contain only its named episode")
            overlap = seen_episodes.intersection(episodes)
            if overlap:
                raise SplitSealedError("episode overlap across split groups")
            seen_episodes.update(episodes)
        expected = canonical_hash(_manifest_identity_payload(self.to_wire(include_id=False)))
        if self.manifest_id != expected:
            raise SplitSealedError("split manifest identity mismatch")

    @classmethod
    def freeze(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        seed: str,
    ) -> "SplitManifest":
        """rows の unique prefix group を stable hash 順で 70/15/15 に一度割り当てる。

        exact/near-duplicate fingerprint が異なる group をまたぐ入力は assignment 前に拒否する。
        """
        if not isinstance(seed, str) or not seed:
            raise SplitSealedError("split seed must be non-empty")
        validated = _validated_split_rows(rows)
        group_members: dict[str, set[str]] = {}
        fingerprint_owner: dict[tuple[str, str], str] = {}
        for row in validated:
            group = _group_id(row)
            group_members.setdefault(group, set()).add(str(row["episode_id"]))
            for field_name in ("source_fingerprint", "near_duplicate_key"):
                fingerprint = row.get(field_name)
                if fingerprint is None:
                    continue
                if not isinstance(fingerprint, str) or not fingerprint:
                    raise SplitSealedError(f"{field_name} must be non-empty")
                key = (field_name, fingerprint)
                owner = fingerprint_owner.setdefault(key, group)
                if owner != group:
                    label = "near-duplicate" if field_name == "near_duplicate_key" else "overlap"
                    raise SplitSealedError(f"{label} rows cross prefix groups")
        groups = sorted(
            group_members,
            key=lambda group: (
                canonical_hash({"seed": seed, "group_id": group}),
                group,
            ),
        )
        count = len(groups)
        if count < 3:
            raise SplitSealedError("at least three episode/prefix groups are required")
        train_count = int(count * SPLIT_RATIOS["train"])
        validation_count = int(count * SPLIT_RATIOS["validation"])
        train_count = max(train_count, 1)
        validation_count = max(validation_count, 1)
        if train_count + validation_count >= count:
            train_count = count - 2
            validation_count = 1
        assignments: dict[str, str] = {}
        for index, group in enumerate(groups):
            if index < train_count:
                split = "train"
            elif index < train_count + validation_count:
                split = "validation"
            else:
                split = "test"
            assignments[group] = split
        members = {
            group: tuple(sorted(episodes))
            for group, episodes in sorted(group_members.items())
        }
        without_identity = {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "seed": seed,
            "ratios": dict(SPLIT_RATIOS),
            "group_assignments": dict(sorted(assignments.items())),
            "group_members": {
                group: list(episodes) for group, episodes in members.items()
            },
            "frozen": True,
            "test_read_policy": "read_test_rows_only_after_method_lock",
        }
        identity = canonical_hash(without_identity)
        return cls(
            seed=seed,
            group_assignments=dict(sorted(assignments.items())),
            group_members=members,
            manifest_id=identity,
        )

    @property
    def partition_group_counts(self) -> dict[str, int]:
        """各 partition の割当 group 数を固定 key 順で返す。

        decision row 数ではなく leakage を防ぐ grouping unit を集計する。
        """
        counts = Counter(self.group_assignments.values())
        return {split: counts[split] for split in ("train", "validation", "test")}

    @property
    def access_audit(self) -> tuple[Mapping[str, Any], ...]:
        """partition reader の呼び出し監査を immutable view で返す。

        data card はこの履歴から test read count を記録できる。
        """
        return tuple(dict(item) for item in self._access_audit)

    def to_wire(self, *, include_id: bool = True) -> dict[str, Any]:
        """split manifest を canonical-JSON-compatible wire object にする。

        identity 計算時だけ manifest ID を空欄相当で除外できる内部 option を持つ。
        """
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "ratios": dict(self.ratios),
            "group_assignments": dict(sorted(self.group_assignments.items())),
            "group_members": {
                group: list(self.group_members[group])
                for group in sorted(self.group_members)
            },
            "frozen": self.frozen,
            "test_read_policy": self.test_read_policy,
        }
        if include_id:
            result["manifest_id"] = self.manifest_id
        return result

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "SplitManifest":
        """persisted manifest の exact schema と identity を検証して復元する。

        unknown field や ratio/assignment 改変を暗黙補正しない。
        """
        if not isinstance(data, Mapping) or frozenset(data) != _MANIFEST_FIELDS:
            raise SplitSealedError("split manifest fields mismatch")
        members = data["group_members"]
        if not isinstance(members, Mapping):
            raise SplitSealedError("group_members must be an object")
        return cls(
            seed=data["seed"],
            group_assignments=dict(data["group_assignments"]),
            group_members={key: tuple(value) for key, value in members.items()},
            manifest_id=data["manifest_id"],
            ratios=dict(data["ratios"]),
            frozen=data["frozen"],
            test_read_policy=data["test_read_policy"],
            schema_version=data["schema_version"],
        )

    @classmethod
    def load(cls, path: Path) -> "SplitManifest":
        """JSON path から split manifest を読み exact validation 後に返す。

        parse error と contract error を SplitSealedError の境界で通知する。
        """
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SplitSealedError(f"cannot load split manifest: {exc}") from exc
        return cls.from_wire(value)

    def commit(self, path: Path) -> None:
        """manifest を canonical bytes で create-once 保存する。

        同一 bytes の再実行だけを idempotent とし、別 assignment の上書きを拒否する。
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = canonical_json_bytes(self.to_wire()) + b"\n"
        try:
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            if destination.read_bytes() != encoded:
                raise SplitSealedError("frozen split manifest cannot overwrite existing content")
            return
        try:
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                raise OSError("short split manifest write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def split_for(self, row: Mapping[str, Any]) -> str:
        """row の frozen group assignment を返し declared split も照合する。

        unknown group、episode/prefix substitution、split substitution は filter 前に拒否する。
        """
        group = _group_id(row)
        if group not in self.group_assignments:
            raise SplitSealedError("row belongs to an unknown split group")
        if row["episode_id"] not in self.group_members[group]:
            raise SplitSealedError("row episode/prefix group membership mismatch")
        expected = self.group_assignments[group]
        declared = row.get("split")
        if declared is not None and declared != expected:
            raise SplitSealedError("row split binding mismatch")
        return expected

    def _bind_rows(self, rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        """reader 入力の全 row を manifest assignment へ束縛する。

        filter 対象外の test row も先に照合し、偽装 split を見逃さない。
        """
        validated = _validated_split_rows(rows)
        for row in validated:
            self.split_for(row)
        return validated

    def read_partition_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        split: str,
        *,
        purpose: str,
    ) -> tuple[Mapping[str, Any], ...]:
        """train/validation の一方だけを読み、test 指定を常に拒否する。

        test の唯一の read path を ``read_test_rows`` に限定する。
        """
        if split not in ALL_SPLITS:
            raise SplitSealedError("unknown split")
        if split == "test":
            raise SplitSealedError("test partition requires read_test_rows")
        if not isinstance(purpose, str) or not purpose:
            raise SplitSealedError("read purpose is required")
        bound = self._bind_rows(rows)
        selected = tuple(row for row in bound if self.split_for(row) == split)
        self._access_audit.append(
            {"split": split, "purpose": purpose, "row_count": len(selected)}
        )
        return selected

    def read_development_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        purpose: str,
    ) -> tuple[Mapping[str, Any], ...]:
        """train と validation を一つの development view として返す。

        learning curve/ablation はこの API のみを利用し test partition を観測しない。
        """
        if not isinstance(purpose, str) or not purpose:
            raise SplitSealedError("read purpose is required")
        bound = self._bind_rows(rows)
        selected = tuple(
            row for row in bound if self.split_for(row) in DEVELOPMENT_SPLITS
        )
        self._access_audit.append(
            {"split": "development", "purpose": purpose, "row_count": len(selected)}
        )
        return selected

    def read_test_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        purpose: str = "sealed_evaluation",
    ) -> tuple[Mapping[str, Any], ...]:
        """method lock 後の明示 purpose に限り test rows を返す唯一の API。

        ablation、ceiling、learning curve の purpose は専用 API からでも拒否する。
        """
        normalized_purpose = str(purpose).strip().lower()
        if (
            not normalized_purpose
            or normalized_purpose in _TEST_FORBIDDEN_PURPOSES
            or any(token in normalized_purpose for token in ("ablation", "ceiling", "learning"))
        ):
            raise SplitSealedError(f"{purpose} cannot read test partition")
        bound = self._bind_rows(rows)
        selected = tuple(row for row in bound if self.split_for(row) == "test")
        self._access_audit.append(
            {"split": "test", "purpose": purpose, "row_count": len(selected)}
        )
        return selected


@dataclass(frozen=True, slots=True)
class TeacherSoftTarget:
    """一 decision の item identity 付き soft target と lineage を表す。

    masked candidate は確率 0、valid probability は正規化済み、reliability は 01-07 slice
    から取得した値だけを保持する。
    """

    candidate_item_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    tie_groups: tuple[tuple[str, ...], ...]
    reliability_weight: float
    temperature: float
    score_scale_id: str
    calibration_id: str
    schema_version: str = SOFT_TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """soft target の長さ、確率和、lineage identity を検証する。

        item permutation 後も position と probability の対応が失われないようにする。
        """
        if self.schema_version != SOFT_TARGET_SCHEMA_VERSION:
            raise TeacherArtifactError("unsupported soft target schema_version")
        if (
            not self.candidate_item_ids
            or len(self.candidate_item_ids) != len(self.probabilities)
            or len(set(self.candidate_item_ids)) != len(self.candidate_item_ids)
        ):
            raise TeacherArtifactError("soft target candidate identities are invalid")
        if any(not math.isfinite(value) or value < 0.0 for value in self.probabilities):
            raise TeacherArtifactError("soft target probabilities are invalid")
        if sum(self.probabilities) != 1.0:
            raise TeacherArtifactError("soft target probability sum must equal one")
        if not 0.0 <= self.reliability_weight <= 1.0:
            raise TeacherArtifactError("reliability weight must be in [0, 1]")
        if self.temperature not in TEMPERATURE_GRID:
            raise TeacherArtifactError("temperature is outside registered grid")
        if not _is_sha256(self.score_scale_id) or not _is_sha256(self.calibration_id):
            raise TeacherArtifactError("soft target lineage identity is invalid")

    @property
    def probability_by_item_id(self) -> dict[str, float]:
        """candidate position に依存しない item ID → probability mapping を返す。

        permutation test と downstream label join はこの view を利用する。
        """
        return dict(zip(self.candidate_item_ids, self.probabilities, strict=True))

    def to_wire(self) -> dict[str, Any]:
        """soft target を versioned canonical wire object に変換する。

        tie group と parent artifact identity を label shard に保存する。
        """
        return {
            "schema_version": self.schema_version,
            "candidate_item_ids": list(self.candidate_item_ids),
            "probabilities": list(self.probabilities),
            "tie_groups": [list(group) for group in self.tie_groups],
            "reliability_weight": self.reliability_weight,
            "temperature": self.temperature,
            "score_scale_id": self.score_scale_id,
            "calibration_id": self.calibration_id,
        }


class TeacherSoftTargetBuilder:
    """committed score scale と reliability calibration から soft target を作る。

    02-01 での scale fitting/substitution を公開 API で拒否し、tie/temperature unit を
    artifact と完全に束縛する。
    """

    def __init__(
        self,
        score_scale: Mapping[str, Any],
        reliability_calibration: Mapping[str, Any],
        *,
        expected_score_scale_id: str | None = None,
        expected_teacher_identity: str | None = None,
        expected_sigma: float | None = None,
        expected_tie_epsilon_z: float = TIE_EPSILON_Z,
    ) -> None:
        """両 parent artifact と全 binding field を検証して builder を初期化する。

        一つでも release blocker slice があれば partial release へ進まず失敗する。
        """
        try:
            validate_teacher_score_scale(score_scale)
            validate_reliability_calibration(reliability_calibration)
            assert_calibration_scale_binding(reliability_calibration, score_scale)
        except (TypeError, ValueError) as exc:
            raise TeacherArtifactError(f"teacher artifact validation failed: {exc}") from exc
        if reliability_calibration["teacher_type"] != score_scale["teacher_type"]:
            raise TeacherArtifactError("teacher type binding mismatch")
        if reliability_calibration["teacher_identity"] != score_scale["teacher_identity"]:
            raise TeacherArtifactError("teacher identity binding mismatch")
        if expected_score_scale_id is not None and score_scale["scale_identity"] != expected_score_scale_id:
            raise TeacherArtifactError("score scale ID substitution rejected")
        if expected_teacher_identity is not None and score_scale["teacher_identity"] != expected_teacher_identity:
            raise TeacherArtifactError("teacher identity substitution rejected")
        if expected_sigma is not None and abs(float(score_scale["sigma"]) - float(expected_sigma)) > 1e-12:
            raise TeacherArtifactError("score sigma substitution rejected")
        if score_scale["tie_epsilon_z"] != expected_tie_epsilon_z or expected_tie_epsilon_z != TIE_EPSILON_Z:
            raise TeacherArtifactError("tie epsilon z substitution rejected")
        blockers = [
            item for item in reliability_calibration["slices"]
            if item["release_blocker"] or float(item["error_ucb_z"]) >= 1.0
        ]
        if blockers:
            raise TeacherArtifactError("reliability release blocker is present")
        self._scale = dict(score_scale)
        self._calibration = dict(reliability_calibration)

    @classmethod
    def from_artifact_paths(
        cls,
        score_scale_path: Path,
        reliability_path: Path,
        **expected: Any,
    ) -> "TeacherSoftTargetBuilder":
        """01-06/01-07 JSON artifact paths を読み検証済み builder を返す。

        scale loader の derived sigma validation と reliability の identity validation を両方通す。
        """
        try:
            scale = load_teacher_score_scale(Path(score_scale_path))
            calibration = json.loads(Path(reliability_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise TeacherArtifactError(f"cannot load teacher artifacts: {exc}") from exc
        return cls(scale, calibration, **expected)

    def refit_scale(self, _: Sequence[Mapping[str, Any]]) -> None:
        """02-01 からの score scale 再 fit 要求を常に拒否する。

        scale の fitting authority は 01-06 artifact producer だけに限定する。
        """
        raise TeacherArtifactError("score scale refit is forbidden in item dataset release")

    def _slice(self, slice_key: str) -> Mapping[str, Any]:
        """事前 calibration の exact requested slice を返す。

        unknown slice を global weight で補完せず release blocker として扱う。
        """
        matches = [
            item for item in self._calibration["slices"]
            if item["slice_key"] == slice_key
        ]
        if len(matches) != 1:
            raise TeacherArtifactError("unknown or duplicate reliability slice")
        item = matches[0]
        if item["release_blocker"] or float(item["error_ucb_z"]) >= 1.0:
            raise TeacherArtifactError("reliability slice is a release blocker")
        return item

    def build(
        self,
        *,
        candidate_item_ids: Sequence[str],
        teacher_scores: Sequence[float | None],
        card_mask: Sequence[bool],
        slice_key: str,
        temperature: float,
    ) -> TeacherSoftTarget:
        """standardized score に tie pooling と masked softmax を適用する。

        降順隣接差が 0.02 未満の連結成分を同率とし、mask 候補は確率 0 に固定する。
        """
        ids = tuple(candidate_item_ids)
        scores = tuple(teacher_scores)
        mask = tuple(card_mask)
        if (
            not ids
            or len(ids) != len(scores)
            or len(ids) != len(mask)
            or len(set(ids)) != len(ids)
            or any(not isinstance(item, str) or not item for item in ids)
            or any(type(item) is not bool for item in mask)
        ):
            raise TeacherArtifactError("candidate score/mask inputs are invalid")
        if temperature not in TEMPERATURE_GRID:
            raise TeacherArtifactError("temperature must come from registered grid")
        for score, valid in zip(scores, mask, strict=True):
            if valid != (score is not None):
                raise TeacherArtifactError("score presence must match card mask")
        standardized = transform_teacher_scores(scores, self._scale)
        valid_indices = [index for index, valid in enumerate(mask) if valid]
        ordered = sorted(valid_indices, key=lambda index: (-float(standardized[index]), ids[index]))
        grouped_indices: list[list[int]] = []
        for index in ordered:
            if not grouped_indices:
                grouped_indices.append([index])
                continue
            previous = grouped_indices[-1][-1]
            gap = float(standardized[previous]) - float(standardized[index])
            if gap < float(self._scale["tie_epsilon_z"]):
                grouped_indices[-1].append(index)
            else:
                grouped_indices.append([index])
        pooled = [None if value is None else float(value) for value in standardized]
        for group in grouped_indices:
            mean = sum(float(standardized[index]) for index in group) / len(group)
            for index in group:
                pooled[index] = mean
        logits = [float(pooled[index]) / temperature for index in valid_indices]
        maximum = max(logits)
        exponentials = [math.exp(value - maximum) for value in logits]
        denominator = sum(exponentials)
        valid_probabilities = [value / denominator for value in exponentials]
        if len(valid_probabilities) == 1:
            valid_probabilities[0] = 1.0
        else:
            valid_probabilities[-1] = 1.0 - sum(valid_probabilities[:-1])
        probabilities = [0.0] * len(ids)
        for index, probability in zip(valid_indices, valid_probabilities, strict=True):
            probabilities[index] = probability
        reliability = self._slice(slice_key)
        return TeacherSoftTarget(
            candidate_item_ids=ids,
            probabilities=tuple(probabilities),
            tie_groups=tuple(
                tuple(ids[index] for index in group) for group in grouped_indices
            ),
            reliability_weight=float(reliability["weight"]),
            temperature=float(temperature),
            score_scale_id=self._scale["scale_identity"],
            calibration_id=self._calibration["calibration_identity"],
        )

    def select_temperature(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        slice_key: str,
    ) -> float:
        """development examples の NLL、次に Brier が最小の登録温度を選ぶ。

        test partition example と unknown target item は選定に使わない。
        """
        if isinstance(examples, (str, bytes)) or not examples:
            raise TeacherArtifactError("temperature selection examples are required")
        objectives: list[tuple[float, float, float]] = []
        for temperature in TEMPERATURE_GRID:
            nll = 0.0
            brier = 0.0
            for example in examples:
                if example.get("split") not in DEVELOPMENT_SPLITS:
                    raise TeacherArtifactError("temperature selection requires development split")
                target = self.build(
                    candidate_item_ids=example["candidate_item_ids"],
                    teacher_scores=example["teacher_scores"],
                    card_mask=example["card_mask"],
                    slice_key=slice_key,
                    temperature=temperature,
                )
                target_id = example.get("observed_best_item_id")
                if target_id not in target.probability_by_item_id:
                    raise TeacherArtifactError("temperature target item is unknown")
                probability = max(target.probability_by_item_id[target_id], 1e-15)
                nll -= math.log(probability)
                for item_id, predicted in target.probability_by_item_id.items():
                    expected = 1.0 if item_id == target_id else 0.0
                    brier += (predicted - expected) ** 2
            objectives.append((nll / len(examples), brier / len(examples), temperature))
        return min(objectives)[2]


@dataclass(frozen=True, slots=True)
class LearningCurvePolicy:
    """ItemSelector decision 収集の停止条件を事前登録する。

    初期 2000 件、2000 件 block、coverage 達成後 NDCG gain 0.005 未満が二回継続、budget
    hard cap の全条件を一つの value object にする。
    """

    initial_decisions: int = 2000
    block_size: int = 2000
    ndcg_gain_threshold: float = 0.005
    consecutive_blocks: int = 2

    def should_stop(
        self,
        points: Sequence[Mapping[str, Any]],
        *,
        coverage_complete: bool,
        hard_cap: int,
    ) -> bool:
        """coverage/plateau または hard cap 到達で収集停止を判定する。

        coverage 未達の plateau だけでは停止せず rare taxonomy の欠落を防ぐ。
        全 point が initial_decisions 以上かつ block_size cadence に従うことを強制する。
        """
        if not points:
            return False
        counts = []
        for point in points:
            c = point.get("decision_count")
            if type(c) is not int:
                raise DatasetReleaseError("learning curve decision_count must be int")
            counts.append(c)
        if counts[0] < self.initial_decisions:
            raise DatasetReleaseError(
                f"first learning curve point must be >= initial_decisions "
                f"({self.initial_decisions}), got {counts[0]}"
            )
        if counts[0] != self.initial_decisions:
            raise DatasetReleaseError(
                f"first learning curve point must equal initial_decisions "
                f"({self.initial_decisions}), got {counts[0]}"
            )
        for idx in range(1, len(counts)):
            expected = self.initial_decisions + idx * self.block_size
            if counts[idx] != expected:
                raise DatasetReleaseError(
                    f"learning curve point {idx} must be at cadence "
                    f"{expected} (initial={self.initial_decisions} + "
                    f"{idx}*block_size={self.block_size}), got {counts[idx]}"
                )
        if counts[-1] >= hard_cap:
            return True
        if not coverage_complete or len(points) < self.consecutive_blocks + 1:
            return False
        gains = [
            float(points[index]["validation_ndcg"])
            - float(points[index - 1]["validation_ndcg"])
            for index in range(len(points) - self.consecutive_blocks, len(points))
        ]
        return all(gain < self.ndcg_gain_threshold for gain in gains)

    def to_wire(self) -> dict[str, Any]:
        """data card に保存する learning curve 条件を返す。

        threshold と連続 block 数を PR ごとの暗黙値にしない。
        """
        return {
            "initial_decisions": self.initial_decisions,
            "block_size": self.block_size,
            "ndcg_gain_threshold": self.ndcg_gain_threshold,
            "consecutive_blocks": self.consecutive_blocks,
            "stop_condition": (
                "coverage_complete and validation_ndcg_gain < threshold "
                "for consecutive_blocks"
            ),
        }


def calculate_hard_cap(
    *,
    throughput_decisions_per_hour: float,
    collection_hours: float,
    storage_budget_bytes: int,
    estimated_row_bytes: int,
) -> dict[str, Any]:
    """throughput 時間 budget と storage budget の小さい方から hard cap を算出する。

    根拠のない固定件数を使わず、両制約と採用 constraint を data card 向けに返す。
    """
    numeric = (throughput_decisions_per_hour, collection_hours)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in numeric
    ):
        raise DatasetReleaseError("throughput and collection hours must be positive finite")
    if type(storage_budget_bytes) is not int or storage_budget_bytes <= 0:
        raise DatasetReleaseError("storage budget must be a positive int")
    if type(estimated_row_bytes) is not int or estimated_row_bytes <= 0:
        raise DatasetReleaseError("estimated row bytes must be a positive int")
    throughput_cap = int(float(throughput_decisions_per_hour) * float(collection_hours))
    storage_cap = storage_budget_bytes // estimated_row_bytes
    value = min(throughput_cap, storage_cap)
    if value <= 0:
        raise DatasetReleaseError("hard cap budget cannot hold one decision")
    return {
        "value": value,
        "throughput_cap": throughput_cap,
        "storage_cap": storage_cap,
        "throughput_decisions_per_hour": float(throughput_decisions_per_hour),
        "collection_hours": float(collection_hours),
        "storage_budget_bytes": storage_budget_bytes,
        "estimated_row_bytes": estimated_row_bytes,
        "formula": "min(throughput_decisions_per_hour*collection_hours, storage_budget_bytes/estimated_row_bytes)",
    }


class ItemSelectorDatasetBuilder:
    """承認済み teacher label を development-only derived shards へ release する。

    source identity、teacher verdict、acyclic ancestry、split binding を全行検証した後だけ
    directory rename で manifest/shard を公開する。
    """

    def __init__(
        self,
        *,
        split_manifest: SplitManifest,
        source_identity: str,
        teacher_identity: str,
        derived_logical_id: str,
    ) -> None:
        """release 全体の immutable parent identities と logical ID を固定する。

        row ごとの値で source/teacher identity を切り替えることは許可しない。
        """
        if not _is_sha256(source_identity) or not _is_sha256(teacher_identity):
            raise DatasetReleaseError("source/teacher identity must be lowercase sha256")
        if not isinstance(derived_logical_id, str) or not derived_logical_id:
            raise DatasetReleaseError("derived logical ID must be non-empty")
        self._split_manifest = split_manifest
        self._source_identity = source_identity
        self._teacher_identity = teacher_identity
        self._derived_logical_id = derived_logical_id

    def _validate_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """一行の source、verdict、DAG、UI ownership、split を検証して canonical copy する。

        test row は source が承認済みでも derived release への書込みを拒否する。
        """
        if not isinstance(row, Mapping):
            raise DatasetReleaseError("dataset row must be an object")
        if row.get("source_identity") != self._source_identity:
            raise DatasetReleaseError("source identity mismatch")
        trace_ref = row.get("source_trace_ref")
        if not isinstance(trace_ref, str) or not trace_ref:
            raise DatasetReleaseError("source trace ref is required")
        verdict = row.get("label_verdict")
        if not isinstance(verdict, Mapping):
            raise DatasetReleaseError("label verdict must be an object")
        if verdict.get("schema_version") == "survivors.label_release_verdict.v1":
            _validate_formal_label_verdict(
                verdict,
                source_identity=self._source_identity,
            )
            action_kind = row.get("label_action_kind")
        else:
            if verdict.get("status") != "approved":
                raise DatasetReleaseError("label verdict must be approved")
            if verdict.get("teacher_identity") != self._teacher_identity:
                raise DatasetReleaseError("label verdict teacher identity mismatch")
            action_kind = verdict.get("action_kind")
        if action_kind != "choose_card":
            raise DatasetReleaseError(
                "selector label must have explicit action_kind='choose_card'; "
                f"got {action_kind!r}"
            )
        ancestors = row.get("ancestor_refs")
        if (
            not isinstance(ancestors, list)
            or any(not isinstance(item, str) or not item for item in ancestors)
        ):
            raise DatasetReleaseError("ancestor refs must be an array")
        if self._derived_logical_id == trace_ref or self._derived_logical_id in ancestors:
            raise DatasetReleaseError("cyclic source reference rejected")
        try:
            split = self._split_manifest.split_for(row)
        except SplitSealedError as exc:
            raise DatasetReleaseError(f"split binding failed: {exc}") from exc
        if split == "test":
            raise DatasetReleaseError("test partition cannot be written to derived dataset")
        try:
            return json.loads(canonical_json_bytes(dict(row)))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DatasetReleaseError(f"row must be finite canonical JSON: {exc}") from exc

    def release(
        self,
        rows: Sequence[Mapping[str, Any]],
        output_dir: Path,
    ) -> dict[str, Any]:
        """全行検証後に JSONL shard と manifest を directory 単位で公開する。

        任意 row の ``source_trace_ref`` は shard に保持し、source trace へ逆参照できる。
        """
        if isinstance(rows, (str, bytes)) or not rows:
            raise DatasetReleaseError("release rows are required")
        normalized = [self._validate_row(row) for row in rows]
        decisions = [row.get("decision_id") for row in normalized]
        if (
            any(not isinstance(item, str) or not item for item in decisions)
            or len(set(decisions)) != len(decisions)
        ):
            raise DatasetReleaseError("release decision IDs must be unique")
        ordered = sorted(normalized, key=lambda row: row["decision_id"])
        payload = b"".join(canonical_json_bytes(row) + b"\n" for row in ordered)
        dataset_identity = canonical_hash(
            {
                "source_identity": self._source_identity,
                "teacher_identity": self._teacher_identity,
                "split_manifest_id": self._split_manifest.manifest_id,
                "derived_logical_id": self._derived_logical_id,
                "rows": ordered,
            }
        )
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_identity": dataset_identity,
            "derived_logical_id": self._derived_logical_id,
            "source_identity": self._source_identity,
            "teacher_identity": self._teacher_identity,
            "split_manifest_id": self._split_manifest.manifest_id,
            "row_count": len(ordered),
            "test_row_count": 0,
            "partitions": dict(sorted(Counter(row["split"] for row in ordered).items())),
            "shards": [
                {
                    "logical_id": "shards/development-00000.jsonl",
                    "row_count": len(ordered),
                    "content_identity": canonical_hash(ordered),
                }
            ],
            "source_trace_index": {
                row["decision_id"]: row["source_trace_ref"] for row in ordered
            },
        }
        destination = Path(output_dir)
        if destination.exists():
            raise DatasetReleaseError("release destination already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            shards = temporary / "shards"
            shards.mkdir()
            (shards / "development-00000.jsonl").write_bytes(payload)
            (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
            os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return manifest

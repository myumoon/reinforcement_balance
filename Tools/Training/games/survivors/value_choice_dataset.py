"""Survivors choice trace dataset の immutable shard writer と reader。

JSONL metadata と pickle-free NPZ array を一つの shard directory として確定し、manifest
更新を最後の commit point にする。再送は canonical record ID で deduplicate し、破損・
中断 shard は dataset へ推測追加せず quarantine へ隔離する。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)

DATASET_SCHEMA_VERSION = "survivors.value_choice_dataset.v1"
SHARD_SCHEMA_VERSION = "survivors.value_choice_shard.v1"
_SHA256_LENGTH = 64
_INTEGER_ARRAY_NAMES = frozenset(
    {
        "action_history",
        "candidate_choice_indices",
        "episode_starts",
        "movement_actions",
    }
)
_EMPTY_HISTOGRAM = {
    "behavior_policy": {},
    "candidate_count": {},
    "selected_choice_id": {},
    "teacher_choice_id": {},
}
_ROW_RESERVED_FIELDS = frozenset(
    {
        "schema_version",
        "record_id",
        "record_content_sha256",
        "source_identity_sha256",
        "arrays",
    }
)


class DatasetError(ValueError):
    """dataset の schema、source binding、shard transaction が不正な場合の例外。

    呼び出し側が部分 artifact を通常の空 shard と誤認しないよう、入出力例外を一つの
    fail-closed 境界へ変換する。
    """


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    """commit 済み manifest、ordered rows、array 群の read-back 結果。

    ``rows[index]`` と ``arrays[index]`` は同じ record を表し、shard 順と shard 内順を
    manifest に記録された順序のまま保持する。
    """

    manifest: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    arrays: tuple[Mapping[str, np.ndarray], ...]


@dataclass(slots=True)
class _PendingRow:
    """active shard 内の検証済み一 record を保持する。

    metadata と ndarray は append 時に copy し、commit 前の caller 側変更で hash と値が
    分離しないようにする。
    """

    row: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _is_sha256(value: Any) -> bool:
    """小文字 64 桁の SHA-256 文字列だけを受理する。

    source、record、array、file hash の全 sibling で同じ形式判定を共有する。
    """

    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    """同一 directory の temp file を fsync 後に置換する。

    manifest と shard commit marker の reader が途中までの JSON bytes を観測しないよう、
    rename を唯一の公開点にする。
    """

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


def _copy_json(value: Any, label: str) -> Any:
    """finite-only canonical JSON を検証して独立 object を返す。

    NumPy scalar や NaN を暗黙変換せず、永続化後に caller が nested metadata を変更する
    aliasing も同時に除く。
    """

    try:
        encoded = canonical_json_bytes(value)
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DatasetError(f"{label} must be finite canonical JSON: {exc}") from exc


def _array_dtype(name: str, array: np.ndarray) -> np.dtype:
    """array 名と入力 kind から float32/int32 の一方を決める。

    action/index/start sibling は int32 に固定し、それ以外でも integer 入力は int32、
    floating 入力は float32 として保存する。
    """

    if name in _INTEGER_ARRAY_NAMES or np.issubdtype(array.dtype, np.integer):
        return np.dtype(np.int32)
    if np.issubdtype(array.dtype, np.floating):
        return np.dtype(np.float32)
    raise DatasetError(
        f"array {name} must be numeric and stored as float32 or int32"
    )


def _normalize_arrays(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """全 ndarray sibling の name、dtype、finite、shape を検証して copy する。

    LSTM state を flatten/stack せず各入力の original shape を保持し、object/string/bool
    array は pickle-free NPZ へ到達する前に拒否する。
    """

    if not isinstance(arrays, Mapping) or not arrays:
        raise DatasetError("arrays must be a non-empty object")
    normalized: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if (
            not isinstance(name, str)
            or not name
            or "__" in name
            or name in normalized
        ):
            raise DatasetError("array names must be unique non-empty strings without '__'")
        raw = np.asarray(value)
        if np.issubdtype(raw.dtype, np.bool_) and name not in _INTEGER_ARRAY_NAMES:
            raise DatasetError(f"array {name} bool dtype is not supported")
        dtype = _array_dtype(name, raw)
        if np.issubdtype(dtype, np.integer):
            limits = np.iinfo(np.int32)
            if raw.size and (
                np.min(raw) < limits.min or np.max(raw) > limits.max
            ):
                raise DatasetError(f"array {name} exceeds int32 range")
        array = np.asarray(raw, dtype=dtype).copy()
        if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(array)):
            raise DatasetError(f"array {name} must contain only finite values")
        normalized[name] = array
    return normalized


def _array_descriptor(array: np.ndarray) -> dict[str, Any]:
    """array の dtype、shape、値を共有 canonical hash へ束縛する。

    raw byte hashing を record identity に使わず、platform 非依存の JSON 表現を
    ``canonical_json.py`` へ渡す。
    """

    payload = {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "values": array.tolist(),
    }
    return {
        "dtype": payload["dtype"],
        "shape": payload["shape"],
        "sha256": canonical_hash(payload),
    }


def record_id_for(
    source_identity_sha256: str,
    episode_logical_id: str,
    decision_id: str,
) -> str:
    """source + logical episode + external decision の canonical SHA-256 を返す。

    retry 回数、process ID、時刻、shard 番号を含めず、同じ logical decision が必ず同じ
    record ID へ収束するようにする。
    """

    if not _is_sha256(source_identity_sha256):
        raise DatasetError("source identity must be lowercase sha256")
    if not isinstance(episode_logical_id, str) or not episode_logical_id:
        raise DatasetError("episode_logical_id must be non-empty")
    if not isinstance(decision_id, str) or not decision_id:
        raise DatasetError("decision_id must be non-empty")
    return canonical_hash(
        {
            "source_identity_sha256": source_identity_sha256,
            "episode_logical_id": episode_logical_id,
            "decision_id": decision_id,
        }
    )


def _validate_choice_binding(metadata: Mapping[str, Any]) -> None:
    """candidate 集合へ behavior selection と teacher label を別々に束縛する。

    selected choice を teacher 正解として補完せず、両 field が存在する場合はそれぞれ
    candidate ID 集合への membership だけを検証する。
    """

    candidates = metadata.get("candidate_choice_ids")
    if (
        not isinstance(candidates, list)
        or len(candidates) < 2
        or any(not isinstance(value, str) or not value for value in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        raise DatasetError(
            "candidate_choice_ids must contain at least two unique choices"
        )
    candidate_ids = set(candidates)
    behavior = metadata.get("behavior")
    teacher = metadata.get("teacher_label")
    if not isinstance(behavior, Mapping):
        raise DatasetError("behavior must be an object")
    if not isinstance(teacher, Mapping):
        raise DatasetError("teacher_label must be a separate object")
    selected = behavior.get("selected_choice_id")
    if selected not in candidate_ids:
        raise DatasetError("behavior selected choice is unknown")
    teacher_best = teacher.get("best_choice_id")
    if teacher_best not in candidate_ids:
        raise DatasetError("teacher best choice is unknown")
    propensity = behavior.get("propensity")
    if (
        isinstance(propensity, bool)
        or not isinstance(propensity, (int, float))
        or not 0.0 < float(propensity) <= 1.0
    ):
        raise DatasetError("behavior propensity must be in (0, 1]")
    ordered = teacher.get("ordered_choice_ids")
    if not isinstance(ordered, list) or set(ordered) != candidate_ids:
        raise DatasetError("teacher ordered choices must match candidates")


def _empty_manifest(dataset_id: str, source_identity_sha256: str) -> dict[str, Any]:
    """新規 dataset の immutable identity と空集計を作る。

    時刻や host path は dataset ID に含めず、source identity と caller が指定した論理名
    だけを manifest の親情報として保存する。
    """

    if not isinstance(dataset_id, str) or not dataset_id:
        raise DatasetError("dataset_id must be non-empty")
    if not _is_sha256(source_identity_sha256):
        raise DatasetError("source identity must be lowercase sha256")
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "source_identity_sha256": source_identity_sha256,
        "record_count": 0,
        "histogram": _copy_json(_EMPTY_HISTOGRAM, "empty histogram"),
        "shards": [],
    }


def _validate_manifest(value: Any) -> dict[str, Any]:
    """dataset manifest の exact top-level schema と derived count を検証する。

    shard summary の hash/count/ID 重複も全件走査し、reader と writer が同じ manifest
    boundary を共有する。
    """

    if not isinstance(value, Mapping):
        raise DatasetError("manifest must be an object")
    expected = {
        "schema_version",
        "dataset_id",
        "source_identity_sha256",
        "record_count",
        "histogram",
        "shards",
    }
    if set(value) != expected:
        raise DatasetError("manifest fields mismatch")
    manifest = _copy_json(dict(value), "manifest")
    if manifest["schema_version"] != DATASET_SCHEMA_VERSION:
        raise DatasetError("unsupported dataset schema version")
    if not isinstance(manifest["dataset_id"], str) or not manifest["dataset_id"]:
        raise DatasetError("manifest dataset_id must be non-empty")
    if not _is_sha256(manifest["source_identity_sha256"]):
        raise DatasetError("manifest source identity must be lowercase sha256")
    if not isinstance(manifest["shards"], list):
        raise DatasetError("manifest shards must be an array")
    shard_ids: set[str] = set()
    record_ids_seen: set[str] = set()
    aggregate_histogram: dict[str, dict[str, int]] = _copy_json(
        _EMPTY_HISTOGRAM,
        "aggregate histogram",
    )
    count = 0
    for shard in manifest["shards"]:
        if not isinstance(shard, Mapping) or set(shard) != {
            "shard_id",
            "row_count",
            "jsonl_sha256",
            "npz_sha256",
            "record_ids",
            "histogram",
        }:
            raise DatasetError("manifest shard summary fields mismatch")
        shard_id = shard["shard_id"]
        if (
            not isinstance(shard_id, str)
            or not shard_id
            or shard_id in shard_ids
        ):
            raise DatasetError("manifest shard IDs must be unique")
        shard_ids.add(shard_id)
        row_count = shard["row_count"]
        record_ids = shard["record_ids"]
        if type(row_count) is not int or row_count <= 0:
            raise DatasetError("manifest shard row_count must be positive")
        if (
            not isinstance(record_ids, list)
            or len(record_ids) != row_count
            or len(set(record_ids)) != row_count
            or not all(_is_sha256(record_id) for record_id in record_ids)
        ):
            raise DatasetError("manifest shard record IDs are invalid")
        if record_ids_seen.intersection(record_ids):
            raise DatasetError("manifest record IDs must be globally unique")
        record_ids_seen.update(record_ids)
        if not _is_sha256(shard["jsonl_sha256"]) or not _is_sha256(
            shard["npz_sha256"]
        ):
            raise DatasetError("manifest shard file hash is invalid")
        _validate_histogram(shard["histogram"])
        _merge_histogram(aggregate_histogram, shard["histogram"])
        count += row_count
    if type(manifest["record_count"]) is not int or manifest["record_count"] != count:
        raise DatasetError("manifest record_count does not match shards")
    _validate_histogram(manifest["histogram"])
    if manifest["histogram"] != aggregate_histogram:
        raise DatasetError("manifest histogram does not match committed shards")
    return manifest


def _validate_histogram(value: Any) -> None:
    """四 histogram sibling の key/count 型を対称に検証する。

    append 中の仮集計や負数を manifest として受理せず、commit 済み shard summary の
    非負 integer count だけを許可する。
    """

    if not isinstance(value, Mapping) or set(value) != set(_EMPTY_HISTOGRAM):
        raise DatasetError("histogram fields mismatch")
    for category, counts in value.items():
        if not isinstance(counts, Mapping):
            raise DatasetError(f"histogram {category} must be an object")
        for key, count in counts.items():
            if not isinstance(key, str) or not key or type(count) is not int or count < 0:
                raise DatasetError(f"histogram {category} entry is invalid")


def _histogram_for(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """commit 対象 rows だけから shard histogram を計算する。

    active transaction の append 時には呼ばず、完成 shard を公開する直前に一回だけ
    derived summary を作る。
    """

    histogram: dict[str, dict[str, int]] = _copy_json(
        _EMPTY_HISTOGRAM,
        "histogram",
    )
    for row in rows:
        values = {
            "behavior_policy": row["behavior"]["policy"],
            "candidate_count": str(len(row["candidate_choice_ids"])),
            "selected_choice_id": row["behavior"]["selected_choice_id"],
            "teacher_choice_id": row["teacher_label"]["best_choice_id"],
        }
        for category, key in values.items():
            counts = histogram[category]
            counts[key] = counts.get(key, 0) + 1
    return histogram


def _merge_histogram(
    destination: dict[str, dict[str, int]],
    addition: Mapping[str, Mapping[str, int]],
) -> None:
    """shard commit histogram を dataset 集計へ加算する。

    category を動的に増やさず、versioned manifest の四 sibling へ同じ integer 加算だけを
    適用する。
    """

    _validate_histogram(destination)
    _validate_histogram(addition)
    for category in _EMPTY_HISTOGRAM:
        for key, count in addition[category].items():
            destination[category][key] = destination[category].get(key, 0) + count


def _quarantine_path(root: Path, source: Path, reason: str) -> Path:
    """破損または中断 path を dataset quarantine へ移動する。

    元 path の basename を保持し、同名が既にある場合だけ連番を加えて既存 evidence を
    上書きしない。
    """

    quarantine = root / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    destination = quarantine / source.name
    suffix = 1
    while destination.exists():
        destination = quarantine / f"{source.name}-{suffix}"
        suffix += 1
    if source.exists():
        shutil.move(os.fspath(source), os.fspath(destination))
    else:
        destination.mkdir()
    _atomic_write(
        destination / "quarantine.json",
        canonical_json_bytes({"reason": reason}),
    )
    return destination


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """canonical JSONL の非空行を ordered row として読む。

    UTF-8/JSON/schema エラーを shard 単位の DatasetError に変換し、壊れた一行を
    読み飛ばして row count を合わせない。
    """

    try:
        lines = path.read_bytes().splitlines()
        if not lines:
            raise DatasetError("shard JSONL is empty")
        rows = [json.loads(line) for line in lines]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"shard JSONL could not be read: {exc}") from exc
    return [_copy_json(row, "shard row") for row in rows]


def _validate_persisted_row(
    row: Any,
    source_identity_sha256: str,
) -> dict[str, Any]:
    """永続 row の identity、choice binding、array descriptor seal を検証する。

    writer append と reader read-back の両経路で同じ record ID/content hash 規則を適用し、
    JSONL の field substitution を NPZ hash が正しくても受理しない。
    """

    if not isinstance(row, Mapping) or not _ROW_RESERVED_FIELDS <= set(row):
        raise DatasetError("persisted row required fields are missing")
    checked = _copy_json(dict(row), "persisted row")
    if checked["schema_version"] != DATASET_SCHEMA_VERSION:
        raise DatasetError("persisted row schema version is unsupported")
    if checked["source_identity_sha256"] != source_identity_sha256:
        raise DatasetError("persisted row source identity mismatch")
    if not _is_sha256(checked["record_id"]) or not _is_sha256(
        checked["record_content_sha256"]
    ):
        raise DatasetError("persisted row hash fields are invalid")
    if (
        type(checked.get("environment_step")) is not int
        or checked["environment_step"] < 0
    ):
        raise DatasetError("persisted row environment step is invalid")
    _validate_choice_binding(checked)
    expected_record_id = record_id_for(
        source_identity_sha256,
        checked.get("episode_logical_id"),
        checked.get("decision_id"),
    )
    if checked["record_id"] != expected_record_id:
        raise DatasetError("persisted row record ID mismatch")
    descriptors = checked["arrays"]
    if not isinstance(descriptors, Mapping) or not descriptors:
        raise DatasetError("persisted row array descriptors are invalid")
    for name, descriptor in descriptors.items():
        if (
            not isinstance(name, str)
            or not name
            or "__" in name
            or not isinstance(descriptor, Mapping)
            or set(descriptor) != {"dtype", "shape", "sha256"}
            or descriptor["dtype"] not in {"float32", "int32"}
            or not isinstance(descriptor["shape"], list)
            or not all(type(dimension) is int and dimension >= 0 for dimension in descriptor["shape"])
            or not _is_sha256(descriptor["sha256"])
        ):
            raise DatasetError("persisted row array descriptor is invalid")
    metadata = {
        key: value
        for key, value in checked.items()
        if key not in _ROW_RESERVED_FIELDS
    }
    expected_content_hash = canonical_hash(
        {
            "source_identity_sha256": source_identity_sha256,
            "metadata": metadata,
            "arrays": descriptors,
        }
    )
    if checked["record_content_sha256"] != expected_content_hash:
        raise DatasetError("persisted row content hash mismatch")
    return checked


def _validate_shard(
    shard_path: Path,
    *,
    source_identity_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    """shard marker、file hash、row/NPZ count、全 array hash を再検証する。

    partial NPZ、JSONL/NPZ count mismatch、dtype/shape/hash substitution の全 sibling を
    同じ read-back gate で拒否する。
    """

    marker_path = shard_path / "commit.json"
    rows_path = shard_path / "rows.jsonl"
    arrays_path = shard_path / "arrays.npz"
    try:
        marker = json.loads(marker_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"shard commit marker could not be read: {exc}") from exc
    if not isinstance(marker, Mapping) or set(marker) != {
        "schema_version",
        "source_identity_sha256",
        "summary",
    }:
        raise DatasetError("shard commit marker fields mismatch")
    if marker["schema_version"] != SHARD_SCHEMA_VERSION:
        raise DatasetError("unsupported shard schema version")
    if marker["source_identity_sha256"] != source_identity_sha256:
        raise DatasetError("shard source identity mismatch")
    summary = marker["summary"]
    if not isinstance(summary, Mapping) or set(summary) != {
        "shard_id",
        "row_count",
        "jsonl_sha256",
        "npz_sha256",
        "record_ids",
        "histogram",
    }:
        raise DatasetError("shard summary fields mismatch")
    if (
        sha256_hex(rows_path.read_bytes()) != summary.get("jsonl_sha256")
        or sha256_hex(arrays_path.read_bytes()) != summary.get("npz_sha256")
    ):
        raise DatasetError("shard file hash mismatch")
    rows = _load_rows(rows_path)
    arrays_by_row: list[dict[str, np.ndarray]] = []
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            if "row_count" not in archive.files:
                raise DatasetError("NPZ row_count is missing")
            row_count_value = archive["row_count"]
            if row_count_value.dtype != np.int32 or row_count_value.shape != ():
                raise DatasetError("NPZ row_count must be scalar int32")
            row_count = int(row_count_value)
            if row_count != len(rows) or row_count != summary.get("row_count"):
                raise DatasetError("JSONL/NPZ count mismatch")
            for index, row in enumerate(rows):
                row = _validate_persisted_row(
                    row,
                    source_identity_sha256,
                )
                rows[index] = row
                descriptors = row.get("arrays")
                if not isinstance(descriptors, Mapping) or not descriptors:
                    raise DatasetError("row array descriptors are invalid")
                restored: dict[str, np.ndarray] = {}
                for name, descriptor in descriptors.items():
                    key = f"r{index:08d}__{name}"
                    if key not in archive.files:
                        raise DatasetError(f"NPZ array is missing: {key}")
                    array = np.asarray(archive[key])
                    if array.dtype not in (np.dtype(np.float32), np.dtype(np.int32)):
                        raise DatasetError(f"NPZ array dtype is invalid: {key}")
                    expected = _array_descriptor(array)
                    if descriptor != expected:
                        raise DatasetError(f"NPZ array descriptor mismatch: {key}")
                    restored[name] = array.copy()
                arrays_by_row.append(restored)
            expected_keys = {"row_count"} | {
                f"r{index:08d}__{name}"
                for index, row in enumerate(rows)
                for name in row["arrays"]
            }
            if set(archive.files) != expected_keys:
                raise DatasetError("NPZ contains unknown arrays")
    except DatasetError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise DatasetError(f"shard NPZ could not be read: {exc}") from exc
    if summary.get("record_ids") != [row.get("record_id") for row in rows]:
        raise DatasetError("shard record order mismatch")
    expected_histogram = _histogram_for(rows)
    if summary.get("histogram") != expected_histogram:
        raise DatasetError("shard histogram mismatch")
    validated_summary = {
        "shard_id": summary.get("shard_id"),
        "row_count": summary.get("row_count"),
        "jsonl_sha256": summary.get("jsonl_sha256"),
        "npz_sha256": summary.get("npz_sha256"),
        "record_ids": summary.get("record_ids"),
        "histogram": summary.get("histogram"),
    }
    return validated_summary, rows, arrays_by_row


class DatasetWriter:
    """Survivors value choice dataset の単一 active-shard writer。

    ``start_shard`` → ``append`` → ``commit_shard`` または ``abort_shard`` を明示し、
    manifest 集計と既存 record ID は commit 成功後にだけ更新する。
    """

    def __init__(
        self,
        dataset_root: Path,
        *,
        dataset_id: str,
        source_identity_sha256: str,
        manifest_path: Path | None = None,
    ) -> None:
        """dataset identity を検証し、中断/orphan shard を recovery scan する。

        既存 manifest の source/dataset ID が caller と異なる場合は、shard を開く前に
        fail-closed で拒否する。
        """

        configured_root = Path(dataset_root)
        if manifest_path is None and configured_root.suffix == ".json":
            manifest_path = configured_root
            configured_root = configured_root.parent
        self.root = configured_root
        self.manifest_path = (
            Path(manifest_path)
            if manifest_path is not None
            else self.root / "manifest.json"
        )
        if self.manifest_path.parent != self.root:
            raise DatasetError("manifest_path must be directly inside dataset root")
        self.shards_path = self.root / "shards"
        self.staging_path = self.root / ".staging"
        self.root.mkdir(parents=True, exist_ok=True)
        self.shards_path.mkdir(exist_ok=True)
        self.staging_path.mkdir(exist_ok=True)
        if self.manifest_path.exists():
            try:
                self._manifest = _validate_manifest(
                    json.loads(self.manifest_path.read_bytes())
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DatasetError(f"manifest could not be read: {exc}") from exc
            if self._manifest["dataset_id"] != dataset_id:
                raise DatasetError("dataset ID does not match existing manifest")
            if (
                self._manifest["source_identity_sha256"]
                != source_identity_sha256
            ):
                raise DatasetError(
                    "source identity does not match existing dataset"
                )
        else:
            self._manifest = _empty_manifest(
                dataset_id,
                source_identity_sha256,
            )
            self._publish_manifest()
        self.source_identity_sha256 = source_identity_sha256
        self._active_id: str | None = None
        self._active_path: Path | None = None
        self._pending: list[_PendingRow] = []
        self._pending_by_id: dict[str, str] = {}
        self._existing_by_id: dict[str, str] = {}
        self._recover()
        self._load_existing_record_hashes()

    @property
    def manifest(self) -> Mapping[str, Any]:
        """現在 commit 済み manifest の detached JSON copy を返す。

        caller が histogram/shard list を変更して次 commit へ混入させないよう mutable
        内部 object を公開しない。
        """

        return _copy_json(self._manifest, "manifest")

    @property
    def active_shard_id(self) -> str | None:
        """active transaction の shard ID、未開始なら ``None`` を返す。

        collector が shard size 境界だけを判断できるよう、pending rows 自体は公開しない。
        """

        return self._active_id

    @property
    def active_row_count(self) -> int:
        """active transaction 内の deduplicate 後 row 数を返す。

        retry append は増えないため、CLI の shard size 判定が logical decision count と一致する。
        """

        return len(self._pending)

    def contains_record(
        self,
        record_id: str,
        *,
        committed_only: bool = False,
    ) -> bool:
        """record ID が committed または active shard に存在するかを返す。

        collector resume が既存 canonical record を異なる retry event 内容でも再 append
        せず、staging quarantine 後だけ再収集できるようにする。
        """

        if not _is_sha256(record_id):
            raise DatasetError("record_id must be lowercase sha256")
        if record_id in self._existing_by_id:
            return True
        return not committed_only and record_id in self._pending_by_id

    def _publish_manifest(self) -> None:
        """内部 manifest を再検証し canonical bytes で atomic publish する。

        shard directory rename 後にだけ呼び、histogram を append 時の仮状態として
        外部公開しない。
        """

        self._manifest = _validate_manifest(self._manifest)
        _atomic_write(
            self.manifest_path,
            canonical_json_bytes(self._manifest),
        )

    def _recover(self) -> None:
        """中断 staging を隔離し、完全 orphan commit だけを manifest へ再適用する。

        directory rename 後・manifest rename 前の process interruption は commit marker と
        全 content を read-back できる場合だけ exactly once recovery する。
        """

        for staged in sorted(self.staging_path.iterdir()):
            _quarantine_path(self.root, staged, "interrupted active shard")
        known = {shard["shard_id"]: shard for shard in self._manifest["shards"]}
        changed = False
        for shard_path in sorted(self.shards_path.iterdir()):
            if not shard_path.is_dir():
                _quarantine_path(self.root, shard_path, "unknown shard entry")
                continue
            shard_id = shard_path.name
            try:
                summary, _, _ = _validate_shard(
                    shard_path,
                    source_identity_sha256=self._manifest[
                        "source_identity_sha256"
                    ],
                )
                if summary["shard_id"] != shard_id:
                    raise DatasetError("shard ID does not match directory")
                if shard_id in known:
                    if summary != known[shard_id]:
                        raise DatasetError("committed shard summary mismatch")
                    continue
                existing_ids = {
                    record_id
                    for shard in self._manifest["shards"]
                    for record_id in shard["record_ids"]
                }
                if existing_ids.intersection(summary["record_ids"]):
                    raise DatasetError("orphan shard duplicates committed records")
                self._manifest["shards"].append(summary)
                self._manifest["record_count"] += summary["row_count"]
                _merge_histogram(
                    self._manifest["histogram"],
                    summary["histogram"],
                )
                known[shard_id] = summary
                changed = True
            except DatasetError as exc:
                _quarantine_path(self.root, shard_path, str(exc))
                if shard_id in known:
                    raise DatasetError(
                        f"manifest referenced shard was quarantined: {shard_id}"
                    ) from exc
        missing = set(known) - {
            path.name
            for path in self.shards_path.iterdir()
            if path.is_dir()
        }
        if missing:
            raise DatasetError(
                f"manifest references missing shards: {sorted(missing)}"
            )
        if changed:
            self._publish_manifest()

    def _load_existing_record_hashes(self) -> None:
        """全 commit 済み row の record/content hash map を構築する。

        source/episode/decision が同じ retry を shard 境界や process resume 後にも
        deduplicate し、内容衝突は黙って既存 row 扱いしない。
        """

        for shard in self._manifest["shards"]:
            _, rows, _ = _validate_shard(
                self.shards_path / shard["shard_id"],
                source_identity_sha256=self.source_identity_sha256,
            )
            for row in rows:
                record_id = row["record_id"]
                content_hash = row["record_content_sha256"]
                if record_id in self._existing_by_id:
                    raise DatasetError("duplicate record ID across committed shards")
                self._existing_by_id[record_id] = content_hash

    def start_shard(self, shard_id: str | None = None) -> str:
        """新しい active shard transaction を開始して ID を返す。

        同時に二 shard を開かず、省略 ID は commit 済み shard 数から deterministic に
        採番する。
        """

        if self._active_id is not None:
            raise DatasetError("another shard transaction is already active")
        selected = shard_id or f"shard-{len(self._manifest['shards']):05d}"
        if (
            not isinstance(selected, str)
            or not selected
            or "/" in selected
            or "\\" in selected
            or selected in {".", ".."}
        ):
            raise DatasetError("shard_id must be a safe non-empty name")
        if (
            (self.shards_path / selected).exists()
            or (self.staging_path / selected).exists()
            or any(
                shard["shard_id"] == selected
                for shard in self._manifest["shards"]
            )
        ):
            raise DatasetError("shard_id already exists")
        active_path = self.staging_path / selected
        active_path.mkdir()
        self._active_id = selected
        self._active_path = active_path
        self._pending = []
        self._pending_by_id = {}
        return selected

    def append(
        self,
        metadata: Mapping[str, Any],
        arrays: Mapping[str, Any] | None = None,
    ) -> str:
        """検証済み record を active shard へ追加し deterministic record ID を返す。

        retry duplicate は内容 hash が一致するときだけ no-op にし、NaN、未知 choice、
        同一 ID の内容衝突は active shard 全体を quarantine する。
        """

        if self._active_id is None or self._active_path is None:
            raise DatasetError("start_shard must be called before append")
        try:
            raw_metadata = dict(metadata)
            if arrays is None and "arrays" in raw_metadata:
                arrays = raw_metadata.pop("arrays")
            if arrays is None:
                raise DatasetError("record arrays are required")
            copied = _copy_json(raw_metadata, "record metadata")
            reserved = set(copied).intersection(_ROW_RESERVED_FIELDS)
            if reserved:
                raise DatasetError(
                    f"record metadata uses reserved fields: {sorted(reserved)}"
                )
            if (
                not isinstance(copied.get("episode_logical_id"), str)
                or not copied["episode_logical_id"]
                or not isinstance(copied.get("decision_id"), str)
                or not copied["decision_id"]
            ):
                raise DatasetError(
                    "episode_logical_id and decision_id must be non-empty"
                )
            if (
                type(copied.get("environment_step")) is not int
                or copied["environment_step"] < 0
            ):
                raise DatasetError("environment_step must be non-negative")
            _validate_choice_binding(copied)
            normalized_arrays = _normalize_arrays(arrays)
            descriptors = {
                name: _array_descriptor(array)
                for name, array in normalized_arrays.items()
            }
            record_id = record_id_for(
                self.source_identity_sha256,
                copied["episode_logical_id"],
                copied["decision_id"],
            )
            content_payload = {
                "source_identity_sha256": self.source_identity_sha256,
                "metadata": copied,
                "arrays": descriptors,
            }
            content_hash = canonical_hash(content_payload)
            existing_hash = self._existing_by_id.get(record_id)
            pending_hash = self._pending_by_id.get(record_id)
            if existing_hash is not None or pending_hash is not None:
                if (existing_hash or pending_hash) != content_hash:
                    raise DatasetError(
                        "retry record ID has conflicting content"
                    )
                return record_id
            row = {
                "schema_version": DATASET_SCHEMA_VERSION,
                "record_id": record_id,
                "record_content_sha256": content_hash,
                "source_identity_sha256": self.source_identity_sha256,
                **copied,
                "arrays": descriptors,
            }
            _copy_json(row, "dataset row")
            self._pending.append(_PendingRow(row, normalized_arrays))
            self._pending_by_id[record_id] = content_hash
            return record_id
        except DatasetError as exc:
            self.abort_shard(reason=str(exc))
            raise

    def commit_shard(self) -> Mapping[str, Any]:
        """active shard の JSONL/NPZ/marker を read-back 後に一回 commit する。

        directory rename を shard 公開点、manifest rename を集計公開点とし、histogram と
        committed record map は両方が成功した後にだけ更新する。
        """

        if (
            self._active_id is None
            or self._active_path is None
            or not self._pending
        ):
            raise DatasetError("active shard must contain at least one row")
        shard_id = self._active_id
        active_path = self._active_path
        rows = [entry.row for entry in self._pending]
        try:
            jsonl_payload = b"".join(
                canonical_json_bytes(row) + b"\n" for row in rows
            )
            rows_path = active_path / "rows.jsonl"
            with rows_path.open("wb") as stream:
                stream.write(jsonl_payload)
                stream.flush()
                os.fsync(stream.fileno())
            npz_values: dict[str, np.ndarray] = {
                "row_count": np.asarray(len(rows), dtype=np.int32)
            }
            for index, entry in enumerate(self._pending):
                for name, array in entry.arrays.items():
                    npz_values[f"r{index:08d}__{name}"] = array
            arrays_path = active_path / "arrays.npz"
            with arrays_path.open("wb") as stream:
                np.savez_compressed(stream, **npz_values)
                stream.flush()
                os.fsync(stream.fileno())
            summary = {
                "shard_id": shard_id,
                "row_count": len(rows),
                "jsonl_sha256": sha256_hex(rows_path.read_bytes()),
                "npz_sha256": sha256_hex(arrays_path.read_bytes()),
                "record_ids": [row["record_id"] for row in rows],
                "histogram": _histogram_for(rows),
            }
            _atomic_write(
                active_path / "commit.json",
                canonical_json_bytes(
                    {
                        "schema_version": SHARD_SCHEMA_VERSION,
                        "source_identity_sha256": self.source_identity_sha256,
                        "summary": summary,
                    }
                ),
            )
            checked_summary, _, _ = _validate_shard(
                active_path,
                source_identity_sha256=self.source_identity_sha256,
            )
            destination = self.shards_path / shard_id
            os.replace(active_path, destination)
            updated = _copy_json(self._manifest, "manifest")
            updated["shards"].append(checked_summary)
            updated["record_count"] += checked_summary["row_count"]
            _merge_histogram(
                updated["histogram"],
                checked_summary["histogram"],
            )
            self._manifest = updated
            self._publish_manifest()
            for row in rows:
                self._existing_by_id[row["record_id"]] = row[
                    "record_content_sha256"
                ]
            result = _copy_json(checked_summary, "shard summary")
            self._clear_active()
            return result
        except Exception as exc:
            if active_path.exists():
                _quarantine_path(self.root, active_path, f"commit failed: {exc}")
            self._clear_active()
            if isinstance(exc, DatasetError):
                raise
            raise DatasetError(f"shard commit failed: {exc}") from exc

    def abort_shard(self, reason: str = "aborted by caller") -> Path:
        """active shard を quarantine へ移し transaction を破棄する。

        manifest、histogram、committed record map は変更せず、失敗理由だけを quarantine
        marker として finite canonical JSON で残す。
        """

        if self._active_id is None or self._active_path is None:
            raise DatasetError("no shard transaction is active")
        destination = _quarantine_path(
            self.root,
            self._active_path,
            reason,
        )
        self._clear_active()
        return destination

    def _clear_active(self) -> None:
        """active transaction の全 in-memory sibling を一括解放する。

        commit/abort の片側だけに stale pending map を残さず、次 shard が前 transaction の
        retry ID を誤って参照しないようにする。
        """

        self._active_id = None
        self._active_path = None
        self._pending = []
        self._pending_by_id = {}


def read_dataset(dataset_root: Path) -> DatasetSnapshot:
    """commit 済み全 shard を manifest 順に検証して read-back する。

    writer recovery を暗黙実行せず、manifest が参照する source/hash/count/order と全 ndarray
    descriptor が一致する snapshot だけを返す。
    """

    configured_root = Path(dataset_root)
    manifest_path = (
        configured_root
        if configured_root.suffix == ".json"
        else configured_root / "manifest.json"
    )
    root = manifest_path.parent
    try:
        manifest = _validate_manifest(json.loads(manifest_path.read_bytes()))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"manifest could not be read: {exc}") from exc
    rows: list[Mapping[str, Any]] = []
    arrays: list[Mapping[str, np.ndarray]] = []
    seen: set[str] = set()
    for summary in manifest["shards"]:
        checked, shard_rows, shard_arrays = _validate_shard(
            root / "shards" / summary["shard_id"],
            source_identity_sha256=manifest["source_identity_sha256"],
        )
        if checked != summary:
            raise DatasetError("manifest and shard summary differ")
        for row in shard_rows:
            if row["record_id"] in seen:
                raise DatasetError("duplicate record ID across shards")
            seen.add(row["record_id"])
        rows.extend(shard_rows)
        arrays.extend(shard_arrays)
    if len(rows) != manifest["record_count"]:
        raise DatasetError("read-back record count mismatch")
    return DatasetSnapshot(manifest, tuple(rows), tuple(arrays))


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DatasetError",
    "DatasetSnapshot",
    "DatasetWriter",
    "read_dataset",
    "record_id_for",
]

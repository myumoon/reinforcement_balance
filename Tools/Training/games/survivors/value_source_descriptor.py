"""Survivors の immutable Value Source descriptor を構築・公開する。

初心者向け:
訓練モデルと、その入力・設定・コード・実行条件を一つの内容 ID にまとめます。時刻や
保存場所、後から得られる検証結果は内容 ID へ入れず、同じ source を安定して参照できます。
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)

SCHEMA_VERSION = "survivors.value_source_descriptor.v1"
_SHA256_LENGTH = 64
_REQUIRED_ARTIFACTS = ("model", "vecnormalize", "package_freeze")
_REQUIRED_CODE_HASHES = (
    "cpp_logic",
    "cpp_base_reward",
    "python_reward",
    "hp_penalty",
    "noveld_config",
    "noveld_callback",
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "identity_sha256",
        "source_run",
        "created_at_utc",
        "git_commit",
        "dirty",
        "patch_artifact_ref",
        "artifacts",
        "model_spec",
        "observation_schema",
        "provenance",
        "runtime",
        "completion",
        "ready_for_probe",
        "blocking_reasons",
    }
)
_FORBIDDEN_FIELD_PARTS = (
    "verdict",
    "ready_for_label",
    "teacher_validation",
)


class ValueSourceDescriptorError(ValueError):
    """Value Source descriptor を安全に構築・検証できない場合の例外。

    初心者向け:
    不明な field や dirty source の黙認を通常の入力ミスと区別して呼び出し元へ伝えます。
    """


def _is_sha256(value: Any) -> bool:
    """小文字 64 桁 SHA-256 文字列だけを受理する。

    初心者向け:
    単なるラベルを content hash と誤認して ready にしないための共通判定です。
    """
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_commit(value: Any) -> bool:
    """SHA-1 / SHA-256 repository の完全 commit ID だけを受理する。

    初心者向け:
    ``HEAD`` や短縮 SHA のように後から別内容を指し得る参照を immutable identity にしません。
    """
    return (
        isinstance(value, str)
        and len(value) in (40, 64)
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    """mapping 型を検査し、違反時は契約エラーに変換する。

    初心者向け:
    AttributeError ではなく descriptor のどの入力が不正かを明示します。
    """
    if not isinstance(value, Mapping):
        raise ValueSourceDescriptorError(f"{label} must be an object")
    return value


def _require_exact_fields(
    value: Any,
    expected: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    """wire object の field 集合を完全一致で検証する。

    初心者向け:
    typo や将来の verdict field を黙って読み捨てず、schema 境界で拒否します。
    """
    data = _require_mapping(value, label)
    unknown = set(data) - expected
    missing = expected - set(data)
    if unknown:
        raise ValueSourceDescriptorError(f"unknown {label} fields: {sorted(unknown)}")
    if missing:
        raise ValueSourceDescriptorError(f"missing {label} fields: {sorted(missing)}")
    return data


def _reject_forbidden_fields(value: Any, path: str = "descriptor") -> None:
    """descriptor 全階層から verdict / label-ready の逆参照を拒否する。

    初心者向け:
    source → teacher verdict → dataset の一方向性を、field 名の全階層 sweep で守ります。
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in _FORBIDDEN_FIELD_PARTS):
                raise ValueSourceDescriptorError(
                    f"forbidden descriptor field {path}.{key}"
                )
            _reject_forbidden_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, f"{path}[{index}]")


def _parse_created_at(value: str) -> None:
    """created_at_utc が timezone 付き ISO-8601 であることを検証する。

    初心者向け:
    identity 外の監査情報でも、曖昧なローカル時刻は保存しません。
    """
    if not isinstance(value, str) or not value:
        raise ValueSourceDescriptorError("created_at_utc must be a non-empty string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueSourceDescriptorError(
            "created_at_utc must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueSourceDescriptorError("created_at_utc must include a timezone")
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueSourceDescriptorError("created_at_utc must use UTC")


def _relative_run_path(run_dir: Path, configured: Any, label: str) -> tuple[str, Path]:
    """run artifact の path を run 相対参照と実 path に解決する。

    初心者向け:
    descriptor へ絶対 path を漏らさず、run 外への ``..`` 脱出も拒否します。
    """
    if not isinstance(configured, (str, Path)) or not str(configured):
        raise ValueSourceDescriptorError(f"{label}.path must be a non-empty path")
    configured_path = Path(configured)
    candidate = (
        configured_path.resolve()
        if configured_path.is_absolute()
        else (run_dir / configured_path).resolve()
    )
    try:
        relative = candidate.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueSourceDescriptorError(f"{label}.path must be inside run_dir") from exc
    if not relative.parts:
        raise ValueSourceDescriptorError(f"{label}.path must reference a file")
    return relative.as_posix(), candidate


def _source_path(source_root: Path, configured: Any, label: str) -> Path:
    """source provenance の file path を source root 配下へ解決する。

    初心者向け:
    code hash の対象を checkout 内に限定し、任意のローカル file を取り込みません。
    """
    if not isinstance(configured, (str, Path)) or not str(configured):
        raise ValueSourceDescriptorError(f"{label}.path must be a non-empty path")
    configured_path = Path(configured)
    candidate = (
        configured_path.resolve()
        if configured_path.is_absolute()
        else (source_root / configured_path).resolve()
    )
    try:
        candidate.relative_to(source_root.resolve())
    except ValueError as exc:
        raise ValueSourceDescriptorError(f"{label}.path must be inside source_root") from exc
    return candidate


def _file_sha256(path: Path) -> str | None:
    """存在する通常 file の raw bytes SHA-256 を返す。

    初心者向け:
    欠落は例外で ready と推測せず ``None`` にし、後段で blocking reason へ変換します。
    """
    if not path.is_file():
        return None
    return sha256_hex(path.read_bytes())


def _artifact_descriptor(
    run_dir: Path,
    artifacts: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    """model 等の run 相対参照と content hash を組み立てる。

    初心者向け:
    保存場所は可搬な参照、同一性は file 内容の hash として別々に保持します。
    """
    configured = artifacts.get(name, {})
    if not isinstance(configured, Mapping):
        configured = {}
    default_path = {
        "model": "result/model.zip",
        "vecnormalize": "result/vecnormalize.pkl",
        "package_freeze": "log/package_freeze.txt",
    }[name]
    relative, path = _relative_run_path(
        run_dir,
        configured.get("path", default_path),
        f"artifacts.{name}",
    )
    return {"path_relative": relative, "sha256": _file_sha256(path)}


def _normalize_model_spec(value: Any) -> dict[str, Any]:
    """algorithm / policy / recurrent 設定を immutable model spec に整形する。

    初心者向け:
    model bytes だけでなく、推論方法を決める設定も source identity に含めます。
    """
    data = value if isinstance(value, Mapping) else {}
    settings = data.get("settings", {})
    if not isinstance(settings, Mapping):
        settings = {}
    return {
        "algorithm": data.get("algorithm"),
        "policy": data.get("policy"),
        "recurrent": data.get("recurrent"),
        "settings": json.loads(canonical_json_bytes(dict(settings)).decode("utf-8")),
    }


def _normalize_obs_schema(value: Mapping[str, Any]) -> dict[str, Any]:
    """obs schema を total dim と順序付き segment 契約へ正規化する。

    初心者向け:
    UE5 の補助 hash を信用するだけでなく、実際の順序付き構造を canonical hash します。
    """
    raw_segments = value.get("segments")
    segments: list[dict[str, Any]] = []
    if isinstance(raw_segments, Sequence) and not isinstance(
        raw_segments, (str, bytes, bytearray)
    ):
        names: set[str] = set()
        for segment in raw_segments:
            if not isinstance(segment, Mapping):
                return {
                    "sha256": None,
                    "reported_hash": value.get("obs_schema_hash")
                    or value.get("schema_hash"),
                    "total_dim": value.get("total_dim"),
                    "ordered_segments": [],
                }
            normalized_segment = dict(segment)
            name = normalized_segment.get("name")
            dimension = normalized_segment.get("dim")
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or type(dimension) is not int
                or dimension <= 0
            ):
                return {
                    "sha256": None,
                    "reported_hash": value.get("obs_schema_hash")
                    or value.get("schema_hash"),
                    "total_dim": value.get("total_dim"),
                    "ordered_segments": [],
                }
            names.add(name)
            segments.append(normalized_segment)
    total_dim = value.get("total_dim")
    if isinstance(total_dim, bool) or not isinstance(total_dim, int) or total_dim <= 0:
        total_dim = None
    if total_dim is None and segments:
        total_dim = sum(segment["dim"] for segment in segments)
    if segments and total_dim != sum(segment["dim"] for segment in segments):
        total_dim = None
    normalized = {
        "total_dim": total_dim,
        "ordered_segments": json.loads(
            canonical_json_bytes(segments).decode("utf-8")
        ),
    }
    schema_hash = canonical_hash(normalized) if total_dim and segments else None
    return {
        "sha256": schema_hash,
        "reported_hash": value.get("obs_schema_hash") or value.get("schema_hash"),
        **normalized,
    }


def _normalize_code_hashes(
    source_root: Path,
    value: Any,
) -> dict[str, str | None]:
    """全必須 code sibling の file hash を対称に収集する。

    初心者向け:
    C++ / Python / NovelD のどれか一経路だけを取りこぼさないよう、固定 key を sweep します。
    """
    code = value if isinstance(value, Mapping) else {}
    hashes: dict[str, str | None] = {}
    for name in _REQUIRED_CODE_HASHES:
        configured = code.get(name, {})
        if isinstance(configured, str):
            configured = {"path": configured}
        if not isinstance(configured, Mapping):
            hashes[name] = None
            continue
        if "sha256" in configured:
            hashes[name] = (
                configured["sha256"] if _is_sha256(configured["sha256"]) else None
            )
            continue
        if "path" not in configured:
            hashes[name] = None
            continue
        path = _source_path(source_root, configured["path"], f"code.{name}")
        hashes[name] = _file_sha256(path)
    return hashes


def _normalize_runtime(value: Any) -> dict[str, Any]:
    """action と物理 tick の実行契約を正規化する。

    初心者向け:
    同じ model でも action 順序や frame skip が違えば別 source として識別します。
    """
    data = value if isinstance(value, Mapping) else {}
    action_map = data.get("ordered_action_map")
    if not isinstance(action_map, Sequence) or isinstance(
        action_map, (str, bytes, bytearray)
    ):
        action_map = []
    normalized_actions = list(action_map)
    action_hash = (
        canonical_hash(
            {
                "action_semantics_version": data.get("action_semantics_version"),
                "ordered_action_map": normalized_actions,
            }
        )
        if data.get("action_semantics_version") and normalized_actions
        else None
    )
    return {
        "action_semantics_version": data.get("action_semantics_version"),
        "physics_dt": data.get("physics_dt"),
        "frame_skip": data.get("frame_skip"),
        "decision_hz": data.get("decision_hz"),
        "ordered_action_map": normalized_actions,
        "ordered_action_map_sha256": action_hash,
    }


def _coverage_count(value: Any) -> int:
    """coverage count を非負の厳密 int に正規化する。

    初心者向け:
    bool や負数を件数として誤受理せず、未証明の 0 として gate を閉じます。
    """
    return value if type(value) is int and value >= 0 else 0


def _normalize_completion(value: Mapping[str, Any]) -> dict[str, Any]:
    """IS2 completion と全 coverage sibling を固定 field へ正規化する。

    初心者向け:
    weapon / passive / evolution / union を同じ規則で扱い、片側だけ ready にしません。
    """
    return {
        "item_stage_key": value.get("item_stage_key"),
        "is2_complete": value.get("is2_complete") is True,
        "weapon_coverage_count": _coverage_count(
            value.get("weapon_coverage_count")
        ),
        "passive_coverage_count": _coverage_count(
            value.get("passive_coverage_count")
        ),
        "evolution_coverage_count": _coverage_count(
            value.get("evolution_coverage_count")
        ),
        "union_coverage_count": _coverage_count(value.get("union_coverage_count")),
    }


def _load_artifact_store_class():
    """既存 Tools/Artifacts の ArtifactStore class を遅延 import する。

    初心者向け:
    dirty 許可時だけ store 実装を読み、通常の clean descriptor には追加依存を持ち込みません。
    """
    tools_root = Path(__file__).resolve().parents[3]
    artifacts_root = tools_root / "Artifacts"
    if str(artifacts_root) not in sys.path:
        sys.path.insert(0, str(artifacts_root))
    try:
        from artifact_store import ArtifactStore
    except ImportError as exc:  # pragma: no cover - 配置破損時の防御
        raise ValueSourceDescriptorError("ArtifactStore is unavailable") from exc
    return ArtifactStore


def _dirty_patch_ref(
    run_dir: Path,
    source_run: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any] | None:
    """dirty source の patch を content-addressed store へ保存する。

    初心者向け:
    明示許可がない dirty tree は拒否し、許可時も patch 内容を source identity に固定します。
    """
    dirty = provenance.get("dirty")
    if dirty is not True:
        return None
    if provenance.get("allow_dirty") is not True:
        raise ValueSourceDescriptorError(
            "source worktree is dirty; set allow_dirty only with an artifact store"
        )
    store_root = provenance.get("artifact_store_root")
    if not isinstance(store_root, (str, Path)) or not str(store_root):
        raise ValueSourceDescriptorError(
            "dirty source requires artifact_store_root"
        )
    patch_text = provenance.get("patch_text")
    patch_path = provenance.get("patch_path")
    if isinstance(patch_text, str):
        patch_bytes = patch_text.encode("utf-8")
    elif isinstance(patch_path, (str, Path)) and str(patch_path):
        configured = Path(patch_path)
        if not configured.is_absolute():
            configured = (run_dir / configured).resolve()
        if not configured.is_file():
            raise ValueSourceDescriptorError("dirty source patch_path is missing")
        patch_bytes = configured.read_bytes()
    else:
        raise ValueSourceDescriptorError(
            "dirty source requires patch_text or patch_path"
        )
    if not patch_bytes:
        raise ValueSourceDescriptorError("dirty source patch must not be empty")
    patch_sha256 = sha256_hex(patch_bytes)
    logical_id = f"value-sources/{source_run}/patches/{patch_sha256}.patch"
    artifact_store = _load_artifact_store_class()(store_root)
    ref = artifact_store.put_bytes(
        logical_id=logical_id,
        data=patch_bytes,
        media_type="text/x-diff",
    )
    return ref.to_wire()


def _blocking_reasons(descriptor: Mapping[str, Any]) -> list[str]:
    """全 probe-ready 条件を列挙し deterministic な blocking reason を返す。

    初心者向け:
    最初の失敗で止めず、修復に必要な不足を一度の audit で全て表示します。
    """
    reasons: list[str] = []
    completion = descriptor["completion"]
    if (
        completion["item_stage_key"] != "IS2"
        or completion["is2_complete"] is not True
    ):
        reasons.append("is2_incomplete")
    for coverage in (
        "weapon_coverage_count",
        "passive_coverage_count",
        "evolution_coverage_count",
        "union_coverage_count",
    ):
        if completion[coverage] <= 0:
            reasons.append(f"{coverage}_missing")
    for artifact in _REQUIRED_ARTIFACTS:
        if not _is_sha256(descriptor["artifacts"][artifact]["sha256"]):
            reasons.append(f"{artifact}_missing")
    model_spec = descriptor["model_spec"]
    if not isinstance(model_spec["algorithm"], str) or not model_spec["algorithm"]:
        reasons.append("model_algorithm_unknown")
    if not isinstance(model_spec["policy"], str) or not model_spec["policy"]:
        reasons.append("model_policy_unknown")
    if not isinstance(model_spec["recurrent"], bool):
        reasons.append("model_recurrent_setting_unknown")
    obs_schema = descriptor["observation_schema"]
    if not _is_sha256(obs_schema["sha256"]):
        reasons.append("obs_schema_hash_unknown")
    if not _is_sha256(descriptor["provenance"]["resolved_config_sha256"]):
        reasons.append("resolved_config_hash_unknown")
    for name, content_hash in descriptor["provenance"]["code_sha256"].items():
        if not _is_sha256(content_hash):
            reasons.append(f"code_hash_unknown:{name}")
    runtime = descriptor["runtime"]
    if not isinstance(runtime["action_semantics_version"], str) or not runtime[
        "action_semantics_version"
    ]:
        reasons.append("action_semantics_version_unknown")
    if not _is_sha256(runtime["ordered_action_map_sha256"]):
        reasons.append("ordered_action_map_hash_unknown")
    action_map = runtime["ordered_action_map"]
    if (
        not isinstance(action_map, list)
        or not action_map
        or not all(isinstance(action, str) and action for action in action_map)
        or len(set(action_map)) != len(action_map)
    ):
        reasons.append("ordered_action_map_invalid")
    if (
        isinstance(runtime["physics_dt"], bool)
        or not isinstance(runtime["physics_dt"], (int, float))
        or runtime["physics_dt"] <= 0
    ):
        reasons.append("physics_dt_unknown")
    if (
        isinstance(runtime["frame_skip"], bool)
        or not isinstance(runtime["frame_skip"], int)
        or runtime["frame_skip"] <= 0
    ):
        reasons.append("frame_skip_unknown")
    if (
        isinstance(runtime["decision_hz"], bool)
        or not isinstance(runtime["decision_hz"], (int, float))
        or runtime["decision_hz"] <= 0
    ):
        reasons.append("decision_hz_unknown")
    elif (
        isinstance(runtime["physics_dt"], (int, float))
        and not isinstance(runtime["physics_dt"], bool)
        and runtime["physics_dt"] > 0
        and isinstance(runtime["frame_skip"], int)
        and not isinstance(runtime["frame_skip"], bool)
        and runtime["frame_skip"] > 0
        and abs(
            float(runtime["decision_hz"])
            - 1.0 / (float(runtime["physics_dt"]) * runtime["frame_skip"])
        )
        > 1e-9
    ):
        reasons.append("decision_hz_inconsistent")
    if descriptor["dirty"] and descriptor["patch_artifact_ref"] is None:
        reasons.append("dirty_patch_artifact_missing")
    return reasons


def _identity_payload(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """volatile path / clock / gate result を除く identity payload を返す。

    初心者向け:
    除外規則を再帰的な key 推測にせず、全 field を明示列挙して意図しない混入を防ぎます。
    """
    artifacts = descriptor["artifacts"]
    patch_ref = descriptor["patch_artifact_ref"]
    return {
        "schema_version": descriptor["schema_version"],
        "source_run": descriptor["source_run"],
        "git_commit": descriptor["git_commit"],
        "dirty": descriptor["dirty"],
        "patch_artifact_ref": patch_ref,
        "artifact_sha256": {
            name: artifacts[name]["sha256"] for name in _REQUIRED_ARTIFACTS
        },
        "model_spec": descriptor["model_spec"],
        "observation_schema": descriptor["observation_schema"],
        "provenance": descriptor["provenance"],
        "runtime": descriptor["runtime"],
        "completion": descriptor["completion"],
    }


def build_value_source_descriptor(
    *,
    run_dir: Path,
    completion: Mapping[str, Any],
    obs_schema: Mapping[str, Any],
    git_commit: str,
    created_at_utc: str,
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """immutable ``survivors.value_source_descriptor.v1`` を構築する。

    初心者向け:
    欠落 artifact は blocking reason として表し、dirty source や schema 違反だけを例外で
    fail-closed にします。identity 計算は Common の canonical JSON 実装へ一任します。
    """
    run_path = Path(run_dir).resolve()
    if not run_path.is_dir():
        raise ValueSourceDescriptorError("run_dir must be an existing directory")
    completion_data = _require_mapping(completion, "completion")
    schema_data = _require_mapping(obs_schema, "obs_schema")
    provenance_data = _require_mapping(source_provenance, "source_provenance")
    _reject_forbidden_fields(completion_data, "completion")
    _reject_forbidden_fields(schema_data, "obs_schema")
    _reject_forbidden_fields(provenance_data, "source_provenance")
    _parse_created_at(created_at_utc)
    if not _is_git_commit(git_commit):
        raise ValueSourceDescriptorError(
            "git_commit must be a full lowercase commit hash"
        )
    dirty = provenance_data.get("dirty")
    if not isinstance(dirty, bool):
        raise ValueSourceDescriptorError("source_provenance.dirty must be bool")
    source_root_value = provenance_data.get("source_root")
    if not isinstance(source_root_value, (str, Path)) or not str(source_root_value):
        raise ValueSourceDescriptorError(
            "source_provenance.source_root must be an explicit path"
        )
    source_root = Path(source_root_value).resolve()
    source_run = run_path.name
    patch_ref = _dirty_patch_ref(
        run_path, source_run, provenance_data
    )
    artifacts_input = provenance_data.get("artifacts", {})
    if not isinstance(artifacts_input, Mapping):
        artifacts_input = {}
    artifacts = {
        name: _artifact_descriptor(run_path, artifacts_input, name)
        for name in _REQUIRED_ARTIFACTS
    }
    resolved_config = provenance_data.get("resolved_config", {})
    if isinstance(resolved_config, str):
        resolved_config = {"path": resolved_config}
    resolved_config_hash = None
    if isinstance(resolved_config, Mapping):
        if "sha256" in resolved_config:
            resolved_config_hash = (
                resolved_config["sha256"]
                if _is_sha256(resolved_config["sha256"])
                else None
            )
        elif "path" in resolved_config:
            _, resolved_config_path = _relative_run_path(
                run_path,
                resolved_config["path"],
                "resolved_config",
            )
            resolved_config_hash = _file_sha256(resolved_config_path)
    descriptor: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": "",
        "source_run": source_run,
        "created_at_utc": created_at_utc,
        "git_commit": git_commit,
        "dirty": dirty,
        "patch_artifact_ref": patch_ref,
        "artifacts": artifacts,
        "model_spec": _normalize_model_spec(provenance_data.get("model_spec")),
        "observation_schema": _normalize_obs_schema(schema_data),
        "provenance": {
            "resolved_config_sha256": resolved_config_hash,
            "code_sha256": _normalize_code_hashes(
                source_root, provenance_data.get("code")
            ),
        },
        "runtime": _normalize_runtime(provenance_data.get("runtime")),
        "completion": _normalize_completion(completion_data),
        "ready_for_probe": False,
        "blocking_reasons": [],
    }
    descriptor["identity_sha256"] = canonical_hash(_identity_payload(descriptor))
    descriptor["blocking_reasons"] = _blocking_reasons(descriptor)
    descriptor["ready_for_probe"] = not descriptor["blocking_reasons"]
    validate_value_source_descriptor(descriptor)
    return descriptor


def validate_value_source_descriptor(
    descriptor: Mapping[str, Any],
) -> None:
    """descriptor の exact schema・hash binding・非循環 field を検証する。

    初心者向け:
    write 前にも同じ validator を通し、呼び出し側が verdict field を追加した payload を
    immutable source として公開できないようにします。
    """
    data = _require_exact_fields(descriptor, _TOP_LEVEL_FIELDS, "descriptor")
    _reject_forbidden_fields(data)
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueSourceDescriptorError("unsupported descriptor schema_version")
    if not _is_sha256(data["identity_sha256"]):
        raise ValueSourceDescriptorError(
            "identity_sha256 must be lowercase 64-hex sha256"
        )
    if not isinstance(data["source_run"], str) or not data["source_run"]:
        raise ValueSourceDescriptorError("source_run must be a non-empty string")
    _parse_created_at(data["created_at_utc"])
    if not _is_git_commit(data["git_commit"]):
        raise ValueSourceDescriptorError(
            "git_commit must be a full lowercase commit hash"
        )
    if not isinstance(data["dirty"], bool):
        raise ValueSourceDescriptorError("dirty must be bool")
    artifacts = _require_exact_fields(
        data["artifacts"], frozenset(_REQUIRED_ARTIFACTS), "artifacts"
    )
    for name in _REQUIRED_ARTIFACTS:
        artifact = _require_exact_fields(
            artifacts[name],
            frozenset({"path_relative", "sha256"}),
            f"artifacts.{name}",
        )
        if (
            not isinstance(artifact["path_relative"], str)
            or not artifact["path_relative"]
            or Path(artifact["path_relative"]).is_absolute()
            or ".." in Path(artifact["path_relative"]).parts
        ):
            raise ValueSourceDescriptorError(
                f"artifacts.{name}.path_relative must be a safe relative path"
            )
        if artifact["sha256"] is not None and not _is_sha256(artifact["sha256"]):
            raise ValueSourceDescriptorError(
                f"artifacts.{name}.sha256 must be sha256 or null"
            )
    patch_ref = data["patch_artifact_ref"]
    if patch_ref is not None:
        patch_data = _require_mapping(patch_ref, "patch_artifact_ref")
        required_patch_fields = frozenset(
            {
                "schema_version",
                "logical_id",
                "sha256",
                "size_bytes",
                "media_type",
                "store_uri",
            }
        )
        _require_exact_fields(
            patch_data, required_patch_fields, "patch_artifact_ref"
        )
        if not _is_sha256(patch_data["sha256"]):
            raise ValueSourceDescriptorError(
                "patch_artifact_ref.sha256 must be sha256"
            )
        if patch_data["schema_version"] != "artifact_ref.v1":
            raise ValueSourceDescriptorError(
                "patch_artifact_ref.schema_version must be artifact_ref.v1"
            )
        if (
            not isinstance(patch_data["logical_id"], str)
            or not patch_data["logical_id"]
        ):
            raise ValueSourceDescriptorError(
                "patch_artifact_ref.logical_id must be non-empty"
            )
        if (
            type(patch_data["size_bytes"]) is not int
            or patch_data["size_bytes"] <= 0
        ):
            raise ValueSourceDescriptorError(
                "patch_artifact_ref.size_bytes must be positive"
            )
        if patch_data["media_type"] != "text/x-diff":
            raise ValueSourceDescriptorError(
                "patch_artifact_ref.media_type must be text/x-diff"
            )
        if patch_data["store_uri"] != (
            f"artifact://sha256/{patch_data['sha256']}"
        ):
            raise ValueSourceDescriptorError(
                "patch_artifact_ref.store_uri must bind its sha256"
            )
    if data["dirty"] != (patch_ref is not None):
        raise ValueSourceDescriptorError(
            "dirty must be true exactly when patch_artifact_ref is present"
        )
    model_spec = _require_exact_fields(
        data["model_spec"],
        frozenset({"algorithm", "policy", "recurrent", "settings"}),
        "model_spec",
    )
    _require_mapping(model_spec["settings"], "model_spec.settings")
    obs = _require_exact_fields(
        data["observation_schema"],
        frozenset({"sha256", "reported_hash", "total_dim", "ordered_segments"}),
        "observation_schema",
    )
    if obs["sha256"] is not None and not _is_sha256(obs["sha256"]):
        raise ValueSourceDescriptorError(
            "observation_schema.sha256 must be sha256 or null"
        )
    if not isinstance(obs["ordered_segments"], list):
        raise ValueSourceDescriptorError(
            "observation_schema.ordered_segments must be a list"
        )
    provenance = _require_exact_fields(
        data["provenance"],
        frozenset({"resolved_config_sha256", "code_sha256"}),
        "provenance",
    )
    if (
        provenance["resolved_config_sha256"] is not None
        and not _is_sha256(provenance["resolved_config_sha256"])
    ):
        raise ValueSourceDescriptorError(
            "provenance.resolved_config_sha256 must be sha256 or null"
        )
    code_hashes = _require_exact_fields(
        provenance["code_sha256"],
        frozenset(_REQUIRED_CODE_HASHES),
        "provenance.code_sha256",
    )
    for name, content_hash in code_hashes.items():
        if content_hash is not None and not _is_sha256(content_hash):
            raise ValueSourceDescriptorError(
                f"provenance.code_sha256.{name} must be sha256 or null"
            )
    _require_exact_fields(
        data["runtime"],
        frozenset(
            {
                "action_semantics_version",
                "physics_dt",
                "frame_skip",
                "decision_hz",
                "ordered_action_map",
                "ordered_action_map_sha256",
            }
        ),
        "runtime",
    )
    _require_exact_fields(
        data["completion"],
        frozenset(
            {
                "item_stage_key",
                "is2_complete",
                "weapon_coverage_count",
                "passive_coverage_count",
                "evolution_coverage_count",
                "union_coverage_count",
            }
        ),
        "completion",
    )
    if not isinstance(data["ready_for_probe"], bool):
        raise ValueSourceDescriptorError("ready_for_probe must be bool")
    if (
        not isinstance(data["blocking_reasons"], list)
        or not all(
            isinstance(reason, str) and reason
            for reason in data["blocking_reasons"]
        )
    ):
        raise ValueSourceDescriptorError(
            "blocking_reasons must be a list of non-empty strings"
        )
    expected_reasons = _blocking_reasons(data)
    if data["blocking_reasons"] != expected_reasons:
        raise ValueSourceDescriptorError(
            "blocking_reasons do not match descriptor gate inputs"
        )
    if data["ready_for_probe"] != (not expected_reasons):
        raise ValueSourceDescriptorError(
            "ready_for_probe does not match blocking_reasons"
        )
    expected_identity = canonical_hash(_identity_payload(data))
    if data["identity_sha256"] != expected_identity:
        raise ValueSourceDescriptorError(
            "identity_sha256 does not match immutable descriptor fields"
        )


def write_value_source_descriptor(
    run_dir: Path,
    descriptor: Mapping[str, Any],
) -> Path:
    """検証済み descriptor を result へ atomic replace で公開する。

    初心者向け:
    同じ directory に temp file を fsync してから rename し、途中 JSON を観測させません。
    """
    validate_value_source_descriptor(descriptor)
    result_dir = Path(run_dir).resolve() / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    destination = result_dir / "value_source_descriptor.json"
    encoded = canonical_json_bytes(dict(descriptor))
    if destination.exists():
        try:
            existing = json.loads(destination.read_text(encoding="utf-8"))
            validate_value_source_descriptor(existing)
        except (OSError, json.JSONDecodeError, ValueSourceDescriptorError) as exc:
            raise ValueSourceDescriptorError(
                "existing value_source_descriptor.json is invalid and immutable"
            ) from exc
        if canonical_json_bytes(existing) != encoded:
            raise ValueSourceDescriptorError(
                "existing value_source_descriptor.json has a different immutable identity"
            )
        return destination
    fd, temp_name = tempfile.mkstemp(
        prefix=".value_source_descriptor.",
        suffix=".tmp",
        dir=str(result_dir),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return destination


def ensure_value_source_run_unreleased(run_dir: Path) -> None:
    """公開済み descriptor を持つ run への追加書込みを拒否する。

    初心者向け:
    訓練の再開や同名 run の衝突を、config や status を書く前に止めるための共通 gate です。
    low-level writer の atomicity に加えて run 全体の provenance も immutable に保ちます。
    """
    destination = Path(run_dir).resolve() / "result" / "value_source_descriptor.json"
    if destination.exists():
        raise ValueSourceDescriptorError(
            "immutable Value Source descriptor が既に存在する run は変更できません: "
            f"{destination}"
        )


def finalize_value_source_descriptor(
    *,
    run_dir: Path,
    exit_reason: str,
    final_model_zip: Path | None,
    completion: Mapping[str, Any] | None,
    obs_schema: Mapping[str, Any] | None,
    git_commit: str | None,
    created_at_utc: str,
    source_provenance: Mapping[str, Any],
) -> Path | None:
    """train 終了理由を gate にして descriptor または incomplete marker を書く。

    初心者向け:
    ``curriculum_complete`` と model/schema の存在が揃う場合だけ result へ atomic publish し、
    SIGINT・例外・欠落時は log に理由を残して ``None`` を返します。
    """
    run_path = Path(run_dir).resolve()
    released_path = run_path / "result" / "value_source_descriptor.json"
    if released_path.exists():
        try:
            released = json.loads(released_path.read_text(encoding="utf-8"))
            validate_value_source_descriptor(released)
        except (OSError, json.JSONDecodeError, ValueSourceDescriptorError) as exc:
            raise ValueSourceDescriptorError(
                "existing value_source_descriptor.json is invalid and immutable"
            ) from exc
        return released_path
    log_dir = run_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    incomplete_path = log_dir / "value_source_descriptor.incomplete.json"
    incomplete_reason = None
    if exit_reason != "curriculum_complete":
        incomplete_reason = exit_reason
    elif final_model_zip is None or not Path(final_model_zip).is_file():
        incomplete_reason = "model_missing"
    elif obs_schema is None:
        incomplete_reason = "obs_schema_missing"
    if incomplete_reason is not None:
        incomplete_path.write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": (
                        "survivors.value_source_descriptor.incomplete.v1"
                    ),
                    "source_run": run_path.name,
                    "status": "incomplete",
                    "reason": incomplete_reason,
                }
            )
        )
        return None
    if completion is None:
        raise ValueSourceDescriptorError(
            "curriculum_complete release requires completion"
        )
    descriptor = build_value_source_descriptor(
        run_dir=run_path,
        completion=completion,
        obs_schema=obs_schema,
        git_commit=git_commit,
        created_at_utc=created_at_utc,
        source_provenance=source_provenance,
    )
    destination = write_value_source_descriptor(run_path, descriptor)
    if incomplete_path.exists():
        incomplete_path.unlink()
    return destination

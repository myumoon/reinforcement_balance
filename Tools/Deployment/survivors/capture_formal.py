"""Survivors capture session の昇格と formal descriptor 発行をまとめる。

収録済み artifact を変更せず、split と content hash を確認して公開する。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import warnings
from typing import Any, Mapping, Sequence

import yaml

from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes
from survivors.annotation_labels import ANNOTATION_ONLY_CLASSES, REGION_LABEL, iter_frame_files, read_label_file
from survivors.capture_dataset import (
    SPLIT_NAMES,
    DatasetWriter,
    SessionManifest,
    SplitFreezer,
    SplitManifest,
    _canonical_json,
    _sha256_file,
    _split_manifest_from_bytes,
    _split_manifest_paths,
    _is_split_manifest_path,
    _validated_component,
)


class PromotionError(ValueError):
    """formal 昇格の前提条件を満たさない session を示す。

    保存前の拒否理由を CLI と呼び出し側へ伝える。
    """


class PublishError(ValueError):
    """formal dataset descriptor を安全に発行できない状態を示す。

    不完全な artifact 世代を release consumer に渡さない。
    """


@dataclass(frozen=True, slots=True)
class ModelExportPaths:
    """model split に束縛する二つの COCO file を保持する。

    world と UI を同じ世代入力として publisher へ渡す。
    """

    world_coco_model: Path
    ui_coco_model: Path


def _utc_now() -> str:
    """UTC の秒精度 ISO 8601 表記を返す。

    operator checkpoint と planned split に同じ区切り文字を使う。
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _planned_prefix(checkpoint: str) -> tuple[str | None, str]:
    """checkpoint の planned split prefix と残り文字列を検証する。

    先頭二 segment だけを予約領域として読み、後続 attestation は保持する。
    """
    parts = checkpoint.split("|", 2)
    if not parts[0].startswith("planned_split:"):
        return None, checkpoint
    if len(parts) < 2 or not parts[1].startswith("planned_at:"):
        raise ValueError("invalid planned split prefix")
    split_name = parts[0][len("planned_split:") :]
    timestamp = parts[1][len("planned_at:") :]
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"unknown planned split: {split_name!r}")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid planned_at timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("planned_at timestamp must include UTC offset")
    tail = parts[2] if len(parts) == 3 else ""
    if tail.startswith("planned_split:") or tail.startswith("planned_at:"):
        raise ValueError("duplicate planned split prefix")
    return split_name, tail


def _replace_planned_prefix(checkpoint: str, split_name: str) -> str:
    """checkpoint の先頭 planned split だけを指定用途へ差し替える。

    昇格 attestation などの後続 checkpoint は一切切り詰めない。
    """
    existing, tail = _planned_prefix(checkpoint)
    if existing == split_name:
        return checkpoint
    return f"planned_split:{split_name}|planned_at:{_utc_now()}|{tail}"


def promote_session(
    store_root: str | Path,
    session_id: str,
    *,
    operator_id: str,
    attestation_note: str,
    expected_target_profile_hash: str,
    expected_game_build_id: str,
    split_manifest_path: str | Path | None = None,
    planned_split: str | None = None,
) -> SessionManifest:
    """session manifest に formal eligibility と operator attestation を保存する。

    frame、metadata、annotation、build identity は読み取りだけに限定する。
    """
    if not operator_id or not attestation_note or any(c in operator_id + attestation_note for c in "|\r\n"):
        raise PromotionError("operator id and note must be non-empty single segments")
    if planned_split is not None and planned_split not in SPLIT_NAMES:
        raise PromotionError(f"unknown planned split: {planned_split!r}")
    root = Path(store_root).expanduser().resolve()
    manifest = DatasetWriter.restore(root, session_id, metadata_only=True)
    if manifest.pixel_source != "lossless_png":
        raise PromotionError("session pixel source must be lossless_png")
    if (
        manifest.game_build_id != expected_game_build_id
        or not expected_game_build_id
        or "TEST_FIXTURE_ONLY" in manifest.game_build_id
        or "MANUAL_ATTESTATION_REQUIRED" in manifest.game_build_id
    ):
        raise PromotionError("session build id is not the expected production build")
    if manifest.target_profile_hash != expected_target_profile_hash:
        raise PromotionError("session target profile hash does not match expected profile")
    if manifest.frame_count <= 0:
        raise PromotionError("session must contain at least one frame")
    if _sha256_file(manifest.session_path / "frames.jsonl") != manifest.metadata_sha256:
        raise PromotionError("session frames metadata hash mismatch")

    if split_manifest_path is not None:
        split_path = Path(split_manifest_path).expanduser()
        if not split_path.is_absolute():
            split_path = root / split_path
        if not _is_split_manifest_path(root, split_path.resolve()):
            raise PromotionError("split manifest path must match the root-level capture split manifest pattern")
    for split_path in _split_manifest_paths(root):
        if not split_path.is_file():
            continue
        try:
            split = _split_manifest_from_bytes(split_path, split_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise PromotionError(f"invalid split manifest: {exc}") from exc
        if any(unit.session_id == manifest.session_id for units in split.splits.values() for unit in units):
            raise PromotionError("frozen split contains this session")

    checkpoint = manifest.operator_checkpoint
    try:
        if planned_split is not None:
            checkpoint = _replace_planned_prefix(checkpoint, planned_split)
        else:
            _planned_prefix(checkpoint)
    except ValueError as exc:
        raise PromotionError(str(exc)) from exc
    if manifest.formal_dataset_eligible and checkpoint == manifest.operator_checkpoint:
        return manifest
    if not manifest.formal_dataset_eligible:
        checkpoint = f"{checkpoint}|formal:{operator_id}:{_utc_now()}:{attestation_note}"

    wire = {
        "schema_version": manifest.schema_version,
        "session_id": manifest.session_id,
        "target_profile_hash": manifest.target_profile_hash,
        "game_build_id": manifest.game_build_id,
        "formal_dataset_eligible": True,
        "operator_checkpoint": checkpoint,
        "pixel_source": manifest.pixel_source,
        "metadata_path": manifest.metadata_path,
        "metadata_sha256": manifest.metadata_sha256,
        "frame_count": manifest.frame_count,
    }
    encoded = _canonical_json(wire)
    path = manifest.session_path / "session_manifest.json"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return DatasetWriter.restore(root, session_id, metadata_only=True)


def assign_from_manifests(
    store_root: str | Path,
    *,
    session_ids: Sequence[str] | None = None,
    manifest_path: str | Path | None = None,
) -> SplitFreezer:
    """保存済み checkpoint から昇格済み session の split 割当を組み立てる。

    未昇格 session は無視し、用途未指定の昇格 session は学習用へ警告付きで送る。
    """
    root = Path(store_root).expanduser().resolve()
    if isinstance(session_ids, (str, bytes)):
        raise ValueError("session_ids must be a sequence of session ids")
    ids = list(session_ids) if session_ids is not None else sorted(
        path.name for path in (root / "capture_sessions").iterdir() if path.is_dir()
    )
    buckets: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for session_id in ids:
        manifest = DatasetWriter.restore(root, session_id, metadata_only=True)
        if manifest.formal_dataset_eligible is not True:
            continue
        try:
            split_name, _ = _planned_prefix(manifest.operator_checkpoint)
        except ValueError as exc:
            raise PublishError(f"invalid planned split for {session_id}: {exc}") from exc
        if split_name is None:
            warnings.warn(
                f"formal session {session_id} has no planned split; assigning model_train",
                RuntimeWarning,
                stacklevel=2,
            )
            split_name = "model_train"
        buckets[split_name].append(session_id)
    freezer = SplitFreezer(root, manifest_path=manifest_path)
    for split_name in SPLIT_NAMES:
        if buckets[split_name]:
            freezer.assign(split_name, buckets[split_name])
    return freezer


def _sha256_bytes_local(content: bytes) -> str:
    """descriptor generation input の SHA-256 を返す。

    canonical artifact generation tag の入力形式を一箇所に揃える。
    """
    return hashlib.sha256(content).hexdigest()


def _put_file(store: ArtifactStore, logical_id: str, path: Path, media_type: str) -> Any:
    """ローカル file を content-addressed store に保存する。

    ArtifactStore 自身の hash 検証と logical-id 冪等性を利用する。
    """
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise PublishError(f"cannot read required artifact {path}: {exc}") from exc
    return store.put_bytes(logical_id=logical_id, data=content, media_type=media_type)


def _annotation_schema(path: Path) -> str | None:
    """JSONL annotation の先頭 schema_version を返す。

    空ファイルは None とし、JSON の破損や未知 schema は publish error にする。
    """
    try:
        first = next(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except StopIteration:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise PublishError(f"cannot read annotations: {path}") from exc
    try:
        row = json.loads(first)
    except json.JSONDecodeError as exc:
        raise PublishError(f"invalid annotation JSONL: {path}") from exc
    if not isinstance(row, dict):
        raise PublishError(f"invalid annotation row: {path}")
    schema = row.get("schema_version")
    if schema is not None and schema != "perception_formal_annotation.v1":
        raise PublishError(f"unknown annotation schema: {schema!r}")
    return schema


def _file_binding(ref: Any, manifest_ref: Any) -> dict[str, str]:
    """consumer が要求する四つの annotation binding key を作る。

    semantic annotation と、その session manifest の参照 hash を対にする。
    """
    return {
        "annotation_logical_id": ref.logical_id,
        "annotation_sha256": ref.sha256,
        "session_manifest_logical_id": manifest_ref.logical_id,
        "session_manifest_sha256": manifest_ref.sha256,
    }


def _validated_window_sessions(value: Any) -> dict[str, list[list[int]]]:
    """04-18 evaluation_windows.v1 の session 区間を検証する。

    各 session は一つ以上の非負な閉区間 pair を持つ。
    """
    if not isinstance(value, Mapping):
        raise PublishError("invalid evaluation windows sessions")
    sessions: dict[str, list[list[int]]] = {}
    for session_id, windows in value.items():
        if not isinstance(session_id, str) or not session_id or not isinstance(windows, list) or not windows:
            raise PublishError(f"invalid evaluation window: {session_id}")
        validated = []
        for window in windows:
            if (
                not isinstance(window, list)
                or len(window) != 2
                or type(window[0]) is not int
                or type(window[1]) is not int
                or window[0] < 0
                or window[1] < window[0]
            ):
                raise PublishError(f"invalid evaluation window: {session_id}")
            validated.append([window[0], window[1]])
        sessions[session_id] = validated
    return sessions


def _load_windows(
    path: Path | None,
) -> tuple[dict[str, list[list[int]]] | None, int | None, str | None, bytes | None]:
    """evaluation windows の schema、sessions、stride、hash を読む。

    descriptor identity には path を含めず、plan の再現に必要な値だけ保存する。
    """
    if path is None:
        return None, None, None, None
    content = path.read_bytes()
    wire = json.loads(content)
    if not isinstance(wire, dict) or wire.get("schema_version") != "evaluation_windows.v1":
        raise PublishError("unknown evaluation windows schema")
    sessions = _validated_window_sessions(wire.get("sessions"))
    stride = wire.get("stride")
    plan_hash = wire.get("plan_canonical_hash")
    if type(stride) is not int or stride < 1 or not isinstance(plan_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", plan_hash):
        raise PublishError("invalid evaluation windows identity")
    return sessions, stride, plan_hash, content


def publish_capture_dataset(
    store: ArtifactStore,
    *,
    store_root: str | Path,
    split_manifest_path: str | Path,
    logical_prefix: str,
    evaluation_windows_path: Path | None = None,
    model_exports: ModelExportPaths | None = None,
    agreement_path: Path | None = None,
    formal_gates_path: Path | None = None,
    allow_missing_formal_annotations: bool = False,
) -> ArtifactDescriptor:
    """frozen capture dataset を descriptor として ArtifactStore に発行する。

    session と外部 evaluation asset の hash を世代 ID と consumer bindings に結ぶ。
    """
    if not isinstance(store, ArtifactStore):
        raise TypeError("store must be ArtifactStore")
    root = Path(store_root).expanduser().resolve()
    split_path = Path(split_manifest_path).expanduser().resolve()
    try:
        split_bytes = split_path.read_bytes()
        split = _split_manifest_from_bytes(split_path, split_bytes)
    except (OSError, ValueError) as exc:
        raise PublishError(f"invalid frozen split manifest: {exc}") from exc
    all_units = [(name, unit) for name in SPLIT_NAMES for unit in split.splits[name]]
    if not all_units:
        raise PublishError("frozen split has no sessions")
    sealed_units = [unit for name, unit in all_units if name in {"error_calibration", "final_e2e_test"}]
    model_units = [unit for name, unit in all_units if name in {"model_train", "model_validation"}]
    if allow_missing_formal_annotations and sealed_units:
        raise PublishError("--allow-missing-formal-annotations is only valid when sealed splits are empty")
    if sealed_units and formal_gates_path is None:
        raise PublishError("formal_gates is required when sealed sessions exist")

    windows, windows_stride, windows_plan_hash, windows_bytes = _load_windows(evaluation_windows_path)
    gates_bytes = None
    gates_schema = None
    if formal_gates_path is not None:
        try:
            gates_bytes = Path(formal_gates_path).read_bytes()
            gates_wire = yaml.safe_load(gates_bytes)
        except (OSError, yaml.YAMLError) as exc:
            raise PublishError("cannot read formal gates YAML") from exc
        if not isinstance(gates_wire, dict) or gates_wire.get("schema_version") != "formal_gates.v1":
            raise PublishError("unknown formal gates schema")
        gates_schema = gates_wire["schema_version"]

    model_contents: dict[str, bytes] = {}
    if model_exports is not None:
        model_contents = {
            "world_coco_model": Path(model_exports.world_coco_model).read_bytes(),
            "ui_coco_model": Path(model_exports.ui_coco_model).read_bytes(),
        }
    agreement_bytes = Path(agreement_path).read_bytes() if agreement_path is not None else None
    if agreement_bytes is not None:
        agreement = json.loads(agreement_bytes)
        if not isinstance(agreement, dict) or agreement.get("schema_version") != "annotation_agreement.v1":
            raise PublishError("unknown annotation agreement schema")
    generation_tag = canonical_hash({
        "formal_gates": _sha256_bytes_local(gates_bytes) if gates_bytes is not None else "absent",
        "evaluation_windows": _sha256_bytes_local(windows_bytes) if windows_bytes is not None else "absent",
        "world_coco_model": _sha256_bytes_local(model_contents["world_coco_model"]) if model_contents else "absent",
        "ui_coco_model": _sha256_bytes_local(model_contents["ui_coco_model"]) if model_contents else "absent",
        "agreement": _sha256_bytes_local(agreement_bytes) if agreement_bytes is not None else "absent",
    })[:8]
    generation = f"{split.manifest_sha256[:16]}-{generation_tag}"
    files: list[Any] = []
    bindings: dict[str, dict[str, str]] = {}
    model_bindings: dict[str, dict[str, str]] = {}
    frames_bindings: dict[str, dict[str, str]] = {}
    hud_bindings: dict[str, dict[str, str]] = {}
    coco_bindings: dict[str, dict[str, str]] = {}
    session_rows: list[tuple[str, str, str, list[str]]] = []
    builds: set[str] = set()
    profiles: set[str] = set()
    sealed_ids = {unit.session_id for unit in sealed_units}
    model_ids = {unit.session_id for unit in model_units}
    windows_sessions = set(windows or {})
    if windows is not None and not sealed_ids.issubset(windows_sessions):
        raise PublishError("evaluation windows omit a sealed session")
    if sealed_ids and windows is None:
        warnings.warn(
            "evaluation windows are absent; downstream sealed formal benchmark cannot run",
            RuntimeWarning,
            stacklevel=2,
        )

    manifest_ref = store.put_bytes(
        logical_id=f"{logical_prefix}/{split.manifest_sha256[:16]}/capture_split_manifest.json",
        data=split_bytes,
        media_type="application/json",
    )
    files.append(manifest_ref)
    for split_name, unit in all_units:
        try:
            session = DatasetWriter.restore(root, unit.session_id, metadata_only=True)
        except (OSError, ValueError) as exc:
            raise PublishError(f"cannot restore session {unit.session_id}: {exc}") from exc
        session_manifest_path = session.session_path / "session_manifest.json"
        if (
            _sha256_file(session_manifest_path) != unit.session_manifest_sha256
            or session.manifest_sha256 != unit.session_manifest_sha256
            or session.formal_dataset_eligible is not True
            or session.pixel_source != "lossless_png"
        ):
            raise PublishError(f"session is not formal or split hash mismatched: {unit.session_id}")
        builds.add(session.game_build_id)
        profiles.add(session.target_profile_hash)
        session_base = f"{logical_prefix}/{split.manifest_sha256[:16]}/sessions/{unit.session_id}"
        session_ref = _put_file(store, f"{session_base}/session_manifest.json", session_manifest_path, "application/json")
        frames_path = session.session_path / session.metadata_path
        frames_ref = _put_file(store, f"{session_base}/frames.jsonl", frames_path, "application/x-ndjson")
        frames_bindings[unit.session_id] = {"frames_logical_id": frames_ref.logical_id, "frames_sha256": frames_ref.sha256}
        files.extend((session_ref, frames_ref))

        annotations_path = session.session_path / "annotations.jsonl"
        annotation_ref = None
        annotation_schema = None
        if annotations_path.is_file():
            annotation_schema = _annotation_schema(annotations_path)
            annotation_ref = _put_file(store, f"{session_base}/annotations.jsonl", annotations_path, "application/x-ndjson")
            files.append(annotation_ref)
        elif unit.session_id in sealed_ids:
            raise PublishError(f"sealed session has no formal annotations: {unit.session_id}")
        if unit.session_id in sealed_ids and annotation_schema != "perception_formal_annotation.v1":
            raise PublishError(f"sealed session has no formal annotations: {unit.session_id}")
        if annotation_ref is not None:
            binding = _file_binding(annotation_ref, session_ref)
            if unit.session_id in sealed_ids:
                bindings[unit.session_id] = binding
            elif unit.session_id in model_ids:
                model_bindings[unit.session_id] = binding

        row_extra: list[str] = [frames_ref.sha256]
        if annotation_ref is not None:
            row_extra.append(annotation_ref.sha256)
        hud_path = session.session_path / "hud_truth.jsonl"
        if hud_path.is_file():
            hud_ref = _put_file(store, f"{session_base}/hud_truth.jsonl", hud_path, "application/x-ndjson")
            files.append(hud_ref)
            hud_bindings[unit.session_id] = {"hud_truth_logical_id": hud_ref.logical_id, "hud_truth_sha256": hud_ref.sha256}
            row_extra.append(hud_ref.sha256)
        if unit.session_id in sealed_ids:
            coco_dir = root / "coco" / unit.session_id
            world_path, ui_path = coco_dir / "world_coco.json", coco_dir / "ui_coco.json"
            if not world_path.is_file() or not ui_path.is_file():
                raise PublishError(f"sealed session COCO exports are missing: {unit.session_id}")
            world_ref = _put_file(store, f"{session_base}/world_coco.json", world_path, "application/json")
            ui_ref = _put_file(store, f"{session_base}/ui_coco.json", ui_path, "application/json")
            files.extend((world_ref, ui_ref))
            coco_bindings[unit.session_id] = {
                "world_coco_logical_id": world_ref.logical_id,
                "world_coco_sha256": world_ref.sha256,
                "ui_coco_logical_id": ui_ref.logical_id,
                "ui_coco_sha256": ui_ref.sha256,
            }
            row_extra.extend((world_ref.sha256, ui_ref.sha256))
        session_rows.append((unit.session_id, unit.session_manifest_sha256, annotation_ref.sha256 if annotation_ref else "absent", row_extra))

    if len(builds) != 1 or len(profiles) != 1:
        raise PublishError("dataset sessions must share game_build_id and target_profile_hash")
    if set(bindings) != sealed_ids or set(model_bindings) != model_ids:
        raise PublishError("annotation bindings do not match formal and model sessions")

    dataset_exports: dict[str, Any] = {}
    if model_contents:
        export_refs = {}
        for key, content in model_contents.items():
            ref = store.put_bytes(
                logical_id=f"{logical_prefix}/{generation}/{key}.json",
                data=content,
                media_type="application/json",
            )
            files.append(ref)
            export_refs[key] = {"logical_id": ref.logical_id, "sha256": ref.sha256}
        dataset_exports.update(export_refs)
    if agreement_bytes is not None:
        ref = store.put_bytes(
            logical_id=f"{logical_prefix}/{generation}/annotation_agreement.json",
            data=agreement_bytes,
            media_type="application/json",
        )
        files.append(ref)
        dataset_exports["annotation_agreement"] = {"logical_id": ref.logical_id, "sha256": ref.sha256}
    if formal_gates_path is not None:
        gates_ref = store.put_bytes(
            logical_id=f"{logical_prefix}/{generation}/formal_gates.yaml",
            data=gates_bytes,
            media_type="application/yaml",
        )
        files.append(gates_ref)
        gates_binding = {"logical_id": gates_ref.logical_id, "sha256": gates_ref.sha256, "schema_version": gates_schema}
    else:
        gates_binding = None
    if windows_bytes is not None:
        windows_ref = store.put_bytes(
            logical_id=f"{logical_prefix}/{generation}/evaluation_windows.json",
            data=windows_bytes,
            media_type="application/json",
        )
        files.append(windows_ref)

    windows_hash = _sha256_bytes_local(windows_bytes) if windows_bytes is not None else None
    gates_hash = _sha256_bytes_local(gates_bytes) if gates_bytes is not None else None
    dataset_hash = canonical_hash({
        "sessions": sorted(session_rows),
        "split_manifest_sha256": split.manifest_sha256,
        "evaluation_windows_hash": windows_hash,
        "formal_gates_sha256": gates_hash,
        "dataset_exports_sha256": {
            key: _sha256_bytes_local(content) for key, content in sorted(model_contents.items())
        },
        "annotation_agreement_sha256": (
            _sha256_bytes_local(agreement_bytes) if agreement_bytes is not None else None
        ),
    })
    identity: dict[str, Any] = {
        "manifest_logical_id": manifest_ref.logical_id,
        "annotation_bindings": bindings,
        "model_session_bindings": model_bindings,
        "hud_truth_bindings": hud_bindings,
        "frames_bindings": frames_bindings,
        "coco_bindings": coco_bindings,
        "dataset_exports": dataset_exports,
        "evaluation_windows": (
            {"sessions": windows, "stride": windows_stride, "plan_canonical_hash": windows_plan_hash}
            if windows is not None else None
        ),
        "evaluation_windows_hash": windows_hash,
        "formal_gates": gates_binding,
        "dataset_identity_hash": dataset_hash,
        "game_build_id": next(iter(builds)),
        "target_profile_hash": next(iter(profiles)),
        "session_count_by_split": {name: len(split.splits[name]) for name in SPLIT_NAMES},
    }
    descriptor = ArtifactDescriptor(
        logical_id=f"{logical_prefix}/{generation}/descriptor.json",
        node_kind="source_descriptor",
        producer_id="capture-dataset-formal-publisher",
        producer_version="d04_capture_dataset.v1",
        parents=(),
        files=files,
        identity_metadata=identity,
    )
    store.put_bytes(
        logical_id=descriptor.logical_id,
        data=canonical_json_bytes(descriptor.to_wire()),
        media_type="application/json",
    )
    verify_capture_dataset_descriptor(store, descriptor, root)
    return descriptor


def _binding_ref(descriptor: ArtifactDescriptor, logical_id: str, sha256: str) -> Any:
    """descriptor file から logical id と hash が一致する ref を引く。

    binding typo や ref 差し替えを consumer と同じ境界で fail-closed にする。
    """
    matches = [ref for ref in descriptor.files if ref.logical_id == logical_id and ref.sha256 == sha256]
    if len(matches) != 1:
        raise PublishError("descriptor binding does not identify one artifact file")
    return matches[0]


def verify_capture_dataset_descriptor(
    store: ArtifactStore, descriptor: ArtifactDescriptor, store_root: str | Path
) -> None:
    """published descriptor の split/session/hash binding を再検証する。

    production module から benchmark consumer を import せず同じ wire 条件を確認する。
    """
    if descriptor.node_kind != "source_descriptor":
        raise PublishError("capture dataset must be a source_descriptor")
    for ref in descriptor.files:
        if not store.verify(ref).ok:
            raise PublishError(f"artifact verification failed: {ref.logical_id}")
    identity = descriptor.identity_metadata
    manifest_id = identity.get("manifest_logical_id")
    refs = [ref for ref in descriptor.files if ref.logical_id == manifest_id]
    if len(refs) != 1:
        raise PublishError("descriptor manifest reference is missing or ambiguous")
    split_bytes = store.object_path(refs[0].store_uri).read_bytes()
    if _sha256_bytes_local(split_bytes) != refs[0].sha256:
        raise PublishError("split manifest sha256 mismatch")
    split = _split_manifest_from_bytes(Path(refs[0].logical_id), split_bytes)
    sessions = {unit.session_id: unit for units in split.splits.values() for unit in units}
    sealed_ids = {unit.session_id for name in ("error_calibration", "final_e2e_test") for unit in split.splits[name]}
    model_ids = {unit.session_id for name in ("model_train", "model_validation") for unit in split.splits[name]}
    annotations = identity.get("annotation_bindings")
    model_annotations = identity.get("model_session_bindings")
    frames = identity.get("frames_bindings")
    hud = identity.get("hud_truth_bindings")
    if not isinstance(annotations, Mapping) or set(annotations) != sealed_ids:
        raise PublishError("annotation bindings do not exactly cover sealed sessions")
    if not isinstance(model_annotations, Mapping) or set(model_annotations) != model_ids:
        raise PublishError("model session bindings do not exactly cover model sessions")
    if not isinstance(frames, Mapping) or set(frames) != set(sessions):
        raise PublishError("frames bindings do not exactly cover split sessions")
    if not isinstance(hud, Mapping) or not set(hud).issubset(sessions):
        raise PublishError("HUD truth bindings reference an unknown session")
    coco = identity.get("coco_bindings")
    if not isinstance(coco, Mapping) or set(coco) != sealed_ids:
        raise PublishError("COCO bindings do not exactly cover sealed sessions")
    root = Path(store_root).expanduser().resolve()
    session_rows: list[tuple[str, str, str, list[str]]] = []
    builds: set[str] = set()
    profiles: set[str] = set()
    for session_id, unit in sessions.items():
        session = DatasetWriter.restore(root, session_id, metadata_only=True)
        if session.formal_dataset_eligible is not True or session.manifest_sha256 != unit.session_manifest_sha256:
            raise PublishError(f"session is not formal or split hash mismatched: {session_id}")
        builds.add(session.game_build_id)
        profiles.add(session.target_profile_hash)
        binding = annotations.get(session_id) if session_id in sealed_ids else model_annotations.get(session_id)
        if not isinstance(binding, Mapping) or set(binding) != {
            "annotation_logical_id", "annotation_sha256", "session_manifest_logical_id", "session_manifest_sha256"
        }:
            raise PublishError(f"annotation binding has wrong shape: {session_id}")
        _binding_ref(descriptor, binding["annotation_logical_id"], binding["annotation_sha256"])
        _binding_ref(descriptor, binding["session_manifest_logical_id"], binding["session_manifest_sha256"])
        if binding["session_manifest_sha256"] != unit.session_manifest_sha256:
            raise PublishError(f"session binding hash mismatch: {session_id}")
        annotations_path = session.session_path / "annotations.jsonl"
        if not annotations_path.is_file() or _sha256_file(annotations_path) != binding["annotation_sha256"]:
            raise PublishError(f"annotation binding hash mismatch: {session_id}")
        if session_id in sealed_ids and _annotation_schema(annotations_path) != "perception_formal_annotation.v1":
            raise PublishError(f"sealed session has no formal annotations: {session_id}")
        frame_binding = frames[session_id]
        if not isinstance(frame_binding, Mapping) or set(frame_binding) != {"frames_logical_id", "frames_sha256"}:
            raise PublishError(f"frames binding has wrong shape: {session_id}")
        _binding_ref(descriptor, frame_binding["frames_logical_id"], frame_binding["frames_sha256"])
        if _sha256_file(session.session_path / session.metadata_path) != frame_binding["frames_sha256"]:
            raise PublishError(f"frames binding hash mismatch: {session_id}")
        extra_hashes = [frame_binding["frames_sha256"], binding["annotation_sha256"]]
        if session_id in hud:
            extra_hashes.append(hud[session_id]["hud_truth_sha256"])
        if session_id in sealed_ids:
            extra_hashes.extend((
                identity["coco_bindings"][session_id]["world_coco_sha256"],
                identity["coco_bindings"][session_id]["ui_coco_sha256"],
            ))
        session_rows.append((session_id, unit.session_manifest_sha256, binding["annotation_sha256"], extra_hashes))
    for session_id, binding in hud.items():
        if not isinstance(binding, Mapping) or set(binding) != {"hud_truth_logical_id", "hud_truth_sha256"}:
            raise PublishError(f"HUD truth binding has wrong shape: {session_id}")
        _binding_ref(descriptor, binding["hud_truth_logical_id"], binding["hud_truth_sha256"])
        truth_path = root / "capture_sessions" / session_id / "hud_truth.jsonl"
        if not truth_path.is_file() or _sha256_file(truth_path) != binding["hud_truth_sha256"]:
            raise PublishError(f"HUD truth binding hash mismatch: {session_id}")
    for session_id, binding in coco.items():
        if not isinstance(binding, Mapping) or set(binding) != {
            "world_coco_logical_id", "world_coco_sha256", "ui_coco_logical_id", "ui_coco_sha256"
        }:
            raise PublishError(f"COCO binding has wrong shape: {session_id}")
        for role in ("world_coco", "ui_coco"):
            digest = binding[f"{role}_sha256"]
            _binding_ref(descriptor, binding[f"{role}_logical_id"], digest)
            path = root / "coco" / session_id / f"{role}.json"
            if not path.is_file() or _sha256_file(path) != digest:
                raise PublishError(f"sealed COCO binding hash mismatch: {session_id}")
    gates = identity.get("formal_gates")
    gates_hash = None
    if sealed_ids and (not isinstance(gates, Mapping) or set(gates) != {"logical_id", "sha256", "schema_version"}):
        raise PublishError("sealed dataset descriptor lacks formal_gates")
    if gates is not None:
        if gates.get("schema_version") != "formal_gates.v1":
            raise PublishError("unknown formal gates schema")
        gates_ref = _binding_ref(descriptor, gates["logical_id"], gates["sha256"])
        try:
            gates_wire = yaml.safe_load(store.object_path(gates_ref.store_uri).read_bytes())
        except yaml.YAMLError as exc:
            raise PublishError("invalid formal gates YAML") from exc
        if not isinstance(gates_wire, Mapping) or gates_wire.get("schema_version") != "formal_gates.v1":
            raise PublishError("unknown formal gates schema")
        gates_hash = gates["sha256"]
    windows_binding = identity.get("evaluation_windows")
    windows_hash = identity.get("evaluation_windows_hash")
    if windows_binding is not None:
        if not isinstance(windows_binding, Mapping) or set(windows_binding) != {
            "sessions", "stride", "plan_canonical_hash"
        }:
            raise PublishError("evaluation windows binding has wrong shape")
        sessions_binding = _validated_window_sessions(windows_binding["sessions"])
        if (
            not sealed_ids.issubset(sessions_binding)
            or type(windows_binding["stride"]) is not int
            or windows_binding["stride"] < 1
            or not isinstance(windows_binding["plan_canonical_hash"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", windows_binding["plan_canonical_hash"])
        ):
            raise PublishError("evaluation windows omit a sealed session")
        windows_refs = [ref for ref in descriptor.files if ref.sha256 == windows_hash]
        if len(windows_refs) != 1:
            raise PublishError("evaluation windows file hash is missing or ambiguous")
        windows_wire = json.loads(store.object_path(windows_refs[0].store_uri).read_bytes())
        if (
            not isinstance(windows_wire, Mapping)
            or windows_wire.get("schema_version") != "evaluation_windows.v1"
            or _validated_window_sessions(windows_wire.get("sessions")) != sessions_binding
            or type(windows_wire.get("stride")) is not int
            or windows_wire["stride"] < 1
            or windows_wire.get("stride") != windows_binding["stride"]
            or windows_wire.get("plan_canonical_hash") != windows_binding["plan_canonical_hash"]
        ):
            raise PublishError("evaluation windows identity does not match file")
    elif windows_hash is not None:
        raise PublishError("evaluation windows hash has no identity binding")

    exports = identity.get("dataset_exports")
    if not isinstance(exports, Mapping) or set(exports) - {
        "world_coco_model", "ui_coco_model", "annotation_agreement"
    }:
        raise PublishError("dataset_exports has unknown fields")
    if ("world_coco_model" in exports) != ("ui_coco_model" in exports):
        raise PublishError("dataset_exports must bind both model COCO files")
    export_hashes: dict[str, str] = {}
    for name in ("world_coco_model", "ui_coco_model", "annotation_agreement"):
        if name not in exports:
            continue
        binding = exports[name]
        if not isinstance(binding, Mapping) or set(binding) != {"logical_id", "sha256"}:
            raise PublishError(f"dataset export binding has wrong shape: {name}")
        ref = _binding_ref(descriptor, binding["logical_id"], binding["sha256"])
        content = store.object_path(ref.store_uri).read_bytes()
        if name == "annotation_agreement":
            agreement = json.loads(content)
            if not isinstance(agreement, Mapping) or agreement.get("schema_version") != "annotation_agreement.v1":
                raise PublishError("unknown annotation agreement schema")
        export_hashes[name] = ref.sha256
    expected_dataset_hash = canonical_hash({
        "sessions": sorted(session_rows),
        "split_manifest_sha256": split.manifest_sha256,
        "evaluation_windows_hash": windows_hash,
        "formal_gates_sha256": gates_hash,
        "dataset_exports_sha256": {
            key: export_hashes[key] for key in ("world_coco_model", "ui_coco_model") if key in export_hashes
        },
        "annotation_agreement_sha256": export_hashes.get("annotation_agreement"),
    })
    if (
        len(builds) != 1
        or len(profiles) != 1
        or identity.get("game_build_id") != next(iter(builds))
        or identity.get("target_profile_hash") != next(iter(profiles))
        or identity.get("dataset_identity_hash") != expected_dataset_hash
    ):
        raise PublishError("dataset identity hash or session build/profile does not match contents")


def _box_iou(first: Any, second: Any) -> float:
    """二つの label box の intersection over union を計算する。

    agreement matcher が使う半開区間の pixel 座標を同じ面積式で扱う。
    """
    width = max(0.0, min(first.right, second.right) - max(first.left, second.left))
    height = max(0.0, min(first.bottom, second.bottom) - max(first.top, second.top))
    intersection = width * height
    first_area = (first.right - first.left) * (first.bottom - first.top)
    second_area = (second.right - second.left) * (second.bottom - second.top)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _checked_boxes(work_root: Path, session_id: str) -> dict[int, list[Any]]:
    """checked frame に属する COCO と同じ母集団の box を返す。

    weapon_target と labeled_region を除外し、部分未確認 label を混入させない。
    """
    session_dir = work_root / session_id
    if not session_dir.is_dir():
        raise ValueError(f"missing annotation session: {session_id}")
    boxes_by_frame: dict[int, list[Any]] = {}
    for frame_id, _, label_path in iter_frame_files(session_dir):
        if label_path is None:
            continue
        boxes, checked = read_label_file(label_path)
        if checked and not any(box.label == REGION_LABEL for box in boxes):
            boxes_by_frame[frame_id] = [
                box for box in boxes if box.label not in ANNOTATION_ONLY_CLASSES
            ]
    return boxes_by_frame


def build_annotation_agreement(
    work_root_a: str | Path,
    work_root_b: str | Path,
    session_ids: Sequence[str],
    *,
    iou_match: float = 0.5,
) -> dict[str, Any]:
    """独立二重 annotation の frame 単位 greedy IoU agreement を集計する。

    class ごとに IoU 最大 pair を採り、未対応 box と class mismatch を数える。
    """
    if not isinstance(iou_match, (int, float)) or isinstance(iou_match, bool) or not 0 < iou_match <= 1:
        raise ValueError("iou_match must be in (0, 1]")
    root_a, root_b = Path(work_root_a).expanduser().resolve(), Path(work_root_b).expanduser().resolve()
    if not root_a.is_dir() or not root_b.is_dir():
        raise ValueError("both work roots must exist")
    session_values = tuple(dict.fromkeys(_validated_component(value, "session_id") for value in session_ids))
    all_ious: list[float] = []
    all_matches: list[bool] = []
    unmatched_first = unmatched_second = 0
    sessions: list[dict[str, Any]] = []
    per_class: dict[str, dict[str, Any]] = defaultdict(lambda: {"pair_count": 0, "iou_sum": 0.0, "class_matches": 0})
    for session_id in session_values:
        first_frames = _checked_boxes(root_a, session_id)
        second_frames = _checked_boxes(root_b, session_id)
        session_ious: list[float] = []
        session_matches: list[bool] = []
        session_unmatched_first = session_unmatched_second = 0
        for frame_id in sorted(first_frames.keys() & second_frames.keys()):
            first_boxes, second_boxes = first_frames[frame_id], second_frames[frame_id]
            candidates = sorted(
                ((_box_iou(left, right), i, j) for i, left in enumerate(first_boxes) for j, right in enumerate(second_boxes)),
                reverse=True,
            )
            used_first: set[int] = set()
            used_second: set[int] = set()
            for score, i, j in candidates:
                if score < iou_match:
                    break
                if i in used_first or j in used_second:
                    continue
                used_first.add(i)
                used_second.add(j)
                same_class = first_boxes[i].label == second_boxes[j].label
                session_ious.append(score)
                session_matches.append(same_class)
                stat = per_class[first_boxes[i].label]
                stat["pair_count"] += 1
                stat["iou_sum"] += score
                stat["class_matches"] += int(same_class)
            session_unmatched_first += len(first_boxes) - len(used_first)
            session_unmatched_second += len(second_boxes) - len(used_second)
        unmatched_first += session_unmatched_first
        unmatched_second += session_unmatched_second
        all_ious.extend(session_ious)
        all_matches.extend(session_matches)
        sessions.append({
            "session_id": session_id,
            "pair_count": len(session_ious),
            "iou_mean": sum(session_ious) / len(session_ious) if session_ious else 0.0,
            "class_agreement": sum(session_matches) / len(session_matches) if session_matches else 0.0,
            "unmatched_first": session_unmatched_first,
            "unmatched_second": session_unmatched_second,
        })
    return {
        "schema_version": "annotation_agreement.v1",
        "pair_count": len(all_ious),
        "iou_mean": sum(all_ious) / len(all_ious) if all_ious else 0.0,
        "class_agreement": sum(all_matches) / len(all_matches) if all_matches else 0.0,
        "unmatched_first": unmatched_first,
        "unmatched_second": unmatched_second,
        "sessions": sessions,
        "per_class": {
            name: {
                "pair_count": values["pair_count"],
                "iou_mean": values["iou_sum"] / values["pair_count"],
                "class_agreement": values["class_matches"] / values["pair_count"],
            }
            for name, values in sorted(per_class.items())
        },
    }


__all__ = [
    "ModelExportPaths",
    "PromotionError",
    "PublishError",
    "assign_from_manifests",
    "build_annotation_agreement",
    "promote_session",
    "publish_capture_dataset",
    "verify_capture_dataset_descriptor",
]

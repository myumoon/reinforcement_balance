"""WorldDetector の formal release identity を検証する。\n\npreflight と verdict の結果を checkpoint、package、runtime 用設定へ束縛する。"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import yaml

from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    RestoreTestVerdict,
    is_sha256_hex,
)
from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes
from survivors.capture_dataset import DatasetWriter, SPLIT_NAMES, load_split_manifest_by_hash
from survivors.vision.world_dataset import SessionSplit, WorldDataset, load_class_map
from survivors.vision.session_cluster_ci import cluster_ci_bounds


class FormalPreflightError(ValueError):
    """formal preflight の入力条件を満たさない。\n\n失敗理由を CLI と test が区別できるよう ValueError として伝える。"""


def _sha256(data: bytes) -> str:
    """bytes の SHA-256 を lower-case hex で返す。

    release identity にファイル内容を結び付けるために使う。
    """
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: pathlib.Path) -> str:
    """file を chunk 単位で読み SHA-256 を返す。

    大きな PNG や annotation export を全量メモリへ載せない。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_config_hash(config: Mapping[str, Any]) -> str:
    """trainer と共通の canonical serialization で config hash を計算する。

    sort key と UTF-8 を固定し、CLI override を含む設定を checkpoint と照合する。
    """
    return _sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def _override_config(config: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """既存 config key に限って CLI override を再帰適用する。

    未知 key を許すと config validator をすり抜けるため、追加と型置換を拒否する。
    """
    result = json.loads(json.dumps(config))
    for key, value in overrides.items():
        if key not in result:
            raise FormalPreflightError(f"unknown config override: {key}")
        if isinstance(result[key], dict):
            if not isinstance(value, Mapping):
                raise FormalPreflightError(f"config override type mismatch: {key}")
            result[key] = _override_config(result[key], value)
        else:
            result[key] = value
    return result


def _descriptor_file(descriptor: ArtifactDescriptor, logical_id: str, digest: str) -> None:
    """descriptor 内の logical id と sha256 が同じ file を束縛する。

    一方だけが一致する曖昧な artifact 参照は formal 入力として採用しない。
    """
    matches = [ref for ref in descriptor.files if ref.logical_id == logical_id and ref.sha256 == digest]
    if len(matches) != 1:
        raise FormalPreflightError(f"descriptor file binding mismatch: {logical_id}")


def _validate_restore_verdict(
    dataset: ArtifactDescriptor, descriptor: ArtifactDescriptor, split_hash: str
) -> None:
    """shared RestoreTestVerdict を再構築して dataset と split hash を照合する。

    caller が渡した descriptor の identity 自体も再計算して検証する。
    """
    meta = descriptor.identity_metadata
    if (
        descriptor.node_kind != "restore_test_verdict"
        or descriptor.producer_id != "artifact-restore-test"
        or descriptor.files
        or descriptor.parents != (dataset.node_ref(),)
    ):
        raise FormalPreflightError("restore verdict is not bound to dataset")
    try:
        verdict = RestoreTestVerdict(
            logical_id=descriptor.logical_id,
            subject=dataset.node_ref(),
            manifest_hash=meta["manifest_hash"],
            primary_root=descriptor.non_identity_metadata["primary_root"],
            backup_root=descriptor.non_identity_metadata["backup_root"],
            verify_mode=meta["verify_mode"],
            checked_object_count=meta["checked_object_count"],
            passed=meta["passed"],
            blocking_reasons=meta["blocking_reasons"],
            producer_id=descriptor.producer_id,
            producer_version=descriptor.producer_version,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FormalPreflightError(f"invalid restore verdict: {exc}") from exc
    if verdict.to_descriptor().identity_hash != descriptor.identity_hash:
        raise FormalPreflightError("restore verdict identity does not reconstruct")
    if meta["verify_mode"] != "full" or meta["passed"] is not True or meta["manifest_hash"] != split_hash:
        raise FormalPreflightError("restore verdict must be full, passed, and bound to split manifest")


def _code_hashes() -> dict[str, str]:
    """formal training/eval/runtime を構成する全 production file の hash を返す。

    必須 file が欠けた状態を別の build identity として扱わない。
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    names = (
        "survivors/vision/world_detector.py",
        "survivors/vision/world_dataset.py",
        "survivors/vision/entity_tracker.py",
        "survivors/vision/world_detector_package.py",
        "survivors/vision/world_formal_release.py",
        "survivors/vision/world_formal_sampling.py",
        "survivors/vision/session_cluster_ci.py",
        "survivors/artifact_restore_test.py",
        "train_survivors_world_detector.py",
        "eval_survivors_world_detector.py",
    )
    result = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FormalPreflightError(f"formal code file is missing: {name}")
        result[name] = _hash_file(path)
    return result


@dataclass(frozen=True)
class FormalPreflightReport:
    """formal dataset と resolved config/build を結んだ immutable report。\n\n後続の training と evaluation が同一の split、config、hardware identity を読む。"""

    dataset_identity: str
    annotations_sha256: str
    config_identity: str
    build_identity: str
    feasibility_verdict_hash: str
    dataset_logical_id: str
    restore_verdict_logical_id: str
    target_profile_hash: str
    game_build_id: str
    hardware_id: str
    split_hash: str
    train_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]
    slice_counts: Mapping[str, int]
    agreement: Mapping[str, Any]
    code_hashes: Mapping[str, str]
    resolved_config: Mapping[str, Any]
    preflight_canonical_hash: str
    schema_version: str = "world_formal_preflight.v1"

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        """wire dict を返し、任意で report canonical hash を含める。

        train/eval の入力を同一 resolved config に固定する。
        """
        value = asdict(self)
        value["train_sessions"] = list(self.train_sessions)
        value["validation_sessions"] = list(self.validation_sessions)
        value["slice_counts"] = dict(self.slice_counts)
        value["agreement"] = dict(self.agreement)
        value["code_hashes"] = dict(self.code_hashes)
        value["resolved_config"] = dict(self.resolved_config)
        if not include_hash:
            value.pop("preflight_canonical_hash")
        return value

    def write(self, path: str | pathlib.Path) -> pathlib.Path:
        """preflight JSON を atomic replace で保存する。

        eval が未完成 report を読み込まないよう一時ファイルから置換する。
        """
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(canonical_json_bytes(self.to_dict()))
        temporary.replace(target)
        return target

    @classmethod
    def from_dict(cls, wire: Mapping[str, Any]) -> "FormalPreflightReport":
        """schema、hash、未知 field を検証して report を復元する。

        JSON からの report 差し替えを canonical hash で検出する。
        """
        fields = {
            "schema_version", "dataset_identity", "annotations_sha256", "config_identity", "build_identity",
            "feasibility_verdict_hash", "dataset_logical_id", "restore_verdict_logical_id",
            "target_profile_hash", "game_build_id", "hardware_id", "split_hash",
            "train_sessions", "validation_sessions", "slice_counts", "agreement",
            "code_hashes", "resolved_config", "preflight_canonical_hash",
        }
        if not isinstance(wire, Mapping) or set(wire) != fields or wire.get("schema_version") != "world_formal_preflight.v1":
            raise FormalPreflightError("unknown or incomplete preflight report schema")
        payload = dict(wire)
        digest = payload.pop("preflight_canonical_hash")
        if not is_sha256_hex(digest) or canonical_hash(payload) != digest:
            raise FormalPreflightError("preflight canonical hash mismatch")
        for key in ("dataset_identity", "annotations_sha256", "config_identity", "build_identity", "feasibility_verdict_hash", "target_profile_hash", "split_hash"):
            if not is_sha256_hex(wire[key]):
                raise FormalPreflightError(f"invalid preflight {key}")
        return cls(
            dataset_identity=wire["dataset_identity"], annotations_sha256=wire["annotations_sha256"],
            config_identity=wire["config_identity"],
            build_identity=wire["build_identity"], feasibility_verdict_hash=wire["feasibility_verdict_hash"],
            dataset_logical_id=wire["dataset_logical_id"], restore_verdict_logical_id=wire["restore_verdict_logical_id"],
            target_profile_hash=wire["target_profile_hash"], game_build_id=wire["game_build_id"],
            hardware_id=wire["hardware_id"], split_hash=wire["split_hash"],
            train_sessions=tuple(wire["train_sessions"]), validation_sessions=tuple(wire["validation_sessions"]),
            slice_counts=dict(wire["slice_counts"]), agreement=dict(wire["agreement"]),
            code_hashes=dict(wire["code_hashes"]), resolved_config=dict(wire["resolved_config"]),
            preflight_canonical_hash=digest,
        )


def read_preflight_report(path: str | pathlib.Path) -> FormalPreflightReport:
    """file から formal preflight report を読み canonical hash を検証する。

    eval CLI はこの入口を通り、caller が直接 bool を渡すのを避ける。
    """
    try:
        value = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalPreflightError(f"cannot read preflight report: {exc}") from exc
    return FormalPreflightReport.from_dict(value)


def verify_validation_annotations(
    preflight: FormalPreflightReport, annotations_path: str | pathlib.Path
) -> None:
    """evaluation が読む COCO file を preflight の export hash と照合する。

    formal validation の model_validation input 差し替えを checkpoint verdict に持ち込ませない。
    """
    report = FormalPreflightReport.from_dict(preflight.to_dict())
    if _hash_file(pathlib.Path(annotations_path)) != report.annotations_sha256:
        raise FormalPreflightError("formal validation annotation hash does not match preflight")


def _minimum(value: Any, key: str, fallback: int) -> int:
    """minimum count を strict positive integer として読む。

    bool や小数の暗黙変換を通さず、検査下限を安定させる。
    """
    if value is None:
        return fallback
    if type(value) is not int or value < 1:
        raise FormalPreflightError(f"minimum_counts.{key} must be a positive int")
    return value


def run_formal_preflight(
    *,
    feasibility_verdict_path: str | pathlib.Path,
    capture_store_root: str | pathlib.Path,
    dataset_descriptor: ArtifactDescriptor,
    restore_verdict_descriptor: ArtifactDescriptor,
    annotations_path: str | pathlib.Path,
    class_map_path: str | pathlib.Path,
    config_path: str | pathlib.Path,
    cli_overrides: Mapping[str, Any],
    target_profile_hash: str,
    hardware_id: str,
    annotation_agreement_path: str | pathlib.Path,
    minimum_counts: Mapping[str, Any],
) -> FormalPreflightReport:
    """feasibility、sealed split、dataset export と restore verdict を検証する。

    formal eligible session 以外の pixel と annotations は読み込み対象にしない。
    """
    if not isinstance(dataset_descriptor, ArtifactDescriptor) or dataset_descriptor.node_kind != "source_descriptor":
        raise FormalPreflightError("dataset_descriptor must be a source_descriptor")
    if not isinstance(restore_verdict_descriptor, ArtifactDescriptor):
        raise FormalPreflightError("restore_verdict_descriptor must use the shared contract")
    try:
        feasibility_bytes = pathlib.Path(feasibility_verdict_path).read_bytes()
        feasibility = json.loads(feasibility_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalPreflightError(f"feasibility verdict is unreadable: {exc}") from exc
    if (
        feasibility.get("schema_version") != "perception_feasibility_verdict.v1"
        or feasibility.get("status") != "PASS"
        or not isinstance(feasibility.get("selected_architecture"), str)
    ):
        raise FormalPreflightError("feasibility verdict must be PASS with selected architecture")

    if dataset_descriptor.node_kind != "source_descriptor" or dataset_descriptor.parents:
        raise FormalPreflightError("formal dataset descriptor must be a parentless source_descriptor")
    identity = dataset_descriptor.identity_metadata
    if identity.get("synthetic") is True or identity.get("development_only") is True:
        raise FormalPreflightError("synthetic or development dataset descriptor is not formal eligible")
    build_id = identity.get("game_build_id")
    if not is_sha256_hex(target_profile_hash) or not isinstance(hardware_id, str) or not hardware_id:
        raise FormalPreflightError("target_profile_hash or hardware_id is invalid")
    if not isinstance(build_id, str) or not build_id:
        raise FormalPreflightError("dataset descriptor lacks game_build_id")
    manifest_id = identity.get("manifest_logical_id")
    manifest_refs = [ref for ref in dataset_descriptor.files if ref.logical_id == manifest_id]
    if len(manifest_refs) != 1:
        raise FormalPreflightError("dataset split manifest reference is missing or ambiguous")
    root = pathlib.Path(capture_store_root).expanduser().resolve()
    try:
        split_manifest = load_split_manifest_by_hash(root, manifest_refs[0].sha256)
    except (OSError, ValueError) as exc:
        if isinstance(exc, FormalPreflightError):
            raise
        raise FormalPreflightError(f"invalid frozen split manifest: {exc}") from exc
    if split_manifest.frozen is not True:
        raise FormalPreflightError("split manifest must be frozen")
    split_hash = split_manifest.manifest_sha256
    _validate_restore_verdict(dataset_descriptor, restore_verdict_descriptor, split_hash)

    try:
        with pathlib.Path(config_path).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        from survivors.vision.world_detector import validate_detector_config
        resolved = _override_config(config, cli_overrides)
        validate_detector_config(resolved)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
        raise FormalPreflightError(f"resolved detector config is invalid: {exc}") from exc
    if resolved["model"]["architecture"] != feasibility["selected_architecture"]:
        raise FormalPreflightError("feasibility architecture does not match resolved config")
    if resolved["schema_version"] != "world_detector.v2":
        raise FormalPreflightError("formal release requires world_detector.v2")
    class_map = load_class_map(class_map_path)
    if class_map.class_map_version != 2 or class_map.num_classes != 16:
        raise FormalPreflightError("formal release requires world_class_map_v2")

    exports = identity.get("dataset_exports")
    if not isinstance(exports, Mapping):
        raise FormalPreflightError("dataset descriptor lacks dataset_exports")
    coco_binding = exports.get("world_coco_model")
    agreement_binding = exports.get("annotation_agreement")
    if not isinstance(coco_binding, Mapping) or not isinstance(agreement_binding, Mapping):
        raise FormalPreflightError("dataset descriptor lacks formal COCO/agreement bindings")
    annotations_path = pathlib.Path(annotations_path)
    agreement_path = pathlib.Path(annotation_agreement_path)
    annotation_hash = _hash_file(annotations_path)
    agreement_hash = _hash_file(agreement_path)
    for binding, digest, label in (
        (coco_binding, annotation_hash, "world_coco_model"),
        (agreement_binding, agreement_hash, "annotation_agreement"),
    ):
        if binding.get("sha256") != digest or not isinstance(binding.get("logical_id"), str):
            raise FormalPreflightError(f"{label} hash does not match dataset descriptor")
        _descriptor_file(dataset_descriptor, binding["logical_id"], digest)

    model_bindings = identity.get("model_session_bindings")
    if not isinstance(model_bindings, Mapping):
        raise FormalPreflightError("dataset descriptor lacks model_session_bindings")
    train_units = split_manifest.splits["model_train"]
    validation_units = split_manifest.splits["model_validation"]
    model_units = tuple(train_units) + tuple(validation_units)
    if not train_units or not validation_units:
        raise FormalPreflightError("model_train and model_validation must both be non-empty")
    model_sessions = {unit.session_id for unit in model_units}
    sealed_sessions = {
        unit.session_id
        for name in SPLIT_NAMES
        if name not in {"model_train", "model_validation"}
        for unit in split_manifest.splits[name]
    }
    if model_sessions & sealed_sessions or set(model_bindings) != model_sessions:
        raise FormalPreflightError("model session bindings do not match frozen model split")
    session_frame_hashes: dict[str, set[str]] = {}
    session_manifest_hashes: dict[str, str] = {}
    for unit in model_units:
        binding = model_bindings.get(unit.session_id)
        if not isinstance(binding, Mapping) or binding.get("session_manifest_sha256") != unit.session_manifest_sha256:
            raise FormalPreflightError(f"model session binding mismatch: {unit.session_id}")
        _descriptor_file(dataset_descriptor, binding.get("session_manifest_logical_id", ""), unit.session_manifest_sha256)
        session_path = root / "capture_sessions" / unit.session_id / "session_manifest.json"
        if _hash_file(session_path) != unit.session_manifest_sha256:
            raise FormalPreflightError(f"session manifest hash mismatch: {unit.session_id}")
        try:
            session = DatasetWriter.restore(root, unit.session_id, metadata_only=True)
        except (OSError, ValueError) as exc:
            raise FormalPreflightError(f"invalid session {unit.session_id}: {exc}") from exc
        if (
            session.formal_dataset_eligible is not True
            or session.pixel_source != "lossless_png"
            or session.target_profile_hash != target_profile_hash
            or session.game_build_id != build_id
            or unit.game_build_id != build_id
        ):
            raise FormalPreflightError(f"session is not formal eligible: {unit.session_id}")
        session_manifest_hashes[unit.session_id] = unit.session_manifest_sha256
        session_frame_hashes[unit.session_id] = {record.object_sha256 for record in session.frame_records}

    coco = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = coco.get("images")
    annotations = coco.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise FormalPreflightError("COCO images and annotations arrays are required")
    image_sessions: dict[int, str] = {}
    image_shas: dict[str, list[str]] = defaultdict(list)
    for image in images:
        if not isinstance(image, Mapping):
            raise FormalPreflightError("COCO image entry must be a mapping")
        image_id = image.get("id")
        session_id = image.get("session_id")
        file_name = image.get("file_name")
        if type(image_id) is not int or not isinstance(session_id, str) or not session_id or not isinstance(file_name, str) or not file_name:
            raise FormalPreflightError("COCO image lacks integer id or session_id")
        if session_id in sealed_sessions or session_id not in model_sessions:
            raise FormalPreflightError(f"COCO references sealed or unbound session: {session_id}")
        if image_id in image_sessions:
            raise FormalPreflightError(f"duplicate COCO image id: {image_id}")
        image_sessions[image_id] = session_id
        image_path = pathlib.Path(file_name)
        if not image_path.is_absolute():
            image_path = annotations_path.parent / image_path
        digest = _hash_file(image_path)
        if digest not in session_frame_hashes[session_id] or (
            "object_sha256" in image and image["object_sha256"] != digest
        ):
            raise FormalPreflightError(f"COCO image is not bound to its session frames: {session_id}")
        image_shas[session_id].append(digest)
    for annotation in annotations:
        session_id = annotation.get("session_id", image_sessions.get(annotation.get("image_id")))
        if annotation.get("image_id") not in image_sessions or session_id != image_sessions[annotation["image_id"]]:
            raise FormalPreflightError("COCO annotation image/session binding mismatch")
        if session_id in sealed_sessions:
            raise FormalPreflightError(f"COCO annotation references sealed session: {session_id}")

    data = json.loads(agreement_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "annotation_agreement.v1":
        raise FormalPreflightError("unknown annotation agreement schema")
    agreement = {key: data.get(key) for key in ("pair_count", "iou_mean", "class_agreement")}
    thresholds = minimum_counts.get("agreement") if isinstance(minimum_counts, Mapping) else None
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(agreement):
        raise FormalPreflightError("minimum_counts.agreement must set pair_count, iou_mean, class_agreement")
    if type(agreement["pair_count"]) is not int or agreement["pair_count"] < thresholds["pair_count"]:
        raise FormalPreflightError("annotation agreement pair_count is below minimum")
    for key in ("iou_mean", "class_agreement"):
        actual, floor = agreement[key], thresholds[key]
        if isinstance(actual, Mapping):
            if not isinstance(floor, Mapping) or set(actual) != set(floor):
                raise FormalPreflightError(f"annotation agreement {key} class set mismatch")
            values = list(actual.items())
            if any(not math.isfinite(value) or value < floor[name] for name, value in values):
                raise FormalPreflightError(f"annotation agreement {key} is below minimum")
        elif isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual < floor:
            raise FormalPreflightError(f"annotation agreement {key} is below minimum")

    dataset = WorldDataset(annotations_path, class_map_path, validate_bounds=True, rejected_sessions=sealed_sessions)
    if not isinstance(minimum_counts, Mapping):
        raise FormalPreflightError("minimum_counts must be a mapping")
    cfg_counts = resolved["training"]["preflight"]
    limits = {
        key: max(cfg_counts[key], _minimum(minimum_counts.get(key), key, cfg_counts[key]))
        for key in cfg_counts
    }
    dataset.run_preflight(
        min_frames=limits["min_frames"], min_entities=limits["min_entities"],
        min_classes=limits["min_classes"], min_time_bands=limits["min_time_bands"],
    )
    split = SessionSplit(
        train=[unit.session_id for unit in train_units],
        validation=[unit.session_id for unit in validation_units],
        error_calibration=sorted(sealed_sessions), final_e2e_test=[],
    )
    dataset.run_split_preflight(
        split,
        min_frames_per_split=limits["min_frames_per_split"],
        min_entities_per_split=limits["min_entities_per_split"],
        min_classes_per_split=limits["min_classes_per_split"],
    )
    counts = {"frames": len(dataset), "entities": len(annotations), "classes": len({a.category_id for s in range(len(dataset)) for a in dataset[s].annotations})}
    heavy_classes = {item["id"] for item in class_map.foreground_classes if item["name"].startswith("weapon_")}
    counts["heavy_effect_slice"] = sum(annotation["category_id"] in heavy_classes for annotation in annotations)
    if counts["heavy_effect_slice"] < _minimum(minimum_counts.get("heavy_effect_slice"), "heavy_effect_slice", 1):
        raise FormalPreflightError("heavy-effect slice is below minimum")

    session_identity = [
        (session_id, session_manifest_hashes[session_id], sorted(image_shas[session_id]))
        for session_id in sorted(model_sessions)
    ]
    dataset_identity = canonical_hash({
        "annotations_sha256": annotation_hash,
        "split_manifest_sha256": split_hash,
        "sessions": session_identity,
    })
    config_identity = resolved_config_hash(resolved)
    code_hashes = _code_hashes()
    build_identity = canonical_hash({
        "target_profile_hash": target_profile_hash,
        "game_build_id": build_id,
        "hardware_id": hardware_id,
        "code_hashes": code_hashes,
    })
    payload = {
        "schema_version": "world_formal_preflight.v1",
        "dataset_identity": dataset_identity,
        "annotations_sha256": annotation_hash,
        "config_identity": config_identity,
        "build_identity": build_identity,
        "feasibility_verdict_hash": _sha256(feasibility_bytes),
        "dataset_logical_id": dataset_descriptor.logical_id,
        "restore_verdict_logical_id": restore_verdict_descriptor.logical_id,
        "target_profile_hash": target_profile_hash,
        "game_build_id": build_id,
        "hardware_id": hardware_id,
        "split_hash": split_hash,
        "train_sessions": sorted(unit.session_id for unit in train_units),
        "validation_sessions": sorted(unit.session_id for unit in validation_units),
        "slice_counts": counts,
        "agreement": agreement,
        "code_hashes": code_hashes,
        "resolved_config": resolved,
    }
    report = FormalPreflightReport(
        **{key: value for key, value in payload.items() if key != "schema_version"},
        preflight_canonical_hash=canonical_hash(payload),
    )
    if report.preflight_canonical_hash != canonical_hash(report.to_dict(include_hash=False)):
        raise FormalPreflightError("internal preflight canonical hash mismatch")
    return report


@dataclass(frozen=True)
class GpuLatencyMeasurement:
    """latency measurement と target hardware の出典を保持する。\n\n未対応 hardware の数値が正式な latency gate に混ざらないよう由来も保存する。"""

    p95_ms: float | None
    sample_count: int
    hardware_id: str
    source: str
    cuda_event: bool

    def __post_init__(self) -> None:
        """wire 値の型と有限性を検証する。

        欠測は p95_ms=None として表し、0 ms と混同しない。
        """
        if type(self.sample_count) is not int or self.sample_count < 0:
            raise ValueError("latency sample_count must be a non-negative int")
        if not isinstance(self.hardware_id, str) or not self.hardware_id:
            raise ValueError("latency hardware_id must be non-empty")
        if self.source not in {"target_gpu", "other"} or not isinstance(self.cuda_event, bool):
            raise ValueError("latency source/cuda_event is invalid")
        if self.p95_ms is not None and (
            isinstance(self.p95_ms, bool)
            or not isinstance(self.p95_ms, (int, float))
            or not math.isfinite(self.p95_ms)
            or self.p95_ms < 0
        ):
            raise ValueError("latency p95_ms must be finite and non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GpuLatencyMeasurement":
        """JSON wire measurement を余分な field なしで復元する。

        CLI 自己申告値の field 名や型を fail-closed に扱う。
        """
        keys = {"p95_ms", "sample_count", "hardware_id", "source", "cuda_event"}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise ValueError("latency JSON schema is incomplete or unknown")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        """latency measurement を schema 固定の JSON mapping にする。\n\n保存する field を限定し、読込側の厳密な schema 検査と対にする。"""
        return {
            "p95_ms": self.p95_ms,
            "sample_count": self.sample_count,
            "hardware_id": self.hardware_id,
            "source": self.source,
            "cuda_event": self.cuda_event,
        }


def session_recall_gate(
    counts_by_session: Mapping[str, tuple[float, float]],
    *,
    floor: float,
    resamples: int = 10_000,
    seed: int = 0,
    min_applicable_sessions: int = 3,
) -> tuple[float | None, bool]:
    """比率型 session-cluster CI の下限を recall floor と比較する。

    denominator 0 の session は共有 helper の規則で non-applicable にする。
    """
    if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or not 0 <= floor <= 1:
        raise ValueError("recall floor must be finite and in [0, 1]")
    try:
        lower, _ = cluster_ci_bounds(
            counts_by_session,
            resamples=resamples,
            seed=seed,
            min_applicable_sessions=min_applicable_sessions,
        )
    except ValueError:
        return None, False
    return lower, lower >= floor


def latency_gate(
    measurement: GpuLatencyMeasurement | None,
    *,
    hardware_id: str,
    max_ms: float = 25.0,
) -> dict[str, Any]:
    """target GPU 由来・100 sample 以上・同一 hardware の latency を判定する。

    欠測値、CUDA event 以外、非有限値は gate failure として記録する。
    """
    valid_limit = not isinstance(max_ms, bool) and isinstance(max_ms, (int, float)) and math.isfinite(max_ms) and max_ms >= 0
    valid = (
        isinstance(measurement, GpuLatencyMeasurement)
        and valid_limit
        and measurement.source == "target_gpu"
        and measurement.cuda_event is True
        and measurement.hardware_id == hardware_id
        and measurement.sample_count >= 100
        and measurement.p95_ms is not None
        and math.isfinite(measurement.p95_ms)
    )
    return {
        "passed": bool(valid and measurement.p95_ms <= max_ms),
        "value": measurement.p95_ms if valid else "missing",
        "max_ms": float(max_ms) if valid_limit else "missing",
        "sample_count": measurement.sample_count if isinstance(measurement, GpuLatencyMeasurement) else 0,
        "hardware_id": measurement.hardware_id if isinstance(measurement, GpuLatencyMeasurement) else "missing",
    }


def select_formal_threshold(
    recall_precision_by_candidate: Mapping[float, tuple[float, float]],
    *,
    precision_floor: float,
) -> tuple[float, dict[str, dict[str, float]]]:
    """precision floor を満たす候補から recall 最大の threshold を選ぶ。

    recall が同値なら小さい threshold を選び、候補表も verdict に固定する。
    """
    if isinstance(precision_floor, bool) or not isinstance(precision_floor, (int, float)) or not math.isfinite(precision_floor) or not 0 <= precision_floor <= 1:
        raise ValueError("precision_floor must be finite and in [0, 1]")
    details: dict[str, dict[str, float]] = {}
    eligible: list[tuple[float, float]] = []
    for threshold, values in recall_precision_by_candidate.items():
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("threshold candidates must be in [0, 1]")
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("each candidate must contain recall and precision")
        recall, precision = values
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 for value in (recall, precision)):
            raise ValueError("candidate recall and precision must be finite in [0, 1]")
        details[f"{float(threshold):.8g}"] = {"mean_recall": float(recall), "precision": float(precision)}
        if precision >= precision_floor:
            eligible.append((float(recall), float(threshold)))
    if not eligible:
        raise ValueError("no formal threshold candidate meets precision_floor")
    recall, threshold = max(eligible, key=lambda item: (item[0], -item[1]))
    return threshold, details


def measure_gpu_latency(
    detector: Any,
    frames: Sequence[np.ndarray],
    *,
    hardware_id: str,
) -> GpuLatencyMeasurement:
    """CUDA event で detector の frame ごとの p95 推論時間を測る。

    CUDA 不在時は target measurement を捏造せず source=other を返す。
    """
    try:
        import torch
        import torchvision.transforms.functional as TF
    except ModuleNotFoundError as exc:
        if exc.name not in {"torch", "torchvision"}:
            raise
        return GpuLatencyMeasurement(None, 0, hardware_id, "other", False)
    if not torch.cuda.is_available():
        return GpuLatencyMeasurement(None, 0, hardware_id, "other", False)
    model = getattr(detector, "_model", detector)
    model = model.to("cuda").eval()
    times: list[float] = []
    with torch.no_grad():
        for frame in frames:
            if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("latency frames must be BGR images")
            image = TF.to_tensor(frame[..., ::-1].copy()).to("cuda")
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            model([image])
            end.record()
            end.synchronize()
            times.append(float(start.elapsed_time(end)))
    return GpuLatencyMeasurement(
        float(np.percentile(times, 95)) if times else None,
        len(times), hardware_id, "target_gpu", True,
    )


@dataclass(frozen=True)
class ValidationVerdict:
    """validation metric、gate と親 identity を結ぶ formal verdict。\n\nPASS は checkpoint weight、preflight report、dataset、build が同一の場合だけ発行する。"""

    status: str
    metrics: Mapping[str, Any]
    gates: Mapping[str, Any]
    score_threshold: float
    threshold_rule: Mapping[str, Any]
    selection_bias_note: str
    weight_hash: str
    checkpoint_manifest_hash: str
    dataset_identity: str
    resolved_config_hash: str
    build_identity: str
    preflight_canonical_hash: str
    code_hashes: Mapping[str, str]
    inference_caps: Mapping[str, Any]
    verdict_canonical_hash: str
    schema_version: str = "world_detector_validation_verdict.v1"

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        """JSON wire mapping を返し、hash の有無を指定する。

        verdict hash は metrics と全親 identity を含む payload に対して計算する。
        """
        result = {
            "schema_version": self.schema_version,
            "status": self.status,
            "metrics": dict(self.metrics),
            "gates": dict(self.gates),
            "score_threshold": self.score_threshold,
            "threshold_rule": dict(self.threshold_rule),
            "selection_bias_note": self.selection_bias_note,
            "weight_hash": self.weight_hash,
            "checkpoint_manifest_hash": self.checkpoint_manifest_hash,
            "dataset_identity": self.dataset_identity,
            "resolved_config_hash": self.resolved_config_hash,
            "build_identity": self.build_identity,
            "preflight_canonical_hash": self.preflight_canonical_hash,
            "code_hashes": dict(self.code_hashes),
            "inference_caps": dict(self.inference_caps),
        }
        if include_hash:
            result["verdict_canonical_hash"] = self.verdict_canonical_hash
        return result

    def write(self, path: str | pathlib.Path) -> pathlib.Path:
        """verdict を atomic replace で保存する。

        package publisher が途中書き込みを PASS として読むのを防ぐ。
        """
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(canonical_json_bytes(self.to_dict()))
        temporary.replace(target)
        return target

    @classmethod
    def from_dict(cls, wire: Mapping[str, Any]) -> "ValidationVerdict":
        """schema と canonical hash を検証して verdict を復元する。

        一つでも未知 field や identity 書き換えがあれば publisher に渡さない。
        """
        fields = {
            "schema_version", "status", "metrics", "gates", "score_threshold", "threshold_rule",
            "selection_bias_note", "weight_hash", "checkpoint_manifest_hash", "dataset_identity",
            "resolved_config_hash", "build_identity", "preflight_canonical_hash", "code_hashes",
            "inference_caps", "verdict_canonical_hash",
        }
        if not isinstance(wire, Mapping) or set(wire) != fields or wire.get("schema_version") != "world_detector_validation_verdict.v1":
            raise ValueError("unknown or incomplete validation verdict schema")
        payload = dict(wire)
        digest = payload.pop("verdict_canonical_hash")
        if not is_sha256_hex(digest) or canonical_hash(payload) != digest:
            raise ValueError("validation verdict canonical hash mismatch")
        if wire["status"] not in {"PASS", "FAIL"}:
            raise ValueError("validation verdict status is invalid")
        for key in ("weight_hash", "checkpoint_manifest_hash", "dataset_identity", "resolved_config_hash", "build_identity", "preflight_canonical_hash"):
            if not is_sha256_hex(wire[key]):
                raise ValueError(f"validation verdict {key} is invalid")
        threshold = wire["score_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("validation verdict score_threshold is invalid")
        return cls(
            status=wire["status"], metrics=dict(wire["metrics"]), gates=dict(wire["gates"]),
            score_threshold=float(threshold), threshold_rule=dict(wire["threshold_rule"]),
            selection_bias_note=wire["selection_bias_note"], weight_hash=wire["weight_hash"],
            checkpoint_manifest_hash=wire["checkpoint_manifest_hash"], dataset_identity=wire["dataset_identity"],
            resolved_config_hash=wire["resolved_config_hash"], build_identity=wire["build_identity"],
            preflight_canonical_hash=wire["preflight_canonical_hash"], code_hashes=dict(wire["code_hashes"]),
            inference_caps=dict(wire["inference_caps"]), verdict_canonical_hash=digest,
        )


DEFAULT_FORMAL_GATES = {
    "enemy_recall_min": 0.90,
    "gem_recall_min": 0.92,
    "boss_recall_min": 0.98,
    "directional_density_correlation_min": 0.85,
    "density_error_ci_upper_max": 0.20,
    "nearest_distance_error_ci_upper_max": 0.04,
    "gpu_p95_latency_ms_max": 25.0,
    "heavy_effect_recall_min": 0.80,
    "late_game_recall_min": 0.80,
    "boss_min_instances": 3,
    "boss_min_sessions": 3,
    "hazard_min_instances": 4,
    "hazard_min_sessions": 4,
    "min_applicable_sessions": 3,
    "ci_resamples": 10_000,
    "ci_seed": 0,
}


def _record_box(annotation: Mapping[str, Any]) -> list[float]:
    """COCO xywh または xyxy wire bbox を xyxy に揃える。

    annotation に無い・非有限座標は正式評価の前に拒否する。
    """
    if "bbox_xyxy" in annotation:
        values = [float(value) for value in annotation["bbox_xyxy"]]
    else:
        x, y, width, height = (float(value) for value in annotation["bbox"])
        values = [x, y, x + width, y + height]
    if len(values) != 4 or not all(math.isfinite(value) for value in values) or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("validation bbox is invalid")
    return values


def _inside_regions(box: Sequence[float], width: int, height: int, regions: Sequence[Sequence[float]]) -> bool:
    """box center が normalized ignore region に含まれるか返す。

    formal GT と prediction に同じ center-based ROI を使う。
    """
    x = (box[0] + box[2]) / 2 / width
    y = (box[1] + box[3]) / 2 / height
    return any(region[0] <= x <= region[2] and region[1] <= y <= region[3] for region in regions)


def _frame_records(
    detections_by_session: Mapping[str, Sequence[Mapping[str, Any]]],
    annotations_by_session: Mapping[str, Sequence[Mapping[str, Any]]],
    ignore_regions: Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    """session/frame keyed record へ GT と raw score prediction を正規化する。

    frame dimension と score は生の入力から保持し、threshold 前の評価を可能にする。
    """
    frames: dict[tuple[str, Any], dict[str, Any]] = {}
    for session_id, annotations in annotations_by_session.items():
        for annotation in annotations:
            image_id = annotation.get("image_id")
            if image_id is None:
                raise ValueError("formal annotation lacks image_id")
            key = (session_id, image_id)
            width, height = int(annotation.get("image_width", 1920)), int(annotation.get("image_height", 1080))
            frame = frames.setdefault(key, {"session_id": session_id, "image_id": image_id, "width": width, "height": height, "gt": [], "pred": []})
            box = _record_box(annotation)
            if _inside_regions(box, width, height, ignore_regions):
                continue
            class_id = annotation.get("category_id", annotation.get("class_id"))
            if type(class_id) is not int or class_id <= 0:
                raise ValueError("formal annotation class id must be positive int")
            frame["gt"].append({"box": box, "class_id": class_id, "track_id": annotation.get("track_id"), "late_game": annotation.get("late_game") is True or annotation.get("time_band") in {"late", "late_game", "late-game"}})
    for session_id, predictions in detections_by_session.items():
        for prediction in predictions:
            image_id = prediction.get("image_id")
            if image_id is None:
                raise ValueError("formal prediction lacks image_id")
            width, height = int(prediction["image_width"]), int(prediction["image_height"])
            key = (session_id, image_id)
            frame = frames.setdefault(key, {"session_id": session_id, "image_id": image_id, "width": width, "height": height, "gt": [], "pred": []})
            boxes = np.asarray(prediction["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
            scores = np.asarray(prediction["scores"], dtype=np.float64).reshape(-1)
            classes = np.asarray(prediction["class_ids"], dtype=np.int64).reshape(-1)
            if len(boxes) != len(scores) or len(scores) != len(classes) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                raise ValueError("formal prediction arrays are inconsistent or non-finite")
            if np.any(scores < 0) or np.any(scores > 1):
                raise ValueError("formal prediction score is outside [0, 1]")
            for box, score, class_id in zip(boxes, scores, classes):
                normalized_box = [float(value) for value in box]
                if _inside_regions(normalized_box, width, height, ignore_regions):
                    continue
                if class_id <= 0:
                    continue
                frame["pred"].append({"box": normalized_box, "score": float(score), "class_id": int(class_id)})
    return [frames[key] for key in sorted(frames, key=lambda pair: (pair[0], str(pair[1])))]


def _match_frame_records(frames: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    """IoU .50 greedy match から session/class/slice の TP と誤差を集計する。

    prediction score 順を固定し、同じ GT を二重に TP と数えない。
    """
    from eval_survivors_world_detector import compute_iou_matrix
    counts: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    heavy: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    late: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    class_totals: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    total_tp = total_fp = total_gt = 0
    count_errors: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    nearest: list[float] = []
    nearest_by_session: dict[str, list[float]] = defaultdict(list)
    grid_gt: list[float] = []
    grid_pred: list[float] = []
    directional_by_session: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"gt": [], "pred": []}
    )
    boss_instances: set[str] = set()
    hazard_instances: set[str] = set()
    boss_count = hazard_count = 0
    for frame in frames:
        session_id = frame["session_id"]
        width, height = frame["width"], frame["height"]
        gt = frame["gt"]
        pred = sorted((item for item in frame["pred"] if item["score"] >= threshold), key=lambda item: -item["score"])
        matched: set[int] = set()
        frame_tp: dict[int, int] = defaultdict(int)
        frame_fp: dict[int, int] = defaultdict(int)
        for item in gt:
            class_totals[item["class_id"]][1] += 1
            counts[item["class_id"]][session_id][1] += 1
            name = str(item["class_id"])
            if item["class_id"] in getattr(_match_frame_records, "weapon_ids", set()):
                heavy[session_id][1] += 1
            if item["late_game"]:
                late[session_id][1] += 1
            if item["class_id"] in getattr(_match_frame_records, "boss_ids", set()):
                boss_count += 1
                boss_instances.add(session_id)
            if item["class_id"] in getattr(_match_frame_records, "hazard_ids", set()):
                hazard_count += 1
                hazard_instances.add(session_id)
        for item in pred:
            same_class = [index for index, target in enumerate(gt) if index not in matched and target["class_id"] == item["class_id"]]
            if same_class:
                overlaps = compute_iou_matrix(
                    np.asarray([gt[index]["box"] for index in same_class], dtype=np.float32),
                    np.asarray([item["box"]], dtype=np.float32),
                )[:, 0]
                best_offset = int(np.argmax(overlaps))
                if overlaps[best_offset] >= 0.5:
                    target_index = same_class[best_offset]
                    matched.add(target_index)
                    class_id = item["class_id"]
                    class_totals[class_id][0] += 1
                    counts[class_id][session_id][0] += 1
                    frame_tp[class_id] += 1
                    if class_id in getattr(_match_frame_records, "weapon_ids", set()):
                        heavy[session_id][0] += 1
                    if gt[target_index]["late_game"]:
                        late[session_id][0] += 1
                else:
                    frame_fp[item["class_id"]] += 1
            else:
                frame_fp[item["class_id"]] += 1
        total_gt += len(gt)
        total_tp += len(matched)
        total_fp += sum(frame_fp.values())
        count_errors[session_id][0] += abs(len(pred) - len(gt))
        count_errors[session_id][1] += 1
        for target in gt:
            candidates = [item for item in pred if item["class_id"] == target["class_id"]]
            if candidates:
                cx = (target["box"][0] + target["box"][2]) / 2
                cy = (target["box"][1] + target["box"][3]) / 2
                distance = min(math.hypot(((item["box"][0] + item["box"][2]) / 2 - cx) / width, ((item["box"][1] + item["box"][3]) / 2 - cy) / height) for item in candidates)
            else:
                distance = 1.0
            nearest.append(distance)
            nearest_by_session[session_id].append(distance)
        session_density = directional_by_session[session_id]
        for y in range(4):
            for x in range(4):
                def in_cell(box):
                    """box center が指定した 4x4 density cell に入るか返す。\n\nGT と予測を同じ grid index で数えて session density 比較に使う。"""
                    cx, cy = (box[0] + box[2]) / 2 / width, (box[1] + box[3]) / 2 / height
                    return min(3, int(cy * 4)) == y and min(3, int(cx * 4)) == x
                gt_count = float(sum(in_cell(item["box"]) for item in gt))
                pred_count = float(sum(in_cell(item["box"]) for item in pred))
                grid_gt.append(gt_count)
                grid_pred.append(pred_count)
                session_density["gt"].append(gt_count)
                session_density["pred"].append(pred_count)
    return {
        "counts_by_class": {key: dict(value) for key, value in counts.items()},
        "class_totals": dict(class_totals), "total_tp": total_tp, "total_fp": total_fp,
        "total_gt": total_gt, "heavy": dict(heavy), "late": dict(late),
        "count_errors": dict(count_errors), "nearest": nearest,
        "nearest_by_session": dict(nearest_by_session),
        "directional_gt": grid_gt, "directional_pred": grid_pred,
        "directional_by_session": dict(directional_by_session),
        "boss_instances": boss_count, "boss_sessions": len(boss_instances),
        "hazard_instances": hazard_count, "hazard_sessions": len(hazard_instances),
    }


def build_validation_verdict(
    *,
    detections_by_session: Mapping[str, Sequence[Mapping[str, Any]]],
    annotations_by_session: Mapping[str, Sequence[Mapping[str, Any]]],
    class_map: Any,
    gates: Mapping[str, Any],
    latency: GpuLatencyMeasurement,
    preflight: FormalPreflightReport,
    checkpoint_manifest: Any,
    weight_hash: str,
    threshold_rule: Mapping[str, Any],
    checkpoint_manifest_hash: str | None = None,
    inference_caps: Mapping[str, Any] | None = None,
) -> ValidationVerdict:
    """raw score predictions を threshold 選択・metric・gate へ固定する。

    validation score、latency、parent identity のいずれかが欠ければ PASS を出さない。
    """
    if not is_sha256_hex(weight_hash):
        raise ValueError("weight_hash must be lowercase sha256")
    if not isinstance(preflight, FormalPreflightReport):
        raise TypeError("preflight must be a verified FormalPreflightReport")
    error_ci_gates = {"density_error_ci_upper_max", "nearest_distance_error_ci_upper_max"}
    if not isinstance(gates, Mapping) or set(gates) - set(DEFAULT_FORMAL_GATES):
        raise ValueError("formal gates contain unknown keys")
    fixed_gates = dict(DEFAULT_FORMAL_GATES)
    fixed_gates.update(gates)
    for key, value in fixed_gates.items():
        if key in error_ci_gates:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"formal gate {key} must be a finite non-negative number")
        elif key.endswith("_instances") or key.endswith("_sessions") or key in {"min_applicable_sessions", "ci_resamples", "ci_seed"}:
            if type(value) is not int or value < (0 if key == "ci_seed" else 1):
                raise ValueError(f"formal gate {key} must be a valid int")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1 and key != "gpu_p95_latency_ms_max":
            raise ValueError(f"formal gate {key} must be finite and in range")
    if fixed_gates["gpu_p95_latency_ms_max"] < 0:
        raise ValueError("GPU latency maximum must be non-negative")
    names = {int(item["id"]): str(item["name"]) for item in class_map.foreground_classes}
    enemy_ids = {key for key, name in names.items() if name.startswith("enemy_") and name != "enemy_boss"}
    gem_ids = {key for key, name in names.items() if name.startswith("gem_")}
    boss_ids = {key for key, name in names.items() if name == "enemy_boss"}
    hazard_ids = {key for key, name in names.items() if name.startswith("hazard_")}
    weapon_ids = {key for key, name in names.items() if name.startswith("weapon_")}
    _match_frame_records.weapon_ids = weapon_ids
    _match_frame_records.boss_ids = boss_ids
    _match_frame_records.hazard_ids = hazard_ids
    regions = preflight.resolved_config.get("input", {}).get("ignore_regions", [])
    frames = _frame_records(detections_by_session, annotations_by_session, regions)
    if not frames or not any(frame["gt"] for frame in frames):
        raise ValueError("formal validation needs at least one model_validation GT")

    if not isinstance(threshold_rule, Mapping) or threshold_rule.get("kind") != "max_mean_recall_subject_to_precision":
        raise ValueError("unsupported formal threshold rule")
    precision_floor = threshold_rule.get("precision_floor")
    candidates = threshold_rule.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("formal threshold candidates are missing")
    candidate_metrics: dict[float, tuple[float, float]] = {}
    for candidate in candidates:
        raw = _match_frame_records(frames, float(candidate))
        per_class_recall = [counts[0] / counts[1] for counts in raw["class_totals"].values() if counts[1]]
        mean_recall = float(np.mean(per_class_recall)) if per_class_recall else 0.0
        precision = raw["total_tp"] / max(raw["total_tp"] + raw["total_fp"], 1)
        candidate_metrics[float(candidate)] = (mean_recall, precision)
    score_threshold, selected_candidates = select_formal_threshold(
        candidate_metrics, precision_floor=precision_floor
    )
    selected_rule = dict(threshold_rule)
    selected_rule["candidate_metrics"] = selected_candidates
    raw = _match_frame_records(frames, score_threshold)

    from eval_survivors_world_detector import compute_ap_coco101
    gt_boxes: list[list[float]] = []
    gt_classes: list[int] = []
    gt_image_ids: list[int] = []
    pred_boxes: list[list[float]] = []
    pred_classes: list[int] = []
    pred_scores: list[float] = []
    pred_image_ids: list[int] = []
    for image_index, frame in enumerate(frames):
        for item in frame["gt"]:
            gt_boxes.append(item["box"]); gt_classes.append(item["class_id"]); gt_image_ids.append(image_index)
        for item in frame["pred"]:
            pred_boxes.append(item["box"]); pred_classes.append(item["class_id"]); pred_scores.append(item["score"]); pred_image_ids.append(image_index)
    map_value = compute_ap_coco101(
        np.asarray(gt_boxes), np.asarray(gt_classes), np.asarray(pred_boxes).reshape(-1, 4),
        np.asarray(pred_scores), np.asarray(pred_classes),
        gt_image_ids=np.asarray(gt_image_ids), pred_image_ids=np.asarray(pred_image_ids),
    )
    ci_config = {"resamples": fixed_gates["ci_resamples"], "seed": fixed_gates["ci_seed"], "min_applicable_sessions": fixed_gates["min_applicable_sessions"]}
    class_metrics: dict[str, Any] = {}
    class_gates: dict[int, tuple[float | None, bool]] = {}
    for class_id, per_session in raw["counts_by_class"].items():
        name = names.get(class_id, str(class_id))
        floor = fixed_gates["boss_recall_min"] if name == "enemy_boss" else fixed_gates["gem_recall_min"] if name.startswith("gem_") else fixed_gates["enemy_recall_min"] if name.startswith("enemy_") else 0.0
        lower, passed = session_recall_gate(per_session, floor=floor, **ci_config)
        pooled = raw["class_totals"][class_id][0] / max(raw["class_totals"][class_id][1], 1)
        class_metrics[name] = {"pooled_recall": pooled, "ci_lower": lower if lower is not None else "missing", "floor": floor, "passed": passed}
        class_gates[class_id] = (lower, passed)
    group_lowers = {
        "enemy": [value[0] for key, value in class_gates.items() if key in enemy_ids and value[0] is not None],
        "gem": [value[0] for key, value in class_gates.items() if key in gem_ids and value[0] is not None],
        "boss": [value[0] for key, value in class_gates.items() if key in boss_ids and value[0] is not None],
    }
    gate_results: dict[str, Any] = {}
    for group, floor_key in (("enemy", "enemy_recall_min"), ("gem", "gem_recall_min"), ("boss", "boss_recall_min")):
        lowers = group_lowers[group]
        floor = fixed_gates[floor_key]
        value = min(lowers) if lowers else None
        gate_results[f"{group}_recall"] = {"value": value if value is not None else "missing", "minimum": floor, "passed": value is not None and value >= floor}

    heavy_ids = weapon_ids
    slice_gates: dict[str, Any] = {}
    heavy_lower, heavy_pass = session_recall_gate(raw["heavy"], floor=fixed_gates["heavy_effect_recall_min"], **ci_config) if raw["heavy"] else (None, False)
    late_lower, late_pass = session_recall_gate(raw["late"], floor=fixed_gates["late_game_recall_min"], **ci_config) if raw["late"] else (None, False)
    slice_gates["heavy_effect_recall"] = {"value": heavy_lower if heavy_lower is not None else "missing", "minimum": fixed_gates["heavy_effect_recall_min"], "passed": heavy_pass}
    slice_gates["late_game_recall"] = {"value": late_lower if late_lower is not None else "missing", "minimum": fixed_gates["late_game_recall_min"], "passed": late_pass}
    slice_gates["boss_slice"] = {"instances": raw["boss_instances"], "sessions": raw["boss_sessions"], "passed": raw["boss_instances"] >= fixed_gates["boss_min_instances"] and raw["boss_sessions"] >= fixed_gates["boss_min_sessions"]}
    slice_gates["hazard_slice"] = {"instances": raw["hazard_instances"], "sessions": raw["hazard_sessions"], "passed": raw["hazard_instances"] >= fixed_gates["hazard_min_instances"] and raw["hazard_sessions"] >= fixed_gates["hazard_min_sessions"]}

    gt_density = raw["directional_gt"]
    pred_density = raw["directional_pred"]
    if len(gt_density) < 2 or np.std(gt_density) == 0 or np.std(pred_density) == 0:
        directional = 1.0 if gt_density == pred_density else 0.0
    else:
        directional = float(np.corrcoef(gt_density, pred_density)[0, 1])
    nearest_median = float(np.median(raw["nearest"])) if raw["nearest"] else None
    count_error_ci = None
    nearest_error_ci = None
    directional_ci = None
    try:
        _, count_error_ci = cluster_ci_bounds(
            {key: tuple(value) for key, value in raw["count_errors"].items()}, **ci_config,
        )
    except ValueError:
        pass
    nearest_counts = {
        key: (float(sum(values)), float(len(values)))
        for key, values in raw["nearest_by_session"].items() if values
    }
    try:
        _, nearest_error_ci = cluster_ci_bounds(nearest_counts, **ci_config)
    except ValueError:
        pass
    directional_counts = {}
    for session_id, values in raw["directional_by_session"].items():
        gt_values = values["gt"]
        pred_values = values["pred"]
        if len(gt_values) < 2 or np.std(gt_values) == 0 or np.std(pred_values) == 0:
            correlation = 1.0 if gt_values == pred_values else 0.0
        else:
            correlation = float(np.corrcoef(gt_values, pred_values)[0, 1])
        directional_counts[session_id] = (correlation * len(gt_values), len(gt_values))
    try:
        directional_ci, _ = cluster_ci_bounds(directional_counts, **ci_config)
    except ValueError:
        pass
    latency_result = latency_gate(latency, hardware_id=preflight.hardware_id, max_ms=fixed_gates["gpu_p95_latency_ms_max"])
    directional_gate_value = directional_ci if directional_ci is not None else None
    gate_results["directional_density_correlation"] = {"value": directional_gate_value if directional_gate_value is not None else "missing", "pooled": directional if math.isfinite(directional) else "missing", "minimum": fixed_gates["directional_density_correlation_min"], "passed": directional_gate_value is not None and directional_gate_value >= fixed_gates["directional_density_correlation_min"]}
    density_error_max = fixed_gates.get("density_error_ci_upper_max")
    nearest_error_max = fixed_gates.get("nearest_distance_error_ci_upper_max")
    gate_results["density_error_ci"] = {"upper": count_error_ci if count_error_ci is not None else "missing", "maximum": density_error_max, "passed": count_error_ci is not None and count_error_ci <= density_error_max}
    gate_results["nearest_distance_error_ci"] = {"upper": nearest_error_ci if nearest_error_ci is not None else "missing", "pooled_median": nearest_median if nearest_median is not None else "missing", "maximum": nearest_error_max, "passed": nearest_error_ci is not None and nearest_error_ci <= nearest_error_max}
    gate_results["gpu_p95_latency"] = latency_result
    gate_results.update({f"class:{name}": value for name, value in class_metrics.items()})
    gate_results.update({f"slice:{name}": value for name, value in slice_gates.items()})

    mean_count_error = float(np.mean([value[0] / max(value[1], 1) for value in raw["count_errors"].values()])) if raw["count_errors"] else None
    switches: str | int = "unsupported"
    if any(item["track_id"] is not None for frame in frames for item in frame["gt"]):
        switches = _track_id_switches(frames, class_map, score_threshold, preflight.resolved_config.get("tracker", {}))
    metrics = {
        "map50_95": map_value,
        "class_recall": class_metrics,
        "count_error": mean_count_error if mean_count_error is not None else "missing",
        "density_error": mean_count_error if mean_count_error is not None else "missing",
        "density_error_ci_upper": count_error_ci if count_error_ci is not None else "missing",
        "nearest_distance_median": nearest_median if nearest_median is not None else "missing",
        "nearest_distance_error_ci_upper": nearest_error_ci if nearest_error_ci is not None else "missing",
        "directional_density_correlation": directional if math.isfinite(directional) else "missing",
        "directional_density_correlation_ci_lower": directional_ci if directional_ci is not None else "missing",
        "track_id_switches": switches,
        "gpu_p95_latency_ms": latency_result["value"],
        "slices": slice_gates,
        "instance_pooled_recall": raw["total_tp"] / max(raw["total_gt"], 1),
    }
    checkpoint_payload = dict(checkpoint_manifest) if isinstance(checkpoint_manifest, Mapping) else asdict(checkpoint_manifest)
    manifest_hash = checkpoint_manifest_hash or canonical_hash(checkpoint_payload)
    if not is_sha256_hex(manifest_hash):
        raise ValueError("checkpoint manifest hash is invalid")
    if (
        checkpoint_payload.get("training_mode") != "formal"
        or checkpoint_payload.get("development_only") is not False
        or checkpoint_payload.get("formal_detector_eligible") is not False
        or checkpoint_payload.get("data_hash") != preflight.dataset_identity
        or checkpoint_payload.get("build_hash") != preflight.build_identity
        or checkpoint_payload.get("resolved_config_hash") != preflight.config_identity
    ):
        raise ValueError("checkpoint manifest does not match formal preflight identity")
    status = "PASS" if all(result.get("passed") is True for result in gate_results.values()) else "FAIL"
    payload = {
        "schema_version": "world_detector_validation_verdict.v1", "status": status,
        "metrics": metrics, "gates": {"thresholds": fixed_gates, "results": gate_results},
        "score_threshold": score_threshold, "threshold_rule": selected_rule,
        "selection_bias_note": "Threshold selection and gate scoring share model_validation; final_e2e_test remains sealed.",
        "weight_hash": weight_hash, "checkpoint_manifest_hash": manifest_hash,
        "dataset_identity": preflight.dataset_identity, "resolved_config_hash": preflight.config_identity,
        "build_identity": preflight.build_identity, "preflight_canonical_hash": preflight.preflight_canonical_hash,
        "code_hashes": dict(preflight.code_hashes), "inference_caps": dict(inference_caps or {}),
    }
    return ValidationVerdict(
        status=status, metrics=metrics, gates=payload["gates"], score_threshold=score_threshold,
        threshold_rule=selected_rule, selection_bias_note=payload["selection_bias_note"],
        weight_hash=weight_hash, checkpoint_manifest_hash=manifest_hash,
        dataset_identity=preflight.dataset_identity, resolved_config_hash=preflight.config_identity,
        build_identity=preflight.build_identity, preflight_canonical_hash=preflight.preflight_canonical_hash,
        code_hashes=dict(preflight.code_hashes), inference_caps=dict(inference_caps or {}),
        verdict_canonical_hash=canonical_hash(payload),
    )


def publish_formal_detector_artifact(
    store: Any,
    *,
    package_dir: str | pathlib.Path,
    preflight: FormalPreflightReport,
    verdict: ValidationVerdict,
    dataset_descriptor: ArtifactDescriptor,
    logical_prefix: str,
    build_attestation_path: str | pathlib.Path | None = None,
) -> ArtifactDescriptor:
    """formal package、verdict、build object と 04-10 envelope を ArtifactStore へ登録する。

    envelope の build hash は session split が使う build object の SHA-256 に揃える。
    """
    from reinbalance_survivors_contracts.artifact_store import ArtifactStore
    from survivors.vision.world_detector_package import (
        PackageManifest,
        _content_key,
    )

    if not isinstance(store, ArtifactStore):
        raise TypeError("store は共有 ArtifactStore が必要です")
    preflight = FormalPreflightReport.from_dict(preflight.to_dict())
    verdict = ValidationVerdict.from_dict(verdict.to_dict())
    if (
        verdict.status != "PASS"
        or verdict.preflight_canonical_hash != preflight.preflight_canonical_hash
        or verdict.dataset_identity != preflight.dataset_identity
        or verdict.resolved_config_hash != preflight.config_identity
        or verdict.build_identity != preflight.build_identity
        or dataset_descriptor.node_kind != "source_descriptor"
        or bool(dataset_descriptor.parents)
        or dataset_descriptor.identity_metadata.get("synthetic") is True
        or dataset_descriptor.identity_metadata.get("development_only") is True
        or dataset_descriptor.logical_id != preflight.dataset_logical_id
        or dataset_descriptor.identity_metadata.get("game_build_id") != preflight.game_build_id
        or preflight.build_identity != canonical_hash({
            "target_profile_hash": preflight.target_profile_hash,
            "game_build_id": preflight.game_build_id,
            "hardware_id": preflight.hardware_id,
            "code_hashes": dict(preflight.code_hashes),
        })
    ):
        raise FormalPreflightError("dataset descriptor、preflight と validation verdict が一致しません")
    package_dir = pathlib.Path(package_dir)
    manifest_path = package_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    raw_manifest = json.loads(manifest_bytes)
    package_manifest = PackageManifest.from_dict(raw_manifest)
    package_manifest.assert_formal_eligible()
    content_key = _content_key(package_manifest)
    if package_dir.name != content_key:
        raise FormalPreflightError("package directory が manifest content key と一致しません")
    package_files = {
        path.name: path for path in package_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    required_files = {"manifest.json", "world_detector_v2.yaml", "world_class_map_v2.yaml", "model.pt", "validation_verdict.json"}
    if set(package_files) != required_files:
        raise FormalPreflightError("formal package の file 集合が不完全または未知です")
    package_verdict_bytes = package_files["validation_verdict.json"].read_bytes()
    package_verdict = ValidationVerdict.from_dict(json.loads(package_verdict_bytes))
    weight_hash = _hash_file(package_files["model.pt"])
    if (
        hashlib.sha256(package_verdict_bytes).hexdigest() != package_manifest.validation_verdict_hash
        or package_verdict.to_dict() != verdict.to_dict()
        or package_manifest.score_threshold != verdict.score_threshold
        or package_manifest.model_hash != verdict.weight_hash
        or weight_hash != verdict.weight_hash
        or package_manifest.data_hash != preflight.dataset_identity
        or package_manifest.build_hash != preflight.build_identity
    ):
        raise FormalPreflightError("formal package manifest、weight と verdict が一致しません")

    game_build_id = preflight.game_build_id
    if is_sha256_hex(game_build_id):
        if build_attestation_path is None:
            raise FormalPreflightError("sha256 game_build_id には build attestation file が必要です")
        build_bytes = pathlib.Path(build_attestation_path).read_bytes()
        if _sha256(build_bytes) != game_build_id:
            raise FormalPreflightError("build attestation の SHA-256 が game_build_id と一致しません")
    else:
        if build_attestation_path is not None:
            raise FormalPreflightError("human-readable game_build_id には attestation file を指定できません")
        build_bytes = game_build_id.encode("utf-8")
    build_hash = _sha256(build_bytes)
    descriptor_build_hash = dataset_descriptor.identity_metadata.get("build_hash")
    if descriptor_build_hash is not None and descriptor_build_hash != build_hash:
        raise FormalPreflightError("dataset descriptor build_hash が session build object と一致しません")

    base = f"{logical_prefix.rstrip('/')}/{content_key[:16]}"
    refs = []
    package_ref_by_name = {}
    media_types = {
        "manifest.json": "application/json",
        "world_detector_v2.yaml": "application/yaml",
        "world_class_map_v2.yaml": "application/yaml",
        "model.pt": "application/octet-stream",
        "validation_verdict.json": "application/json",
    }
    for name in sorted(required_files):
        ref = store.put(
            logical_id=f"{base}/package/{name}",
            source_path=package_files[name],
            media_type=media_types[name],
        )
        refs.append(ref)
        package_ref_by_name[name] = ref
    refs.append(store.put_bytes(
        logical_id=f"{base}/preflight.json",
        data=canonical_json_bytes(preflight.to_dict()),
        media_type="application/json",
    ))
    refs.append(store.put_bytes(
        logical_id=f"{base}/validation_verdict.json",
        data=canonical_json_bytes(verdict.to_dict()),
        media_type="application/json",
    ))
    build_ref = store.put_bytes(
        logical_id=f"{base}/build_object",
        data=build_bytes,
        media_type="application/octet-stream",
    )
    refs.append(build_ref)
    build_identity = {
        "target_profile_hash": preflight.target_profile_hash,
        "game_build_id": game_build_id,
        "hardware_id": preflight.hardware_id,
        "code_hashes": dict(preflight.code_hashes),
    }
    refs.append(store.put_bytes(
        logical_id=f"{base}/build_identity.json",
        data=canonical_json_bytes(build_identity),
        media_type="application/json",
    ))
    envelope = {
        "schema_version": "world_detector_package.v1",
        "development_only": False,
        "formal_eligible": True,
        "detector_artifact_hash": package_ref_by_name["manifest.json"].sha256,
        "model_hash": package_ref_by_name["model.pt"].sha256,
        "build_hash": build_ref.sha256,
        "files": {
            "detector_artifact_hash": package_ref_by_name["manifest.json"].logical_id,
            "model_hash": package_ref_by_name["model.pt"].logical_id,
            "build_hash": build_ref.logical_id,
        },
    }
    envelope_ref = store.put_bytes(
        logical_id=f"{base}/detector_package.json",
        data=canonical_json_bytes(envelope),
        media_type="application/json",
    )
    refs.append(envelope_ref)
    descriptor = ArtifactDescriptor(
        logical_id=f"{base}/descriptor",
        node_kind="source_descriptor",
        producer_id="world-detector-formal-release",
        producer_version="world_detector_package.v1",
        identity_metadata={
            "manifest_logical_id": envelope_ref.logical_id,
            "content_key": content_key,
            "weight_hash": weight_hash,
            "score_threshold": verdict.score_threshold,
            "dataset_identity": preflight.dataset_identity,
            "dataset_logical_id": dataset_descriptor.logical_id,
            "resolved_config_hash": preflight.config_identity,
            "build_identity": build_identity,
            "verdict_canonical_hash": verdict.verdict_canonical_hash,
            "preflight_canonical_hash": preflight.preflight_canonical_hash,
        },
        parents=(),
        files=tuple(refs),
    )
    store.put_bytes(
        logical_id=descriptor.logical_id,
        data=canonical_json_bytes(descriptor.to_wire()),
        media_type="application/json",
    )
    return descriptor


def emit_resolved_detector_yaml(
    verdict: ValidationVerdict,
    descriptor: ArtifactDescriptor,
    restore_verdict_descriptor: ArtifactDescriptor,
    preflight: FormalPreflightReport,
    out_path: str | pathlib.Path,
) -> pathlib.Path:
    """PASS verdict と restore descriptor に束縛した target detector YAML を出力する。

    出力には runtime が必要とする architecture、threshold、tracker、release lineage を記録する。
    """
    preflight = FormalPreflightReport.from_dict(preflight.to_dict())
    verdict = ValidationVerdict.from_dict(verdict.to_dict())
    identity = descriptor.identity_metadata
    restore = restore_verdict_descriptor.identity_metadata
    manifest_id = identity.get("manifest_logical_id")
    envelope_refs = [ref for ref in descriptor.files if ref.logical_id == manifest_id]
    if (
        verdict.status != "PASS"
        or verdict.preflight_canonical_hash != preflight.preflight_canonical_hash
        or descriptor.node_kind != "source_descriptor"
        or descriptor.parents
        or identity.get("preflight_canonical_hash") != preflight.preflight_canonical_hash
        or identity.get("verdict_canonical_hash") != verdict.verdict_canonical_hash
        or identity.get("dataset_identity") != preflight.dataset_identity
        or identity.get("resolved_config_hash") != preflight.config_identity
        or identity.get("build_identity") != {
            "target_profile_hash": preflight.target_profile_hash,
            "game_build_id": preflight.game_build_id,
            "hardware_id": preflight.hardware_id,
            "code_hashes": dict(preflight.code_hashes),
        }
        or len(envelope_refs) != 1
        or restore_verdict_descriptor.node_kind != "restore_test_verdict"
        or restore.get("subject_logical_id") != descriptor.logical_id
        or restore.get("subject_identity_hash") != descriptor.identity_hash
        or restore.get("verify_mode") != "full"
        or restore.get("passed") is not True
        or restore.get("manifest_hash") != envelope_refs[0].sha256
    ):
        raise FormalPreflightError("resolved yaml inputs do not identify one successful formal release")
    model_cfg = preflight.resolved_config.get("model")
    if not isinstance(model_cfg, Mapping) or model_cfg.get("architecture") != "ssdlite320":
        raise FormalPreflightError("resolved yaml requires the supported ssdlite320 architecture")
    document = {
        "schema_version": "world_detector_target_config.v1",
        "architecture": model_cfg["architecture"],
        "score_threshold": verdict.score_threshold,
        "tracker": dict(preflight.resolved_config.get("tracker", {})),
        "package_logical_id": descriptor.logical_id,
        "content_key": identity["content_key"],
        "restore_verdict_logical_id": restore_verdict_descriptor.logical_id,
        "dataset_identity": preflight.dataset_identity,
        "verdict_canonical_hash": verdict.verdict_canonical_hash,
        "ignore_regions": list(preflight.resolved_config.get("input", {}).get("ignore_regions", [])),
        "code_hashes": dict(preflight.code_hashes),
    }
    target = pathlib.Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(document, sort_keys=True, allow_unicode=True), encoding="utf-8", newline="\n")
    temporary.replace(target)
    return target


def _track_id_switches(
    frames: Sequence[Mapping[str, Any]], class_map: Any, threshold: float, tracker_config: Mapping[str, Any]
) -> int:
    """EntityTracker の ID と annotation track_id の IoU 対応で switch 数を数える。

    session ごとに tracker を新規作成し、別 capture の ID state を混ぜない。
    """
    from survivors.vision.entity_tracker import EntityTracker
    from survivors.vision.world_detector import DetectionResult
    age_by_class = {
        class_map.name_to_id(name): age
        for name, age in tracker_config.get("max_age_by_class", {}).items()
    }
    previous: dict[tuple[str, Any], int] = {}
    switches = 0
    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for frame in frames:
        by_session[frame["session_id"]].append(frame)
    for session_id, session_frames in by_session.items():
        tracker = EntityTracker(
            age_by_class,
            tracker_config.get("max_match_cost", 0.7),
            tracker_config.get("velocity_ema_alpha", 0.6),
            tracker_config.get("confidence_decay_per_frame", 0.9),
            coarse_by_class_id=class_map.coarse_by_class_id(),
        )
        for frame_index, frame in enumerate(session_frames):
            predictions = [item for item in frame["pred"] if item["score"] >= threshold]
            detection = DetectionResult(
                boxes_xyxy=np.asarray([item["box"] for item in predictions], dtype=np.float32).reshape(-1, 4),
                scores=np.asarray([item["score"] for item in predictions], dtype=np.float32),
                class_ids=np.asarray([item["class_id"] for item in predictions], dtype=np.int32),
                image_width=frame["width"], image_height=frame["height"],
            )
            state = tracker.update(detection, frame_index=frame_index, timestamp_ns=frame_index)
            for target in frame["gt"]:
                if target["track_id"] is None:
                    continue
                candidates = [track for track in state.tracks if track.class_id == target["class_id"]]
                if not candidates:
                    continue
                overlaps = [float(_iou_pair(target["box"], track.box_xyxy)) for track in candidates]
                best = int(np.argmax(overlaps))
                if overlaps[best] < 0.5:
                    continue
                identity = (session_id, target["track_id"])
                track_id = candidates[best].track_id
                if identity in previous and previous[identity] != track_id:
                    switches += 1
                previous[identity] = track_id
    return switches


def _iou_pair(left: Sequence[float], right: Sequence[float]) -> float:
    """二つの xyxy box の IoU を返す。

    tracker ID と annotation track ID の frame ごとの対応付けに使う。
    """
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(area_left + area_right - intersection, 1e-12)

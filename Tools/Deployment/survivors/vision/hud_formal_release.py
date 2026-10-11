"""HUD truth を正式 parser package と release artifact に結び付ける。

正式 data の preflight、実 frame atlas、parser validation と publish をまとめます。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    ArtifactRef,
    RestoreTestVerdict,
)
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes
from survivors.capture_dataset import DatasetWriter, SPLIT_NAMES, _split_manifest_from_bytes
from survivors.vision.hud_calibration import (
    CalibrationConfig,
    FitResult,
    GateResult,
    ValidationReport,
    _macro_f1,
    _report_from_wire,
    validate,
)
from survivors.vision.hud_parser import INV_SLOT_COUNT, SCREEN_STATES, HudParser
from survivors.vision.hud_truth import (
    HUD_ROI_NAME,
    LEVEL_UP_STATES,
    HudTruthRecord,
    expected_roi_for_state,
    load_frame_pixels,
    read_hud_truth,
)
from survivors.vision.icon_matcher import (
    ATLAS_SCHEMA_VERSION,
    AtlasManifest,
    IconMatcher,
    TemplateEntry,
    build_template_feature,
    serialize_manifest,
)
from survivors.vision.roi_layout import (
    HP_BAR_ROI,
    INV_SLOT_ROIS,
    XP_BAR_ROI,
    card_icon_roi,
    norm_to_pixels,
)
from survivors.vision.screen_layout import _card_count, _detect_card_rows
from survivors.vision.session_cluster_ci import cluster_ci_bounds


class FormalPreflightError(ValueError):
    """正式 dataset の split、truth、descriptor が不適格である。

    入口の検査結果をこの例外へ集約し、CLI の失敗を明示する。
    """


class FormalAtlasError(ValueError):
    """正式 atlas の crop 出典または vocabulary が不適格である。

    実 frame と manifest の不整合を publish 前に拒否する。
    """


class FormalGateError(ValueError):
    """正式 parser validation の gate または CI が不合格である。

    合格しない report は正式 package に昇格できない。
    """


def _sha256(data: bytes) -> str:
    """bytes の小文字 SHA-256 を返す。

    descriptor binding との比較に使うため hex 表記を固定する。
    """
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    """code/data file を全量メモリ化せず hash 化する。

    大きな capture metadata を一括で読み込まずに済ませる。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value(record: object, name: str, default: Any = None) -> Any:
    """truth record または mapping から field を取り出す。

    annotation dataclass とテスト用 wire mapping の両方を受ける。
    """
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _wire(record: object) -> dict[str, Any]:
    """HUD truth record を unknown field を保った wire にする。

    04-21 が追加した slot level なども hash 対象から落とさない。
    """
    if isinstance(record, HudTruthRecord):
        return record.to_wire()
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError("HUD truth record must be a HudTruthRecord or mapping")


def _code_hashes() -> dict[str, str]:
    """formal parser と release tooling を構成する production file の hash を返す。

    実装が欠けている optional module は code identity に含めない。
    """
    root = Path(__file__).resolve().parents[2]
    names = (
        "survivors/vision/hud_parser.py",
        "survivors/vision/hud_calibration.py",
        "survivors/vision/roi_layout.py",
        "survivors/vision/icon_matcher.py",
        "survivors/vision/digit_parser.py",
        "survivors/vision/bar_parser.py",
        "survivors/vision/choice_parser.py",
        "survivors/vision/slot_level_parser.py",
        "survivors/vision/screen_layout.py",
        "survivors/vision/hud_types.py",
        "survivors/vision/hud_truth.py",
        "survivors/vision/hud_formal_release.py",
        "survivors/vision/session_cluster_ci.py",
        "survivors/artifact_restore_test.py",
    )
    required = set(names) - {
        "survivors/vision/slot_level_parser.py",
        "survivors/vision/hud_types.py",
    }
    result = {}
    for name in names:
        path = root / name
        if not path.is_file():
            if name in required:
                raise FormalPreflightError(f"required code hash input is missing: {name}")
            continue
        result[name] = _hash_file(path)
    return result


def _descriptor_ref(descriptor: ArtifactDescriptor, logical_id: str, digest: str) -> ArtifactRef:
    """descriptor に束縛された論理 ID と content hash の file を返す。

    ID と digest の片方だけが一致する参照は採用しない。
    """
    matches = [
        ref for ref in descriptor.files
        if ref.logical_id == logical_id and ref.sha256 == digest
    ]
    if len(matches) != 1:
        raise FormalPreflightError(f"descriptor file binding mismatch: {logical_id}")
    return matches[0]


def _restore_verdict_matches(
    dataset: ArtifactDescriptor, verdict: ArtifactDescriptor
) -> None:
    """restore verdict を共有型で再構築し、dataset identity と照合する。

    自己申告 wire が full verification の結果として通るのを防ぐ。
    """
    metadata = verdict.identity_metadata
    if (
        verdict.node_kind != "restore_test_verdict"
        or verdict.producer_id != "artifact-restore-test"
        or verdict.files
        or verdict.parents != (dataset.node_ref(),)
    ):
        raise FormalPreflightError("restore verdict is not bound to the dataset descriptor")
    try:
        rebuilt = RestoreTestVerdict(
            logical_id=verdict.logical_id,
            subject=dataset.node_ref(),
            manifest_hash=metadata["manifest_hash"],
            primary_root=verdict.non_identity_metadata["primary_root"],
            backup_root=verdict.non_identity_metadata["backup_root"],
            verify_mode=metadata["verify_mode"],
            checked_object_count=metadata["checked_object_count"],
            passed=metadata["passed"],
            blocking_reasons=metadata["blocking_reasons"],
            producer_id=verdict.producer_id,
            producer_version=verdict.producer_version,
            schema_version=metadata["restore_test_schema_version"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FormalPreflightError(f"invalid restore verdict: {exc}") from exc
    if rebuilt.to_descriptor().identity_hash != verdict.identity_hash:
        raise FormalPreflightError("restore verdict identity hash mismatch")
    if metadata.get("subject_logical_id") != dataset.logical_id or metadata.get(
        "subject_identity_hash"
    ) != dataset.identity_hash:
        raise FormalPreflightError("restore verdict subject does not match dataset")
    if (
        metadata.get("verify_mode") != "full"
        or metadata.get("passed") is not True
        or metadata.get("blocking_reasons") != []
        or type(metadata.get("checked_object_count")) is not int
        or metadata["checked_object_count"] < len(dataset.files)
        or verdict.producer_version != "restore_test_verdict.v1"
    ):
        raise FormalPreflightError("restore verdict must pass full verification")


def _confirmed_truth(
    capture_root: Path,
    dataset: ArtifactDescriptor,
    split_name: str,
    session_id: str,
    bindings: Mapping[str, Any],
    frame_ids: set[int],
    sealed_sessions: set[str],
) -> list[HudTruthRecord]:
    """1 session の bound truth を hash と frame id 付きで読む。

    unconfirmed 行は除外し、sealed split への参照は拒否する。
    """
    if session_id in sealed_sessions:
        raise FormalPreflightError("model truth references a sealed split session")
    binding = bindings.get(session_id)
    if not isinstance(binding, Mapping):
        raise FormalPreflightError(f"missing hud_truth binding: {session_id}")
    logical_id, digest = binding.get("hud_truth_logical_id"), binding.get("hud_truth_sha256")
    if not isinstance(logical_id, str) or not isinstance(digest, str):
        raise FormalPreflightError(f"invalid hud_truth binding: {session_id}")
    _descriptor_ref(dataset, logical_id, digest)
    session_path = capture_root / "capture_sessions" / session_id
    truth_path = session_path / "hud_truth.jsonl"
    try:
        if _hash_file(truth_path) != digest:
            raise FormalPreflightError(f"hud_truth sha256 mismatch: {session_id}")
        records = read_hud_truth(session_path)
    except (OSError, ValueError) as exc:
        if isinstance(exc, FormalPreflightError):
            raise
        raise FormalPreflightError(f"cannot read hud_truth for {session_id}: {exc}") from exc
    confirmed = [record for record in records if record.confirmed is True]
    for record in confirmed:
        if (
            record.schema_version != "hud_truth.v1"
            or record.session_id != session_id
            or record.session_id in sealed_sessions
            or type(record.frame_id) is not int
            or record.frame_id < 0
            or not record.annotator_id
            or not isinstance(record.confirmed_at, str)
        ):
            raise FormalPreflightError("HUD truth session binding is invalid")
        if record.expected_screen_state not in SCREEN_STATES:
            raise FormalPreflightError(f"unknown screen state in {session_id}:{record.frame_id}")
        if record.frame_id not in frame_ids:
            raise FormalPreflightError(f"HUD truth frame does not exist: {session_id}:{record.frame_id}")
        if record.expected_items is not None:
            if not isinstance(record.expected_items, (tuple, list)) or len(record.expected_items) != INV_SLOT_COUNT:
                raise FormalPreflightError(f"invalid expected_items: {session_id}:{record.frame_id}")
            if any(item is None for item in record.expected_items):
                raise FormalPreflightError(f"confirmed item slots contain null: {session_id}:{record.frame_id}")
            if any(not isinstance(item, str) or not item for item in record.expected_items):
                raise FormalPreflightError(f"invalid item id in HUD truth: {session_id}:{record.frame_id}")
        if record.expected_choice is not None:
            if not isinstance(record.expected_choice, (tuple, list)) or record.expected_screen_state not in LEVEL_UP_STATES:
                raise FormalPreflightError(f"invalid choice truth: {session_id}:{record.frame_id}")
            if any(not isinstance(item, str) or not item for item in record.expected_choice):
                raise FormalPreflightError(f"invalid choice truth: {session_id}:{record.frame_id}")
        for name in ("expected_timer_seconds", "expected_hp_ratio", "expected_xp_ratio"):
            value = getattr(record, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (name != "expected_timer_seconds" and value > 1)
            ):
                raise FormalPreflightError(f"invalid {name}: {session_id}:{record.frame_id}")
        if record.expected_level is not None and (
            type(record.expected_level) is not int or record.expected_level < 1
        ):
            raise FormalPreflightError(f"invalid expected_level: {session_id}:{record.frame_id}")
        if record.roi_name != HUD_ROI_NAME or record.expected_roi != expected_roi_for_state(record.expected_screen_state):
            raise FormalPreflightError(f"invalid ROI truth: {session_id}:{record.frame_id}")
        if not isinstance(record.draft_source, Mapping) or not all(
            isinstance(record.draft_source.get(key), str)
            for key in ("parser_artifact_hash", "atlas_content_hash")
        ):
            raise FormalPreflightError(f"invalid draft source: {session_id}:{record.frame_id}")
    if not confirmed:
        raise FormalPreflightError(f"session has no confirmed HUD truth: {session_id}")
    return confirmed


def _time_band(value: object, selector: str) -> bool:
    """`start-end` selector が秒単位の timer を含むか判定する。

    欠損 timer はどの coverage band にも割り当てない。
    """
    match = re.fullmatch(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)", selector)
    if match is None or value is None:
        return False
    seconds = float(value)
    return float(match.group(1)) <= seconds < float(match.group(2))


def _slice_counts(records: Sequence[HudTruthRecord]) -> dict[str, Any]:
    """validation truth から正式 release の coverage slice 件数を集計する。

    card、item、button、fallback、ROI 負例を別の母集団として数える。
    """
    states = Counter(record.expected_screen_state for record in records)
    cards = Counter(
        str(len(record.expected_choice))
        for record in records
        if record.expected_choice is not None
    )
    items = Counter(
        item
        for record in records
        if record.expected_items is not None
        for item in record.expected_items
    )
    buttons = Counter()
    for record in records:
        values = record.extra.get("expected_buttons", ())
        if isinstance(values, (list, tuple)):
            buttons.update(value for value in values if isinstance(value, str))
    return {
        "time_bands": {},
        "screen_states": dict(states),
        "card_counts": dict(cards),
        "item_classes": dict(items),
        "buttons": dict(buttons),
        "fallback": sum(record.expected_screen_state == "level_up_fallback" for record in records),
        "roi_negative": sum(record.expected_roi is None for record in records),
    }


def _check_minimum_counts(
    records: Sequence[HudTruthRecord], minimum_counts: Mapping[str, Any]
) -> dict[str, Any]:
    """設定された全 coverage slice の最小 validation 件数を検査する。

    未知の selector と不足件数は preflight で止める。
    """
    expected_keys = {
        "time_bands", "screen_states", "card_counts", "item_classes", "buttons",
        "fallback", "roi_negative",
    }
    if not isinstance(minimum_counts, Mapping) or set(minimum_counts) != expected_keys:
        raise FormalPreflightError(f"minimum_counts keys must be {sorted(expected_keys)}")
    counts = _slice_counts(records)
    for category in ("time_bands", "screen_states", "card_counts", "item_classes", "buttons"):
        requirements = minimum_counts[category]
        if not isinstance(requirements, Mapping) or (category != "buttons" and not requirements):
            raise FormalPreflightError(f"minimum_counts.{category} must be a non-empty mapping")
        if any(not isinstance(selector, str) for selector in requirements):
            raise FormalPreflightError(f"minimum_counts.{category} keys must be strings")
        if category == "time_bands":
            observed = {
                selector: sum(_time_band(record.expected_timer_seconds, selector) for record in records)
                for selector in requirements
            }
            counts[category] = observed
        for selector, minimum in requirements.items():
            if type(minimum) is not int or minimum < 1:
                raise FormalPreflightError(f"minimum count must be a positive integer: {category}.{selector}")
            count = counts[category].get(str(selector), 0)
            if count < minimum:
                raise FormalPreflightError(
                    f"validation slice below minimum: {category}.{selector}={count} < {minimum}"
                )
    for category in ("fallback", "roi_negative"):
        minimum = minimum_counts[category]
        if type(minimum) is not int or minimum < 1 or counts[category] < minimum:
            raise FormalPreflightError(f"validation slice below minimum: {category}={counts[category]}")
    return counts


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """formal data preflight の hash-bound summary。

    session truth と code identity を同じ canonical hash に束ねる。
    """

    dataset_logical_id: str
    dataset_identity_hash: str
    restore_verdict_logical_id: str
    restore_verdict_identity_hash: str
    target_profile_hash: str
    game_build_id: str
    train_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]
    train_record_count: int
    validation_record_count: int
    train_truth_canonical_hash: str
    validation_truth_canonical_hash: str
    slice_counts: Mapping[str, Any]
    code_hashes: Mapping[str, str]
    preflight_canonical_hash: str
    schema_version: str = "hud_formal_preflight.v1"

    def to_wire(self) -> dict[str, Any]:
        """preflight report を package 用 wire にする。

        canonical hash は再計算に使う payload だけを含める。
        """
        return {
            "schema_version": self.schema_version,
            "dataset_logical_id": self.dataset_logical_id,
            "dataset_identity_hash": self.dataset_identity_hash,
            "restore_verdict_logical_id": self.restore_verdict_logical_id,
            "restore_verdict_identity_hash": self.restore_verdict_identity_hash,
            "target_profile_hash": self.target_profile_hash,
            "game_build_id": self.game_build_id,
            "train_sessions": list(self.train_sessions),
            "validation_sessions": list(self.validation_sessions),
            "train_record_count": self.train_record_count,
            "validation_record_count": self.validation_record_count,
            "train_truth_canonical_hash": self.train_truth_canonical_hash,
            "validation_truth_canonical_hash": self.validation_truth_canonical_hash,
            "slice_counts": dict(self.slice_counts),
            "code_hashes": dict(self.code_hashes),
            "preflight_canonical_hash": self.preflight_canonical_hash,
        }

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> "PreflightReport":
        """strict wire を復元し preflight hash を検証する。

        未知 schema や欠けた field は release lineage に入れない。
        """
        expected = {
            "schema_version", "dataset_logical_id", "dataset_identity_hash",
            "restore_verdict_logical_id", "restore_verdict_identity_hash", "target_profile_hash",
            "game_build_id", "train_sessions", "validation_sessions", "train_record_count",
            "validation_record_count", "train_truth_canonical_hash", "validation_truth_canonical_hash",
            "slice_counts", "code_hashes", "preflight_canonical_hash",
        }
        if not isinstance(wire, Mapping) or set(wire) != expected or wire["schema_version"] != "hud_formal_preflight.v1":
            raise ValueError("preflight report fields do not match schema")
        identity = {key: value for key, value in wire.items() if key != "preflight_canonical_hash"}
        if canonical_hash(identity) != wire["preflight_canonical_hash"]:
            raise ValueError("preflight report hash mismatch")
        return cls(
            wire["dataset_logical_id"], wire["dataset_identity_hash"],
            wire["restore_verdict_logical_id"], wire["restore_verdict_identity_hash"],
            wire["target_profile_hash"], wire["game_build_id"],
            tuple(wire["train_sessions"]), tuple(wire["validation_sessions"]),
            wire["train_record_count"], wire["validation_record_count"],
            wire["train_truth_canonical_hash"], wire["validation_truth_canonical_hash"],
            wire["slice_counts"], wire["code_hashes"], wire["preflight_canonical_hash"],
        )


def run_formal_preflight(
    *,
    capture_store_root: str | Path,
    dataset_descriptor: ArtifactDescriptor,
    restore_verdict_descriptor: ArtifactDescriptor,
    target_profile_hash: str,
    minimum_counts: Mapping[str, Any],
) -> PreflightReport:
    """model_train／model_validation だけの sealed capture と HUD truth を検証する。

    error_calibration と final_e2e_test の pixel は開封しない。
    """
    if not isinstance(dataset_descriptor, ArtifactDescriptor) or dataset_descriptor.node_kind != "source_descriptor":
        raise FormalPreflightError("dataset must be a source_descriptor")
    _restore_verdict_matches(dataset_descriptor, restore_verdict_descriptor)
    identity = dataset_descriptor.identity_metadata
    if identity.get("target_profile_hash") != target_profile_hash:
        raise FormalPreflightError("target profile hash does not match dataset descriptor")
    build_id = identity.get("game_build_id")
    dataset_hash = identity.get("dataset_identity_hash")
    if (
        not isinstance(build_id, str)
        or not build_id
        or not isinstance(dataset_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", dataset_hash)
    ):
        raise FormalPreflightError("dataset identity is incomplete")
    manifest_id = identity.get("manifest_logical_id")
    manifest_refs = [ref for ref in dataset_descriptor.files if ref.logical_id == manifest_id]
    if len(manifest_refs) != 1:
        raise FormalPreflightError("dataset split manifest reference is missing or ambiguous")
    root = Path(capture_store_root).expanduser().resolve()
    split_path = root / "capture_split_manifest.json"
    try:
        split_bytes = split_path.read_bytes()
        if _sha256(split_bytes) != manifest_refs[0].sha256:
            raise FormalPreflightError("split manifest sha256 mismatch")
        split = _split_manifest_from_bytes(split_path, split_bytes)
    except (OSError, ValueError) as exc:
        if isinstance(exc, FormalPreflightError):
            raise
        raise FormalPreflightError(f"invalid frozen split manifest: {exc}") from exc
    if split.frozen is not True:
        raise FormalPreflightError("split manifest must be frozen")
    model_bindings = identity.get("model_session_bindings")
    truth_bindings = identity.get("hud_truth_bindings")
    frame_bindings = identity.get("frames_bindings")
    if (
        not isinstance(model_bindings, Mapping)
        or not isinstance(truth_bindings, Mapping)
        or not isinstance(frame_bindings, Mapping)
    ):
        raise FormalPreflightError("dataset descriptor lacks model/truth session bindings")
    if restore_verdict_descriptor.identity_metadata.get("manifest_hash") != manifest_refs[0].sha256:
        raise FormalPreflightError("restore verdict manifest hash does not match split manifest")

    sealed_sessions = {
        unit.session_id
        for split_name in SPLIT_NAMES
        if split_name not in {"model_train", "model_validation"}
        for unit in split.splits[split_name]
    }
    collected: dict[str, list[HudTruthRecord]] = {"model_train": [], "model_validation": []}
    session_sets: dict[str, tuple[str, ...]] = {}
    for split_name in collected:
        units = split.splits[split_name]
        if not units:
            raise FormalPreflightError(f"split has no sessions: {split_name}")
        ids = tuple(unit.session_id for unit in units)
        session_sets[split_name] = ids
        for unit in units:
            binding = model_bindings.get(unit.session_id)
            if not isinstance(binding, Mapping):
                raise FormalPreflightError(f"missing model session binding: {unit.session_id}")
            session_path = root / "capture_sessions" / unit.session_id
            session_manifest_path = session_path / "session_manifest.json"
            try:
                manifest_bytes = session_manifest_path.read_bytes()
                if _sha256(manifest_bytes) != unit.session_manifest_sha256:
                    raise FormalPreflightError(f"session manifest hash mismatch: {unit.session_id}")
                if binding.get("session_manifest_sha256") != unit.session_manifest_sha256:
                    raise FormalPreflightError(f"model session binding hash mismatch: {unit.session_id}")
                _descriptor_ref(
                    dataset_descriptor,
                    binding["session_manifest_logical_id"],
                    binding["session_manifest_sha256"],
                )
                if _hash_file(session_path / "annotations.jsonl") != binding["annotation_sha256"]:
                    raise FormalPreflightError(f"annotation hash mismatch: {unit.session_id}")
                _descriptor_ref(
                    dataset_descriptor,
                    binding["annotation_logical_id"],
                    binding["annotation_sha256"],
                )
                manifest = DatasetWriter.restore(root, unit.session_id, metadata_only=True)
                frame_binding = frame_bindings.get(unit.session_id)
                if not isinstance(frame_binding, Mapping):
                    raise FormalPreflightError(f"missing frames binding: {unit.session_id}")
                frames_sha = frame_binding.get("frames_sha256")
                frames_id = frame_binding.get("frames_logical_id")
                if (
                    not isinstance(frames_sha, str)
                    or not isinstance(frames_id, str)
                    or _hash_file(session_path / manifest.metadata_path) != frames_sha
                ):
                    raise FormalPreflightError(f"frames metadata hash mismatch: {unit.session_id}")
                _descriptor_ref(dataset_descriptor, frames_id, frames_sha)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                if isinstance(exc, FormalPreflightError):
                    raise
                raise FormalPreflightError(f"invalid model session {unit.session_id}: {exc}") from exc
            if (
                manifest.formal_dataset_eligible is not True
                or manifest.pixel_source != "lossless_png"
                or manifest.target_profile_hash != target_profile_hash
                or manifest.game_build_id != build_id
                or unit.game_build_id != build_id
            ):
                raise FormalPreflightError(f"session is not eligible for this formal build: {unit.session_id}")
            collected[split_name].extend(
                _confirmed_truth(
                    root,
                    dataset_descriptor,
                    split_name,
                    unit.session_id,
                    truth_bindings,
                    {record.frame_id for record in manifest.frame_records},
                    sealed_sessions,
                )
            )
    train_ids = set(session_sets["model_train"])
    validation_ids = set(session_sets["model_validation"])
    if train_ids & validation_ids:
        raise FormalPreflightError("model_train and model_validation sessions overlap")
    slice_counts = _check_minimum_counts(collected["model_validation"], minimum_counts)
    train_truth_hash = canonical_hash([_wire(record) for record in sorted(collected["model_train"], key=lambda r: (r.session_id, r.frame_id))])
    validation_truth_hash = canonical_hash([_wire(record) for record in sorted(collected["model_validation"], key=lambda r: (r.session_id, r.frame_id))])
    payload = {
        "schema_version": "hud_formal_preflight.v1",
        "dataset_logical_id": dataset_descriptor.logical_id,
        "dataset_identity_hash": dataset_hash,
        "restore_verdict_logical_id": restore_verdict_descriptor.logical_id,
        "restore_verdict_identity_hash": restore_verdict_descriptor.identity_hash,
        "target_profile_hash": target_profile_hash,
        "game_build_id": build_id,
        "train_sessions": sorted(train_ids),
        "validation_sessions": sorted(validation_ids),
        "train_record_count": len(collected["model_train"]),
        "validation_record_count": len(collected["model_validation"]),
        "train_truth_canonical_hash": train_truth_hash,
        "validation_truth_canonical_hash": validation_truth_hash,
        "slice_counts": slice_counts,
        "code_hashes": _code_hashes(),
    }
    return PreflightReport(
        dataset_logical_id=payload["dataset_logical_id"],
        dataset_identity_hash=dataset_hash,
        restore_verdict_logical_id=payload["restore_verdict_logical_id"],
        restore_verdict_identity_hash=payload["restore_verdict_identity_hash"],
        target_profile_hash=target_profile_hash,
        game_build_id=build_id,
        train_sessions=tuple(payload["train_sessions"]),
        validation_sessions=tuple(payload["validation_sessions"]),
        train_record_count=payload["train_record_count"],
        validation_record_count=payload["validation_record_count"],
        train_truth_canonical_hash=train_truth_hash,
        validation_truth_canonical_hash=validation_truth_hash,
        slice_counts=slice_counts,
        code_hashes=payload["code_hashes"],
        preflight_canonical_hash=canonical_hash(payload),
    )


def _load_vocabulary(path: str | Path) -> tuple[dict[str, str], str]:
    """正式 atlas 用 vocabulary を fail-closed で読む。

    未登録 item を近い class へ暗黙に割り当てない。
    """
    try:
        wire = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise FormalAtlasError(f"cannot read item vocabulary: {exc}") from exc
    if not isinstance(wire, Mapping) or not isinstance(wire.get("weapons"), Mapping) or not isinstance(wire.get("passives"), Mapping):
        raise FormalAtlasError("vocabulary requires weapons and passives mappings")
    items = {str(key): "weapon" for key in wire["weapons"]}
    items.update({str(key): "passive" for key in wire["passives"]})
    non_items = wire.get("non_items", ())
    empty_slot = wire.get("empty_slot")
    if not isinstance(non_items, list) or not isinstance(empty_slot, str) or empty_slot != "empty_slot":
        raise FormalAtlasError("vocabulary fallback and empty_slot fields are invalid")
    items.update({item_id: "fallback" for item_id in non_items if isinstance(item_id, str)})
    items[empty_slot] = "unknown"
    return items, empty_slot


def _session_frames(value: object) -> tuple[Path, Mapping[int, Any]]:
    """metadata-only SessionManifest から遅延 frame lookup を作る。

    必要な frame_id だけを後段の loader へ渡す。
    """
    session_path = getattr(value, "session_path", None)
    frame_records = getattr(value, "frame_records", None)
    if not isinstance(session_path, Path) or not isinstance(frame_records, (tuple, list)):
        raise FormalAtlasError("frames_by_session values must be metadata-only SessionManifest objects")
    records = {record.frame_id: record for record in frame_records}
    return session_path, records


def _validate_formal_inputs(
    frames_by_session: Mapping[str, object],
    truth: Sequence[object],
    preflight: PreflightReport,
    split_name: str,
) -> None:
    """formal atlas／予測の session と truth を preflight identity に結び付ける。

後段へ渡す値が preflight 後に差し替わっていないことを同じ入口条件で確認します。
    """
    if not isinstance(preflight, PreflightReport):
        raise FormalPreflightError("formal inputs require a PreflightReport")
    try:
        preflight = PreflightReport.from_wire(preflight.to_wire())
    except (AttributeError, TypeError, ValueError) as exc:
        raise FormalPreflightError("invalid formal preflight report") from exc
    if split_name == "model_train":
        expected_sessions = preflight.train_sessions
        expected_count = preflight.train_record_count
        expected_hash = preflight.train_truth_canonical_hash
    elif split_name == "model_validation":
        expected_sessions = preflight.validation_sessions
        expected_count = preflight.validation_record_count
        expected_hash = preflight.validation_truth_canonical_hash
    else:
        raise FormalPreflightError(f"invalid formal split: {split_name}")
    expected_session_set = set(expected_sessions)
    if (
        len(expected_session_set) != len(expected_sessions)
        or set(frames_by_session) != expected_session_set
    ):
        raise FormalPreflightError(f"{split_name} sessions differ from preflight")
    frame_ids: dict[str, set[int]] = {}
    for session_id, manifest in frames_by_session.items():
        if (
            getattr(manifest, "session_id", None) != session_id
            or getattr(manifest, "formal_dataset_eligible", None) is not True
            or getattr(manifest, "pixel_source", None) != "lossless_png"
            or getattr(manifest, "target_profile_hash", None) != preflight.target_profile_hash
            or getattr(manifest, "game_build_id", None) != preflight.game_build_id
        ):
            raise FormalPreflightError(f"session is not eligible for this formal build: {session_id}")
        _, records = _session_frames(manifest)
        if len(records) != len(manifest.frame_records):
            raise FormalPreflightError(f"session contains duplicate frame ids: {session_id}")
        frame_ids[session_id] = set(records)
    confirmed = [record for record in truth if _value(record, "confirmed") is True]
    keys = [(_value(record, "session_id"), _value(record, "frame_id")) for record in confirmed]
    if (
        len(confirmed) != expected_count
        or len(set(keys)) != len(keys)
        or any(session_id not in expected_session_set or frame_id not in frame_ids[session_id] for session_id, frame_id in keys)
        or canonical_hash(sorted((_wire(record) for record in confirmed), key=lambda item: (item["session_id"], item["frame_id"]))) != expected_hash
    ):
        raise FormalPreflightError(f"{split_name} truth changed after preflight")


def _frame(session: object, frame_id: int, cache: dict[tuple[str, int], np.ndarray]) -> tuple[np.ndarray, Any]:
    """必要 frame を一度だけ decode し、隣接処理で共有する。

    hash を確かめてから pixel を返し、session 全体は保持しない。
    """
    session_path, frame_records = _session_frames(session)
    record = frame_records.get(frame_id)
    if record is None:
        raise ValueError(f"missing frame id {frame_id}")
    key = (str(session_path), frame_id)
    if key not in cache:
        cache[key] = load_frame_pixels(session_path, record)
    return cache[key], record


def build_formal_atlas(
    *,
    train_frames_by_session: Mapping[str, object],
    train_truth: Sequence[object],
    vocabulary_path: str | Path,
    preflight: PreflightReport,
    profile_hash: str,
    build_hash: str,
    min_crops: int = 3,
    output_path: str | Path | None = None,
) -> tuple[AtlasManifest, dict[str, Any]]:
    """model_train truth が指す実 frame の crop から formal atlas を作る。

    inventory と card の surface ごとに代表 feature と provenance を残す。
    """
    if type(min_crops) is not int or min_crops < 1:
        raise FormalAtlasError("min_crops must be a positive integer")
    _validate_formal_inputs(train_frames_by_session, train_truth, preflight, "model_train")
    if profile_hash != preflight.target_profile_hash or build_hash != preflight.game_build_id:
        raise FormalPreflightError("formal atlas profile/build differs from preflight")
    vocabulary, empty_slot = _load_vocabulary(vocabulary_path)
    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    frame_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in train_truth:
        if _value(record, "confirmed") is not True:
            continue
        session_id, frame_id = _value(record, "session_id"), _value(record, "frame_id")
        session = train_frames_by_session.get(session_id)
        if session is None:
            raise FormalAtlasError(f"train frame has no session metadata: {session_id}")
        frame, frame_record = _frame(session, frame_id, {})
        expected_items = _value(record, "expected_items")
        if expected_items is not None:
            if len(expected_items) != INV_SLOT_COUNT or any(item is None for item in expected_items):
                raise FormalAtlasError(f"invalid confirmed inventory truth: {session_id}:{frame_id}")
            for index, item_id in enumerate(expected_items):
                if item_id not in vocabulary:
                    raise FormalAtlasError(f"item id is not in vocabulary: {item_id}")
                roi = norm_to_pixels(INV_SLOT_ROIS[index], frame.shape[1], frame.shape[0])
                key = (item_id, "inventory")
                grouped[key].append(build_template_feature(roi.crop(frame)))
                frame_hashes[key].add(frame_record.object_sha256)
        choice = _value(record, "expected_choice")
        if _value(record, "expected_screen_state") in LEVEL_UP_STATES and choice is not None:
            if len(choice) == _card_count(_detect_card_rows(frame)):
                for index, item_id in enumerate(choice):
                    if item_id not in vocabulary:
                        raise FormalAtlasError(f"card item id is not in vocabulary: {item_id}")
                    key = (item_id, "card")
                    grouped[key].append(build_template_feature(card_icon_roi(index).crop(frame)))
                    frame_hashes[key].add(frame_record.object_sha256)
    empty_key = (empty_slot, "inventory")
    if not grouped.get(empty_key):
        raise FormalAtlasError("formal atlas requires confirmed empty_slot crops")
    entries: list[TemplateEntry] = []
    provenance_entries: dict[str, Any] = {}
    for (item_id, surface), features in sorted(grouped.items()):
        if len(features) < min_crops:
            raise FormalAtlasError(f"{item_id}@{surface} has {len(features)} crops; need {min_crops}")
        kind = vocabulary[item_id]
        feature = np.median(np.stack(features), axis=0).astype(np.float32)
        entries.append(TemplateEntry(item_id, kind, 1, 1, feature, surface))
        provenance_entries[f"{item_id}@{surface}"] = {
            "crop_count": len(features),
            "source_frame_hashes": sorted(frame_hashes[(item_id, surface)]),
        }
    atlas_content_hash = hashlib.sha256(
        np.concatenate([entry.feature for entry in entries]).tobytes()
    ).hexdigest()
    atlas = AtlasManifest(
        ATLAS_SCHEMA_VERSION,
        profile_hash,
        build_hash,
        False,
        True,
        atlas_content_hash,
        tuple(entries),
    )
    provenance = {
        "schema_version": "hud_atlas_provenance.v1",
        "entries": provenance_entries,
        "atlas_content_hash": atlas_content_hash,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(serialize_manifest(atlas))
    return atlas, provenance


def _hud_roi() -> tuple[int, int, int, int]:
    """固定 HUD 帯を 1920×1080 pixel の bounding box で返す。

    formal ROI gate では HUD presence の基準矩形として使う。
    """
    hp = norm_to_pixels(HP_BAR_ROI)
    xp = norm_to_pixels(XP_BAR_ROI)
    return (min(hp.x0, xp.x0), min(hp.y0, xp.y0), max(hp.x1, xp.x1), max(hp.y1, xp.y1))


def predict_validation_records(
    *,
    validation_frames_by_session: Mapping[str, object],
    validation_truth: Sequence[object],
    atlas: IconMatcher,
    preflight: PreflightReport,
) -> list[dict[str, Any]]:
    """HudParser を session ごとに reset し、truth と独立した予測値を追加する。

    frame 順を維持し prediction の値を expected field へ流用しない。
    """
    if not isinstance(atlas, IconMatcher) or atlas.manifest.development_only or not atlas.manifest.formal_parser_eligible:
        raise FormalAtlasError("prediction requires a formal icon atlas")
    _validate_formal_inputs(validation_frames_by_session, validation_truth, preflight, "model_validation")
    if (
        atlas.manifest.profile_hash != preflight.target_profile_hash
        or atlas.manifest.build_hash != preflight.game_build_id
    ):
        raise FormalPreflightError("formal prediction atlas profile/build differs from preflight")
    records_by_session: dict[str, list[object]] = defaultdict(list)
    for truth in validation_truth:
        if _value(truth, "confirmed") is True:
            records_by_session[_value(truth, "session_id")].append(truth)
    output: list[dict[str, Any]] = []
    for session_id in sorted(records_by_session):
        session = validation_frames_by_session.get(session_id)
        if session is None:
            raise ValueError(f"validation session not found: {session_id}")
        parser = HudParser(parser_artifact_hash="formal-validation-placeholder", icon_matcher=atlas)
        parser.reset_temporal_state()
        for truth in sorted(records_by_session[session_id], key=lambda record: _value(record, "frame_id")):
            frame_id = _value(truth, "frame_id")
            frame, frame_record = _frame(session, frame_id, {})
            started = perf_counter()
            state = parser.parse(
                frame,
                session_id=session_id,
                frame_index=frame_id,
                captured_monotonic_ns=frame_record.captured_monotonic_ns,
            )
            latency = (perf_counter() - started) * 1000.0
            wire = _wire(truth)
            cards = "|".join(card.item_id for card in state.cards) if state.cards else None
            hud_present = state.screen_state in {"gameplay", "level_up_items", "level_up_fallback", "chest"}
            wire.update({
                "sample_id": f"{session_id}:{frame_id}",
                "split": "model_validation",
                "predicted_timer_seconds": state.timer_seconds,
                "predicted_level": state.level,
                "predicted_hp_ratio": state.hp_ratio,
                "predicted_xp_ratio": state.xp_ratio,
                "predicted_items": list(state.inventory),
                "predicted_choice": cards,
                "predicted_screen_state": state.screen_state,
                "predicted_roi": list(_hud_roi()) if hud_present else None,
                "latency_ms": latency,
                "_hud_truth_wire": _wire(truth),
            })
            output.append(wire)
    return output


def _roi_value(value: object) -> tuple[float, float, float, float] | None:
    """ROI wire を有限座標の tuple に正規化する。

    null は HUD 不在を表す値としてそのまま保持する。
    """
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("ROI must contain four coordinates")
    coords = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in coords):
        raise ValueError("ROI coordinates must be finite")
    return coords


def _roi_center_error(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """二つの ROI 中心間の pixel 距離を返す。

    4 辺の差ではなく矩形中心のずれだけを測る。
    """
    return math.hypot((left[0] + left[2] - right[0] - right[2]) / 2, (left[1] + left[3] - right[1] - right[3]) / 2)


def _formal_report_hash_payload(report: "FormalValidationReport") -> dict[str, Any]:
    """formal report canonical hash の対象 field を返す。

    hash 自身を payload から外して循環参照を避ける。
    """
    return report.to_wire(include_hash=False)


@dataclass(frozen=True, slots=True)
class FormalValidationReport:
    """pooled gate、session CI と lineage hash を束ねる正式 report。

    pooled 判定と session 偏りの両方を package の入口で検査する。
    """

    pooled_report: ValidationReport
    session_values: Mapping[str, Mapping[str, tuple[float, float]]]
    ci_bounds: Mapping[str, tuple[float, float]]
    ci_passed: Mapping[str, bool]
    preflight_canonical_hash: str
    validation_truth_canonical_hash: str
    atlas_content_hash: str
    fit_canonical_hash: str
    code_hashes: Mapping[str, str]
    diagnostics: Mapping[str, Any]
    formal_report_canonical_hash: str
    roi_gate_definition: str = "presence"
    schema_version: str = "hud_formal_validation.v1"

    @property
    def all_passed(self) -> bool:
        """pooled 11 gate と 10 本の session CI がすべて合格か返す。

        CI が欠けている古い report は合格として扱わない。
        """
        return self.pooled_report.all_passed and len(self.ci_passed) == 10 and all(self.ci_passed.values())

    def to_wire(self, *, include_hash: bool = True) -> dict[str, Any]:
        """formal report を strict JSON wire へ変換する。

        include_hash=False は canonical payload の再検証専用である。
        """
        wire = {
            "schema_version": self.schema_version,
            "pooled_report": self.pooled_report.to_wire(),
            "session_values": {
                name: {session: list(pair) for session, pair in values.items()}
                for name, values in self.session_values.items()
            },
            "ci_bounds": {name: list(bounds) for name, bounds in self.ci_bounds.items()},
            "ci_passed": dict(self.ci_passed),
            "preflight_canonical_hash": self.preflight_canonical_hash,
            "validation_truth_canonical_hash": self.validation_truth_canonical_hash,
            "atlas_content_hash": self.atlas_content_hash,
            "fit_canonical_hash": self.fit_canonical_hash,
            "code_hashes": dict(self.code_hashes),
            "diagnostics": dict(self.diagnostics),
            "roi_gate_definition": self.roi_gate_definition,
            "all_passed": self.all_passed,
        }
        if include_hash:
            wire["formal_report_canonical_hash"] = self.formal_report_canonical_hash
        return wire

    @classmethod
    def from_wire(cls, wire: Mapping[str, Any]) -> "FormalValidationReport":
        """strict formal report wire を復元して gate と hash chain を検証する。

        保存された all_passed 値ではなく CI から合否を再計算する。
        """
        expected = {
            "schema_version", "pooled_report", "session_values", "ci_bounds", "ci_passed",
            "preflight_canonical_hash", "validation_truth_canonical_hash", "atlas_content_hash",
            "fit_canonical_hash", "code_hashes", "diagnostics", "roi_gate_definition",
            "all_passed", "formal_report_canonical_hash",
        }
        if not isinstance(wire, Mapping) or set(wire) != expected or wire["schema_version"] != "hud_formal_validation.v1":
            raise ValueError("formal validation report fields do not match schema")
        pooled = _report_from_wire(wire["pooled_report"])
        session_values = {
            name: {session: tuple(pair) for session, pair in values.items()}
            for name, values in wire["session_values"].items()
        }
        bounds = {name: tuple(value) for name, value in wire["ci_bounds"].items()}
        report = cls(
            pooled, session_values, bounds, wire["ci_passed"],
            wire["preflight_canonical_hash"], wire["validation_truth_canonical_hash"],
            wire["atlas_content_hash"], wire["fit_canonical_hash"], wire["code_hashes"],
            wire["diagnostics"], wire["formal_report_canonical_hash"],
            wire["roi_gate_definition"], wire["schema_version"],
        )
        expected_ci = {
            "timer_exact", "level", "hp_mae", "xp_mae", "item", "choice",
            "screen_state_f1", "roi_center_error", "roi_inside_rate", "roi_false_positive",
        }
        if (
            report.roi_gate_definition != "presence"
            or set(report.ci_bounds) != expected_ci
            or set(report.ci_passed) != expected_ci
            or set(report.session_values) != expected_ci
        ):
            raise ValueError("formal report CI fields do not match schema")
        gates = {gate.name: gate for gate in report.pooled_report.gates}
        for name in expected_ci:
            bound = report.ci_bounds[name]
            if len(bound) != 2 or any(not math.isfinite(float(value)) for value in bound) or bound[0] > bound[1]:
                raise ValueError(f"invalid confidence interval for {name}")
            if not isinstance(report.session_values[name], Mapping):
                raise ValueError(f"invalid session values for {name}")
            gate = gates[name]
            expected_pass = bound[0] >= gate.threshold if gate.direction == "floor" else bound[1] <= gate.threshold
            if type(report.ci_passed[name]) is not bool or report.ci_passed[name] is not expected_pass:
                raise ValueError(f"formal CI decision mismatch for {name}")
        if wire["all_passed"] is not report.all_passed:
            raise ValueError("formal validation report all_passed mismatch")
        if canonical_hash(report.to_wire(include_hash=False)) != report.formal_report_canonical_hash:
            raise ValueError("formal validation report hash mismatch")
        return report


def validate_formal(
    records: Sequence[Mapping[str, Any]],
    *,
    config: CalibrationConfig | None = None,
    preflight: PreflightReport,
    atlas_content_hash: str,
    fit_canonical_hash: str,
) -> FormalValidationReport:
    """pooled validation と session-cluster CI を計算し、presence ROI gate を適用する。

    判定対象のない session は各 gate の分母から除外する。
    """
    if not records:
        raise ValueError("formal validation requires records")
    cfg = CalibrationConfig() if config is None else config
    sessions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    truth_wires = []
    scoring = []
    for record in records:
        session_id = record.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("formal record lacks session_id")
        truth_wire = record.get("_hud_truth_wire")
        if not isinstance(truth_wire, Mapping):
            raise ValueError("formal record lacks source truth wire")
        truth_wires.append(dict(truth_wire))
        sessions[session_id].append(record)
        sample = dict(record)
        choice = sample.get("expected_choice")
        if isinstance(choice, (list, tuple)):
            sample["expected_choice"] = "|".join(choice)
        scoring.append(sample)
    validation_truth_hash = canonical_hash(
        sorted(truth_wires, key=lambda item: (item["session_id"], item["frame_id"]))
    )
    if validation_truth_hash != preflight.validation_truth_canonical_hash:
        raise FormalPreflightError("validation truth changed after preflight")
    pooled = validate(scoring, cfg, split="model_validation", formal_mode=True, formal_dataset_eligible=True)
    gate_by_name = {gate.name: gate for gate in pooled.gates}
    session_values: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    custom_metrics = {
        "timer_exact": "ratio",
        "level": "ratio",
        "hp_mae": "mean_error",
        "xp_mae": "mean_error",
        "item": "item_ratio",
        "choice": "choice_ratio",
        "screen_state_f1": "f1",
        "roi_center_error": "roi_center",
        "roi_inside_rate": "roi_inside",
        "roi_false_positive": "roi_false_positive",
    }
    for session_id, session_records in sessions.items():
        for name, metric in custom_metrics.items():
            numerator = denominator = 0.0
            if metric == "ratio":
                field = "timer_seconds" if name == "timer_exact" else "level"
                expected_name = "expected_timer_seconds" if name == "timer_exact" else "expected_level"
                predicted_name = "predicted_timer_seconds" if name == "timer_exact" else "predicted_level"
                pairs = [(r.get(expected_name), r.get(predicted_name)) for r in session_records]
                pairs = [(left, right) for left, right in pairs if left is not None]
                numerator = sum(left == right for left, right in pairs)
                denominator = len(pairs)
            elif metric == "mean_error":
                expected_name = "expected_hp_ratio" if name == "hp_mae" else "expected_xp_ratio"
                predicted_name = "predicted_hp_ratio" if name == "hp_mae" else "predicted_xp_ratio"
                errors = [
                    1.0 if record.get(predicted_name) is None else abs(float(record[expected_name]) - float(record[predicted_name]))
                    for record in session_records if record.get(expected_name) is not None
                ]
                numerator, denominator = sum(errors), len(errors)
            elif metric == "item_ratio":
                pairs = [record for record in session_records if record.get("expected_items") is not None]
                for record in pairs:
                    expected, predicted = record["expected_items"], record.get("predicted_items") or ()
                    denominator += max(len(expected), len(predicted))
                    numerator += sum(index < len(predicted) and expected[index] == predicted[index] for index in range(len(expected)))
            elif metric == "choice_ratio":
                pairs = [(r.get("expected_choice"), r.get("predicted_choice")) for r in session_records]
                pairs = [(left, right) for left, right in pairs if left is not None]
                numerator = sum("|".join(left) == right if isinstance(left, (list, tuple)) else left == right for left, right in pairs)
                denominator = len(pairs)
            elif metric == "f1":
                value = _macro_f1(session_records)
                numerator, denominator = value * len(session_records), len(session_records)
            elif metric == "roi_center":
                pairs = [(_roi_value(r.get("expected_roi")), _roi_value(r.get("predicted_roi"))) for r in session_records]
                errors = [_roi_center_error(left, right) for left, right in pairs if left is not None and right is not None]
                numerator, denominator = sum(errors), len(errors)
            elif metric == "roi_inside":
                pairs = [(_roi_value(r.get("expected_roi")), _roi_value(r.get("predicted_roi"))) for r in session_records]
                applicable = [(left, right) for left, right in pairs if left is not None]
                numerator = sum(right is not None for _, right in applicable)
                denominator = len(applicable)
            else:
                pairs = [(_roi_value(r.get("expected_roi")), _roi_value(r.get("predicted_roi"))) for r in session_records]
                applicable = [(left, right) for left, right in pairs if left is None]
                numerator = sum(right is not None for _, right in applicable)
                denominator = len(applicable)
            session_values[name][session_id] = (float(numerator), float(denominator))

    bounds: dict[str, tuple[float, float]] = {}
    ci_passed: dict[str, bool] = {}
    for name, values in session_values.items():
        try:
            lower, upper = cluster_ci_bounds(values)
        except ValueError as exc:
            raise FormalGateError(f"cannot compute {name} session-cluster CI: {exc}") from exc
        bounds[name] = (lower, upper)
        gate = gate_by_name[name]
        ci_passed[name] = lower >= gate.threshold if gate.direction == "floor" else upper <= gate.threshold

    custom_pooled: dict[str, tuple[float, float]] = {}
    for name, values in session_values.items():
        numerator = sum(pair[0] for pair in values.values())
        denominator = sum(pair[1] for pair in values.values())
        custom_pooled[name] = (numerator / denominator if denominator else (0.0 if gate_by_name[name].direction == "floor" else 1.0e9), denominator)
    latency = [float(record["latency_ms"]) for record in records if record.get("latency_ms") is not None]
    if not latency:
        latency_p95 = 1.0e9
    else:
        latency_p95 = float(np.percentile(latency, 95))
    replacement = {
        "timer_exact": custom_pooled["timer_exact"][0],
        "level": custom_pooled["level"][0],
        "hp_mae": custom_pooled["hp_mae"][0],
        "xp_mae": custom_pooled["xp_mae"][0],
        "item": custom_pooled["item"][0],
        "choice": custom_pooled["choice"][0],
        "screen_state_f1": _macro_f1(scoring),
        "latency": latency_p95,
        "roi_center_error": custom_pooled["roi_center_error"][0],
        "roi_inside_rate": custom_pooled["roi_inside_rate"][0],
        "roi_false_positive": custom_pooled["roi_false_positive"][0],
    }
    pooled_gates = tuple(
        GateResult(
            gate.name,
            replacement[gate.name],
            gate.threshold,
            gate.direction,
            replacement[gate.name] >= gate.threshold if gate.direction == "floor" else replacement[gate.name] <= gate.threshold,
        )
        for gate in pooled.gates
    )
    pooled_identity = {
        "schema_version": pooled.schema_version,
        "sample_count": pooled.sample_count,
        "config": cfg.to_wire(),
        "gates": [gate.to_wire() for gate in pooled_gates],
    }
    pooled = ValidationReport(
        pooled.schema_version, pooled.sample_count, cfg, pooled_gates, canonical_hash(pooled_identity)
    )
    report = FormalValidationReport(
        pooled_report=pooled,
        session_values=dict(session_values),
        ci_bounds=bounds,
        ci_passed=ci_passed,
        preflight_canonical_hash=preflight.preflight_canonical_hash,
        validation_truth_canonical_hash=validation_truth_hash,
        atlas_content_hash=atlas_content_hash,
        fit_canonical_hash=fit_canonical_hash,
        code_hashes=dict(preflight.code_hashes),
        diagnostics={"layout_offset": {"label_bbox_count": 0, "different_bbox_count": 0, "mean_center_error_px": None}},
        formal_report_canonical_hash="0" * 64,
    )
    object.__setattr__(report, "formal_report_canonical_hash", canonical_hash(_formal_report_hash_payload(report)))
    return report


def _serialize_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """canonical JSON を同一 directory の一時 file 経由で保存する。

    最終 path は書き込み完了後の replace でのみ置き換える。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp")
    try:
        with temp.open("wb") as stream:
            stream.write(canonical_json_bytes(value))
            stream.flush()
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def publish_formal_parser_artifact(
    store: ArtifactStore,
    *,
    package_path: str | Path,
    atlas_manifest_path: str | Path,
    atlas_provenance_path: str | Path,
    preflight: PreflightReport,
    formal_report: FormalValidationReport,
    dataset_descriptor: ArtifactDescriptor,
    logical_prefix: str,
) -> ArtifactDescriptor:
    """formal package、atlas、reports と 04-10 envelope を immutable store へ publish する。

    全 artifact を束ねた descriptor を最後に登録する。
    """
    from survivors.vision.hud_calibration import PackageLoader

    if (
        not isinstance(store, ArtifactStore)
        or not isinstance(preflight, PreflightReport)
        or not isinstance(formal_report, FormalValidationReport)
        or not isinstance(dataset_descriptor, ArtifactDescriptor)
        or not formal_report.all_passed
    ):
        raise FormalGateError("formal artifact publishing requires passing formal gates")
    preflight = PreflightReport.from_wire(preflight.to_wire())
    formal_report = FormalValidationReport.from_wire(formal_report.to_wire())
    dataset_identity = dataset_descriptor.identity_metadata
    if (
        dataset_descriptor.node_kind != "source_descriptor"
        or preflight.dataset_logical_id != dataset_descriptor.logical_id
        or preflight.dataset_identity_hash != dataset_identity.get("dataset_identity_hash")
        or dataset_identity.get("target_profile_hash") != preflight.target_profile_hash
        or dataset_identity.get("game_build_id") != preflight.game_build_id
    ):
        raise FormalPreflightError("dataset descriptor does not match formal preflight")
    atlas = IconMatcher.load_formal(Path(atlas_manifest_path)).manifest
    actual_atlas_hash = hashlib.sha256(
        np.concatenate([entry.feature for entry in atlas.entries]).tobytes()
    ).hexdigest() if atlas.entries else hashlib.sha256(b"").hexdigest()
    if (
        atlas.atlas_content_hash != actual_atlas_hash
        or atlas.atlas_content_hash != formal_report.atlas_content_hash
        or atlas.profile_hash != preflight.target_profile_hash
        or atlas.build_hash != preflight.game_build_id
    ):
        raise FormalAtlasError("published atlas does not match the formal report/preflight")
    try:
        provenance = json.loads(Path(atlas_provenance_path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalAtlasError("invalid atlas provenance JSON") from exc
    expected_provenance_keys = {f"{entry.item_id}@{entry.surface}" for entry in atlas.entries}
    if (
        not any(entry.item_id == "empty_slot" and entry.surface == "inventory" for entry in atlas.entries)
        or
        not isinstance(provenance, dict)
        or set(provenance) != {"schema_version", "entries", "atlas_content_hash"}
        or provenance["schema_version"] != "hud_atlas_provenance.v1"
        or provenance["atlas_content_hash"] != atlas.atlas_content_hash
        or not isinstance(provenance["entries"], dict)
        or set(provenance["entries"]) != expected_provenance_keys
    ):
        raise FormalAtlasError("atlas provenance does not match the formal manifest")
    for item_id, entry in provenance["entries"].items():
        if (
            not isinstance(entry, dict)
            or type(entry.get("crop_count")) is not int
            or entry["crop_count"] < 3
            or not isinstance(entry.get("source_frame_hashes"), list)
            or not entry["source_frame_hashes"]
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in entry["source_frame_hashes"])
        ):
            raise FormalAtlasError(f"invalid provenance for {item_id}")
    package = PackageLoader.load_formal(package_path)
    package_report = package.formal_validation_report
    package_preflight = package.preflight_report
    if (
        package.meta.preflight_canonical_hash != preflight.preflight_canonical_hash
        or package.meta.formal_report_canonical_hash != formal_report.formal_report_canonical_hash
        or package_report.to_wire() != formal_report.to_wire()
        or package_preflight.to_wire() != preflight.to_wire()
        or dict(package_report.code_hashes) != dict(preflight.code_hashes)
        or dict(package_preflight.code_hashes) != dict(preflight.code_hashes)
        or package.meta.atlas_content_hash != formal_report.atlas_content_hash
    ):
        raise FormalPreflightError("package does not match the formal report/preflight")
    package_hash = package.meta.package_hash
    package_bytes = Path(package_path).read_bytes()
    package_sha = _sha256(package_bytes)
    package_id = f"{logical_prefix}/{package_hash[:16]}/parser_package.json"
    envelope_path = Path(package_path).with_name("hud_parser_package.json")
    envelope = {
        "schema_version": "hud_parser_package.v1",
        "development_only": False,
        "formal_eligible": True,
        "parser_artifact_hash": package_sha,
        "files": {"parser_artifact_hash": package_id},
    }
    _serialize_json(envelope_path, envelope)
    base = f"{logical_prefix}/{package_hash[:16]}"
    refs = [
        store.put(logical_id=package_id, source_path=package_path, media_type="application/json"),
        store.put(logical_id=f"{base}/icon_atlas.json", source_path=atlas_manifest_path, media_type="application/json"),
        store.put(logical_id=f"{base}/atlas_provenance.json", source_path=atlas_provenance_path, media_type="application/json"),
        store.put_bytes(logical_id=f"{base}/preflight.json", data=canonical_json_bytes(preflight.to_wire()), media_type="application/json"),
        store.put_bytes(logical_id=f"{base}/formal_report.json", data=canonical_json_bytes(formal_report.to_wire()), media_type="application/json"),
        store.put(logical_id=f"{base}/hud_parser_package.json", source_path=envelope_path, media_type="application/json"),
    ]
    identity = dataset_descriptor.identity_metadata
    descriptor = ArtifactDescriptor(
        logical_id=f"{base}/descriptor",
        node_kind="source_descriptor",
        producer_id="hud-formal-release",
        producer_version="1",
        identity_metadata={
            "manifest_logical_id": refs[-1].logical_id,
            "package_hash": package_hash,
            "fit_canonical_hash": package.meta.fit_canonical_hash,
            "formal_report_canonical_hash": formal_report.formal_report_canonical_hash,
            "preflight_canonical_hash": preflight.preflight_canonical_hash,
            "atlas_content_hash": formal_report.atlas_content_hash,
            "dataset_logical_id": dataset_descriptor.logical_id,
            "dataset_identity_hash": identity["dataset_identity_hash"],
            "code_hashes": dict(preflight.code_hashes),
            "target_profile_hash": preflight.target_profile_hash,
            "game_build_id": preflight.game_build_id,
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


def emit_resolved_layout_yaml(
    fit_result: FitResult,
    formal_report: FormalValidationReport,
    preflight: PreflightReport,
    descriptor: ArtifactDescriptor,
    restore_verdict_descriptor: ArtifactDescriptor,
    out_path: str | Path,
) -> Path:
    """fit 値と release lineage を HUD layout yaml に記録する。

    この出力は正式 dataset 実行時だけの release artifact とする。
    """
    if not formal_report.all_passed:
        raise FormalGateError("cannot emit resolved layout for a failing formal report")
    identity = descriptor.identity_metadata
    if (
        descriptor.node_kind != "source_descriptor"
        or descriptor.parents
        or formal_report.preflight_canonical_hash != preflight.preflight_canonical_hash
        or formal_report.validation_truth_canonical_hash != preflight.validation_truth_canonical_hash
        or formal_report.atlas_content_hash != identity.get("atlas_content_hash")
        or formal_report.formal_report_canonical_hash != identity.get("formal_report_canonical_hash")
        or formal_report.code_hashes != preflight.code_hashes
        or identity.get("preflight_canonical_hash") != preflight.preflight_canonical_hash
        or identity.get("fit_canonical_hash") != fit_result.fit_canonical_hash
        or identity.get("dataset_logical_id") != preflight.dataset_logical_id
        or identity.get("dataset_identity_hash") != preflight.dataset_identity_hash
        or identity.get("target_profile_hash") != preflight.target_profile_hash
        or identity.get("game_build_id") != preflight.game_build_id
        or identity.get("code_hashes") != dict(preflight.code_hashes)
        or not isinstance(identity.get("package_hash"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", identity["package_hash"])
    ):
        raise ValueError("formal package descriptor is incomplete")
    _restore_verdict_matches(descriptor, restore_verdict_descriptor)
    manifest_id = identity.get("manifest_logical_id")
    manifest_refs = [ref for ref in descriptor.files if ref.logical_id == manifest_id]
    package_refs = [ref for ref in descriptor.files if ref.logical_id.endswith("/parser_package.json")]
    if len(manifest_refs) != 1 or len(package_refs) != 1:
        raise ValueError("formal package descriptor references are incomplete")
    if restore_verdict_descriptor.identity_metadata.get("manifest_hash") != manifest_refs[0].sha256:
        raise ValueError("restore verdict does not verify the formal package envelope")
    document = {
        "schema_version": "hud_layout.v1",
        "layout_source": "survivors.vision.roi_layout.py constants (fixed)",
        "roi_anchors": {name: list(roi) for name, roi in fit_result.roi_anchors},
        "thresholds": formal_report.pooled_report.config.to_wire(),
        "source": {
            "dataset_logical_id": preflight.dataset_logical_id,
            "dataset_identity_hash": preflight.dataset_identity_hash,
            "package_logical_id": package_refs[0].logical_id,
            "package_hash": identity["package_hash"],
            "restore_verdict_logical_id": restore_verdict_descriptor.logical_id,
            "preflight_canonical_hash": preflight.preflight_canonical_hash,
            "formal_report_canonical_hash": formal_report.formal_report_canonical_hash,
            "atlas_content_hash": formal_report.atlas_content_hash,
            "code_hashes": dict(preflight.code_hashes),
        },
    }
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(document, sort_keys=True, allow_unicode=True), encoding="utf-8", newline="\n")
    return target

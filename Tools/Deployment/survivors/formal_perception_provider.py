"""verified formal packages から prediction-only Survivors snapshot を作る。

GT annotation を受け取らず、parser・detector・tracker・assembler の出力だけを返す。
"""

from __future__ import annotations

import json
import hashlib
import math
import tempfile
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from survivors.capture.captured_frame import CapturedFrame
from survivors.perception_snapshot import PerceptionSnapshot
from survivors.real_obs_assembler import RealObsAssembler
from survivors.vision.entity_tracker import EntityTracker, TrackedWorldStateV2
from survivors.vision.hud_calibration import PackageLoader
from survivors.vision.hud_parser import HudParser
from survivors.vision.icon_matcher import IconMatcher
from survivors.vision.world_dataset import load_class_map
from survivors.vision.world_detector_package import PackageManifest, load_formal_detector
from survivors.formal_ground_truth import normalize_snapshot_ids

_SHA256 = frozenset("0123456789abcdef")


def _materialize_files(root: Path, files: Mapping[str, bytes]) -> None:
    """verified logical file bytes を root 以下の相対 path に展開する。

    absolute path、parent traversal、重複 path を拒否し package boundary を保つ。
    """
    if not isinstance(files, Mapping) or not files:
        raise ValueError("formal package has no files")
    root.mkdir(parents=True, exist_ok=True)
    targets: set[Path] = set()
    for logical_id, content in files.items():
        if type(logical_id) is not str or not logical_id or not isinstance(content, bytes):
            raise ValueError("formal package file entry is invalid")
        posix, windows = PurePosixPath(logical_id), PureWindowsPath(logical_id)
        if posix.is_absolute() or windows.is_absolute() or windows.drive or ".." in posix.parts or ".." in windows.parts:
            raise ValueError("formal package logical id is not a safe relative path")
        target = root.joinpath(*posix.parts)
        resolved = target.resolve()
        if not resolved.is_relative_to(root.resolve()) or resolved in targets:
            raise ValueError("formal package logical ids collide or escape their root")
        targets.add(resolved)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _json_files(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """root 内の JSON file を parse し、壊れた package JSON を拒否する。

    package file を見落として development fallback へ進まないよう全候補を検査する。
    """
    parsed = []
    for path in sorted(root.rglob("*.json")):
        try:
            wire = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid formal package JSON: {path.name}") from exc
        if not isinstance(wire, dict):
            raise ValueError(f"formal package JSON must be an object: {path.name}")
        parsed.append((path, wire))
    return parsed


def _find_package_files(root: Path) -> tuple[Path, Path, Path]:
    """parser package、atlas、detector manifest の path を特定する。

    formal package 固有 schema が見つからない場合に曖昧な JSON を選ばない。
    """
    files = _json_files(root)
    parser_candidates = [
        path for path, wire in files
        if path.name in {"parser_package.json", "parser.json"}
        or {"metadata", "fit_result", "validation_report"} <= set(wire)
    ]
    atlas_candidates = [
        path for path, wire in files if wire.get("schema_version") == "icon_atlas.v1"
    ]
    detector_candidates = [
        path for path, wire in files
        if path.name == "manifest.json" and wire.get("schema_version") == "world_detector_package.v1"
    ]
    if any(len(values) != 1 for values in (parser_candidates, atlas_candidates, detector_candidates)):
        raise ValueError("formal package files are missing or ambiguous")
    return parser_candidates[0], atlas_candidates[0], detector_candidates[0]


class _FormalPredictor:
    """formal inference components と一 session 分の temporal state を保持する。

    reset は parser、tracker、assembler の履歴を消し、verified model weight は再利用する。
    """

    def __init__(
        self,
        *,
        temp_directory: tempfile.TemporaryDirectory[str],
        detector_dir: Path,
        parser: HudParser,
        detector: Any,
        tracker: EntityTracker,
        detector_manifest: PackageManifest,
        parser_artifact_hash: str,
        detector_artifact_hash: str,
    ) -> None:
        """verified package components と envelope hashes を保持する。

        temporary directory は class lifetime に束縛し、class map が replay 中に消えない。
        """
        self._temp_directory = temp_directory
        self._detector_dir = detector_dir
        self._parser = parser
        self._detector = detector
        self._tracker = tracker
        self._detector_manifest = detector_manifest
        self._parser_artifact_hash = parser_artifact_hash
        self._detector_artifact_hash = detector_artifact_hash
        self._class_map = load_class_map(detector_dir / "world_class_map_v2.yaml")
        self._schema = DeployObsSchema.default_v2()
        self._assembler = RealObsAssembler()
        self._session_id: str | None = None

    def reset(self) -> None:
        """parser・tracker・assembler の temporal history を破棄する。

        detector model と package file は評価窓ごとに再ロードせず保持する。
        """
        self._parser.reset_temporal_state()
        self._tracker._tracks.clear()
        self._assembler = RealObsAssembler()
        self._session_id = None

    def predict_frame(
        self,
        frame: CapturedFrame,
        session_id: str,
        frame_index: int,
    ) -> tuple[PerceptionSnapshot | None, float]:
        """frame pixels だけから prediction snapshot と処理時間を返す。

        formal package threshold を使い、GT・annotation bytes を読む経路は持たない。
        """
        started = time.perf_counter()
        if not isinstance(frame, CapturedFrame) or frame.session_frame_index != frame_index:
            raise ValueError("captured frame index does not match prediction request")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be non-empty")
        if self._session_id is not None and self._session_id != session_id:
            self.reset()
        self._session_id = session_id
        hud = self._parser.parse(
            frame.frame_bgra, session_id=session_id,
            frame_index=frame_index,
            captured_monotonic_ns=frame.captured_monotonic_ns,
        )
        detections = self._detector.infer(
            frame.frame_bgra[:, :, :3],
            score_threshold=self._detector_manifest.score_threshold,
        )
        tracked = self._tracker.update(
            detections, frame_index=frame_index,
            timestamp_ns=frame.captured_monotonic_ns,
        )
        world = TrackedWorldStateV2.from_state(
            tracked, frame_index=frame_index,
            timestamp_ns=frame.captured_monotonic_ns,
            class_map_path=self._detector_dir / "world_class_map_v2.yaml",
        )
        snapshot = self._assembler.assemble(hud, world, self._schema, (1920, 1080))
        latency_ms = (time.perf_counter() - started) * 1000.0
        if snapshot is None:
            return None, latency_ms
        filtered_hud = self._assembler.temporal.last_hud
        if filtered_hud is None:
            raise RuntimeError("temporal assembler did not retain prediction HUD")
        snapshot = normalize_snapshot_ids(
            snapshot, session_id=session_id, frame_index=frame_index,
            viewport=(1920, 1080), hud=filtered_hud,
        )
        predicted_boxes = []
        for box, score, class_id in zip(
            detections.boxes_xyxy, detections.scores, detections.class_ids, strict=True,
        ):
            coords = [float(value) for value in box]
            confidence = float(score)
            if any(not math.isfinite(value) for value in (*coords, confidence)):
                raise ValueError("detector returned non-finite prediction values")
            predicted_boxes.append({
                "class": self._class_map.id_to_name(int(class_id)),
                "xyxy": coords,
                "score": confidence,
            })
        diagnostics = dict(snapshot.diagnostics)
        diagnostics["predicted_boxes"] = predicted_boxes
        return replace_snapshot_hashes(
            snapshot, diagnostics, self._detector_artifact_hash,
        ), latency_ms


def replace_snapshot_hashes(
    snapshot: PerceptionSnapshot,
    diagnostics: Mapping[str, Any],
    detector_artifact_hash: str,
) -> PerceptionSnapshot:
    """snapshot の frozen diagnostics と detector envelope hash を再束縛する。

    constructor を通して MappingProxy を作り直し、prediction boxes を ROI-free のまま格納する。
    """
    from dataclasses import replace

    return replace(
        snapshot,
        diagnostics=diagnostics,
        detector_artifact_hash=detector_artifact_hash,
    )


def build_formal_predictor(
    restored: Mapping[str, Any],
) -> _FormalPredictor:
    """verified parser/detector artifacts から prediction-only predictor を構築する。

    loader 名により formal-only restore を選び、development package へ fallback しない。
    """
    parser_artifact = restored["parser_package"]
    detector_artifact = restored["detector_package"]
    parser_hash = parser_artifact.manifest.get("parser_artifact_hash")
    detector_hash = detector_artifact.manifest.get("detector_artifact_hash")
    model_hash = detector_artifact.manifest.get("model_hash")
    for name, digest in (
        ("parser_artifact_hash", parser_hash),
        ("detector_artifact_hash", detector_hash),
        ("model_hash", model_hash),
    ):
        if type(digest) is not str or len(digest) != 64 or not set(digest) <= _SHA256:
            raise ValueError(f"{name} envelope is not a SHA-256")
    temp_directory = tempfile.TemporaryDirectory(prefix="survivors-formal-")
    root = Path(temp_directory.name)
    parser_root, detector_root = root / "parser_package", root / "detector_package"
    try:
        _materialize_files(parser_root, parser_artifact.files)
        _materialize_files(detector_root, detector_artifact.files)
        parser_path, atlas_path, detector_manifest_path = _find_package_files(root)
        if hashlib.sha256(parser_path.read_bytes()).hexdigest() != parser_hash:
            raise ValueError("parser package file hash does not match artifact envelope")
        PackageLoader.load_formal(parser_path)
        icon_matcher = IconMatcher.load_formal(atlas_path)
        detector_manifest_bytes = detector_manifest_path.read_bytes()
        if hashlib.sha256(detector_manifest_bytes).hexdigest() != detector_hash:
            raise ValueError("detector manifest file hash does not match artifact envelope")
        detector_manifest_wire = json.loads(detector_manifest_bytes)
        if detector_manifest_wire.get("model_hash") != model_hash:
            raise ValueError("detector manifest model_hash does not match artifact envelope")
        detector, tracker, detector_manifest = load_formal_detector(detector_manifest_path.parent)
        if detector_manifest.model_hash != model_hash:
            raise ValueError("loaded detector model_hash does not match artifact envelope")
        parser = HudParser(parser_artifact_hash=parser_hash, icon_matcher=icon_matcher)
        return _FormalPredictor(
            temp_directory=temp_directory,
            detector_dir=detector_manifest_path.parent,
            parser=parser,
            detector=detector,
            tracker=tracker,
            detector_manifest=detector_manifest,
            parser_artifact_hash=parser_hash,
            detector_artifact_hash=detector_hash,
        )
    except BaseException:
        temp_directory.cleanup()
        raise

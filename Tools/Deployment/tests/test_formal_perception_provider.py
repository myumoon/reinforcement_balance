"""formal predictor が verified package から prediction-only snapshot を作る。

GT・annotation を入力せず、parser と detector の envelope identity を保持する。
"""

from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes
from survivors.capture.captured_frame import CapturedFrame
from survivors.formal_perception_provider import build_formal_predictor
from survivors.vision.entity_tracker import DEFAULT_CLASS_MAP_PATH, EntityTracker, default_class_map
from survivors.vision.hud_calibration import (
    FormalPackageMixingError,
    PackageLoader,
    PackageWriter,
    fit,
)
from survivors.vision.hud_formal_release import PreflightReport, validate_formal
from survivors.vision.hud_parser import HUD_STATE_SCHEMA_VERSION, INV_SLOT_COUNT, HudStateV1, _compute_candidate_set_hash
from survivors.vision.icon_matcher import AtlasManifest, TemplateEntry, serialize_manifest
from survivors.vision.world_detector import DetectionResult
from survivors.vision.world_detector_package import (
    PACKAGE_SCHEMA_VERSION,
    FormalPackageRejectedError,
    PackageManifest,
    load_formal_detector,
)

PARSER_BYTES = b"{}"
PARSER_HASH = hashlib.sha256(PARSER_BYTES).hexdigest()
MODEL_BYTES = b"fixture model"
MODEL_HASH = hashlib.sha256(MODEL_BYTES).hexdigest()
DETECTOR_MANIFEST_BYTES = canonical_json_bytes({
    "schema_version": PACKAGE_SCHEMA_VERSION, "model_hash": MODEL_HASH,
})
DETECTOR_HASH = hashlib.sha256(DETECTOR_MANIFEST_BYTES).hexdigest()


class _FixtureHudParser:
    """formal HudParser API を使う固定 gameplay observation。

    reset と artifact hash は実 provider の frame 境界を通して確認する。
    """

    def __init__(self, *, parser_artifact_hash: str, icon_matcher) -> None:
        """parser artifact identity と reset count を初期化する。

        parse 結果へ provider の file hash を伝え、reset の回数を記録する。
        """
        self.parser_artifact_hash = parser_artifact_hash
        self.resets = 0

    def reset_temporal_state(self) -> None:
        """session 境界で parser state reset を記録する。

        テスト用 parser は state を保持しないが、provider の reset 呼び出しを観測する。
        """
        self.resets += 1

    def parse(self, frame_bgra, *, session_id: str, frame_index: int, captured_monotonic_ns: int):
        """最小 gameplay HUD を指定 frame identity で返す。

        認識器の値に依存せず、provider の結合・identity 正規化だけを試す。
        """
        inventory = (None,) * INV_SLOT_COUNT
        return HudStateV1(
            HUD_STATE_SCHEMA_VERSION, session_id, frame_index, captured_monotonic_ns,
            self.parser_artifact_hash, "gameplay", 1.0, "fixture",
            30.0, 1.0, "fixture", False,
            0.75, 1.0, "fixture", 0.5, 1.0, "fixture", 2, 1.0, "fixture",
            inventory, 0.0, canonical_hash({"slots": list(inventory)}), (),
            _compute_candidate_set_hash("gameplay", ()), (),
            False, False, False, 0.0, "fixture",
        )


class _FixtureDetector:
    """固定検出を返す detector double。

    detector output は正式 class map の ID を使い、tracker は実装を使う。
    """

    def __init__(self) -> None:
        """formal class map に対応する固定 detection を構築する。

        player anchor と boss を含め、world snapshot の正規化を試す。
        """
        class_map = default_class_map()
        self.result = DetectionResult(
            np.asarray([[900, 500, 1020, 620], [1100, 500, 1200, 600]], dtype=np.float32),
            np.asarray([0.99, 0.9], dtype=np.float32),
            np.asarray([
                class_map.name_to_id("player_anchor"),
                class_map.name_to_id("enemy_boss"),
            ], dtype=np.int32),
            1920, 1080,
        )

    def infer(self, frame_bgr, *, score_threshold: float):
        """固定 detection result を score threshold にかかわらず返す。

        threshold の受け渡しは provider の API 契約として実引数に記録する。
        """
        self.threshold = score_threshold
        return self.result


def _restored_packages(
    class_map_bytes: bytes,
    *,
    parser_bytes: bytes = PARSER_BYTES,
    parser_artifact_hash: str | None = None,
    detector_manifest_bytes: bytes = DETECTOR_MANIFEST_BYTES,
    detector_artifact_hash: str | None = None,
    model_hash: str = MODEL_HASH,
):
    """provider が展開する相対 file path と envelope identity を作る。

    parser package、formal atlas、detector package を別 artifact に分ける。
    """
    return {
        "parser_package": SimpleNamespace(
            files={
                "parser.json": parser_bytes,
                "atlas.json": b'{"schema_version":"icon_atlas.v1"}',
            },
            manifest={
                "parser_artifact_hash": parser_artifact_hash
                or hashlib.sha256(parser_bytes).hexdigest(),
            },
        ),
        "detector_package": SimpleNamespace(
            files={
                "manifest.json": detector_manifest_bytes,
                "model.pt": MODEL_BYTES,
                "world_detector_v2.yaml": b"fixture: true\n",
                "world_class_map_v2.yaml": class_map_bytes,
            },
            manifest={
                "detector_artifact_hash": detector_artifact_hash
                or hashlib.sha256(detector_manifest_bytes).hexdigest(),
                "model_hash": model_hash,
            },
        ),
    }


def _formal_parser_package_bytes(tmp_path: Path) -> bytes:
    """valid formal HUD reports から PackageWriter 形式の package を作る。

    public PackageLoader が内部 hash chain を検証できる最小成果物を返す。
    """
    truth_wires = []
    records = []
    for session_index in range(3):
        session_id = f"validation-{session_index}"
        for frame_id, state in ((0, "gameplay"), (1, "unknown")):
            positive = state == "gameplay"
            truth = {
                "schema_version": "hud_truth.v1",
                "session_id": session_id,
                "frame_id": frame_id,
                "confirmed": True,
                "expected_screen_state": state,
                "expected_timer_seconds": 10.0 if positive else None,
                "expected_level": 1 if positive else None,
                "expected_hp_ratio": 0.5 if positive else None,
                "expected_xp_ratio": 0.25 if positive else None,
                "expected_items": ["whip"] * 12 if positive else None,
                "expected_choice": ["whip"] if positive else None,
                "roi_name": "hud",
                "expected_roi": [0, 32, 1920, 75] if positive else None,
                "confirmed_at": "2026-10-10T00:00:00Z",
                "draft_source": {"parser_artifact_hash": "p", "atlas_content_hash": "a"},
            }
            truth_wires.append(truth)
            records.append({
                **truth,
                "sample_id": f"{session_id}:{frame_id}",
                "split": "model_validation",
                "predicted_timer_seconds": 10.0 if positive else None,
                "predicted_level": 1 if positive else None,
                "predicted_hp_ratio": 0.5 if positive else None,
                "predicted_xp_ratio": 0.25 if positive else None,
                "predicted_items": ["whip"] * 12 if positive else None,
                "predicted_choice": "whip" if positive else None,
                "predicted_screen_state": state,
                "predicted_roi": [0, 32, 1920, 75] if positive else None,
                "latency_ms": 1.0,
                "_hud_truth_wire": truth,
            })
    truth_hash = canonical_hash(truth_wires)
    payload = {
        "schema_version": "hud_formal_preflight.v1",
        "dataset_logical_id": "dataset/source",
        "dataset_identity_hash": "d" * 64,
        "restore_verdict_logical_id": "dataset/restore",
        "restore_verdict_identity_hash": "e" * 64,
        "target_profile_hash": "a" * 64,
        "game_build_id": "b" * 64,
        "train_sessions": ["train-a", "train-b", "train-c"],
        "validation_sessions": [f"validation-{index}" for index in range(3)],
        "train_record_count": 3,
        "validation_record_count": 6,
        "train_truth_canonical_hash": "f" * 64,
        "validation_truth_canonical_hash": truth_hash,
        "slice_counts": {},
        "code_hashes": {},
    }
    preflight = PreflightReport(
        payload["dataset_logical_id"], payload["dataset_identity_hash"],
        payload["restore_verdict_logical_id"], payload["restore_verdict_identity_hash"],
        payload["target_profile_hash"], payload["game_build_id"],
        tuple(payload["train_sessions"]), tuple(payload["validation_sessions"]),
        payload["train_record_count"], payload["validation_record_count"],
        payload["train_truth_canonical_hash"], truth_hash, {}, {}, canonical_hash(payload),
    )
    fit_result = fit([{
        "sample_id": "train:0", "split": "model_train", "expected_roi": [0, 32, 1920, 75],
    }])
    feature = np.zeros(88, dtype=np.float32)
    empty_feature = np.ones(88, dtype=np.float32)
    atlas_hash = hashlib.sha256(np.concatenate([empty_feature, feature]).tobytes()).hexdigest()
    atlas = AtlasManifest(
        "icon_atlas.v1", "a" * 64, "b" * 64, False, True, atlas_hash,
        (
            TemplateEntry("empty_slot", "unknown", 1, 1, empty_feature),
            TemplateEntry("whip", "weapon", 1, 1, feature),
        ),
    )
    report = validate_formal(
        records, preflight=preflight, atlas_content_hash=atlas_hash,
        fit_canonical_hash=fit_result.fit_canonical_hash,
    )
    if not report.all_passed:
        raise AssertionError("formal package fixture did not pass its release gates")
    package_path = tmp_path / "formal-parser-package.json"
    PackageWriter.publish_formal(
        package_path, fit_result, report, preflight=preflight, atlas_manifest=atlas,
    )
    return package_path.read_bytes()


def _frame(session_id_index: int = 0) -> CapturedFrame:
    """formal 1920x1080 BGRA frame を作る。

    timestamp を十分に進め、実 temporal assembler の cadence を通す。
    """
    return CapturedFrame(
        np.zeros((1080, 1920, 4), dtype=np.uint8), 1_000_000_000,
        session_id_index, (0, 0, 1920, 1080), True, "c" * 64, "build",
    )


def test_formal_predictor_returns_normalized_prediction_and_package_hashes(monkeypatch) -> None:
    """formal predictor が parser/detector の hash と predicted box を返す。

    provider は GT bytes を受け取らず、実 tracker と assembler の出力を正規化する。
    """
    import survivors.formal_perception_provider as provider

    package = _restored_packages(DEFAULT_CLASS_MAP_PATH.read_bytes())
    parser_package = SimpleNamespace(
        package_hash=PARSER_HASH, meta=SimpleNamespace(atlas_content_hash="a" * 64),
    )
    icon_matcher = SimpleNamespace(manifest=parser_package.meta)
    monkeypatch.setattr(provider.PackageLoader, "load_formal", lambda _path: parser_package)
    monkeypatch.setattr(provider.IconMatcher, "load_formal", lambda _path: icon_matcher)
    detector = _FixtureDetector()
    class_map = default_class_map()
    tracker = EntityTracker(
        {}, 0.7, 0.6, 0.9,
        coarse_by_class_id=class_map.coarse_by_class_id(),
    )
    monkeypatch.setattr(
        provider, "load_formal_detector",
        lambda _path: (
            detector, tracker,
            SimpleNamespace(score_threshold=0.2, model_hash=MODEL_HASH),
        ),
    )
    monkeypatch.setattr(provider, "HudParser", _FixtureHudParser)
    predictor = build_formal_predictor(package)
    snapshot, latency = predictor.predict_frame(_frame(), "session-1", 0)
    assert snapshot is not None and latency >= 0
    assert snapshot.frame_id == "session-1:0"
    assert snapshot.parser_artifact_hash == PARSER_HASH
    assert snapshot.ui_presentation.parser_artifact_hash == PARSER_HASH
    assert snapshot.detector_artifact_hash == DETECTOR_HASH
    assert snapshot.diagnostics["predicted_boxes"][1]["class"] == "enemy_boss"
    assert detector.threshold == 0.2
    predictor.reset()


def test_formal_predictor_loads_actual_formal_parser_package(tmp_path: Path, monkeypatch) -> None:
    """PackageWriter の formal package を public loader で復元する。

    envelope は package 内部 hash と異なる parser file の SHA-256 に束縛する。
    """
    import survivors.formal_perception_provider as provider

    parser_bytes = _formal_parser_package_bytes(tmp_path)
    parser_path = tmp_path / "published-formal-parser.json"
    parser_path.write_bytes(parser_bytes)
    parser_hash = hashlib.sha256(parser_bytes).hexdigest()
    parser_package = PackageLoader.load_formal(parser_path)
    assert parser_package.package_hash != parser_hash
    package = _restored_packages(
        DEFAULT_CLASS_MAP_PATH.read_bytes(), parser_bytes=parser_bytes,
    )
    monkeypatch.setattr(
        provider.IconMatcher, "load_formal",
        lambda _path: SimpleNamespace(
            manifest=SimpleNamespace(atlas_content_hash=parser_package.meta.atlas_content_hash),
        ),
    )
    detector = _FixtureDetector()
    class_map = default_class_map()
    tracker = EntityTracker(
        {}, 0.7, 0.6, 0.9,
        coarse_by_class_id=class_map.coarse_by_class_id(),
    )
    monkeypatch.setattr(
        provider, "load_formal_detector",
        lambda _path: (
            detector, tracker,
            SimpleNamespace(score_threshold=0.2, model_hash=MODEL_HASH),
        ),
    )
    monkeypatch.setattr(provider, "HudParser", _FixtureHudParser)
    predictor = build_formal_predictor(package)
    snapshot, _latency = predictor.predict_frame(_frame(), "session-1", 0)
    assert snapshot is not None and snapshot.parser_artifact_hash == parser_hash
    predictor.reset()


def test_formal_predictor_rejects_parser_file_hash_mismatch(monkeypatch) -> None:
    """package JSON の file bytes が envelope hash と違えば拒否する。

    JSON として有効な trailing byte でも parser package をすり替えられない。
    """
    package = _restored_packages(
        DEFAULT_CLASS_MAP_PATH.read_bytes(),
        parser_bytes=PARSER_BYTES + b" ",
        parser_artifact_hash=PARSER_HASH,
    )
    with pytest.raises(ValueError, match="parser package file hash"):
        build_formal_predictor(package)


def test_formal_predictor_rejects_detector_manifest_file_hash_mismatch(monkeypatch) -> None:
    """loaded detector manifest が envelope hash に束縛されない場合を拒否する。

    package 内の別 manifest を読み替える経路を封じる。
    """
    package = _restored_packages(
        DEFAULT_CLASS_MAP_PATH.read_bytes(), detector_artifact_hash="0" * 64,
    )
    import survivors.formal_perception_provider as provider
    monkeypatch.setattr(
        provider.PackageLoader, "load_formal",
        lambda _path: SimpleNamespace(
            package_hash=PARSER_HASH, meta=SimpleNamespace(atlas_content_hash="a" * 64),
        ),
    )
    monkeypatch.setattr(
        provider.IconMatcher, "load_formal",
        lambda _path: SimpleNamespace(manifest=SimpleNamespace(atlas_content_hash="a" * 64)),
    )
    with pytest.raises(ValueError, match="detector manifest file hash"):
        build_formal_predictor(package)


def test_formal_predictor_rejects_detector_model_hash_mismatch(monkeypatch) -> None:
    """detector manifest の model hash が envelope と違えば拒否する。

    package loader が検証する model.pt を別の model identity に差し替えさせない。
    """
    manifest = canonical_json_bytes({
        "schema_version": PACKAGE_SCHEMA_VERSION, "model_hash": "c" * 64,
    })
    package = _restored_packages(
        DEFAULT_CLASS_MAP_PATH.read_bytes(), detector_manifest_bytes=manifest,
    )
    import survivors.formal_perception_provider as provider
    monkeypatch.setattr(
        provider.PackageLoader, "load_formal",
        lambda _path: SimpleNamespace(
            package_hash=PARSER_HASH, meta=SimpleNamespace(atlas_content_hash="a" * 64),
        ),
    )
    monkeypatch.setattr(
        provider.IconMatcher, "load_formal",
        lambda _path: SimpleNamespace(manifest=SimpleNamespace(atlas_content_hash="a" * 64)),
    )
    with pytest.raises(ValueError, match="model_hash"):
        build_formal_predictor(package)


def test_formal_predictor_uses_formal_parser_loader(monkeypatch) -> None:
    """開発用 parser package は strict formal loader で拒否される。

    formal 経路が開発用 parser や atlas へ黙って fallback しないことを確認する。
    """
    import survivors.formal_perception_provider as provider

    package = _restored_packages(b"fixture class map")
    monkeypatch.setattr(
        provider.PackageLoader, "load_formal",
        lambda _path: (_ for _ in ()).throw(FormalPackageMixingError("development package")),
    )
    with pytest.raises(FormalPackageMixingError):
        build_formal_predictor(package)


def test_formal_predictor_rejects_atlas_from_another_formal_package(
    tmp_path: Path, monkeypatch,
) -> None:
    """別の正式 atlas が parser package の lineage と異なれば拒否する。

    有効な parser package に異なる atlas hash を持つ正式 manifest を差し込み、provider 境界を試す。
    """
    import survivors.formal_perception_provider as provider

    parser_bytes = _formal_parser_package_bytes(tmp_path)
    parser_path = tmp_path / "parser.json"
    parser_path.write_bytes(parser_bytes)
    parser_package = PackageLoader.load_formal(parser_path)
    expected_atlas_hash = parser_package.meta.atlas_content_hash
    other_atlas_hash = "0" * 64 if expected_atlas_hash != "0" * 64 else "1" * 64
    package = _restored_packages(
        DEFAULT_CLASS_MAP_PATH.read_bytes(), parser_bytes=parser_bytes,
    )
    package["parser_package"].files["atlas.json"] = serialize_manifest(
        AtlasManifest("icon_atlas.v1", "a" * 64, "b" * 64, False, True, other_atlas_hash, ()),
    )
    class_map = default_class_map()
    tracker = EntityTracker(
        {}, 0.7, 0.6, 0.9,
        coarse_by_class_id=class_map.coarse_by_class_id(),
    )
    monkeypatch.setattr(
        provider, "load_formal_detector",
        lambda _path: (
            _FixtureDetector(), tracker,
            SimpleNamespace(score_threshold=0.2, model_hash=MODEL_HASH),
        ),
    )
    monkeypatch.setattr(provider, "HudParser", _FixtureHudParser)
    with pytest.raises(ValueError, match="atlas_content_hash"):
        build_formal_predictor(package)


def test_formal_detector_loader_rejects_development_manifest() -> None:
    """development eligibility flags は formal detector loader で拒否される。

    model や class map を読む前に package 用途境界を確認する。
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        package_dir = Path(temp_dir) / "development"
        package_dir.mkdir()
        manifest = PackageManifest(
            PACKAGE_SCHEMA_VERSION, False, "c" * 64, "d" * 64,
            "e" * 64, "f" * 64, "1" * 64, "2" * 64,
            {}, {}, False, 0.5,
        ).to_dict()
        (package_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(FormalPackageRejectedError):
            load_formal_detector(package_dir)

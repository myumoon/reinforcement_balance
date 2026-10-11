"""formal predictor が verified package から prediction-only snapshot を作る。

GT・annotation を入力せず、parser と detector の envelope identity を保持する。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reinbalance_survivors_contracts.canonical_json import canonical_hash
from survivors.capture.captured_frame import CapturedFrame
from survivors.formal_perception_provider import build_formal_predictor
from survivors.vision.entity_tracker import DEFAULT_CLASS_MAP_PATH, EntityTracker, default_class_map
from survivors.vision.hud_calibration import FormalPackageMixingError
from survivors.vision.hud_parser import HUD_STATE_SCHEMA_VERSION, INV_SLOT_COUNT, HudStateV1, _compute_candidate_set_hash
from survivors.vision.world_detector import DetectionResult
from survivors.vision.world_detector_package import (
    PACKAGE_SCHEMA_VERSION,
    FormalPackageRejectedError,
    PackageManifest,
    load_formal_detector,
)

PARSER_HASH = "a" * 64
DETECTOR_HASH = "b" * 64


class _FixtureHudParser:
    """formal HudParser API を使う固定 gameplay observation。

    reset と artifact hash は実 provider の frame 境界を通して確認する。
    """

    def __init__(self, *, parser_artifact_hash: str, icon_matcher) -> None:
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


def _restored_packages(class_map_bytes: bytes):
    """provider が展開する相対 file path と envelope identity を作る。

    parser package、formal atlas、detector package を別 artifact に分ける。
    """
    return {
        "parser_package": SimpleNamespace(
            files={
                "parser.json": b"{}",
                "atlas.json": b'{"schema_version":"icon_atlas.v1"}',
            },
            manifest={"parser_artifact_hash": PARSER_HASH},
        ),
        "detector_package": SimpleNamespace(
            files={
                "manifest.json": b'{"schema_version":"world_detector_package.v1"}',
                "world_detector_v2.yaml": b"fixture: true\n",
                "world_class_map_v2.yaml": class_map_bytes,
            },
            manifest={"detector_artifact_hash": DETECTOR_HASH},
        ),
    }


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
    monkeypatch.setattr(provider.PackageLoader, "load_formal", lambda _path: SimpleNamespace(package_hash=PARSER_HASH))
    monkeypatch.setattr(provider.IconMatcher, "load_formal", lambda _path: object())
    detector = _FixtureDetector()
    class_map = default_class_map()
    tracker = EntityTracker(
        {}, 0.7, 0.6, 0.9,
        coarse_by_class_id=class_map.coarse_by_class_id(),
    )
    monkeypatch.setattr(
        provider, "load_formal_detector",
        lambda _path: (detector, tracker, SimpleNamespace(score_threshold=0.2)),
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


def test_formal_predictor_uses_formal_parser_loader(monkeypatch) -> None:
    """development parser package rejection propagates from strict formal loader.

    formal boundary must not silently fall back to a development parser or atlas.
    """
    import survivors.formal_perception_provider as provider

    package = _restored_packages(b"fixture class map")
    monkeypatch.setattr(
        provider.PackageLoader, "load_formal",
        lambda _path: (_ for _ in ()).throw(FormalPackageMixingError("development package")),
    )
    with pytest.raises(FormalPackageMixingError):
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

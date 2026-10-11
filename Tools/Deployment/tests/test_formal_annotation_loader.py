"""formal annotation writer と loader の byte-to-snapshot 一致を検証する。

HUD truth、frames、COCO を同じ verified bytes から復元する。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from build_survivors_formal_annotations import main as build_annotations
from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes
from survivors.capture.captured_frame import CapturedFrame
from survivors.formal_annotation_loader import build_ground_truth_loader
from survivors.formal_ground_truth import CaptureGroundTruthFiles, semantic_hash
from survivors.vision.entity_tracker import DEFAULT_CLASS_MAP_PATH
from survivors.vision.hud_truth import HUD_TRUTH_SCHEMA_VERSION, HudTruthRecord


def _fixture(tmp_path: Path, *, coco_hash: str | None = None):
    """session の三つの input file を最小 COCO export として書き出す。

    object SHA、annotation 順、frame index を揃え、hash mismatch のみを変えられる。
    """
    session_id = "formal-session"
    object_hash = hashlib.sha256(b"png bytes").hexdigest()
    truth = HudTruthRecord(
        HUD_TRUTH_SCHEMA_VERSION, session_id, 0, "annotator", True,
        "level_up_items", 20.0, 3, 0.8, 0.4,
        ("whip",) + ("empty_slot",) * 11, None,
        ("whip", "garlic", "knife"), "hud", (0, 32, 1920, 75),
        "2026-10-11T00:00:00Z",
        {"parser_artifact_hash": "c" * 64, "atlas_content_hash": "d" * 64},
    )
    session_root = tmp_path / "store" / "capture_sessions" / session_id
    session_root.mkdir(parents=True)
    truth_bytes = (json.dumps(truth.to_wire(), sort_keys=True, separators=(",", ":")) + "\n").encode()
    frame_wire = {
        "frame_id": 0, "captured_monotonic_ns": 2_000_000_000,
        "object_path": "objects/frame.png", "object_sha256": object_hash,
        "client_rect_screen_px": [0, 0, 1920, 1080], "foreground": True,
        "target_profile_hash": "a" * 64, "game_build_id": "build",
    }
    frames_bytes = (json.dumps(frame_wire, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (session_root / "hud_truth.jsonl").write_bytes(truth_bytes)
    (session_root / "frames.jsonl").write_bytes(frames_bytes)
    coco_wire = {
        "images": [{
            "id": 1, "frame_id": 0, "session_id": session_id,
            "object_sha256": coco_hash or object_hash,
        }],
        "categories": [
            {"id": 1, "name": "player_anchor"},
            {"id": 2, "name": "enemy_boss"},
            {"id": 3, "name": "hazard_area"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [900, 500, 100, 100]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [1100, 500, 100, 100]},
            {"id": 3, "image_id": 1, "category_id": 3, "bbox": [200, 200, 40, 40]},
        ],
    }
    coco_bytes = canonical_json_bytes(coco_wire)
    coco_root = tmp_path / "coco"
    (coco_root / session_id).mkdir(parents=True)
    (coco_root / session_id / "world_coco.json").write_bytes(coco_bytes)
    frame = CapturedFrame(
        np.zeros((1080, 1920, 4), dtype=np.uint8), 2_000_000_000, 0,
        (0, 0, 1920, 1080), True, "a" * 64, "build",
    )
    files = CaptureGroundTruthFiles(truth_bytes, coco_bytes, frames_bytes)
    return session_id, tmp_path / "store", coco_root, truth, frame, files


def test_annotation_writer_and_loader_use_the_same_snapshot(tmp_path: Path) -> None:
    """CLI annotation hash と loader snapshot hash が一致する。

    snapshot diagnostics も同一 COCO labels から同じ foreground/slice values を作る。
    """
    session_id, store_root, coco_root, truth, frame, files = _fixture(tmp_path)
    assert build_annotations([
        "--store-root", str(store_root), "--session-id", session_id,
        "--coco-dir", str(coco_root), "--parser-artifact-hash", "b" * 64,
        "--class-map", str(DEFAULT_CLASS_MAP_PATH),
    ]) == 0
    row = json.loads((store_root / "capture_sessions" / session_id / "annotations.jsonl").read_text("utf-8"))
    loader = build_ground_truth_loader(
        {session_id: (canonical_json_bytes(row) + b"\n")},
        {"parser_package": {"parser_artifact_hash": "b" * 64}},
        {session_id: files},
    )
    snapshot = loader.load_frame(frame, session_id, 0)
    assert snapshot is not None
    assert row["ground_truth_semantic_hash"] == semantic_hash(snapshot)
    assert snapshot.diagnostics["foreground_entity_classes"] == {
        "formal-session:0:0": "player_anchor",
        "formal-session:0:1": "enemy_boss",
        "formal-session:0:2": "hazard_area",
    }
    assert snapshot.diagnostics["slice_boss_present"] is True
    assert snapshot.diagnostics["slice_hazard_present"] is True


def test_loader_rejects_coco_object_hash_mismatch(tmp_path: Path) -> None:
    """COCO の object SHA が captured frame と違えば loader が拒否する。

    同一 image id を再利用した別 PNG の box を ground truth にしない。
    """
    session_id, _store_root, _coco_root, _truth, frame, files = _fixture(
        tmp_path, coco_hash="0" * 64,
    )
    loader = build_ground_truth_loader(
        {}, {"parser_package": {"parser_artifact_hash": "b" * 64}},
        {session_id: files},
    )
    with pytest.raises(ValueError, match="object_sha256"):
        loader.load_frame(frame, session_id, 0)


def test_loader_rejects_duplicate_coco_frame_ids(tmp_path: Path) -> None:
    """同じ frame_id を指す COCO image を拒否する。

    連番 image id から frame への対応が曖昧にならないよう index 時に検査する。
    """
    session_id, _store_root, _coco_root, _truth, _frame, files = _fixture(tmp_path)
    coco = json.loads(files.world_coco)
    coco["images"].append({**coco["images"][0], "id": 2})
    duplicate_files = CaptureGroundTruthFiles(
        files.hud_truth, canonical_json_bytes(coco), files.frames_jsonl,
    )
    with pytest.raises(ValueError, match="frame_id"):
        build_ground_truth_loader(
            {}, {"parser_package": {"parser_artifact_hash": "b" * 64}},
            {session_id: duplicate_files},
        )


def test_loader_rejects_coco_session_id_mismatch(tmp_path: Path) -> None:
    """別 session に結び付いた COCO image を拒否する。

    capture binding と同じ session_id の label だけを formal ground truth に使う。
    """
    session_id, _store_root, _coco_root, _truth, _frame, files = _fixture(tmp_path)
    coco = json.loads(files.world_coco)
    coco["images"][0]["session_id"] = "another-session"
    mismatched_files = CaptureGroundTruthFiles(
        files.hud_truth, canonical_json_bytes(coco), files.frames_jsonl,
    )
    with pytest.raises(ValueError, match="session_id"):
        build_ground_truth_loader(
            {}, {"parser_package": {"parser_artifact_hash": "b" * 64}},
            {session_id: mismatched_files},
        )

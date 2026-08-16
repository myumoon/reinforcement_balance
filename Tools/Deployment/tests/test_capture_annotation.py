"""アノテーション・スプリット・CLIに関するテスト。

DatasetWriter / AnnotationWriter / SplitFreezer の主要な挙動を検証する。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from survivors.capture.captured_frame import CapturedFrame
from survivors.capture_dataset import (
    AnnotationWriter,
    DatasetWriter,
    SplitConflictError,
    SplitFreezer,
    SplitFrozenError,
)


PROFILE_HASH = "c" * 64


def _publish_session(tmp_path, session_id: str, build_id: str = "build-1"):
    """1フレームを書き込んで公開済みセッションを作成するヘルパー。"""
    writer = DatasetWriter(tmp_path, session_id, PROFILE_HASH, build_id)
    pixels = np.zeros((1080, 1920, 4), dtype=np.uint8)
    pixels[..., 3] = 255
    writer.write_frame(
        CapturedFrame(
            frame_bgra=pixels,
            captured_monotonic_ns=1,
            session_frame_index=0,
            client_rect_screen_px=(0, 0, 1920, 1080),
            foreground=True,
            target_profile_hash=PROFILE_HASH,
            game_build_id=build_id,
        )
    )
    return writer.publish()


def test_bbox_class_annotation_round_trip(tmp_path):
    _publish_session(tmp_path, "session-a")
    writer = AnnotationWriter(tmp_path, "session-a")

    expected = writer.write_annotation(
        frame_id=0,
        bbox=(10, 20, 110, 220),
        class_name="enemy",
        annotator_id="operator-1",
        hud_roi=(0, 0, 1920, 240),
        screen_state="gameplay",
        second_review=True,
    )

    assert AnnotationWriter.read_annotations(tmp_path, "session-a") == (expected,)


def test_unknown_semantic_class_is_rejected(tmp_path):
    _publish_session(tmp_path, "session-a")
    writer = AnnotationWriter(tmp_path, "session-a")

    with pytest.raises(ValueError, match="semantic class"):
        writer.write_annotation(0, (0, 0, 10, 10), "dragon", "operator")


def test_duplicate_frame_and_class_annotation_is_rejected(tmp_path):
    _publish_session(tmp_path, "session-a")
    writer = AnnotationWriter(tmp_path, "session-a")
    writer.write_annotation(0, (0, 0, 10, 10), "enemy", "operator")

    with pytest.raises(ValueError, match="duplicate annotation"):
        writer.write_annotation(0, (20, 20, 30, 30), "enemy", "operator")


def test_undo_removes_last_annotation(tmp_path):
    _publish_session(tmp_path, "session-a")
    writer = AnnotationWriter(tmp_path, "session-a")
    first = writer.write_annotation(0, (0, 0, 10, 10), "player", "operator")
    writer.write_annotation(0, (20, 20, 30, 30), "enemy", "operator")

    removed = writer.undo()

    assert removed.class_name == "enemy"
    assert AnnotationWriter.read_annotations(tmp_path, "session-a") == (first,)


def test_annotation_write_autosaves_jsonl(tmp_path):
    session = _publish_session(tmp_path, "session-a")
    writer = AnnotationWriter(tmp_path, "session-a")

    writer.write_annotation(0, (0, 0, 10, 10), "xp_gem", "operator")

    assert (session.session_path / "annotations.jsonl").is_file()


def test_publish_empty_session_is_rejected(tmp_path):
    """フレームなしでpublishしようとするとエラーになる。"""
    writer = DatasetWriter(tmp_path, "session-a", PROFILE_HASH, "build-1")
    with pytest.raises(ValueError, match="no frames"):
        writer.publish()


def test_write_annotation_rejects_unknown_frame_id(tmp_path):
    """公開セッションに存在しないframe_idへのアノテーションは拒否される。"""
    _publish_session(tmp_path, "session-a")
    writer = AnnotationWriter(tmp_path, "session-a")
    with pytest.raises(ValueError, match="frame_id"):
        writer.write_annotation(99, (0, 0, 10, 10), "enemy", "operator")


def test_second_review_appends_and_keeps_audit_trail(tmp_path):
    """second_review=Trueで追記され、JSONLには両方のレコードが残る。"""
    _publish_session(tmp_path, "session-a")
    writer = AnnotationWriter(tmp_path, "session-a")
    writer.write_annotation(0, (0, 0, 10, 10), "enemy", "operator")
    updated = writer.write_annotation(0, (5, 5, 15, 15), "enemy", "reviewer", second_review=True)

    records = AnnotationWriter.read_annotations(tmp_path, "session-a")
    jsonl_lines = [
        ln for ln in (writer.session_path / "annotations.jsonl").read_text().splitlines()
        if ln.strip()
    ]

    assert len(records) == 1
    assert records[0].bbox == updated.bbox
    assert records[0].second_review is True
    assert len(jsonl_lines) == 2  # 初回 + second_review の両レコードが保持される


def test_write_skip_persists_to_jsonl(tmp_path):
    """write_skip()がskips.jsonlに記録を保存する。"""
    _publish_session(tmp_path, "session-a")
    writer = AnnotationWriter(tmp_path, "session-a")
    writer.write_skip(0)
    skip_path = writer.session_path / "skips.jsonl"
    assert skip_path.is_file()
    assert b'"frame_id"' in skip_path.read_bytes()


def test_annotation_writer_loads_skipped_frame_ids_on_resume(tmp_path):
    """resume時にskips.jsonlからスキップ済みframe_idをロードする。"""
    _publish_session(tmp_path, "session-a")
    writer = AnnotationWriter(tmp_path, "session-a")
    writer.write_skip(0)

    resumed = AnnotationWriter(tmp_path, "session-a", resume=True)

    assert 0 in resumed._skipped_frame_ids


def test_annotation_writer_rejects_tampered_metadata(tmp_path):
    """frames.jsonlが改ざんされていたらAnnotationWriter初期化で拒否される。"""
    session = _publish_session(tmp_path, "session-a")
    frames_path = session.session_path / "frames.jsonl"
    frames_path.write_bytes(frames_path.read_bytes() + b"\n{tampered}")

    with pytest.raises(ValueError, match="integrity"):
        AnnotationWriter(tmp_path, "session-a")


def test_split_assigns_four_uses_then_freezes_with_matching_hash(tmp_path):
    for index, session_id in enumerate(("train", "validation", "calibration", "final")):
        _publish_session(tmp_path, session_id, f"build-{index}")
    freezer = SplitFreezer(tmp_path)
    freezer.assign("model_train", ["train"])
    freezer.assign("model_validation", ["validation"])
    freezer.assign("error_calibration", ["calibration"])
    freezer.assign("final_e2e_test", ["final"])

    frozen = freezer.freeze()

    assert set(frozen.splits) == {
        "model_train",
        "model_validation",
        "error_calibration",
        "final_e2e_test",
    }
    assert hashlib.sha256(frozen.manifest_path.read_bytes()).hexdigest() == frozen.manifest_sha256
    assert freezer.get_manifest() == frozen


def test_split_rejects_append_after_freeze(tmp_path):
    _publish_session(tmp_path, "train")
    freezer = SplitFreezer(tmp_path)
    freezer.assign("model_train", ["train"])
    freezer.freeze()

    with pytest.raises(SplitFrozenError):
        freezer.assign("model_validation", ["train"])


def test_split_rejects_session_overlap_between_uses(tmp_path):
    _publish_session(tmp_path, "shared")
    freezer = SplitFreezer(tmp_path)
    freezer.assign("model_train", ["shared"])

    with pytest.raises(SplitConflictError, match="already assigned"):
        freezer.assign("model_validation", ["shared"])


@pytest.mark.parametrize("earlier_split", ["model_train", "error_calibration"])
def test_final_session_cannot_appear_in_train_or_calibration(tmp_path, earlier_split):
    _publish_session(tmp_path, "held-out")
    freezer = SplitFreezer(tmp_path)
    freezer.assign("final_e2e_test", ["held-out"])

    with pytest.raises(SplitConflictError, match="already assigned"):
        freezer.assign(earlier_split, ["held-out"])


@pytest.mark.parametrize(
    "script_name", ["capture_survivors.py", "annotate_survivors_frames.py"]
)
def test_capture_tools_expose_help(script_name):
    deployment_root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, str(deployment_root / script_name), "--help"],
        cwd=deployment_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_capture_cli_rejects_missing_store_root_with_exit_one():
    deployment_root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, str(deployment_root / "capture_survivors.py"), "--synthetic"],
        cwd=deployment_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "--store-root is required" in result.stderr

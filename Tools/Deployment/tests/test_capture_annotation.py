from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from survivors.capture_dataset import (
    AnnotationWriter,
    DatasetWriter,
    SplitConflictError,
    SplitFreezer,
    SplitFrozenError,
)


PROFILE_HASH = "c" * 64


def _publish_empty_session(tmp_path, session_id: str, build_id: str = "build-1"):
    return DatasetWriter(tmp_path, session_id, PROFILE_HASH, build_id).publish()


def test_bbox_class_annotation_round_trip(tmp_path):
    _publish_empty_session(tmp_path, "session-a")
    writer = AnnotationWriter(tmp_path, "session-a")

    expected = writer.write_annotation(
        frame_id=3,
        bbox=(10, 20, 110, 220),
        class_name="enemy",
        annotator_id="operator-1",
        hud_roi=(0, 0, 1920, 240),
        screen_state="gameplay",
        second_review=True,
    )

    assert AnnotationWriter.read_annotations(tmp_path, "session-a") == (expected,)


def test_unknown_semantic_class_is_rejected(tmp_path):
    _publish_empty_session(tmp_path, "session-a")
    writer = AnnotationWriter(tmp_path, "session-a")

    with pytest.raises(ValueError, match="semantic class"):
        writer.write_annotation(0, (0, 0, 10, 10), "dragon", "operator")


def test_duplicate_frame_and_class_annotation_is_rejected(tmp_path):
    _publish_empty_session(tmp_path, "session-a")
    writer = AnnotationWriter(tmp_path, "session-a")
    writer.write_annotation(0, (0, 0, 10, 10), "enemy", "operator")

    with pytest.raises(ValueError, match="duplicate annotation"):
        writer.write_annotation(0, (20, 20, 30, 30), "enemy", "operator")


def test_undo_removes_last_annotation(tmp_path):
    _publish_empty_session(tmp_path, "session-a")
    writer = AnnotationWriter(tmp_path, "session-a")
    first = writer.write_annotation(0, (0, 0, 10, 10), "player", "operator")
    writer.write_annotation(0, (20, 20, 30, 30), "enemy", "operator")

    removed = writer.undo()

    assert removed.class_name == "enemy"
    assert AnnotationWriter.read_annotations(tmp_path, "session-a") == (first,)


def test_annotation_write_autosaves_jsonl(tmp_path):
    session = _publish_empty_session(tmp_path, "session-a")
    writer = AnnotationWriter(tmp_path, "session-a")

    writer.write_annotation(0, (0, 0, 10, 10), "xp_gem", "operator")

    assert (session.session_path / "annotations.jsonl").is_file()


def test_split_assigns_four_uses_then_freezes_with_matching_hash(tmp_path):
    for index, session_id in enumerate(("train", "validation", "calibration", "final")):
        _publish_empty_session(tmp_path, session_id, f"build-{index}")
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
    _publish_empty_session(tmp_path, "train")
    freezer = SplitFreezer(tmp_path)
    freezer.assign("model_train", ["train"])
    freezer.freeze()

    with pytest.raises(SplitFrozenError):
        freezer.assign("model_validation", ["train"])


def test_split_rejects_session_overlap_between_uses(tmp_path):
    _publish_empty_session(tmp_path, "shared")
    freezer = SplitFreezer(tmp_path)
    freezer.assign("model_train", ["shared"])

    with pytest.raises(SplitConflictError, match="already assigned"):
        freezer.assign("model_validation", ["shared"])


@pytest.mark.parametrize("earlier_split", ["model_train", "error_calibration"])
def test_final_session_cannot_appear_in_train_or_calibration(tmp_path, earlier_split):
    _publish_empty_session(tmp_path, "held-out")
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

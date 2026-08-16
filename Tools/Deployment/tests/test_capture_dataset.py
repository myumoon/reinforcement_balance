from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from survivors.capture.captured_frame import CapturedFrame
from survivors.capture_dataset import (
    DatasetWriter,
    FormalDatasetRejectedError,
    Mp4NotAPixelSourceError,
    compare_png_mp4_domain,
    load_formal_dataset,
    load_mp4_as_pixel_source,
)


PROFILE_HASH = "a" * 64
BUILD_ID = "survivors-test-build"


def _frame(
    frame_id: int,
    timestamp_ns: int,
    *,
    profile_hash: str = PROFILE_HASH,
    build_id: str = BUILD_ID,
) -> CapturedFrame:
    pixels = np.zeros((1080, 1920, 4), dtype=np.uint8)
    pixels[0, 0] = [frame_id, 17, 31, 255]
    return CapturedFrame(
        frame_bgra=pixels,
        captured_monotonic_ns=timestamp_ns,
        session_frame_index=frame_id,
        client_rect_screen_px=(0, 0, 1920, 1080),
        foreground=True,
        target_profile_hash=profile_hash,
        game_build_id=build_id,
    )


def _writer(tmp_path, session_id: str = "session-001") -> DatasetWriter:
    return DatasetWriter(tmp_path, session_id, PROFILE_HASH, BUILD_ID)


def _rewrite_metadata(session_path, mutate) -> None:
    manifest_path = session_path / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_path = session_path / manifest["metadata_path"]
    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    mutate(records)
    encoded = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode()
    metadata_path.write_bytes(encoded)
    manifest["metadata_sha256"] = hashlib.sha256(encoded).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_png_jsonl_round_trip_preserves_pixels_and_metadata(tmp_path):
    writer = _writer(tmp_path)
    expected = [_frame(0, 100), _frame(1, 200)]
    for frame in expected:
        writer.write_frame(frame)

    published = writer.publish(operator_checkpoint="synthetic-smoke")
    restored = DatasetWriter.restore(tmp_path, "session-001")

    assert published.session_id == "session-001"
    assert restored.operator_checkpoint == "synthetic-smoke"
    assert restored.formal_dataset_eligible is False
    assert restored.target_profile_hash == PROFILE_HASH
    assert restored.game_build_id == BUILD_ID
    assert len(restored.frames) == 2
    for actual, wanted in zip(restored.frames, expected, strict=True):
        assert np.array_equal(actual.frame_bgra, wanted.frame_bgra)
        assert replace(actual, frame_bgra=wanted.frame_bgra) == wanted


def test_publish_failure_leaves_no_partial_published_session(tmp_path, monkeypatch):
    writer = _writer(tmp_path)
    writer.write_frame(_frame(0, 100))

    def fail_before_publish(self):
        raise RuntimeError("injected temp failure")

    monkeypatch.setattr(DatasetWriter, "_before_atomic_publish", fail_before_publish)
    with pytest.raises(RuntimeError, match="injected"):
        writer.publish()

    published_root = tmp_path / "capture_sessions"
    assert not published_root.exists() or list(published_root.iterdir()) == []


def test_writer_requires_explicit_store_root():
    for value in (None, "", "   "):
        with pytest.raises(ValueError, match="store_root"):
            DatasetWriter(value, "session", PROFILE_HASH, BUILD_ID)


def test_duplicate_frame_id_is_rejected(tmp_path):
    writer = _writer(tmp_path)
    writer.write_frame(_frame(0, 100))
    with pytest.raises(ValueError, match="duplicate frame_id"):
        writer.write_frame(_frame(0, 200))


def test_timestamp_regression_is_rejected(tmp_path):
    writer = _writer(tmp_path)
    writer.write_frame(_frame(0, 100))
    with pytest.raises(ValueError, match="timestamp"):
        writer.write_frame(_frame(1, 100))


@pytest.mark.parametrize(
    ("profile_hash", "build_id"),
    [("b" * 64, BUILD_ID), (PROFILE_HASH, "another-build")],
)
def test_profile_or_build_mixing_is_rejected(tmp_path, profile_hash, build_id):
    writer = _writer(tmp_path)
    with pytest.raises(ValueError, match="session identity"):
        writer.write_frame(_frame(0, 100, profile_hash=profile_hash, build_id=build_id))


def test_restore_rejects_missing_referenced_png(tmp_path):
    manifest = _writer(tmp_path)
    manifest.write_frame(_frame(0, 100))
    published = manifest.publish()
    (published.session_path / published.frame_records[0].object_path).unlink()

    with pytest.raises(ValueError, match="missing object"):
        DatasetWriter.restore(tmp_path, "session-001")


def test_restore_rejects_root_escape_in_object_path(tmp_path):
    writer = _writer(tmp_path)
    writer.write_frame(_frame(0, 100))
    published = writer.publish()
    _rewrite_metadata(
        published.session_path,
        lambda records: records[0].__setitem__("object_path", "../outside.png"),
    )

    with pytest.raises(ValueError, match="escape"):
        DatasetWriter.restore(tmp_path, "session-001")


def test_restore_rejects_png_hash_substitution(tmp_path):
    writer = _writer(tmp_path)
    writer.write_frame(_frame(0, 100))
    published = writer.publish()
    png_path = published.session_path / published.frame_records[0].object_path
    png_path.write_bytes(png_path.read_bytes() + b"substitution")

    with pytest.raises(ValueError, match="hash"):
        DatasetWriter.restore(tmp_path, "session-001")


def test_formal_loader_rejects_ineligible_session(tmp_path):
    writer = _writer(tmp_path)
    writer.write_frame(_frame(0, 100))
    manifest = writer.publish()

    with pytest.raises(FormalDatasetRejectedError, match="not formally eligible"):
        load_formal_dataset(manifest)


def test_png_round_trip_has_zero_pixel_mae(tmp_path):
    writer = _writer(tmp_path)
    expected = _frame(0, 100)
    expected.frame_bgra[100:110, 200:210] = [9, 80, 177, 255]
    writer.write_frame(expected)
    writer.publish()

    actual = DatasetWriter.restore(tmp_path, "session-001").frames[0].frame_bgra

    mae = np.abs(actual.astype(np.int16) - expected.frame_bgra.astype(np.int16)).mean()
    assert mae == 0


def test_publish_rejects_empty_session(tmp_path):
    """フレームなしでpublishを呼ぶとValueErrorになる。"""
    writer = _writer(tmp_path)
    with pytest.raises(ValueError, match="no frames"):
        writer.publish()


def test_writer_context_manager_cleans_up_temp_on_exception(tmp_path):
    """capture失敗時にcontextmanagerがtempディレクトリを削除する。"""
    try:
        with DatasetWriter(tmp_path, "session-001", PROFILE_HASH, BUILD_ID) as writer:
            writer.write_frame(_frame(0, 100))
            raise RuntimeError("injected failure")
    except RuntimeError:
        pass
    temp_root = tmp_path / ".capture-tmp"
    assert not temp_root.exists() or list(temp_root.iterdir()) == []


def test_mp4_is_never_a_formal_pixel_source(tmp_path):
    with pytest.raises(Mp4NotAPixelSourceError, match="review-only"):
        load_mp4_as_pixel_source(tmp_path / "review.real.mp4")


def test_png_mp4_domain_comparison_reports_metrics(tmp_path, monkeypatch):
    writer = _writer(tmp_path)
    expected = _frame(0, 100)
    writer.write_frame(expected)
    manifest = writer.publish()
    png_path = manifest.session_path / manifest.frame_records[0].object_path
    mp4_path = tmp_path / "review.real.mp4"
    mp4_path.write_bytes(b"review-only-placeholder")
    monkeypatch.setattr(
        "survivors.capture_dataset._read_mp4_first_bgra",
        lambda _path: expected.frame_bgra.copy(),
    )

    comparison = compare_png_mp4_domain(png_path, mp4_path)

    assert comparison == {
        "mae": 0.0,
        "small_object_edge_diff": 0.0,
        "hud_roi_diff": 0.0,
    }

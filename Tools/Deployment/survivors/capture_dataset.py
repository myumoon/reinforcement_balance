"""Versioned real-vs-capture dataset storage confined to Deployment.

The writer keeps every frame in a private temporary session and exposes the
session only with one directory rename.  Published sessions remain explicitly
ineligible for formal training until the separate post-merge operator gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import struct
from typing import Any, Iterable
import uuid
import zlib

import numpy as np
from numpy.typing import NDArray

from .capture.captured_frame import CapturedFrame


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_FRAME_SHAPE = (1080, 1920, 4)
_MANIFEST_VERSION = 1


class FormalDatasetRejectedError(ValueError):
    """Raised when a non-gated capture is requested as a formal dataset."""


class Mp4NotAPixelSourceError(ValueError):
    """Raised for every attempt to treat review-only MP4 as source pixels."""


class SplitFrozenError(ValueError):
    """Raised when a frozen split is mutated."""


class SplitConflictError(ValueError):
    """Raised when a session/build unit is assigned inconsistently."""


@dataclass(frozen=True, slots=True)
class FrameRecord:
    frame_id: int
    captured_monotonic_ns: int
    object_path: str
    object_sha256: str
    client_rect_screen_px: tuple[int, int, int, int]
    foreground: bool
    target_profile_hash: str
    game_build_id: str


@dataclass(frozen=True, slots=True)
class SessionManifest:
    schema_version: int
    session_id: str
    target_profile_hash: str
    game_build_id: str
    formal_dataset_eligible: bool
    operator_checkpoint: str
    pixel_source: str
    metadata_path: str
    metadata_sha256: str
    frame_count: int
    session_path: Path
    manifest_sha256: str
    frame_records: tuple[FrameRecord, ...]
    frames: tuple[CapturedFrame, ...]


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_store_root(store_root: os.PathLike[str] | str | None) -> Path:
    if store_root is None or not str(store_root).strip():
        raise ValueError("store_root must be explicitly provided")
    root = Path(store_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise ValueError("store_root must be a directory")
    return root


def _validated_component(value: str, field: str) -> str:
    if not isinstance(value, str) or _COMPONENT_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a safe non-empty path component")
    return value


def _validated_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a sha256 hex digest")
    return value


def _resolve_relative(root: Path, relative: object, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{field} must be a non-empty relative path")
    rel_path = Path(relative)
    if rel_path.is_absolute() or ".." in rel_path.parts or "." in rel_path.parts:
        raise ValueError(f"{field} attempts root escape")
    candidate = (root / rel_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} attempts root escape") from exc
    return candidate


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _encode_lossless_png(frame_bgra: NDArray[np.uint8]) -> bytes:
    if frame_bgra.dtype != np.uint8 or frame_bgra.shape != _FRAME_SHAPE:
        raise ValueError("PNG frame must be 1920x1080 uint8 BGRA")
    height, width, _ = frame_bgra.shape
    rgba = np.ascontiguousarray(frame_bgra[..., [2, 1, 0, 3]])
    scanlines = b"".join(b"\x00" + row.tobytes() for row in rgba)
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return b"".join(
        (
            _PNG_SIGNATURE,
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(scanlines, level=6)),
            _png_chunk(b"IEND", b""),
        )
    )


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= up_distance and left_distance <= upper_left_distance:
        return left
    if up_distance <= upper_left_distance:
        return up
    return upper_left


def _decode_lossless_png(encoded: bytes) -> NDArray[np.uint8]:
    if not encoded.startswith(_PNG_SIGNATURE):
        raise ValueError("object is not a PNG")
    offset = len(_PNG_SIGNATURE)
    width = height = None
    compressed = bytearray()
    saw_end = False
    while offset < len(encoded):
        if offset + 12 > len(encoded):
            raise ValueError("truncated PNG chunk")
        length = struct.unpack(">I", encoded[offset : offset + 4])[0]
        kind = encoded[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(encoded):
            raise ValueError("truncated PNG payload")
        payload = encoded[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", encoded[offset + 8 + length : end])[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise ValueError("PNG chunk checksum mismatch")
        if kind == b"IHDR":
            if len(payload) != 13:
                raise ValueError("invalid PNG header")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if (depth, color, compression, filtering, interlace) != (8, 6, 0, 0, 0):
                raise ValueError("PNG must be non-interlaced 8-bit RGBA")
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            saw_end = True
            break
        offset = end
    if (height, width) != _FRAME_SHAPE[:2] or not saw_end:
        raise ValueError("PNG dimensions or terminator do not match capture contract")
    try:
        raw = zlib.decompress(bytes(compressed))
    except zlib.error as exc:
        raise ValueError("invalid PNG compressed pixels") from exc
    stride = width * 4
    if len(raw) != height * (stride + 1):
        raise ValueError("PNG pixel payload has the wrong size")
    decoded = np.empty((height, stride), dtype=np.uint8)
    previous = np.zeros(stride, dtype=np.uint8)
    cursor = 0
    for row_index in range(height):
        filter_type = raw[cursor]
        source = raw[cursor + 1 : cursor + 1 + stride]
        cursor += stride + 1
        current = np.empty(stride, dtype=np.uint8)
        for byte_index, source_value in enumerate(source):
            left = int(current[byte_index - 4]) if byte_index >= 4 else 0
            up = int(previous[byte_index])
            upper_left = int(previous[byte_index - 4]) if byte_index >= 4 else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = up
            elif filter_type == 3:
                predictor = (left + up) // 2
            elif filter_type == 4:
                predictor = _paeth(left, up, upper_left)
            else:
                raise ValueError("unsupported PNG row filter")
            current[byte_index] = (source_value + predictor) & 0xFF
        decoded[row_index] = current
        previous = current
    rgba = decoded.reshape(height, width, 4)
    return np.ascontiguousarray(rgba[..., [2, 1, 0, 3]])


class DatasetWriter:
    """Stream CapturedFrame objects into a private, atomically published session."""

    def __init__(
        self,
        store_root: os.PathLike[str] | str | None,
        session_id: str,
        target_profile_hash: str,
        game_build_id: str,
    ) -> None:
        self.store_root = _validated_store_root(store_root)
        self.session_id = _validated_component(session_id, "session_id")
        self.target_profile_hash = _validated_sha256(
            target_profile_hash, "target_profile_hash"
        )
        self.game_build_id = _validated_component(game_build_id, "game_build_id")
        self._published_path = self.store_root / "capture_sessions" / self.session_id
        if self._published_path.exists():
            raise ValueError(f"session already exists: {self.session_id}")
        temp_parent = self.store_root / ".capture-tmp"
        temp_parent.mkdir(parents=True, exist_ok=True)
        self._temp_path = temp_parent / f"{self.session_id}-{uuid.uuid4().hex}"
        (self._temp_path / "frames").mkdir(parents=True)
        self._metadata_path = self._temp_path / "frames.jsonl"
        self._metadata_stream = self._metadata_path.open("xb")
        self._frame_ids: set[int] = set()
        self._last_timestamp: int | None = None
        self._frame_count = 0
        self._sealed = False

    def write_frame(self, frame: CapturedFrame) -> FrameRecord:
        if self._sealed:
            raise ValueError("dataset writer is already sealed")
        if not isinstance(frame, CapturedFrame):
            raise ValueError("frame must be a CapturedFrame")
        frame_id = frame.session_frame_index
        if frame_id in self._frame_ids:
            raise ValueError(f"duplicate frame_id: {frame_id}")
        if self._last_timestamp is not None and frame.captured_monotonic_ns <= self._last_timestamp:
            raise ValueError("captured timestamp must increase monotonically")
        if (
            frame.target_profile_hash != self.target_profile_hash
            or frame.game_build_id != self.game_build_id
        ):
            raise ValueError("frame does not match session identity profile/build")

        object_path = f"frames/{frame_id:08d}.png"
        destination = _resolve_relative(self._temp_path, object_path, "object_path")
        if destination.exists():
            raise ValueError(f"duplicate frame object: {object_path}")
        encoded = _encode_lossless_png(frame.frame_bgra)
        destination.write_bytes(encoded)
        record = FrameRecord(
            frame_id=frame_id,
            captured_monotonic_ns=frame.captured_monotonic_ns,
            object_path=object_path,
            object_sha256=_sha256_bytes(encoded),
            client_rect_screen_px=frame.client_rect_screen_px,
            foreground=frame.foreground,
            target_profile_hash=frame.target_profile_hash,
            game_build_id=frame.game_build_id,
        )
        try:
            self._metadata_stream.write(_canonical_json(_record_to_wire(record)))
            self._metadata_stream.flush()
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        self._frame_ids.add(frame_id)
        self._last_timestamp = frame.captured_monotonic_ns
        self._frame_count += 1
        return record

    def publish(self, operator_checkpoint: str = "") -> SessionManifest:
        if self._sealed:
            raise ValueError("dataset writer is already sealed")
        if not isinstance(operator_checkpoint, str):
            raise ValueError("operator_checkpoint must be a string")
        self._metadata_stream.flush()
        os.fsync(self._metadata_stream.fileno())
        self._metadata_stream.close()
        self._sealed = True
        metadata_hash = _sha256_file(self._metadata_path)
        manifest_wire = {
            "schema_version": _MANIFEST_VERSION,
            "session_id": self.session_id,
            "target_profile_hash": self.target_profile_hash,
            "game_build_id": self.game_build_id,
            "formal_dataset_eligible": False,
            "operator_checkpoint": operator_checkpoint,
            "pixel_source": "lossless_png",
            "metadata_path": "frames.jsonl",
            "metadata_sha256": metadata_hash,
            "frame_count": self._frame_count,
        }
        (self._temp_path / "session_manifest.json").write_bytes(
            _canonical_json(manifest_wire)
        )
        self._before_atomic_publish()
        self._published_path.parent.mkdir(parents=True, exist_ok=True)
        if self._published_path.exists():
            raise ValueError(f"session already exists: {self.session_id}")
        os.replace(self._temp_path, self._published_path)
        return self.restore(self.store_root, self.session_id)

    def _before_atomic_publish(self) -> None:
        """Test injection point after all temp writes and before the only publish rename."""

    @classmethod
    def restore(
        cls, store_root: os.PathLike[str] | str | None, session_id: str
    ) -> SessionManifest:
        root = _validated_store_root(store_root)
        safe_session_id = _validated_component(session_id, "session_id")
        session_path = _resolve_relative(
            root, f"capture_sessions/{safe_session_id}", "session_path"
        )
        if not session_path.is_dir():
            raise ValueError(f"missing session: {safe_session_id}")
        manifest_path = session_path / "session_manifest.json"
        if not manifest_path.is_file():
            raise ValueError("missing session manifest")
        manifest_bytes = manifest_path.read_bytes()
        try:
            wire = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid session manifest") from exc
        _validate_manifest_wire(wire, safe_session_id)
        metadata_path = _resolve_relative(
            session_path, wire["metadata_path"], "metadata_path"
        )
        if not metadata_path.is_file():
            raise ValueError("missing frame metadata")
        if _sha256_file(metadata_path) != wire["metadata_sha256"]:
            raise ValueError("frame metadata hash mismatch")

        records: list[FrameRecord] = []
        frames: list[CapturedFrame] = []
        seen_ids: set[int] = set()
        last_timestamp: int | None = None
        with metadata_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = _record_from_wire(json.loads(line))
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"invalid frame metadata at line {line_number}") from exc
                if record.frame_id in seen_ids:
                    raise ValueError(f"duplicate frame_id: {record.frame_id}")
                if last_timestamp is not None and record.captured_monotonic_ns <= last_timestamp:
                    raise ValueError("captured timestamp regression in metadata")
                if (
                    record.target_profile_hash != wire["target_profile_hash"]
                    or record.game_build_id != wire["game_build_id"]
                ):
                    raise ValueError("profile/build mixing in session metadata")
                object_path = _resolve_relative(
                    session_path, record.object_path, "object_path"
                )
                if not object_path.is_file():
                    raise ValueError(f"missing object: {record.object_path}")
                if _sha256_file(object_path) != record.object_sha256:
                    raise ValueError(f"object hash mismatch: {record.object_path}")
                pixels = _decode_lossless_png(object_path.read_bytes())
                frames.append(
                    CapturedFrame(
                        frame_bgra=pixels,
                        captured_monotonic_ns=record.captured_monotonic_ns,
                        session_frame_index=record.frame_id,
                        client_rect_screen_px=record.client_rect_screen_px,
                        foreground=record.foreground,
                        target_profile_hash=record.target_profile_hash,
                        game_build_id=record.game_build_id,
                    )
                )
                records.append(record)
                seen_ids.add(record.frame_id)
                last_timestamp = record.captured_monotonic_ns
        if len(records) != wire["frame_count"]:
            raise ValueError("manifest frame_count does not match metadata")
        return SessionManifest(
            schema_version=wire["schema_version"],
            session_id=wire["session_id"],
            target_profile_hash=wire["target_profile_hash"],
            game_build_id=wire["game_build_id"],
            formal_dataset_eligible=wire["formal_dataset_eligible"],
            operator_checkpoint=wire["operator_checkpoint"],
            pixel_source=wire["pixel_source"],
            metadata_path=wire["metadata_path"],
            metadata_sha256=wire["metadata_sha256"],
            frame_count=wire["frame_count"],
            session_path=session_path,
            manifest_sha256=_sha256_bytes(manifest_bytes),
            frame_records=tuple(records),
            frames=tuple(frames),
        )


def _record_to_wire(record: FrameRecord) -> dict[str, object]:
    return {
        "frame_id": record.frame_id,
        "captured_monotonic_ns": record.captured_monotonic_ns,
        "object_path": record.object_path,
        "object_sha256": record.object_sha256,
        "client_rect_screen_px": list(record.client_rect_screen_px),
        "foreground": record.foreground,
        "target_profile_hash": record.target_profile_hash,
        "game_build_id": record.game_build_id,
    }


def _record_from_wire(wire: object) -> FrameRecord:
    if not isinstance(wire, dict):
        raise ValueError("frame metadata must be an object")
    expected = {
        "frame_id",
        "captured_monotonic_ns",
        "object_path",
        "object_sha256",
        "client_rect_screen_px",
        "foreground",
        "target_profile_hash",
        "game_build_id",
    }
    if set(wire) != expected:
        raise ValueError("frame metadata fields do not match schema")
    frame_id = wire["frame_id"]
    timestamp = wire["captured_monotonic_ns"]
    rect = wire["client_rect_screen_px"]
    if type(frame_id) is not int or frame_id < 0:
        raise ValueError("frame_id must be a non-negative integer")
    if type(timestamp) is not int or timestamp < 0:
        raise ValueError("captured timestamp must be a non-negative integer")
    if (
        not isinstance(rect, list)
        or len(rect) != 4
        or not all(type(value) is int for value in rect)
        or (rect[2] - rect[0], rect[3] - rect[1]) != (1920, 1080)
    ):
        raise ValueError("client rect does not match capture contract")
    if type(wire["foreground"]) is not bool:
        raise ValueError("foreground must be a bool")
    if not isinstance(wire["object_path"], str):
        raise ValueError("object_path must be a string")
    if not isinstance(wire["game_build_id"], str) or not wire["game_build_id"]:
        raise ValueError("game_build_id must be a non-empty string")
    return FrameRecord(
        frame_id=frame_id,
        captured_monotonic_ns=timestamp,
        object_path=wire["object_path"],
        object_sha256=_validated_sha256(wire["object_sha256"], "object_sha256"),
        client_rect_screen_px=tuple(rect),
        foreground=wire["foreground"],
        target_profile_hash=_validated_sha256(
            wire["target_profile_hash"], "target_profile_hash"
        ),
        game_build_id=wire["game_build_id"],
    )


def _validate_manifest_wire(wire: object, session_id: str) -> None:
    expected = {
        "schema_version",
        "session_id",
        "target_profile_hash",
        "game_build_id",
        "formal_dataset_eligible",
        "operator_checkpoint",
        "pixel_source",
        "metadata_path",
        "metadata_sha256",
        "frame_count",
    }
    if not isinstance(wire, dict) or set(wire) != expected:
        raise ValueError("session manifest fields do not match schema")
    if wire["schema_version"] != _MANIFEST_VERSION:
        raise ValueError("unsupported session manifest version")
    if wire["session_id"] != session_id:
        raise ValueError("session manifest identity mismatch")
    _validated_sha256(wire["target_profile_hash"], "target_profile_hash")
    _validated_sha256(wire["metadata_sha256"], "metadata_sha256")
    _validated_component(wire["game_build_id"], "game_build_id")
    if type(wire["formal_dataset_eligible"]) is not bool:
        raise ValueError("formal_dataset_eligible must be a bool")
    if not isinstance(wire["operator_checkpoint"], str):
        raise ValueError("operator_checkpoint must be a string")
    if wire["pixel_source"] != "lossless_png":
        raise ValueError("session pixel_source must be lossless_png")
    if not isinstance(wire["metadata_path"], str):
        raise ValueError("metadata_path must be a string")
    if type(wire["frame_count"]) is not int or wire["frame_count"] < 0:
        raise ValueError("frame_count must be a non-negative integer")


def restore(
    store_root: os.PathLike[str] | str | None, session_id: str
) -> SessionManifest:
    """Convenience wrapper for callers that do not retain the writer class."""

    return DatasetWriter.restore(store_root, session_id)


def load_formal_dataset(manifest: SessionManifest) -> tuple[CapturedFrame, ...]:
    if not isinstance(manifest, SessionManifest):
        raise ValueError("manifest must be a SessionManifest")
    if manifest.formal_dataset_eligible is not True:
        raise FormalDatasetRejectedError("session is not formally eligible")
    if manifest.pixel_source != "lossless_png":
        raise FormalDatasetRejectedError("formal dataset pixel source must be lossless PNG")
    return manifest.frames


def _import_cv2():
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv-python==4.10.0.84 is required for MP4 review output") from exc


def load_mp4_as_pixel_source(*_args, **_kwargs):
    raise Mp4NotAPixelSourceError(
        "MP4 is review-only and cannot be loaded as a formal pixel source"
    )


class Mp4VideoWriter:
    """Lossy review-video writer; output from this class is never a pixel source."""

    def __init__(
        self,
        output_path: os.PathLike[str] | str,
        *,
        fps: float = 30.0,
        codec: str = "mp4v",
    ) -> None:
        if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
            raise ValueError("fps must be positive")
        if codec not in {"mp4v", "avc1"}:
            raise ValueError("MP4 codec must be mp4v or avc1")
        path = Path(output_path)
        if path.suffix.lower() != ".mp4":
            raise ValueError("review video path must end in .mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2 = _import_cv2()
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(path), fourcc, float(fps), (1920, 1080))
        if not self._writer.isOpened():
            self._writer.release()
            raise ValueError(f"could not open MP4 review writer: {path}")
        self.output_path = path
        self._closed = False

    def write_frame(self, frame: CapturedFrame | NDArray[np.uint8]) -> None:
        if self._closed:
            raise ValueError("MP4 review writer is closed")
        pixels = frame.frame_bgra if isinstance(frame, CapturedFrame) else frame
        if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.shape != _FRAME_SHAPE:
            raise ValueError("MP4 frame must be 1920x1080 uint8 BGRA")
        self._writer.write(np.ascontiguousarray(pixels[..., :3]))

    def close(self) -> None:
        if not self._closed:
            self._writer.release()
            self._closed = True

    def __enter__(self) -> "Mp4VideoWriter":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


def _read_mp4_first_bgra(mp4_path: Path) -> NDArray[np.uint8]:
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(mp4_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open review MP4: {mp4_path}")
        success, bgr = capture.read()
        if not success or not isinstance(bgr, np.ndarray) or bgr.shape != (1080, 1920, 3):
            raise ValueError("review MP4 has no 1920x1080 frame")
        alpha = np.full((1080, 1920, 1), 255, dtype=np.uint8)
        return np.ascontiguousarray(np.concatenate((bgr, alpha), axis=2))
    finally:
        capture.release()


def _edge_strength(bgra: NDArray[np.uint8]) -> NDArray[np.float32]:
    gray = (
        bgra[..., 0].astype(np.float32) * 0.114
        + bgra[..., 1].astype(np.float32) * 0.587
        + bgra[..., 2].astype(np.float32) * 0.299
    )
    horizontal = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    vertical = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    return horizontal + vertical


def compare_png_mp4_domain(
    png_path: os.PathLike[str] | str,
    mp4_path: os.PathLike[str] | str,
) -> dict[str, float]:
    """Compare lossless source PNG to the first lossy MP4 frame for review only."""

    png = Path(png_path)
    mp4 = Path(mp4_path)
    if not png.is_file():
        raise ValueError(f"missing PNG: {png}")
    if not mp4.is_file():
        raise ValueError(f"missing MP4: {mp4}")
    source = _decode_lossless_png(png.read_bytes())
    review = _read_mp4_first_bgra(mp4)
    if review.dtype != np.uint8 or review.shape != source.shape:
        raise ValueError("PNG and MP4 frames do not share the capture shape")
    source_rgb = source[..., :3].astype(np.float32)
    review_rgb = review[..., :3].astype(np.float32)
    mae = float(np.mean(np.abs(source_rgb - review_rgb)))
    edge_diff = float(np.mean(np.abs(_edge_strength(source) - _edge_strength(review))))
    hud_height = source.shape[0] // 4
    hud_diff = float(
        np.mean(np.abs(source_rgb[:hud_height] - review_rgb[:hud_height]))
    )
    return {
        "mae": mae,
        "small_object_edge_diff": edge_diff,
        "hud_roi_diff": hud_diff,
    }


SEMANTIC_CLASSES = (
    "player",
    "enemy",
    "xp_gem",
    "chest",
    "hud_hp",
    "hud_xp",
    "card",
    "button",
    "death_result",
)
SPLIT_NAMES = (
    "model_train",
    "model_validation",
    "error_calibration",
    "final_e2e_test",
)


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    frame_id: int
    bbox: tuple[int | float, int | float, int | float, int | float]
    class_name: str
    annotator_id: str
    hud_roi: tuple[int | float, int | float, int | float, int | float] | None
    screen_state: str | None
    second_review: bool


def _validated_bbox(
    bbox: object, field: str
) -> tuple[int | float, int | float, int | float, int | float]:
    if (
        not isinstance(bbox, (tuple, list))
        or len(bbox) != 4
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in bbox
        )
    ):
        raise ValueError(f"{field} must contain four numeric coordinates")
    left, top, right, bottom = bbox
    if not (0 <= left < right <= 1920 and 0 <= top < bottom <= 1080):
        raise ValueError(f"{field} must be inside the 1920x1080 client")
    return left, top, right, bottom


def _annotation_to_wire(record: AnnotationRecord) -> dict[str, object]:
    return {
        "frame_id": record.frame_id,
        "bbox": list(record.bbox),
        "class_name": record.class_name,
        "annotator_id": record.annotator_id,
        "hud_roi": list(record.hud_roi) if record.hud_roi is not None else None,
        "screen_state": record.screen_state,
        "second_review": record.second_review,
    }


def _annotation_from_wire(wire: object) -> AnnotationRecord:
    expected = {
        "frame_id",
        "bbox",
        "class_name",
        "annotator_id",
        "hud_roi",
        "screen_state",
        "second_review",
    }
    if not isinstance(wire, dict) or set(wire) != expected:
        raise ValueError("annotation fields do not match schema")
    frame_id = wire["frame_id"]
    if type(frame_id) is not int or frame_id < 0:
        raise ValueError("annotation frame_id must be a non-negative integer")
    class_name = wire["class_name"]
    if class_name not in SEMANTIC_CLASSES:
        raise ValueError(f"unknown semantic class: {class_name!r}")
    annotator_id = wire["annotator_id"]
    if not isinstance(annotator_id, str) or not annotator_id.strip():
        raise ValueError("annotator_id must be non-empty")
    screen_state = wire["screen_state"]
    if screen_state is not None and (
        not isinstance(screen_state, str) or not screen_state.strip()
    ):
        raise ValueError("screen_state must be None or a non-empty string")
    if type(wire["second_review"]) is not bool:
        raise ValueError("second_review must be a bool")
    return AnnotationRecord(
        frame_id=frame_id,
        bbox=_validated_bbox(wire["bbox"], "bbox"),
        class_name=class_name,
        annotator_id=annotator_id,
        hud_roi=(
            None
            if wire["hud_roi"] is None
            else _validated_bbox(wire["hud_roi"], "hud_roi")
        ),
        screen_state=screen_state,
        second_review=wire["second_review"],
    )


class AnnotationWriter:
    """Append-only annotation JSONL with an atomic last-record undo."""

    def __init__(
        self,
        store_root: os.PathLike[str] | str | None,
        session_id: str,
        *,
        resume: bool = False,
    ) -> None:
        self.store_root = _validated_store_root(store_root)
        self.session_id = _validated_component(session_id, "session_id")
        self.session_path = _resolve_relative(
            self.store_root, f"capture_sessions/{self.session_id}", "session_path"
        )
        if not (self.session_path / "session_manifest.json").is_file():
            raise ValueError(f"missing published session: {self.session_id}")
        self.annotation_path = self.session_path / "annotations.jsonl"
        if self.annotation_path.exists() and not resume:
            raise ValueError("annotations already exist; use resume=True")
        self._annotations = list(self.read_annotations(self.store_root, self.session_id))
        self._keys = {(item.frame_id, item.class_name) for item in self._annotations}

    def write_annotation(
        self,
        frame_id: int,
        bbox: tuple[int | float, int | float, int | float, int | float],
        class_name: str,
        annotator_id: str,
        hud_roi: tuple[int | float, int | float, int | float, int | float] | None = None,
        screen_state: str | None = None,
        second_review: bool = False,
    ) -> AnnotationRecord:
        record = _annotation_from_wire(
            {
                "frame_id": frame_id,
                "bbox": bbox,
                "class_name": class_name,
                "annotator_id": annotator_id,
                "hud_roi": hud_roi,
                "screen_state": screen_state,
                "second_review": second_review,
            }
        )
        key = record.frame_id, record.class_name
        if key in self._keys:
            raise ValueError(
                f"duplicate annotation for frame_id/class: {record.frame_id}/{record.class_name}"
            )
        with self.annotation_path.open("ab") as stream:
            stream.write(_canonical_json(_annotation_to_wire(record)))
            stream.flush()
            os.fsync(stream.fileno())
        self._annotations.append(record)
        self._keys.add(key)
        return record

    def undo(self) -> AnnotationRecord | None:
        if not self._annotations:
            return None
        removed = self._annotations.pop()
        self._keys.remove((removed.frame_id, removed.class_name))
        temp_path = self.session_path / ".annotations.jsonl.tmp"
        with temp_path.open("wb") as stream:
            for record in self._annotations:
                stream.write(_canonical_json(_annotation_to_wire(record)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, self.annotation_path)
        return removed

    @classmethod
    def read_annotations(
        cls, store_root: os.PathLike[str] | str | None, session_id: str
    ) -> tuple[AnnotationRecord, ...]:
        root = _validated_store_root(store_root)
        safe_session_id = _validated_component(session_id, "session_id")
        session_path = _resolve_relative(
            root, f"capture_sessions/{safe_session_id}", "session_path"
        )
        path = session_path / "annotations.jsonl"
        if not path.exists():
            return ()
        records: list[AnnotationRecord] = []
        keys: set[tuple[int, str]] = set()
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = _annotation_from_wire(json.loads(line))
                except (json.JSONDecodeError, TypeError, KeyError) as exc:
                    raise ValueError(f"invalid annotation at line {line_number}") from exc
                key = record.frame_id, record.class_name
                if key in keys:
                    raise ValueError("duplicate annotation in JSONL")
                keys.add(key)
                records.append(record)
        return tuple(records)


@dataclass(frozen=True, slots=True)
class SplitUnit:
    session_id: str
    game_build_id: str
    session_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class SplitManifest:
    schema_version: int
    frozen: bool
    splits: dict[str, tuple[SplitUnit, ...]]
    manifest_path: Path
    manifest_sha256: str


def _split_unit_to_wire(unit: SplitUnit) -> dict[str, str]:
    return {
        "session_id": unit.session_id,
        "game_build_id": unit.game_build_id,
        "session_manifest_sha256": unit.session_manifest_sha256,
    }


def _split_manifest_from_bytes(path: Path, encoded: bytes) -> SplitManifest:
    try:
        wire = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid split manifest") from exc
    if (
        not isinstance(wire, dict)
        or set(wire) != {"schema_version", "frozen", "splits"}
        or wire["schema_version"] != 1
        or wire["frozen"] is not True
        or not isinstance(wire["splits"], dict)
        or set(wire["splits"]) != set(SPLIT_NAMES)
    ):
        raise ValueError("split manifest fields do not match schema")
    splits: dict[str, tuple[SplitUnit, ...]] = {}
    seen: set[str] = set()
    for split_name in SPLIT_NAMES:
        values = wire["splits"][split_name]
        if not isinstance(values, list):
            raise ValueError("split entries must be lists")
        units = []
        for value in values:
            if not isinstance(value, dict) or set(value) != {
                "session_id",
                "game_build_id",
                "session_manifest_sha256",
            }:
                raise ValueError("split unit fields do not match schema")
            session_id = _validated_component(value["session_id"], "session_id")
            if session_id in seen:
                raise SplitConflictError(f"session already assigned: {session_id}")
            seen.add(session_id)
            units.append(
                SplitUnit(
                    session_id=session_id,
                    game_build_id=_validated_component(value["game_build_id"], "game_build_id"),
                    session_manifest_sha256=_validated_sha256(
                        value["session_manifest_sha256"], "session_manifest_sha256"
                    ),
                )
            )
        splits[split_name] = tuple(units)
    return SplitManifest(1, True, splits, path, _sha256_bytes(encoded))


class SplitFreezer:
    """Freeze disjoint session/build assignments into a hash-addressed manifest."""

    def __init__(self, store_root: os.PathLike[str] | str | None) -> None:
        self.store_root = _validated_store_root(store_root)
        self.manifest_path = self.store_root / "capture_split_manifest.json"
        self._splits: dict[str, list[SplitUnit]] = {name: [] for name in SPLIT_NAMES}
        self._session_ids: set[str] = set()
        self._manifest: SplitManifest | None = None
        if self.manifest_path.exists():
            self._manifest = _split_manifest_from_bytes(
                self.manifest_path, self.manifest_path.read_bytes()
            )
            self._splits = {
                name: list(self._manifest.splits[name]) for name in SPLIT_NAMES
            }
            self._session_ids = {
                unit.session_id for units in self._splits.values() for unit in units
            }

    def assign(self, split_name: str, session_ids: Iterable[str]) -> None:
        if self._manifest is not None:
            raise SplitFrozenError("split manifest is frozen")
        if split_name not in SPLIT_NAMES:
            raise ValueError(f"unknown split name: {split_name!r}")
        if isinstance(session_ids, (str, bytes)):
            raise ValueError("session_ids must be an iterable of session ids")
        pending: list[SplitUnit] = []
        pending_ids: set[str] = set()
        for session_id in session_ids:
            safe_id = _validated_component(session_id, "session_id")
            if safe_id in self._session_ids or safe_id in pending_ids:
                raise SplitConflictError(f"session already assigned: {safe_id}")
            manifest_path = _resolve_relative(
                self.store_root,
                f"capture_sessions/{safe_id}/session_manifest.json",
                "session_manifest_path",
            )
            if not manifest_path.is_file():
                raise ValueError(f"missing session manifest: {safe_id}")
            encoded = manifest_path.read_bytes()
            try:
                wire = json.loads(encoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid session manifest: {safe_id}") from exc
            _validate_manifest_wire(wire, safe_id)
            pending.append(
                SplitUnit(safe_id, wire["game_build_id"], _sha256_bytes(encoded))
            )
            pending_ids.add(safe_id)
        self._splits[split_name].extend(pending)
        self._session_ids.update(pending_ids)

    def freeze(self) -> SplitManifest:
        if self._manifest is not None:
            raise SplitFrozenError("split manifest is frozen")
        wire = {
            "schema_version": 1,
            "frozen": True,
            "splits": {
                name: [_split_unit_to_wire(unit) for unit in self._splits[name]]
                for name in SPLIT_NAMES
            },
        }
        encoded = _canonical_json(wire)
        temp_path = self.store_root / ".capture_split_manifest.json.tmp"
        temp_path.write_bytes(encoded)
        os.replace(temp_path, self.manifest_path)
        self._manifest = _split_manifest_from_bytes(self.manifest_path, encoded)
        return self._manifest

    def get_manifest(self) -> SplitManifest:
        if self._manifest is None:
            raise ValueError("split manifest has not been frozen")
        encoded = self.manifest_path.read_bytes()
        if _sha256_bytes(encoded) != self._manifest.manifest_sha256:
            raise ValueError("frozen split manifest hash mismatch")
        current = _split_manifest_from_bytes(self.manifest_path, encoded)
        if current != self._manifest:
            raise ValueError("frozen split manifest was substituted")
        for units in current.splits.values():
            for unit in units:
                session_manifest = _resolve_relative(
                    self.store_root,
                    f"capture_sessions/{unit.session_id}/session_manifest.json",
                    "session_manifest_path",
                )
                if (
                    not session_manifest.is_file()
                    or _sha256_file(session_manifest) != unit.session_manifest_sha256
                ):
                    raise ValueError(
                        f"frozen session manifest hash mismatch: {unit.session_id}"
                    )
        return self._manifest

"""descriptor-backed HUD truth と COCO label から formal GT を読む。

各 session の JSONL は offset index だけを保持し、snapshot は要求時に一 frame ずつ作る。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from survivors.capture.captured_frame import CapturedFrame
from survivors.capture_dataset import FrameRecord, _record_from_wire
from survivors.formal_ground_truth import (
    CaptureGroundTruthFiles,
    CocoImageEntry,
    coco_image_entries_from_wire,
    ground_truth_snapshot,
)
from survivors.vision.entity_tracker import default_class_map
from survivors.vision.hud_truth import HudTruthRecord, validate_record


def _jsonl_offsets(data: bytes, kind: str, session_id: str) -> dict[int, tuple[int, int]]:
    """JSONL を一行ずつ検証し、frame index から byte 範囲への index を作る。

    行の内容を snapshot として保持せず、必要な frame の JSON だけを後から decode する。
    """
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{kind} must be UTF-8") from exc
    offsets: dict[int, tuple[int, int]] = {}
    start = 0
    while start < len(data):
        end = data.find(b"\n", start)
        if end < 0:
            end = len(data)
        raw = data[start:end]
        if raw:
            try:
                wire = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid {kind} JSONL row") from exc
            if kind == "hud_truth":
                record = HudTruthRecord.from_wire(wire)
                if record.session_id != session_id:
                    raise ValueError("HUD truth session_id does not match capture binding")
                errors = validate_record(record)
                if record.confirmed and errors:
                    raise ValueError(f"invalid confirmed HUD truth: {errors}")
                frame_index = record.frame_id
            else:
                record = _record_from_wire(wire)
                frame_index = record.frame_id
            if frame_index in offsets:
                raise ValueError(f"duplicate {kind} frame index")
            offsets[frame_index] = (start, end)
        start = end + 1
    return offsets


@dataclass(frozen=True, slots=True)
class _SessionIndex:
    """session bytes と frame index を持つ軽量 reader。

    annotation 行を要求された時だけ byte slice を decode し、全 snapshot を保持しない。
    """

    session_id: str
    files: CaptureGroundTruthFiles
    truth_offsets: Mapping[int, tuple[int, int]]
    frame_offsets: Mapping[int, tuple[int, int]]
    coco_images: Mapping[int, CocoImageEntry]

    def truth(self, frame_index: int) -> HudTruthRecord | None:
        """一 frame の HUD truth を必要時に読み出す。

        未注釈 frame は None を返し、未確認 draft は GT snapshot へ渡さない。
        """
        span = self.truth_offsets.get(frame_index)
        if span is None:
            return None
        record = HudTruthRecord.from_wire(json.loads(self.files.hud_truth[span[0]:span[1]]))
        return record

    def frame_record(self, frame_index: int) -> FrameRecord:
        """一 frame の capture record を byte offset から復元する。

        metadata が欠けた capture frame を有効な GT として扱わない。
        """
        span = self.frame_offsets.get(frame_index)
        if span is None:
            raise ValueError(f"frames.jsonl is missing frame {frame_index}")
        return _record_from_wire(json.loads(self.files.frames_jsonl[span[0]:span[1]]))


def _index_session(session_id: str, files: CaptureGroundTruthFiles) -> _SessionIndex:
    """三つの verified capture files を frame index へ束ねる。

    COCO image と frame record は object hash を ground_truth_snapshot 内で再照合する。
    """
    if not isinstance(files, CaptureGroundTruthFiles):
        raise TypeError("capture ground truth files have an invalid type")
    if not all(isinstance(value, bytes) for value in (files.hud_truth, files.world_coco, files.frames_jsonl)):
        raise TypeError("capture ground truth file contents must be bytes")
    try:
        coco_wire = json.loads(files.world_coco)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("world_coco.json is invalid UTF-8 JSON") from exc
    return _SessionIndex(
        session_id, files,
        _jsonl_offsets(files.hud_truth, "hud_truth", session_id),
        _jsonl_offsets(files.frames_jsonl, "frames", session_id),
        coco_image_entries_from_wire(coco_wire, session_id=session_id),
    )


class _FormalGroundTruthLoader:
    """verified capture inputs を frame 単位で GT snapshot に変換する。

    session ごとの parser hash、class map、JSONL offset index を初期化時に固定する。
    """

    def __init__(
        self,
        sessions: Mapping[str, _SessionIndex],
        parser_artifact_hash: str,
    ) -> None:
        """verified package identity と session index を保存する。

        loader 作成後に別 parser hash や未束縛 bytes を差し込めないようにする。
        """
        self._sessions = dict(sessions)
        self._parser_artifact_hash = parser_artifact_hash
        self._class_map = default_class_map()

    def load_frame(
        self,
        frame: CapturedFrame,
        session_id: str,
        frame_index: int,
    ):
        """capture frame を同じ index の verified HUD truth/COCO へ結合する。

        truth row または COCO image がない frame は formal GT として返さない。
        """
        if not isinstance(frame, CapturedFrame) or type(frame_index) is not int or frame_index < 0:
            raise ValueError("capture frame and non-negative frame_index are required")
        if frame.session_frame_index != frame_index:
            raise ValueError("capture frame index does not match loader request")
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError("session has no verified ground truth bytes")
        frame_record = session.frame_record(frame_index)
        if frame_record.captured_monotonic_ns != frame.captured_monotonic_ns:
            raise ValueError("frames.jsonl timestamp does not match captured frame")
        truth = session.truth(frame_index)
        coco_image = session.coco_images.get(frame_record.frame_id)
        return ground_truth_snapshot(
            truth, coco_image, frame_record,
            parser_artifact_hash=self._parser_artifact_hash,
            session_id=session_id, class_map=self._class_map,
        )


def build_ground_truth_loader(
    annotation_payloads: Mapping[str, bytes],
    payloads: Mapping[str, Mapping[str, object]],
    capture_files: Mapping[str, CaptureGroundTruthFiles],
) -> _FormalGroundTruthLoader:
    """verified session bytes から formal ground-truth loader を作る。

    parser artifact hash は package manifest から固定し、annotation と capture の session を揃える。
    """
    if not isinstance(annotation_payloads, Mapping) or not isinstance(capture_files, Mapping):
        raise TypeError("annotation_payloads and capture_files must be mappings")
    if not set(annotation_payloads) <= set(capture_files):
        raise ValueError("annotation session has no verified capture ground truth")
    parser = payloads.get("parser_package")
    if not isinstance(parser, Mapping):
        raise ValueError("parser_package payload is required")
    parser_hash = parser.get("parser_artifact_hash")
    if (
        type(parser_hash) is not str or len(parser_hash) != 64
        or any(character not in "0123456789abcdef" for character in parser_hash)
    ):
        raise ValueError("parser_package parser_artifact_hash must be a SHA-256")
    sessions = {
        session_id: _index_session(session_id, files)
        for session_id, files in capture_files.items()
    }
    return _FormalGroundTruthLoader(sessions, parser_hash)

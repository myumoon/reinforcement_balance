"""Survivors perception benchmark の typed session split 検証。"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Mapping, Sequence

from .capture_dataset import SPLIT_NAMES, SessionManifest, SplitManifest

SessionKind = Literal[
    "model_train", "model_validation", "error_calibration", "final_e2e_test"
]
SourcePolicy = Literal["raw", "lossless", "mp4"]

_CALIBRATION_KIND: Final[str] = "error_calibration"
_FINAL_KIND: Final[str] = "final_e2e_test"
_BENCHMARK_KINDS: Final[frozenset[str]] = frozenset({_CALIBRATION_KIND, _FINAL_KIND})
_ALL_KINDS: Final[frozenset[str]] = frozenset(SPLIT_NAMES)
_ALLOWED_SOURCE_POLICIES: Final[frozenset[str]] = frozenset({"raw", "lossless", "mp4"})
_MIN_SESSION_SECONDS: Final[float] = 1800.0
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


class SplitOverlapError(ValueError):
    """session identity/content が split 内外で重複している。"""


class DuplicateFrameError(SplitOverlapError):
    """個別 frame content または元 capture lineage が split 間で重複している。"""


class MixedBuildError(ValueError):
    """benchmark session の build/profile/resolution が混在している。"""


class UnderpoweredSliceError(ValueError):
    """benchmark split の session cluster 数が最低数を下回る。"""


class ShortSessionError(ValueError):
    """benchmark session の収録時間が 30 分未満である。"""


class ClockRegressionError(ValueError):
    """frame timestamp が strict monotonic でない。"""


class MissingBenchmarkSplitError(ValueError):
    """calibration/final split の片側が欠落している。"""


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def _require_nonempty(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """検証済み manifest から作る benchmark session の不変 view。"""

    session_id: str
    session_hash: str
    build_hash: str
    target_profile_hash: str
    resolution_wh: tuple[int, int]
    duration_seconds: float
    kind: SessionKind
    source_policy: SourcePolicy
    expected_frame_ids: tuple[str, ...] = ()
    session_manifest_path: Path | None = None
    # session_hash はmanifest SHA（session_id に依存）。
    # content_fingerprint はframe object hash列（session_id非依存）でコピー検出に使う。
    content_fingerprint: str = ""
    # 個別 frame 単位の content hash と、transcode 前の capture identity。
    # 集合全体の fingerprint だけでは部分重複を検出できないため両方を保持する。
    frame_content_hashes: tuple[str, ...] = ()
    capture_lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.session_id, "session_id")
        _require_sha256(self.session_hash, "session_hash")
        if self.content_fingerprint:
            _require_sha256(self.content_fingerprint, "content_fingerprint")
        for label, values in (
            ("frame_content_hashes", self.frame_content_hashes),
            ("capture_lineage", self.capture_lineage),
        ):
            if not isinstance(values, tuple):
                raise ValueError(f"{label} must be a tuple")
            for index, value in enumerate(values):
                _require_sha256(value, f"{label}[{index}]")
        # frame_content_hashes（生ピクセル hash）は pause/menu 等の正当な静止 frame で
        # session 内に重複しうるため一意性を要求しない。source 由来の capture_lineage だけ
        # frame ごとに異なる immutable identity を要求し、session 内一意にする。
        if len(self.capture_lineage) != len(set(self.capture_lineage)):
            raise DuplicateFrameError("duplicate value within capture_lineage")
        if self.frame_content_hashes and len(self.frame_content_hashes) != len(
            self.capture_lineage
        ):
            raise ValueError(
                "frame_content_hashes and capture_lineage must have equal lengths"
            )
        _require_sha256(self.build_hash, "build_hash")
        _require_sha256(self.target_profile_hash, "target_profile_hash")
        if self.kind not in _ALL_KINDS:
            raise ValueError(f"unsupported session kind {self.kind!r}")
        if self.source_policy not in _ALLOWED_SOURCE_POLICIES:
            raise ValueError(f"unsupported source policy {self.source_policy!r}")
        if (
            not isinstance(self.resolution_wh, tuple)
            or len(self.resolution_wh) != 2
            or any(type(value) is not int or value <= 0 for value in self.resolution_wh)
        ):
            raise ValueError("resolution_wh must contain two positive integers")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(float(self.duration_seconds))
            or float(self.duration_seconds) < 0.0
        ):
            raise ValueError("duration_seconds must be a finite non-negative number")
        object.__setattr__(self, "duration_seconds", float(self.duration_seconds))
        if (
            not isinstance(self.expected_frame_ids, tuple)
            or not all(type(value) is str and value for value in self.expected_frame_ids)
            or len(self.expected_frame_ids) != len(set(self.expected_frame_ids))
        ):
            raise ValueError("expected_frame_ids must be a unique non-empty string tuple")


@dataclass(frozen=True, slots=True)
class SessionSplit:
    """lineage と内容を検証済みの split。"""

    sessions: tuple[SessionRecord, ...]
    split_manifest_hash: str = ""

    @property
    def expected_ticks(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (session.session_id, frame_id)
            for session in self.sessions
            if session.kind in _BENCHMARK_KINDS
            for frame_id in session.expected_frame_ids
        )


def validate_frame_timestamps(timestamps_ns: Sequence[int]) -> None:
    """全 timestamp の型・非負・strict monotonicity を検証する。"""
    for index, timestamp in enumerate(timestamps_ns):
        if type(timestamp) is not int or timestamp < 0:
            raise ClockRegressionError(
                f"invalid captured timestamp at frame {index}: {timestamp!r}"
            )
        if index and timestamp <= timestamps_ns[index - 1]:
            raise ClockRegressionError(
                f"clock regression at frame {index}: "
                f"{timestamp} <= {timestamps_ns[index - 1]}"
            )


def _source_policy(manifest: SessionManifest) -> SourcePolicy:
    if manifest.pixel_source == "lossless_png":
        return "lossless"
    if manifest.pixel_source in _ALLOWED_SOURCE_POLICIES:
        return manifest.pixel_source  # type: ignore[return-value]
    raise ValueError(f"unsupported SessionManifest.pixel_source {manifest.pixel_source!r}")


def _record_from_manifest(
    kind: str,
    unit_session_hash: str,
    manifest: SessionManifest,
) -> SessionRecord:
    if not isinstance(manifest, SessionManifest):
        raise TypeError("session_manifests must contain SessionManifest values")
    if manifest.schema_version != 1:
        raise ValueError("unsupported SessionManifest schema_version")
    _require_nonempty(manifest.session_id, "SessionManifest.session_id")
    _require_sha256(manifest.manifest_sha256, "SessionManifest.manifest_sha256")
    _require_sha256(manifest.target_profile_hash, "SessionManifest.target_profile_hash")
    _require_sha256(manifest.metadata_sha256, "SessionManifest.metadata_sha256")
    if manifest.manifest_sha256 != unit_session_hash:
        raise ValueError(
            f"split/session manifest lineage mismatch for {manifest.session_id!r}"
        )
    manifest_path = manifest.session_path / "session_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing restored session manifest file: {manifest_path}")
    import hashlib

    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest.manifest_sha256:
        raise ValueError("SessionManifest content changed after restore")
    if type(manifest.frame_count) is not int or manifest.frame_count <= 0:
        raise ValueError("SessionManifest.frame_count must be a positive integer")
    if len(manifest.frame_records) != manifest.frame_count:
        raise ValueError("SessionManifest frame_count/frame_records mismatch")

    timestamps = [record.captured_monotonic_ns for record in manifest.frame_records]
    validate_frame_timestamps(timestamps)
    frame_ids = [record.frame_id for record in manifest.frame_records]
    if any(type(value) is not int or value < 0 for value in frame_ids):
        raise ValueError("frame_id must be a non-negative integer")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("frame_id must be unique within a session")

    resolutions: set[tuple[int, int]] = set()
    for record in manifest.frame_records:
        _require_sha256(record.object_sha256, "FrameRecord.object_sha256")
        if (
            record.target_profile_hash != manifest.target_profile_hash
            or record.game_build_id != manifest.game_build_id
        ):
            raise ValueError("frame lineage does not match SessionManifest")
        left, top, right, bottom = record.client_rect_screen_px
        resolution = (right - left, bottom - top)
        if any(type(value) is not int or value <= 0 for value in resolution):
            raise ValueError("frame resolution must contain positive integers")
        resolutions.add(resolution)
    if len(resolutions) != 1:
        raise MixedBuildError("resolution changes within one session")

    duration_seconds = (timestamps[-1] - timestamps[0]) / 1_000_000_000.0
    # formal capture は実 build artifact SHA を game_build_id に保持する。旧 development
    # capture の human-readable id だけ canonical hash 化して互換性を維持する。
    build_hash = (
        manifest.game_build_id
        if _SHA256_RE.fullmatch(manifest.game_build_id)
        else hashlib.sha256(manifest.game_build_id.encode("utf-8")).hexdigest()
    )
    # session_id・manifest SHA に非依存なフレーム内容フィンガープリント（コピー検出用）
    sorted_frames = sorted(manifest.frame_records, key=lambda r: r.frame_id)
    frame_content_hashes = tuple(r.object_sha256 for r in sorted_frames)
    content_fingerprint = hashlib.sha256(
        b"".join(bytes.fromhex(value) for value in frame_content_hashes)
    ).hexdigest()
    # capture_lineage は session metadata から再生成せず、frame 固有の immutable source
    # identity（stored pixel content hash + 収録 monotonic timestamp）から導出する。
    # timestamp は session 内で strict monotonic に検証済みなので session 内で一意になり、
    # 同じ source frame を別 session の metadata へ入れ替えても lineage が保存される。
    capture_lineage = tuple(
        hashlib.sha256(
            b"\0".join(
                (
                    record.object_sha256.encode("ascii"),
                    str(record.captured_monotonic_ns).encode("ascii"),
                )
            )
        ).hexdigest()
        for record in sorted_frames
    )
    return SessionRecord(
        session_id=manifest.session_id,
        session_hash=manifest.manifest_sha256,
        build_hash=build_hash,
        target_profile_hash=manifest.target_profile_hash,
        resolution_wh=next(iter(resolutions)),
        duration_seconds=duration_seconds,
        kind=kind,  # type: ignore[arg-type]
        source_policy=_source_policy(manifest),
        expected_frame_ids=tuple(str(value) for value in frame_ids),
        session_manifest_path=manifest_path,
        content_fingerprint=content_fingerprint,
        frame_content_hashes=frame_content_hashes,
        capture_lineage=capture_lineage,
    )


def _normalize_typed_split(
    split_manifest: SplitManifest,
    session_manifests: Mapping[str, SessionManifest] | Sequence[SessionManifest] | None,
) -> tuple[list[SessionRecord], str]:
    if not isinstance(split_manifest, SplitManifest):
        raise TypeError("split_manifest must be a SplitManifest")
    if split_manifest.schema_version != 1 or split_manifest.frozen is not True:
        raise ValueError("SplitManifest must be frozen schema v1")
    _require_sha256(split_manifest.manifest_sha256, "SplitManifest.manifest_sha256")
    if set(split_manifest.splits) != _ALL_KINDS:
        raise ValueError("SplitManifest split names do not match capture contract")
    if session_manifests is None:
        raise TypeError("session_manifests are required with SplitManifest")
    if isinstance(session_manifests, Mapping):
        manifests = dict(session_manifests)
    else:
        manifests = {manifest.session_id: manifest for manifest in session_manifests}
        if len(manifests) != len(session_manifests):
            raise SplitOverlapError("duplicate SessionManifest session_id")

    records: list[SessionRecord] = []
    referenced_ids: list[str] = []
    for kind in SPLIT_NAMES:
        units = split_manifest.splits[kind]
        if not isinstance(units, tuple):
            raise ValueError("SplitManifest entries must be tuples")
        for unit in units:
            manifest = manifests.get(unit.session_id)
            if manifest is None:
                raise ValueError(f"missing restored SessionManifest {unit.session_id!r}")
            if unit.game_build_id != manifest.game_build_id:
                raise ValueError(
                    f"split/session build lineage mismatch for {unit.session_id!r}"
                )
            referenced_ids.append(unit.session_id)
            records.append(
                _record_from_manifest(kind, unit.session_manifest_sha256, manifest)
            )
    if set(manifests) != set(referenced_ids):
        extra = sorted(set(manifests) - set(referenced_ids))
        raise ValueError(f"unreferenced SessionManifest entries: {extra}")
    return records, split_manifest.manifest_sha256


def _validate_records(
    records: Sequence[SessionRecord], *, min_benchmark_sessions: int
) -> None:
    if type(min_benchmark_sessions) is not int or min_benchmark_sessions <= 0:
        raise ValueError("min_benchmark_sessions must be a positive integer")
    if not all(isinstance(record, SessionRecord) for record in records):
        raise TypeError("sessions must contain SessionRecord values")

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    seen_fingerprints: set[str] = set()
    seen_capture_lineage: dict[str, str] = {}
    for record in records:
        if record.session_id in seen_ids:
            raise SplitOverlapError(f"duplicate session_id {record.session_id!r}")
        if record.session_hash in seen_hashes:
            raise SplitOverlapError(f"duplicate session_hash {record.session_hash!r}")
        # content_fingerprint は session_id 非依存。同一フレーム列を別 session として登録するコピーを検出する。
        if record.content_fingerprint and record.content_fingerprint in seen_fingerprints:
            raise DuplicateFrameError(
                f"duplicate frame content fingerprint for session {record.session_id!r}"
            )
        # 生ピクセル hash 単体の split 間一致は pause/menu 等で正当に起こりうるため
        # 重複検出には使わない。source 由来の capture_lineage の intersection だけで
        # split/session 間の frame 再利用（同一 source-capture の使い回し）を検出する。
        for lineage_hash in record.capture_lineage:
            previous_session = seen_capture_lineage.get(lineage_hash)
            if previous_session is not None:
                raise DuplicateFrameError(
                    "duplicate capture lineage across sessions: "
                    f"{previous_session!r} and {record.session_id!r}"
                )
        seen_ids.add(record.session_id)
        seen_hashes.add(record.session_hash)
        if record.content_fingerprint:
            seen_fingerprints.add(record.content_fingerprint)
        seen_capture_lineage.update(
            (lineage_hash, record.session_id) for lineage_hash in record.capture_lineage
        )

    benchmark = [record for record in records if record.kind in _BENCHMARK_KINDS]
    if benchmark:
        reference = benchmark[0]
        for record in benchmark[1:]:
            if (
                record.build_hash != reference.build_hash
                or record.target_profile_hash != reference.target_profile_hash
                or record.resolution_wh != reference.resolution_wh
            ):
                raise MixedBuildError(
                    f"session {record.session_id!r} has different "
                    "build/profile/resolution"
                )
        for record in benchmark:
            if record.duration_seconds < _MIN_SESSION_SECONDS:
                raise ShortSessionError(
                    f"session {record.session_id!r} is {record.duration_seconds:.1f}s "
                    f"(minimum {_MIN_SESSION_SECONDS:.0f}s)"
                )
            if record.source_policy == "mp4":
                raise ValueError(
                    f"session {record.session_id!r}: mp4 is review-only"
                )

    counts = {
        kind: sum(record.kind == kind for record in records)
        for kind in _BENCHMARK_KINDS
    }
    if any(counts.values()):
        if counts[_CALIBRATION_KIND] == 0:
            raise MissingBenchmarkSplitError("error_calibration sessions are required")
        if counts[_FINAL_KIND] == 0:
            raise MissingBenchmarkSplitError("final_e2e_test sessions are required")
        for kind, count in counts.items():
            if count < min_benchmark_sessions:
                raise UnderpoweredSliceError(
                    f"{kind} has only {count} session(s), "
                    f"minimum is {min_benchmark_sessions}"
                )


def validate_split(
    split_manifest: SplitManifest | Sequence[SessionRecord],
    session_manifests: Mapping[str, SessionManifest] | Sequence[SessionManifest] | None = None,
    *,
    min_benchmark_sessions: int = 3,
) -> SessionSplit:
    """typed capture manifests を restore 後の内容と照合して split を seal する。

    `Sequence[SessionRecord]` は既存 synthetic unit test 用の互換入口であり、
    formal runner は必ず `SplitManifest` / `SessionManifest` 経路を使用する。
    両入口は最終的に `_validate_records` の同じ fail-closed gate を通る。
    """
    if isinstance(split_manifest, SplitManifest):
        records, split_hash = _normalize_typed_split(split_manifest, session_manifests)
    else:
        if session_manifests is not None:
            raise TypeError("session_manifests can only be used with SplitManifest")
        records = list(split_manifest)
        split_hash = ""
    _validate_records(records, min_benchmark_sessions=min_benchmark_sessions)
    return SessionSplit(tuple(records), split_hash)


def validate_source_policy_consistency(
    sessions: Sequence[SessionRecord], *, benchmark_kind: SessionKind
) -> None:
    """指定 benchmark kind の source policy を同じ規則で再検証する。"""
    if benchmark_kind not in _ALL_KINDS:
        raise ValueError("unsupported benchmark_kind")
    for session in sessions:
        if session.kind == benchmark_kind and session.source_policy == "mp4":
            raise ValueError(
                f"session {session.session_id!r}: mp4 decode not allowed as benchmark input"
            )

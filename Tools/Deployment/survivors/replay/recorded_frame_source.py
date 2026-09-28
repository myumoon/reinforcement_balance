"""記録済み capture session を CapturedFrame 列として決定的に再生する capture 代替。

記録された「controller が poll した結果」の列(frame / duplicate / timeout / focus_lost)を、
記録時の完了順(completion_seq)どおり1件ずつ返します。受け取った event の並びが
async queue の都合で入れ替わっていても、completion_seq で並べ直すので結果は変わりません。
PyTorch の決定性設定(DeterminismManifest)も session に固定し、再生環境との食い違いを拒否します。
実時計と実入力デバイス(survivors.input)には一切依存しません。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..capture.captured_frame import CapturedFrame
from ..capture.frame_capture import LatestFrameQueue
from .virtual_clock import VirtualClock

EVENT_KINDS = ("frame", "duplicate", "timeout", "focus_lost")
_FRAME_KINDS = ("frame", "duplicate")


class DeterminismMismatchError(ValueError):
    """記録時と再生時の決定性設定が一致しないときに送出する。"""


def _from_mapping(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    """dataclass cls のフィールドと data のキーが完全一致することを確かめて dict で返す。

    未知キーや欠けたキーを黙って捨てると記録の取り違えに気付けないため、どちらも拒否します。
    """
    names = {f.name for f in fields(cls)}
    if set(data) != names:
        raise ValueError(f"{cls.__name__} keys mismatch: expected {sorted(names)}, got {sorted(data)}")
    return dict(data)


@dataclass(frozen=True)
class DeterminismManifest:
    """推論結果を左右する PyTorch 設定(決定的アルゴリズム・device・NMS・thread 数)の記録。

    同じ frame でも device や thread 数が違うと数値がずれ、NMS の実装差で検出結果が変わります。
    記録時の値を session に保存し、再生前に ``require_match()`` で今の環境と突き合わせます。
    """

    deterministic_algorithms: bool
    device: str
    nms_backend: str
    num_threads: int
    num_interop_threads: int

    def __post_init__(self) -> None:
        """各フィールドの型と値域を検証する。"""
        if type(self.deterministic_algorithms) is not bool:
            raise ValueError("deterministic_algorithms must be a bool")
        for name in ("device", "nms_backend"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("num_threads", "num_interop_threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_torch(cls, *, device: str, nms_backend: str) -> "DeterminismManifest":
        """いま読み込まれている torch の決定性設定と thread 数を読み取って manifest を作る。

        device と NMS 実装は torch の global 設定から分からないため、呼び出し側が指定します。
        """
        import torch  # 重い import なので実際に使うときだけ読み込む

        return cls(
            deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
            device=device,
            nms_backend=nms_backend,
            num_threads=int(torch.get_num_threads()),
            num_interop_threads=int(torch.get_num_interop_threads()),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeterminismManifest":
        """JSON 由来の dict から manifest を復元する(キーの過不足は拒否)。"""
        return cls(**_from_mapping(cls, data))

    def to_dict(self) -> dict[str, Any]:
        """JSON へ書ける dict に変換する。"""
        return asdict(self)

    def require_match(self, current: "DeterminismManifest") -> None:
        """current(再生環境)が記録時の設定と完全一致しなければ DeterminismMismatchError を送出する。"""
        diffs = [
            f"{f.name}: recorded={getattr(self, f.name)!r} replay={getattr(current, f.name)!r}"
            for f in fields(self)
            if getattr(self, f.name) != getattr(current, f.name)
        ]
        if diffs:
            raise DeterminismMismatchError("determinism manifest mismatch: " + "; ".join(diffs))


@dataclass(frozen=True)
class RecordedEvent:
    """controller の poll 1回分の記録(何が届いたか・いつ・何番目に完了したか)。

    kind は次の4種です。
    - frame: 新しい frame が届いた(session_frame_index と correlation_id が必須)
    - duplicate: 既に届けた frame がもう一度届いた(controller は stale として捨てる)
    - timeout: 何も届かなかった
    - focus_lost: target window がフォアグラウンドを失い capture が一時停止した
    取りこぼし(drop)は frame 番号の飛びとして表現します。
    """

    completion_seq: int
    kind: str
    timestamp_ns: int
    session_frame_index: int | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        """kind ごとに必要なフィールドが揃っているかを検証する。"""
        if type(self.completion_seq) is not int or self.completion_seq < 0:
            raise ValueError("completion_seq must be a non-negative integer")
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"kind must be one of {EVENT_KINDS}")
        if type(self.timestamp_ns) is not int or self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a non-negative integer")
        has_frame = self.session_frame_index is not None or self.correlation_id is not None
        if self.kind in _FRAME_KINDS:
            if type(self.session_frame_index) is not int or self.session_frame_index < 0:
                raise ValueError(f"{self.kind} event needs a non-negative session_frame_index")
            if not isinstance(self.correlation_id, str) or not self.correlation_id:
                raise ValueError(f"{self.kind} event needs a correlation_id")
        elif has_frame:
            raise ValueError(f"{self.kind} event must not carry frame index or correlation_id")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecordedEvent":
        """JSON 由来の dict から event を復元する(キーの過不足は拒否)。"""
        return cls(**_from_mapping(cls, data))


@dataclass(frozen=True)
class RecordedSession:
    """1回分の記録済み capture session(target の同一性・決定性設定・poll event 列)。

    events は受け取った順ではなく completion_seq(記録時に poll が完了した順)で並べ直して保持します。
    async queue の都合で event の到着順が入れ替わった fixture でも、同じ再生結果になります。
    completion_seq の重複、時刻の逆行、correlation_id と frame 番号の食い違い、
    まだ届いていない frame の duplicate は、記録の破損として構築時に拒否します。
    """

    session_id: str
    target_profile_hash: str
    game_build_id: str
    client_rect_screen_px: tuple[int, int, int, int]
    determinism: DeterminismManifest
    events: tuple[RecordedEvent, ...]

    def __post_init__(self) -> None:
        """events を completion_seq 順に並べ直し、session 全体の整合性を検証する。"""
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        if not isinstance(self.determinism, DeterminismManifest):
            raise ValueError("determinism must be a DeterminismManifest")
        ordered = tuple(sorted(self.events, key=lambda event: event.completion_seq))
        object.__setattr__(self, "events", ordered)
        seqs = [event.completion_seq for event in ordered]
        if len(set(seqs)) != len(seqs):
            raise ValueError("completion_seq must be unique")
        last_ns = 0
        delivered: set[int] = set()
        last_index = -1
        for event in ordered:
            if event.timestamp_ns < last_ns:
                raise ValueError("timestamp_ns must be non-decreasing in completion order")
            last_ns = event.timestamp_ns
            if event.kind not in _FRAME_KINDS:
                continue
            index = event.session_frame_index
            if event.correlation_id != f"{self.session_id}:{index}":
                raise ValueError(f"correlation_id {event.correlation_id!r} does not match frame {index}")
            if event.kind == "frame":
                if index <= last_index:
                    raise ValueError("frame indices must strictly increase; use kind='duplicate' for re-delivery")
                last_index = index
                delivered.add(index)
            elif index not in delivered:
                raise ValueError(f"duplicate event references undelivered frame {index}")

    @property
    def dropped_indices(self) -> tuple[int, ...]:
        """frame 番号の飛びから分かる取りこぼし frame 番号(controller と同じく -1 起点で数える)。"""
        dropped: list[int] = []
        last_index = -1
        for event in self.events:
            if event.kind == "frame":
                dropped.extend(range(last_index + 1, event.session_frame_index))
                last_index = event.session_frame_index
        return tuple(dropped)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecordedSession":
        """JSON 由来の dict から session を復元する(キーの過不足は拒否)。"""
        values = _from_mapping(cls, data)
        values["client_rect_screen_px"] = tuple(values["client_rect_screen_px"])
        values["determinism"] = DeterminismManifest.from_dict(values["determinism"])
        values["events"] = tuple(RecordedEvent.from_dict(event) for event in values["events"])
        return cls(**values)


class RecordedFrameSource:
    """RecordedSession を CaptureSession と同じ duck-type(start/capture_next/frames/paused/close)で返す。

    ``capture_next()`` のたびに event を1件進め、仮想時計を記録時刻まで進めてから
    frame を LatestFrameQueue へ積みます。画素は ``pixels(session_frame_index)`` から読み込みます
    (実 frame は local artifact、synthetic fixture は同じ配列の使い回しで構いません)。
    event を使い切ると ``exhausted`` が True になり、以後は None を返し続けます。
    """

    def __init__(
        self,
        session: RecordedSession,
        clock: VirtualClock,
        pixels: Callable[[int], NDArray[np.uint8]],
    ) -> None:
        """session・仮想時計・画素ローダーを受け取り、再生位置を先頭にする。"""
        self.session = session
        self.frames = LatestFrameQueue()
        self.started = False
        self.closed = False
        self._clock = clock
        self._pixels = pixels
        self._position = 0
        self._paused = False
        # duplicate 再送用に frame 番号→記録時刻だけを覚える(画素は毎回ローダーから読む)
        self._delivered_ns: dict[int, int] = {}

    @property
    def paused(self) -> bool:
        """focus_lost 後、次の frame が届くまで True(実 CaptureSession.paused と同じ意味)。"""
        return self._paused

    @property
    def exhausted(self) -> bool:
        """記録済み event を全て返し終えたかどうか。"""
        return self._position >= len(self.session.events)

    def start(self) -> None:
        """controller から呼ばれる開始通知。"""
        self.started = True

    def capture_next(self) -> CapturedFrame | None:
        """次の event を再生し、frame を積んだらそれを、そうでなければ None を返す。"""
        if self.exhausted:
            return None
        event = self.session.events[self._position]
        self._position += 1
        self._clock.advance_to(event.timestamp_ns)
        if event.kind == "focus_lost":
            self._paused = True
            return None
        if event.kind == "timeout":
            return None
        self._paused = False
        index = event.session_frame_index
        captured_ns = self._delivered_ns.setdefault(index, event.timestamp_ns)
        frame = CapturedFrame(
            frame_bgra=self._pixels(index),
            captured_monotonic_ns=captured_ns,
            session_frame_index=index,
            client_rect_screen_px=self.session.client_rect_screen_px,
            foreground=True,
            target_profile_hash=self.session.target_profile_hash,
            game_build_id=self.session.game_build_id,
        )
        self.frames.put_latest(frame)
        return frame

    def close(self) -> None:
        """controller から呼ばれる停止通知(queue の中身は controller が drain する)。"""
        self.closed = True

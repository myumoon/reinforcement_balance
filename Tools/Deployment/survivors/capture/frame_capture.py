"""target client 領域だけを 30 FPS BGRA で取得する latest-only session。

各取得の前後で window 状態を再検証し、focus・解像度・同一性が変わった frame は consumer へ渡しません。
"""

from __future__ import annotations

import importlib
from queue import Empty, Full, Queue
import threading
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .captured_frame import CapturedFrame
from .window_locator import LocatedTargetWindow, ScreenRect, WindowLocator


TARGET_FPS = 30
EXPECTED_FRAME_SHAPE = (1080, 1920, 4)


class CaptureFrameError(RuntimeError):
    pass


@runtime_checkable
class CaptureBackend(Protocol):
    def start(self, *, region: ScreenRect, target_fps: int) -> None: ...

    def get_latest_frame(self) -> tuple[NDArray[np.uint8], int] | None: ...

    def stop(self) -> None: ...

    def release(self) -> None: ...


class DxcamCaptureBackend:
    """DXGI output と client rect に一意に束縛された DXcam capture backend。

    output_idx と expected_client_rect を生成時に固定し、
    start() でその exact region 以外を拒否して誤 output capture を防ぎます。
    """

    def __init__(self, camera: object, expected_client_rect: ScreenRect) -> None:
        self._camera = camera
        self._expected_client_rect = expected_client_rect

    @classmethod
    def create(
        cls,
        output_idx: int,
        expected_client_rect: ScreenRect,
        *,
        device_idx: int = 0,
        dxcam_module=None,
    ) -> "DxcamCaptureBackend":
        """device_idx/output_idx と expected_client_rect を bundle した backend を生成する。

        output_idx は MonitorInfo.dxgi_output_idx (per-device index)、
        device_idx は MonitorInfo.dxgi_device_idx を渡す。
        expected_client_rect と異なる region を start() に渡すと例外になります。
        """
        if not isinstance(output_idx, int) or output_idx < 0:
            raise ValueError("output_idx must be a non-negative integer")
        if not isinstance(device_idx, int) or device_idx < 0:
            raise ValueError("device_idx must be a non-negative integer")
        if (
            not isinstance(expected_client_rect, tuple)
            or len(expected_client_rect) != 4
            or not all(type(v) is int for v in expected_client_rect)
            or (
                expected_client_rect[2] - expected_client_rect[0],
                expected_client_rect[3] - expected_client_rect[1],
            )
            != (1920, 1080)
        ):
            raise ValueError("expected_client_rect must be a 1920x1080 screen rect")
        module = dxcam_module or importlib.import_module("dxcam")
        camera = module.create(
            device_idx=device_idx,
            output_idx=output_idx,
            backend="dxgi",
            processor_backend="numpy",
            output_color="BGRA",
            max_buffer_len=1,
        )
        return cls(camera, expected_client_rect)

    def start(self, *, region: ScreenRect, target_fps: int) -> None:
        if target_fps != TARGET_FPS:
            raise ValueError("DXcam target_fps must be 30")
        if region != self._expected_client_rect:
            raise ValueError(
                f"capture region {region!r} does not match bound client rect "
                f"{self._expected_client_rect!r}"
            )
        self._camera.start(region=region, target_fps=TARGET_FPS)

    def get_latest_frame(self) -> tuple[NDArray[np.uint8], int] | None:
        result = self._camera.get_latest_frame(with_timestamp=True)
        if result is None:
            return None
        frame, ts = result
        return frame, ts

    def stop(self) -> None:
        self._camera.stop()

    def release(self) -> None:
        self._camera.release()


class LatestFrameQueue:
    def __init__(self) -> None:
        self._queue: Queue[CapturedFrame] = Queue(maxsize=1)
        self._producer_lock = threading.Lock()

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize

    def put_latest(self, frame: CapturedFrame) -> None:
        with self._producer_lock:
            while True:
                try:
                    self._queue.put_nowait(frame)
                    return
                except Full:
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        continue

    def get_latest_nowait(self) -> CapturedFrame:
        return self._queue.get_nowait()

    def get_latest(self, timeout: float | None = None) -> CapturedFrame:
        return self._queue.get(timeout=timeout)

    def clear(self) -> None:
        """queue 内の未消費 frame を全て破棄する。close 後に stale frame が読まれるのを防ぐ。"""
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return


class CaptureSession:
    def __init__(
        self,
        locator: WindowLocator,
        target: LocatedTargetWindow,
        backend: CaptureBackend,
        *,
        frame_queue: LatestFrameQueue | None = None,
    ) -> None:
        self._locator = locator
        self.target = target
        self._backend = backend
        self.frames = frame_queue or LatestFrameQueue()
        self._started = False
        self._closed = False
        self._next_frame_index = 0
        self._last_monotonic_ns: int | None = None

    def start(self) -> None:
        if self._closed:
            raise CaptureFrameError("capture session is closed")
        if self._started:
            raise CaptureFrameError("capture session is already started")
        self._locator.validate(self.target, require_foreground=True)
        self._backend.start(
            region=self.target.client_rect_screen_px,
            target_fps=TARGET_FPS,
        )
        try:
            self._locator.validate(self.target, require_foreground=True)
        except Exception:
            self._closed = True  # close() による二重 release を防ぐ
            try:
                self._backend.stop()
            finally:
                self._backend.release()
            raise
        self._started = True

    def capture_next(self) -> CapturedFrame | None:
        if not self._started or self._closed:
            raise CaptureFrameError("capture session is not active")
        # P2-1: DXGI 再列挙を避ける軽量検証を frame ごとに使う
        self._locator.validate_lightweight(self.target, require_foreground=True)
        backend_result = self._backend.get_latest_frame()
        self._locator.validate_lightweight(self.target, require_foreground=True)
        if backend_result is None:
            return None
        frame_bgra, captured_ns = backend_result  # P2-3: backend が capture 時刻を持つ
        self._validate_frame(frame_bgra)

        if (
            type(captured_ns) is not int
            or captured_ns < 0
            or (
                self._last_monotonic_ns is not None
                and captured_ns <= self._last_monotonic_ns
            )
        ):
            raise CaptureFrameError("captured monotonic timestamp did not increase")
        captured = CapturedFrame(
            frame_bgra=frame_bgra,
            captured_monotonic_ns=captured_ns,
            session_frame_index=self._next_frame_index,
            client_rect_screen_px=self.target.client_rect_screen_px,
            foreground=True,
            target_profile_hash=self.target.target_profile_hash,
            game_build_id=self.target.game_build_id,
        )
        self.frames.put_latest(captured)
        self._last_monotonic_ns = captured_ns
        self._next_frame_index += 1
        return captured

    def close(self) -> None:
        if self._closed:
            return
        was_started = self._started
        self._closed = True
        self._started = False
        self.frames.clear()  # P1-2: stale frame を consumer が読めないよう破棄
        try:
            if was_started:
                self._backend.stop()
        finally:
            self._backend.release()

    @staticmethod
    def _validate_frame(frame_bgra: object) -> None:
        if (
            not isinstance(frame_bgra, np.ndarray)
            or frame_bgra.dtype != np.uint8
            or frame_bgra.shape != EXPECTED_FRAME_SHAPE
        ):
            raise CaptureFrameError("backend frame must be 1920x1080 uint8 BGRA")

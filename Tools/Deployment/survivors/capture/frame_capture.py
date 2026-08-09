"""target client 領域だけを 30 FPS BGRA で取得する latest-only session。

各取得の前後で window 状態を再検証し、focus・解像度・同一性が変わった frame は consumer へ渡しません。
"""

from __future__ import annotations

import importlib
from queue import Empty, Full, Queue
import threading
import time
from typing import Callable, Protocol, runtime_checkable

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

    def get_latest_frame(self) -> NDArray[np.uint8] | None: ...

    def stop(self) -> None: ...

    def release(self) -> None: ...


class DxcamCaptureBackend:
    def __init__(self, camera: object) -> None:
        self._camera = camera

    @classmethod
    def create(cls, *, dxcam_module=None) -> "DxcamCaptureBackend":
        module = dxcam_module or importlib.import_module("dxcam")
        camera = module.create(
            backend="dxgi",
            processor_backend="numpy",
            output_color="BGRA",
            max_buffer_len=1,
        )
        return cls(camera)

    def start(self, *, region: ScreenRect, target_fps: int) -> None:
        if target_fps != TARGET_FPS:
            raise ValueError("DXcam target_fps must be 30")
        if (
            not isinstance(region, tuple)
            or len(region) != 4
            or not all(type(value) is int for value in region)
            or (region[2] - region[0], region[3] - region[1]) != (1920, 1080)
        ):
            raise ValueError("DXcam requires an explicit target region")
        self._camera.start(region=region, target_fps=TARGET_FPS)

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        return self._camera.get_latest_frame()

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


class CaptureSession:
    def __init__(
        self,
        locator: WindowLocator,
        target: LocatedTargetWindow,
        backend: CaptureBackend,
        *,
        frame_queue: LatestFrameQueue | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._locator = locator
        self.target = target
        self._backend = backend
        self.frames = frame_queue or LatestFrameQueue()
        self._monotonic_ns = monotonic_ns
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
            self._backend.stop()
            raise
        self._started = True

    def capture_next(self) -> CapturedFrame | None:
        if not self._started or self._closed:
            raise CaptureFrameError("capture session is not active")
        self._locator.validate(self.target, require_foreground=True)
        frame_bgra = self._backend.get_latest_frame()
        self._locator.validate(self.target, require_foreground=True)
        if frame_bgra is None:
            return None
        self._validate_frame(frame_bgra)

        captured_ns = self._monotonic_ns()
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
        if self._started:
            self._backend.stop()
        self._backend.release()
        self._closed = True
        self._started = False

    @staticmethod
    def _validate_frame(frame_bgra: object) -> None:
        if (
            not isinstance(frame_bgra, np.ndarray)
            or frame_bgra.dtype != np.uint8
            or frame_bgra.shape != EXPECTED_FRAME_SHAPE
        ):
            raise CaptureFrameError("backend frame must be 1920x1080 uint8 BGRA")

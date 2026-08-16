"""Survivors 本家画面を識別して低遅延で取得する公開 API。

後続の perception consumer は、この package の CapturedFrame 契約だけに依存します。
"""

from .captured_frame import CapturedFrame
from .frame_capture import (
    CaptureBackend,
    CaptureFrameError,
    CaptureSession,
    DxcamCaptureBackend,
    LatestFrameQueue,
)
from .window_locator import (
    CtypesWin32Api,
    LocatedTargetWindow,
    MonitorInfo,
    TargetWindowNotFound,
    TargetWindowPolicy,
    TargetWindowStateError,
    Win32Api,
    WindowLocator,
)

__all__ = [
    "CaptureBackend",
    "CaptureFrameError",
    "CaptureSession",
    "CapturedFrame",
    "CtypesWin32Api",
    "DxcamCaptureBackend",
    "LatestFrameQueue",
    "LocatedTargetWindow",
    "MonitorInfo",
    "TargetWindowNotFound",
    "TargetWindowPolicy",
    "TargetWindowStateError",
    "Win32Api",
    "WindowLocator",
]

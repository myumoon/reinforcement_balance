"""Windows capture core を実機なしで検証する fake 群。

Win32 と DXcam の境界を差し替え、Linux のテスト環境でも対象識別と capture 契約を再現します。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pytest

from survivors.capture.window_locator import MonitorInfo, TargetWindowPolicy
from survivors.target_profile import load_target_profile


@dataclass
class FakeWindow:
    hwnd: int
    pid: int
    executable: str
    window_class: str
    title: str
    client_rect: tuple[int, int, int, int]
    monitor: MonitorInfo
    visible: bool = True


class FakeWin32Api:
    def __init__(self, windows: list[FakeWindow], foreground_hwnd: int | None):
        self.windows = {window.hwnd: window for window in windows}
        self.foreground_hwnd = foreground_hwnd

    def enum_windows(self):
        return tuple(self.windows)

    def is_window(self, hwnd):
        return hwnd in self.windows

    def is_window_visible(self, hwnd):
        return self.windows[hwnd].visible

    def window_process_id(self, hwnd):
        return self.windows[hwnd].pid

    def process_executable(self, pid):
        matches = [window.executable for window in self.windows.values() if window.pid == pid]
        if not matches:
            raise OSError("unknown process")
        return matches[0]

    def window_class(self, hwnd):
        return self.windows[hwnd].window_class

    def window_title(self, hwnd):
        return self.windows[hwnd].title

    def client_rect_screen_px(self, hwnd):
        return self.windows[hwnd].client_rect

    def monitor_info(self, hwnd):
        return self.windows[hwnd].monitor

    def foreground_window(self):
        return self.foreground_hwnd


class FakeCaptureBackend:
    def __init__(self, frames, after_read=None):
        self.frames = deque(frames)
        self.after_read = after_read
        self.started = []
        self.read_count = 0
        self.stopped = False
        self.released = False

    def start(self, *, region, target_fps):
        self.started.append({"region": region, "target_fps": target_fps})

    def get_latest_frame(self):
        self.read_count += 1
        frame = self.frames.popleft() if self.frames else None
        if self.after_read is not None:
            self.after_read()
        return frame

    def stop(self):
        self.stopped = True

    def release(self):
        self.released = True


@pytest.fixture
def profile():
    return load_target_profile()


@pytest.fixture
def policy():
    return TargetWindowPolicy(
        process_executable="VampireSurvivors.exe",
        window_class="YYGameMakerYY",
        window_title="Vampire Survivors",
    )


@pytest.fixture
def monitor():
    return MonitorInfo(
        rect_screen_px=(0, 0, 1920, 1080),
        device_name=r"\\.\DISPLAY1",
        primary=True,
        dxgi_output_idx=0,
    )


@pytest.fixture
def target_window(monitor):
    return FakeWindow(
        hwnd=101,
        pid=2001,
        executable=r"C:\Games\Vampire Survivors\VampireSurvivors.exe",
        window_class="YYGameMakerYY",
        title="Vampire Survivors",
        client_rect=(0, 0, 1920, 1080),
        monitor=monitor,
    )


@pytest.fixture
def fake_api(target_window):
    return FakeWin32Api([target_window], foreground_hwnd=target_window.hwnd)


@pytest.fixture
def golden_bgra():
    return np.zeros((1080, 1920, 4), dtype=np.uint8)

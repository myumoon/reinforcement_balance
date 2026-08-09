"""Win32 window の process・class・title を完全一致で束縛する locator。

OS 呼び出しを Protocol の外側へ隔離し、実機 adapter と fake の双方で同じ fail-closed 判定を使います。
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import ntpath
import sys
from typing import Iterable, Protocol, runtime_checkable

from survivors.target_profile import TargetProfile


ScreenRect = tuple[int, int, int, int]


class TargetWindowError(RuntimeError):
    pass


class TargetWindowNotFound(TargetWindowError):
    pass


class TargetWindowStateError(TargetWindowError):
    pass


@dataclass(frozen=True, slots=True)
class TargetWindowPolicy:
    process_executable: str
    window_class: str
    window_title: str

    def __post_init__(self) -> None:
        values = (self.process_executable, self.window_class, self.window_title)
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError("target window policy values must be non-empty strings")
        if ntpath.basename(self.process_executable) != self.process_executable:
            raise ValueError("process_executable must be an executable basename")


@dataclass(frozen=True, slots=True)
class MonitorInfo:
    rect_screen_px: ScreenRect
    device_name: str
    primary: bool
    dxgi_output_idx: int


@dataclass(frozen=True, slots=True)
class LocatedTargetWindow:
    hwnd: int
    pid: int
    process_executable_path: str
    window_class: str
    window_title: str
    client_rect_screen_px: ScreenRect
    monitor: MonitorInfo
    target_profile_hash: str
    game_build_id: str


@runtime_checkable
class Win32Api(Protocol):
    def enum_windows(self) -> Iterable[int]: ...

    def is_window(self, hwnd: int) -> bool: ...

    def is_window_visible(self, hwnd: int) -> bool: ...

    def window_process_id(self, hwnd: int) -> int: ...

    def process_executable(self, pid: int) -> str: ...

    def window_class(self, hwnd: int) -> str: ...

    def window_title(self, hwnd: int) -> str: ...

    def client_rect_screen_px(self, hwnd: int) -> ScreenRect: ...

    def monitor_info(self, hwnd: int) -> MonitorInfo: ...

    def foreground_window(self) -> int | None: ...


class WindowLocator:
    def __init__(
        self,
        api: Win32Api,
        target_profile: TargetProfile,
        policy: TargetWindowPolicy,
    ) -> None:
        self._api = api
        self._profile = target_profile
        self._policy = policy

    def locate(self) -> LocatedTargetWindow:
        self._validate_capture_profile()
        matches = [
            hwnd
            for hwnd in self._api.enum_windows()
            if self._identity_matches(hwnd)
        ]
        if not matches:
            raise TargetWindowNotFound("target window was not found")
        if len(matches) != 1:
            raise TargetWindowStateError("ambiguous exact target windows")
        return self._observe(matches[0])

    def validate(
        self,
        target: LocatedTargetWindow,
        *,
        require_foreground: bool,
    ) -> LocatedTargetWindow:
        current = self._observe(target.hwnd)
        if current != target:
            raise TargetWindowStateError("target window identity or geometry changed")
        if require_foreground and self._api.foreground_window() != target.hwnd:
            raise TargetWindowStateError("target window lost foreground")
        return current

    def _identity_matches(self, hwnd: int) -> bool:
        try:
            if not self._api.is_window(hwnd) or not self._api.is_window_visible(hwnd):
                return False
            pid = self._api.window_process_id(hwnd)
            executable = self._api.process_executable(pid)
            return (
                type(pid) is int
                and pid > 0
                and self._executable_matches(executable)
                and self._api.window_class(hwnd) == self._policy.window_class
                and self._api.window_title(hwnd) == self._policy.window_title
            )
        except (OSError, ValueError):
            return False

    def _observe(self, hwnd: int) -> LocatedTargetWindow:
        self._validate_capture_profile()
        if not self._api.is_window(hwnd) or not self._api.is_window_visible(hwnd):
            raise TargetWindowStateError("target window is no longer available")
        pid = self._api.window_process_id(hwnd)
        executable = self._api.process_executable(pid)
        window_class = self._api.window_class(hwnd)
        window_title = self._api.window_title(hwnd)
        if (
            type(pid) is not int
            or pid <= 0
            or not self._executable_matches(executable)
            or window_class != self._policy.window_class
            or window_title != self._policy.window_title
        ):
            raise TargetWindowStateError("target window process/class/title changed")

        client_rect = self._api.client_rect_screen_px(hwnd)
        expected_size = self._expected_client_size()
        self._validate_rect(client_rect, "client")
        if self._rect_size(client_rect) != expected_size:
            raise TargetWindowStateError("target client resolution changed")

        monitor = self._api.monitor_info(hwnd)
        self._validate_rect(monitor.rect_screen_px, "monitor")
        if not self._contains(monitor.rect_screen_px, client_rect):
            raise TargetWindowStateError("target window crosses monitor bounds")
        self._validate_monitor_binding(monitor)

        profile_hash = self._profile.target_hash
        build_id = self._profile.sections["build"]["build_id"]
        if not isinstance(profile_hash, str) or not profile_hash:
            raise TargetWindowStateError("target profile hash is invalid")
        if not isinstance(build_id, str) or not build_id:
            raise TargetWindowStateError("game build id is invalid")
        return LocatedTargetWindow(
            hwnd=hwnd,
            pid=pid,
            process_executable_path=executable,
            window_class=window_class,
            window_title=window_title,
            client_rect_screen_px=client_rect,
            monitor=monitor,
            target_profile_hash=profile_hash,
            game_build_id=build_id,
        )

    def _expected_client_size(self) -> tuple[int, int]:
        resolution = self._profile.sections["base"]["client_resolution"]
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or not all(type(value) is int and value > 0 for value in resolution)
            or resolution != [1920, 1080]
        ):
            raise TargetWindowStateError("target profile client resolution is invalid")
        return resolution[0], resolution[1]

    def _validate_capture_profile(self) -> None:
        base = self._profile.sections["base"]
        hardware = self._profile.sections["hardware"]
        if base["platform"] != "windows_win64":
            raise TargetWindowStateError("capture target must be Windows Win64")
        if base["window_mode"] != "borderless":
            raise TargetWindowStateError("capture target must use borderless mode")
        if hardware["capture_backend"] != "dxcam":
            raise TargetWindowStateError("target profile capture backend must be dxcam")

    def _validate_monitor_binding(self, monitor: MonitorInfo) -> None:
        expected_output = self._profile.sections["display"]["monitor_output"]
        if expected_output != "primary" or monitor.primary is not True:
            raise TargetWindowStateError("capture supports only the configured primary monitor")

    def _executable_matches(self, executable_path: str) -> bool:
        return (
            isinstance(executable_path, str)
            and ntpath.basename(executable_path).casefold()
            == self._policy.process_executable.casefold()
        )

    @staticmethod
    def _validate_rect(rect: ScreenRect, name: str) -> None:
        if (
            not isinstance(rect, tuple)
            or len(rect) != 4
            or not all(type(value) is int for value in rect)
            or rect[2] <= rect[0]
            or rect[3] <= rect[1]
        ):
            raise TargetWindowStateError(f"invalid {name} rectangle")

    @staticmethod
    def _rect_size(rect: ScreenRect) -> tuple[int, int]:
        return rect[2] - rect[0], rect[3] - rect[1]

    @staticmethod
    def _contains(outer: ScreenRect, inner: ScreenRect) -> bool:
        return (
            outer[0] <= inner[0]
            and outer[1] <= inner[1]
            and inner[2] <= outer[2]
            and inner[3] <= outer[3]
        )


class _Point(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class _Rect(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class _MonitorInfoEx(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", _Rect),
        ("rcWork", _Rect),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


class CtypesWin32Api:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("CtypesWin32Api is available only on Windows")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        self._user32.IsWindow.argtypes = [wintypes.HWND]
        self._user32.IsWindow.restype = wintypes.BOOL
        self._user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self._user32.IsWindowVisible.restype = wintypes.BOOL
        self._user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self._user32.GetClassNameW.argtypes = [
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        ]
        self._user32.GetClassNameW.restype = ctypes.c_int
        self._user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        self._user32.GetWindowTextLengthW.restype = ctypes.c_int
        self._user32.GetWindowTextW.argtypes = [
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int,
        ]
        self._user32.GetWindowTextW.restype = ctypes.c_int
        self._user32.GetClientRect.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(_Rect),
        ]
        self._user32.GetClientRect.restype = wintypes.BOOL
        self._user32.ClientToScreen.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(_Point),
        ]
        self._user32.ClientToScreen.restype = wintypes.BOOL
        self._user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        self._user32.MonitorFromWindow.restype = wintypes.HANDLE
        self._user32.GetMonitorInfoW.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_MonitorInfoEx),
        ]
        self._user32.GetMonitorInfoW.restype = wintypes.BOOL
        self._user32.GetForegroundWindow.argtypes = []
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._user32.EnumDisplayMonitors.argtypes = [
            wintypes.HDC,
            ctypes.POINTER(_Rect),
            ctypes.c_void_p,
            wintypes.LPARAM,
        ]
        self._user32.EnumDisplayMonitors.restype = wintypes.BOOL
        self._kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    def enum_windows(self) -> tuple[int, ...]:
        windows: list[int] = []
        callback_type = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
        )

        def collect(hwnd, _lparam):
            windows.append(int(hwnd))
            return True

        if not self._user32.EnumWindows(callback_type(collect), 0):
            self._raise_last_error("EnumWindows")
        return tuple(windows)

    def is_window(self, hwnd: int) -> bool:
        return bool(self._user32.IsWindow(hwnd))

    def is_window_visible(self, hwnd: int) -> bool:
        return bool(self._user32.IsWindowVisible(hwnd))

    def window_process_id(self, hwnd: int) -> int:
        pid = wintypes.DWORD()
        if self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)) == 0:
            self._raise_last_error("GetWindowThreadProcessId")
        return int(pid.value)

    def process_executable(self, pid: int) -> str:
        process = self._kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            self._raise_last_error("OpenProcess")
        try:
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not self._kernel32.QueryFullProcessImageNameW(
                process, 0, buffer, ctypes.byref(size)
            ):
                self._raise_last_error("QueryFullProcessImageNameW")
            return buffer.value
        finally:
            self._kernel32.CloseHandle(process)

    def window_class(self, hwnd: int) -> str:
        buffer = ctypes.create_unicode_buffer(256)
        if self._user32.GetClassNameW(hwnd, buffer, len(buffer)) == 0:
            self._raise_last_error("GetClassNameW")
        return buffer.value

    def window_title(self, hwnd: int) -> str:
        length = self._user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        copied = self._user32.GetWindowTextW(hwnd, buffer, len(buffer))
        if copied == 0 and length > 0:
            self._raise_last_error("GetWindowTextW")
        return buffer.value

    def client_rect_screen_px(self, hwnd: int) -> ScreenRect:
        rect = _Rect()
        if not self._user32.GetClientRect(hwnd, ctypes.byref(rect)):
            self._raise_last_error("GetClientRect")
        top_left = _Point(rect.left, rect.top)
        bottom_right = _Point(rect.right, rect.bottom)
        if not self._user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
            self._raise_last_error("ClientToScreen")
        if not self._user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
            self._raise_last_error("ClientToScreen")
        return top_left.x, top_left.y, bottom_right.x, bottom_right.y

    def monitor_info(self, hwnd: int) -> MonitorInfo:
        monitor_handle = self._user32.MonitorFromWindow(hwnd, 0)
        if not monitor_handle:
            raise OSError("window does not intersect a monitor")
        info = _MonitorInfoEx()
        info.cbSize = ctypes.sizeof(info)
        if not self._user32.GetMonitorInfoW(monitor_handle, ctypes.byref(info)):
            self._raise_last_error("GetMonitorInfoW")
        rect = info.rcMonitor
        device_name = info.szDevice
        output_idx = self._dxgi_output_idx(device_name)
        return MonitorInfo(
            rect_screen_px=(rect.left, rect.top, rect.right, rect.bottom),
            device_name=device_name,
            primary=bool(info.dwFlags & 1),
            dxgi_output_idx=output_idx,
        )

    def _dxgi_output_idx(self, device_name: str) -> int:
        """EnumDisplayMonitors でモニターを列挙し、device_name の位置インデックスを返す。

        left,top でソートした安定順で各 DXGI output_idx を割り当て、
        呼び出し元が正確な output に DXcam を向けられるようにします。
        unknown device_name は fail-closed で例外を返します。
        """
        monitors: list[tuple[int, int, str]] = []

        _MonitorEnumProc = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(_Rect),
            wintypes.LPARAM,
        )

        def _collect(hmon: int, _hdc: int, _rect_ptr, _lparam: int) -> bool:
            mi = _MonitorInfoEx()
            mi.cbSize = ctypes.sizeof(mi)
            if self._user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                r = mi.rcMonitor
                monitors.append((r.left, r.top, mi.szDevice))
            return True

        cb = _MonitorEnumProc(_collect)
        self._user32.EnumDisplayMonitors(None, None, cb, 0)
        monitors.sort()
        for idx, (_, _, name) in enumerate(monitors):
            if name == device_name:
                return idx
        raise OSError(f"monitor {device_name!r} not found in EnumDisplayMonitors result")

    def foreground_window(self) -> int | None:
        hwnd = self._user32.GetForegroundWindow()
        return int(hwnd) if hwnd else None

    @staticmethod
    def _raise_last_error(operation: str) -> None:
        error = ctypes.get_last_error()
        raise OSError(error, f"{operation} failed")

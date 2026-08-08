"""helper process 専用の ctypes Win32 SendInput backend。
WASD・Enter・Escape・left click だけを注入し、foreground binding と arm hotkey を同じ process で検査します。
"""
from __future__ import annotations
import ctypes
import os
from typing import Iterable
ALLOWED_INPUTS = frozenset({"W", "A", "S", "D", "ENTER", "ESCAPE", "LEFT_CLICK"})
_VK = {"W": 0x57, "A": 0x41, "S": 0x53, "D": 0x44, "ENTER": 0x0D, "ESCAPE": 0x1B}
_KEYUP = _MOUSE_LEFT_DOWN = 0x0002
_MOUSE_LEFT_UP = 0x0004
class _MouseInput(ctypes.Structure):
    """Win32 MOUSEINPUT の ctypes layout。
    SendInput union へ渡す native field 幅を定義します。
    """
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("extra", ctypes.c_size_t)]
class _KeyboardInput(ctypes.Structure):
    """Win32 KEYBDINPUT の ctypes layout。
    allowlist の virtual-key press/release を native INPUT へ格納します。
    """
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("extra", ctypes.c_size_t)]
class _InputUnion(ctypes.Union):
    """keyboard と mouse payload を共有する INPUT union。
    desktop absolute pointer field は定義しても public/wire API から到達できません。
    """
    _fields_ = [("mi", _MouseInput), ("ki", _KeyboardInput)]
class _Input(ctypes.Structure):
    """Win32 INPUT の ctypes layout。
    type と keyboard/mouse union を1要素として SendInput へ渡します。
    """
    _anonymous_ = ("value",)
    _fields_ = [("type", ctypes.c_ulong), ("value", _InputUnion)]
class Win32InputBackend:
    """production helper だけが所有する SendInput adapter。
    起動時は必ず disarmed で、foreground PID/HWND と hotkey edge を注入直前まで監視します。
    """
    test_only = False
    def __init__(self) -> None:
        """Windows user32 API を結び、default-disarmed 状態を作る。
        非 Windows で誤起動した場合は入力処理を始めず明示的に失敗します。
        """
        if os.name != "nt":
            raise RuntimeError("Win32 input helper requires Windows")
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.GetForegroundWindow.restype = ctypes.c_void_p
        self._user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        self._user32.SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(_Input), ctypes.c_int]
        self._user32.SendInput.restype = ctypes.c_uint
        self.armed = False
        self.pressed: set[str] = set()
        self._arm_chord_previous = False
    def poll_arm_toggle(self) -> bool:
        """Ctrl+Shift+F12 の chord edge だけで arm を反転する。
        GetAsyncKeyState の high bit を3キーすべてで確認し、押下保持中の再反転を防ぎます。
        """
        chord = all(self._user32.GetAsyncKeyState(vk) & 0x8000 for vk in (0x11, 0x10, 0x7B))
        if chord and not self._arm_chord_previous:
            self.armed = not self.armed
        self._arm_chord_previous = chord
        return self.armed
    def target_is_safe(self, pid: int, hwnd: int) -> bool:
        """foreground HWND とその owning PID が lease target と一致するか返す。
        focus・PID・HWND を1回の gate として扱い、一部だけ一致する window を拒否します。
        """
        foreground = int(self._user32.GetForegroundWindow())
        owner_pid = ctypes.c_ulong()
        self._user32.GetWindowThreadProcessId(ctypes.c_void_p(foreground), ctypes.byref(owner_pid))
        return foreground == hwnd and owner_pid.value == pid
    def apply_inputs(
        self, inputs: Iterable[str], *, sequence: int | None = None, monotonic_ns: int | None = None,
    ) -> None:
        """allowlist 検査後に差分 key/mouse event だけを SendInput する。
        release を先に並べて chord 遷移を安全にし、部分送信は runtime error として停止させます。
        """
        del sequence, monotonic_ns
        desired = set(inputs)
        if not desired <= ALLOWED_INPUTS:
            raise ValueError("input outside allowlist")
        released, pressed = self.pressed - desired, desired - self.pressed
        events = [self._event(key, False) for key in sorted(released)]
        events.extend(self._event(key, True) for key in sorted(pressed))
        if not events:
            return
        array = (_Input * len(events))(*events)
        sent = self._user32.SendInput(len(events), array, ctypes.sizeof(_Input))
        if sent != len(events):
            releases = [self._event(key, False) for key in sorted(ALLOWED_INPUTS)]
            release_array = (_Input * len(releases))(*releases)
            self._user32.SendInput(len(releases), release_array, ctypes.sizeof(_Input))
            self.pressed = set()
            raise RuntimeError(f"SendInput partial failure: {sent}/{len(events)}")
        self.pressed = desired
    def emergency_release_all(self) -> None:
        """tracking 状態に依存せず allowlist 全入力の key-up を送る。
        元 helper が異常終了した後に新しい release-only helper から使用します。
        """
        releases = [self._event(key, False) for key in sorted(ALLOWED_INPUTS)]
        array = (_Input * len(releases))(*releases)
        sent = self._user32.SendInput(len(releases), array, ctypes.sizeof(_Input))
        self.pressed = set()
        if sent != len(releases):
            raise RuntimeError(f"emergency SendInput partial failure: {sent}/{len(releases)}")
    def _event(self, key: str, down: bool) -> _Input:
        """allowlisted symbolic input を native INPUT 1件へ変換する。
        pointer は left button の相対 down/up だけで、座標や desktop click を扱いません。
        """
        if key == "LEFT_CLICK":
            flags = _MOUSE_LEFT_DOWN if down else _MOUSE_LEFT_UP
            return _Input(type=0, value=_InputUnion(mi=_MouseInput(0, 0, 0, flags, 0, 0)))
        flags = 0 if down else _KEYUP
        return _Input(type=1, value=_InputUnion(ki=_KeyboardInput(_VK[key], 0, flags, 0, 0)))

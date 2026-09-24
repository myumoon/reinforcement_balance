"""helper process 専用の ctypes Win32 SendInput backend。
WASD・Enter・Escape・left click だけを注入し、foreground binding と arm hotkey を同じ process で検査します。
UI action(ROI click/Enter/Escape)はatomicなmouse move+click、またはVK down+upとして送ります。
"""
from __future__ import annotations
import ctypes
import os
from typing import Iterable
ALLOWED_INPUTS = frozenset({"W", "A", "S", "D", "ENTER", "ESCAPE", "LEFT_CLICK"})
_UI_ACTIONS = frozenset({"CLICK", "ENTER", "ESCAPE"})
_VK = {"W": 0x57, "A": 0x41, "S": 0x53, "D": 0x44, "ENTER": 0x0D, "ESCAPE": 0x1B}
_KEYUP = _MOUSE_LEFT_DOWN = 0x0002
_MOUSE_LEFT_UP = 0x0004
_MOUSEEVENTF_MOVE = 0x0001
_MOUSEEVENTF_ABSOLUTE = 0x8000
_MOUSEEVENTF_VIRTUALDESK = 0x4000
_SM_XVIRTUALSCREEN, _SM_YVIRTUALSCREEN = 76, 77
_SM_CXVIRTUALSCREEN, _SM_CYVIRTUALSCREEN = 78, 79
_GA_ROOT = 2
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
class _Rect(ctypes.Structure):
    """Win32 RECT の ctypes layout。
    `GetClientRect` が返すclient領域の左上・右下座標を保持します。
    """
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
class _Point(ctypes.Structure):
    """Win32 POINT の ctypes layout。
    `ClientToScreen`/`WindowFromPoint` が使う1点のscreen/client座標です。
    """
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
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
        self._user32.GetClientRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Rect)]
        self._user32.GetClientRect.restype = ctypes.c_int
        self._user32.ClientToScreen.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Point)]
        self._user32.ClientToScreen.restype = ctypes.c_int
        self._user32.WindowFromPoint.argtypes = [_Point]
        self._user32.WindowFromPoint.restype = ctypes.c_void_p
        self._user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self._user32.GetAncestor.restype = ctypes.c_void_p
        self._user32.GetSystemMetrics.argtypes = [ctypes.c_int]
        self._user32.GetSystemMetrics.restype = ctypes.c_int
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
    def _client_rect_to_screen(self, hwnd: int) -> tuple[int, int, int, int] | None:
        """target hwndの現在のclient rectをscreen座標へ変換して返す。
        取得失敗や幅・高さが0以下ならNoneを返し、呼び出し側をfail closedなno-opへ導きます。
        """
        rect = _Rect()
        if not self._user32.GetClientRect(ctypes.c_void_p(hwnd), ctypes.byref(rect)):
            return None
        top_left = _Point(rect.left, rect.top)
        if not self._user32.ClientToScreen(ctypes.c_void_p(hwnd), ctypes.byref(top_left)):
            return None
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width <= 0 or height <= 0:
            return None
        return (top_left.x, top_left.y, top_left.x + width, top_left.y + height)
    @staticmethod
    def _map_normalized_to_screen(
        rect: tuple[int, int, int, int], normalized_x: float, normalized_y: float,
    ) -> tuple[int, int]:
        """0.0〜1.0のROI座標をclient rect内のscreen座標へ写像する。
        端の1.0でも矩形の外へはみ出さないよう `right/bottom - 1` で丸めます。
        """
        left, top, right, bottom = rect
        width, height = right - left, bottom - top
        screen_x = min(left + round(normalized_x * width), right - 1)
        screen_y = min(top + round(normalized_y * height), bottom - 1)
        return screen_x, screen_y
    def _root_owner_at(self, screen_x: int, screen_y: int) -> int:
        """指定screen座標にある window の root owner HWNDを返す。
        `target_hwnd` と比較し、他windowへ意図せずclickが飛ぶ状況をno-opにできるようにします。
        """
        hwnd = self._user32.WindowFromPoint(_Point(screen_x, screen_y))
        if not hwnd:
            return 0
        root = self._user32.GetAncestor(hwnd, _GA_ROOT)
        return int(root) if root else int(hwnd)
    def _pointer_move_event(self, screen_x: int, screen_y: int) -> _Input:
        """virtual desktop全体を基準にした0〜65535の絶対座標へ変換したmouse move eventを作る。
        多重モニタでも `GetSystemMetrics(SM_*VIRTUALSCREEN)` の原点・幅・高さから一意に写像します。
        """
        origin_x = self._user32.GetSystemMetrics(_SM_XVIRTUALSCREEN)
        origin_y = self._user32.GetSystemMetrics(_SM_YVIRTUALSCREEN)
        width = self._user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN)
        height = self._user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN)
        norm_x = round((screen_x - origin_x) * 65535 / max(width - 1, 1))
        norm_y = round((screen_y - origin_y) * 65535 / max(height - 1, 1))
        flags = _MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE | _MOUSEEVENTF_VIRTUALDESK
        return _Input(type=0, value=_InputUnion(mi=_MouseInput(norm_x, norm_y, 0, flags, 0, 0)))
    def apply_ui_action(
        self, ui_action: str, normalized_x: float | None, normalized_y: float | None, *,
        target_hwnd: int, sequence: int | None = None, monotonic_ns: int | None = None,
    ) -> None:
        """CLICK/ENTER/ESCAPEを1組のatomic SendInputとして送る。
        CLICKはclient rect写像とWindowFromPoint一致検査を都度やり直し、
        不一致・取得失敗ならno-opでfail closedし、`self.pressed`は変更しません(held状態を残さない)。
        """
        del sequence, monotonic_ns
        if ui_action not in _UI_ACTIONS:
            raise ValueError("ui action outside allowlist")
        if ui_action == "CLICK":
            rect = self._client_rect_to_screen(target_hwnd)
            if rect is None:
                return
            screen_x, screen_y = self._map_normalized_to_screen(rect, normalized_x, normalized_y)
            if self._root_owner_at(screen_x, screen_y) != target_hwnd:
                return
            events = [self._pointer_move_event(screen_x, screen_y),
                      self._event("LEFT_CLICK", True), self._event("LEFT_CLICK", False)]
        else:
            events = [self._event(ui_action, True), self._event(ui_action, False)]
        array = (_Input * len(events))(*events)
        sent = self._user32.SendInput(len(events), array, ctypes.sizeof(_Input))
        if sent != len(events):
            self.emergency_release_all()
            raise RuntimeError(f"SendInput partial failure: {sent}/{len(events)}")

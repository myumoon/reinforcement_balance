"""capture producer と 04-02 consumer の間で共有する単一 frame 契約。

画素・取得時刻・window と target の同一性だけを運び、保存や annotation の責務は持ちません。
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
from numpy.typing import NDArray


_SHA256 = re.compile(r"[0-9a-f]{64}")
_BGRA_SHAPE = (1080, 1920, 4)
_CLIENT_SIZE = (1920, 1080)


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    frame_bgra: NDArray[np.uint8]
    captured_monotonic_ns: int
    session_frame_index: int
    client_rect_screen_px: tuple[int, int, int, int]
    foreground: bool
    target_profile_hash: str
    game_build_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.frame_bgra, np.ndarray):
            raise ValueError("frame_bgra must be a numpy array")
        if self.frame_bgra.dtype != np.uint8 or self.frame_bgra.shape != _BGRA_SHAPE:
            raise ValueError("frame_bgra must be 1920x1080 uint8 BGRA")
        if type(self.captured_monotonic_ns) is not int or self.captured_monotonic_ns < 0:
            raise ValueError("captured_monotonic_ns must be a non-negative integer")
        if type(self.session_frame_index) is not int or self.session_frame_index < 0:
            raise ValueError("session_frame_index must be a non-negative integer")
        rect = self.client_rect_screen_px
        if (
            not isinstance(rect, tuple)
            or len(rect) != 4
            or not all(type(value) is int for value in rect)
            or (rect[2] - rect[0], rect[3] - rect[1]) != _CLIENT_SIZE
        ):
            raise ValueError("client_rect_screen_px must describe a 1920x1080 client")
        if type(self.foreground) is not bool:
            raise ValueError("foreground must be a bool")
        if (
            not isinstance(self.target_profile_hash, str)
            or _SHA256.fullmatch(self.target_profile_hash) is None
        ):
            raise ValueError("target_profile_hash must be a sha256 hex digest")
        if not isinstance(self.game_build_id, str) or not self.game_build_id:
            raise ValueError("game_build_id must be a non-empty string")

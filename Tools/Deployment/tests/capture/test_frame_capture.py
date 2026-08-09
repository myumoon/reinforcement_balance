"""BGRA frame 契約と latest-only capture session のテスト。

Fake backend の連続フレームだけを使い、04-02 consumer が安全に受け取れる入力境界を確認します。
"""

from __future__ import annotations

from dataclasses import fields, FrozenInstanceError
from queue import Empty

import numpy as np
import pytest

from survivors.capture.captured_frame import CapturedFrame
from survivors.capture.frame_capture import (
    CaptureFrameError,
    CaptureSession,
    DxcamCaptureBackend,
    LatestFrameQueue,
)
from survivors.capture.window_locator import TargetWindowStateError, WindowLocator


class FakeDxcamCamera:
    def __init__(self):
        self.start_calls = []

    def start(self, **kwargs):
        self.start_calls.append(kwargs)

    def get_latest_frame(self):
        return None

    def stop(self):
        pass

    def release(self):
        pass


class FakeDxcamModule:
    def __init__(self):
        self.create_calls = []
        self.camera = FakeDxcamCamera()

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.camera


def _session(profile, policy, fake_api, backend, clock_values):
    locator = WindowLocator(fake_api, profile, policy)
    return CaptureSession(
        locator,
        locator.locate(),
        backend,
        monotonic_ns=lambda: next(clock_values),
    )


def test_captured_frame_has_exact_frozen_seven_field_contract(golden_bgra):
    frame = CapturedFrame(
        frame_bgra=golden_bgra,
        captured_monotonic_ns=100,
        session_frame_index=0,
        client_rect_screen_px=(0, 0, 1920, 1080),
        foreground=True,
        target_profile_hash="a" * 64,
        game_build_id="build-1",
    )

    assert [field.name for field in fields(CapturedFrame)] == [
        "frame_bgra",
        "captured_monotonic_ns",
        "session_frame_index",
        "client_rect_screen_px",
        "foreground",
        "target_profile_hash",
        "game_build_id",
    ]
    with pytest.raises(FrozenInstanceError):
        frame.foreground = False


def test_dxcam_is_configured_for_dxgi_numpy_bgra_at_30_fps():
    dxcam_module = FakeDxcamModule()
    backend = DxcamCaptureBackend.create(dxcam_module=dxcam_module)

    assert dxcam_module.create_calls == [
        {
            "backend": "dxgi",
            "processor_backend": "numpy",
            "output_color": "BGRA",
            "max_buffer_len": 1,
        }
    ]
    backend.start(region=(0, 0, 1920, 1080), target_fps=30)
    assert dxcam_module.camera.start_calls == [
        {"region": (0, 0, 1920, 1080), "target_fps": 30}
    ]


@pytest.mark.parametrize(
    "region",
    [None, (0, 0, 1280, 720), (0, 0, 3840, 1080)],
)
def test_dxcam_backend_rejects_desktop_and_wrong_sized_regions(region):
    backend = DxcamCaptureBackend.create(dxcam_module=FakeDxcamModule())

    with pytest.raises(ValueError, match="region"):
        backend.start(region=region, target_fps=30)


def test_golden_fake_sequence_has_bgra_shape_monotonic_identity_and_consumer_contract(
    profile, policy, fake_api, golden_bgra
):
    from conftest import FakeCaptureBackend

    backend = FakeCaptureBackend([golden_bgra.copy(), golden_bgra.copy()])
    session = _session(profile, policy, fake_api, backend, iter([100, 200]))
    session.start()
    produced = [session.capture_next(), session.capture_next()]

    assert backend.started == [
        {"region": (0, 0, 1920, 1080), "target_fps": 30}
    ]
    assert all(frame.frame_bgra.dtype == np.uint8 for frame in produced)
    assert all(frame.frame_bgra.shape == (1080, 1920, 4) for frame in produced)
    assert [frame.captured_monotonic_ns for frame in produced] == [100, 200]
    assert [frame.session_frame_index for frame in produced] == [0, 1]
    assert all(frame.target_profile_hash == profile.target_hash for frame in produced)
    assert all(
        frame.game_build_id == profile.sections["build"]["build_id"]
        for frame in produced
    )

    def consume_04_02(frames_sequence):
        return [
            (
                frame.session_frame_index,
                frame.frame_bgra.shape,
                frame.target_profile_hash,
                frame.game_build_id,
            )
            for frame in frames_sequence
        ]

    assert [item[0] for item in consume_04_02(produced)] == [0, 1]


def test_latest_only_queue_discards_stale_frames(
    profile, policy, fake_api, golden_bgra
):
    from conftest import FakeCaptureBackend

    backend = FakeCaptureBackend([golden_bgra.copy() for _ in range(3)])
    queue = LatestFrameQueue()
    locator = WindowLocator(fake_api, profile, policy)
    session = CaptureSession(
        locator,
        locator.locate(),
        backend,
        frame_queue=queue,
        monotonic_ns=iter([100, 200, 300]).__next__,
    )
    session.start()
    for _ in range(3):
        session.capture_next()

    assert queue.get_latest_nowait().session_frame_index == 2
    with pytest.raises(Empty):
        queue.get_latest_nowait()


@pytest.mark.parametrize(
    "bad_frame",
    [
        np.zeros((720, 1280, 4), dtype=np.uint8),
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        np.zeros((1080, 1920, 4), dtype=np.float32),
    ],
)
def test_capture_rejects_wrong_resolution_channels_and_dtype(
    profile, policy, fake_api, bad_frame
):
    from conftest import FakeCaptureBackend

    backend = FakeCaptureBackend([bad_frame])
    session = _session(profile, policy, fake_api, backend, iter([100]))
    session.start()

    with pytest.raises(CaptureFrameError):
        session.capture_next()


@pytest.mark.parametrize("state_change", ["focus", "resize"])
def test_capture_revalidates_after_read_and_emits_nothing_on_state_loss(
    profile, policy, fake_api, target_window, golden_bgra, state_change
):
    from conftest import FakeCaptureBackend

    def mutate_state():
        if state_change == "focus":
            fake_api.foreground_hwnd = 999
        else:
            target_window.client_rect = (0, 0, 1280, 720)

    backend = FakeCaptureBackend([golden_bgra], after_read=mutate_state)
    session = _session(profile, policy, fake_api, backend, iter([100]))
    session.start()

    with pytest.raises(TargetWindowStateError):
        session.capture_next()
    with pytest.raises(Empty):
        session.frames.get_latest_nowait()


def test_capture_rejects_non_increasing_monotonic_timestamp(
    profile, policy, fake_api, golden_bgra
):
    from conftest import FakeCaptureBackend

    backend = FakeCaptureBackend([golden_bgra.copy(), golden_bgra.copy()])
    session = _session(profile, policy, fake_api, backend, iter([100, 100]))
    session.start()
    session.capture_next()

    with pytest.raises(CaptureFrameError, match="monotonic"):
        session.capture_next()


def test_capture_start_never_requests_whole_desktop(
    profile, policy, fake_api, golden_bgra
):
    from conftest import FakeCaptureBackend

    backend = FakeCaptureBackend([golden_bgra])
    session = _session(profile, policy, fake_api, backend, iter([100]))
    session.start()

    assert backend.started[0]["region"] == session.target.client_rect_screen_px
    assert backend.started[0]["region"] is not None


def test_close_stops_and_releases_backend(profile, policy, fake_api):
    from conftest import FakeCaptureBackend

    backend = FakeCaptureBackend([])
    session = _session(profile, policy, fake_api, backend, iter([]))
    session.start()
    session.close()

    assert backend.stopped is True
    assert backend.released is True

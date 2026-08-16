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
from conftest import FakeDxcamCamera, FakeDxcamModule  # noqa: F401 — re-exported for test_window_locator
from survivors.capture.window_locator import TargetWindowStateError, WindowLocator



def _session(profile, policy, fake_api, backend):
    """timestamp は FakeCaptureBackend の (frame, ts) タプルから取得するため clock 注入不要。"""
    locator = WindowLocator(fake_api, profile, policy)
    return CaptureSession(locator, locator.locate(), backend)


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
    backend = DxcamCaptureBackend.create(
        output_idx=0,
        expected_client_rect=(0, 0, 1920, 1080),
        dxcam_module=dxcam_module,
    )

    assert dxcam_module.create_calls == [
        {
            "device_idx": 0,
            "output_idx": 0,
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
    backend = DxcamCaptureBackend.create(
        output_idx=0,
        expected_client_rect=(0, 0, 1920, 1080),
        dxcam_module=FakeDxcamModule(),
    )

    with pytest.raises(ValueError):
        backend.start(region=region, target_fps=30)


def test_dxcam_backend_rejects_same_size_but_wrong_position():
    """同サイズでも origin が異なる region は拒否されることを確認する。

    (100, 0, 2020, 1080) は幅1920px・高さ1080pxだが target window の位置ではないため
    ValueError になること。
    """
    backend = DxcamCaptureBackend.create(
        output_idx=0,
        expected_client_rect=(0, 0, 1920, 1080),
        dxcam_module=FakeDxcamModule(),
    )
    with pytest.raises(ValueError, match="bound client rect"):
        backend.start(region=(100, 0, 2020, 1080), target_fps=30)


def test_dxcam_backend_uses_output_idx_from_monitor():
    """output_idx=1 (secondary monitor) が dxcam.create に正確に渡ることを確認する。

    プライマリ以外のモニターに window がある場合でも、
    LocatedTargetWindow.monitor.dxgi_output_idx を経由して
    正しい DXGI output が選択されることを保証します。
    """
    dxcam_module = FakeDxcamModule()
    DxcamCaptureBackend.create(
        output_idx=1,
        expected_client_rect=(1920, 0, 3840, 1080),
        dxcam_module=dxcam_module,
    )
    assert dxcam_module.create_calls[0]["output_idx"] == 1


def test_golden_fake_sequence_has_bgra_shape_monotonic_identity_and_consumer_contract(
    profile, policy, fake_api, golden_bgra
):
    from conftest import FakeCaptureBackend

    backend = FakeCaptureBackend([(golden_bgra.copy(), 100), (golden_bgra.copy(), 200)])
    session = _session(profile, policy, fake_api, backend)
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

    backend = FakeCaptureBackend([(golden_bgra.copy(), ts) for ts in [100, 200, 300]])
    queue = LatestFrameQueue()
    locator = WindowLocator(fake_api, profile, policy)
    session = CaptureSession(locator, locator.locate(), backend, frame_queue=queue)
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

    backend = FakeCaptureBackend([(bad_frame, 100)])
    session = _session(profile, policy, fake_api, backend)
    session.start()

    with pytest.raises(CaptureFrameError):
        session.capture_next()


def test_capture_clears_queue_on_state_loss_even_with_prior_frames(
    profile, policy, fake_api, target_window, golden_bgra
):
    """状態異常前に成功した frame も queue から破棄されることを確認する。

    正常な capture の後に状態異常が起きた場合、直前の stale frame が consumer に渡らないことを保証します。
    """
    from conftest import FakeCaptureBackend

    calls = [0]

    def mutate_on_second():
        calls[0] += 1
        if calls[0] >= 2:
            target_window.client_rect = (0, 0, 1280, 720)

    backend = FakeCaptureBackend(
        [(golden_bgra.copy(), 100), (golden_bgra.copy(), 200)],
        after_read=mutate_on_second,
    )
    session = _session(profile, policy, fake_api, backend)
    session.start()

    session.capture_next()  # 成功 → frame A が queue に残る（consumer はまだ読んでいない）

    with pytest.raises(TargetWindowStateError):
        session.capture_next()  # 2 枚目 read 後に resize → 状態異常

    # queue は clear されており frame A も含まれない
    with pytest.raises(Empty):
        session.frames.get_latest_nowait()


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

    backend = FakeCaptureBackend([(golden_bgra, 100)], after_read=mutate_state)
    session = _session(profile, policy, fake_api, backend)
    session.start()

    with pytest.raises(TargetWindowStateError):
        session.capture_next()
    with pytest.raises(Empty):
        session.frames.get_latest_nowait()


def test_capture_rejects_non_increasing_monotonic_timestamp(
    profile, policy, fake_api, golden_bgra
):
    from conftest import FakeCaptureBackend

    backend = FakeCaptureBackend([(golden_bgra.copy(), 100), (golden_bgra.copy(), 100)])
    session = _session(profile, policy, fake_api, backend)
    session.start()
    session.capture_next()

    with pytest.raises(CaptureFrameError, match="monotonic"):
        session.capture_next()


def test_capture_start_never_requests_whole_desktop(
    profile, policy, fake_api, golden_bgra
):
    from conftest import FakeCaptureBackend

    backend = FakeCaptureBackend([(golden_bgra, 100)])
    session = _session(profile, policy, fake_api, backend)
    session.start()

    assert backend.started[0]["region"] == session.target.client_rect_screen_px
    assert backend.started[0]["region"] is not None


def test_close_stops_and_releases_backend(profile, policy, fake_api):
    from conftest import FakeCaptureBackend

    backend = FakeCaptureBackend([])
    session = _session(profile, policy, fake_api, backend)
    session.start()
    session.close()

    assert backend.stopped is True
    assert backend.released is True

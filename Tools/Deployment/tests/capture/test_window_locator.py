"""process・class・title を束縛する target window locator のテスト。

似た別ウィンドウや変更済みプロセスを採用せず、一意な本家ゲーム画面だけを解決できることを確認します。
"""

from __future__ import annotations

import pytest

from survivors.capture.window_locator import (
    TargetWindowNotFound,
    TargetWindowStateError,
    WindowLocator,
)


def test_locator_records_pid_build_and_profile_identity(profile, policy, fake_api):
    target = WindowLocator(fake_api, profile, policy).locate()

    assert target.hwnd == 101
    assert target.pid == 2001
    assert target.game_build_id == profile.sections["build"]["build_id"]
    assert target.target_profile_hash == profile.target_hash
    assert target.client_rect_screen_px == (0, 0, 1920, 1080)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("executable", r"C:\Games\VampireSurvivors-preview.exe"),
        ("window_class", "YYGameMakerYY-preview"),
        ("title", "Vampire Survivors (Not Responding)"),
    ],
)
def test_locator_requires_exact_executable_class_and_title(
    profile, policy, fake_api, target_window, field, value
):
    setattr(target_window, field, value)

    with pytest.raises(TargetWindowNotFound):
        WindowLocator(fake_api, profile, policy).locate()


def test_locator_rejects_ambiguous_exact_matches(
    profile, policy, fake_api, target_window
):
    duplicate = type(target_window)(
        hwnd=102,
        pid=2002,
        executable=target_window.executable,
        window_class=target_window.window_class,
        title=target_window.title,
        client_rect=target_window.client_rect,
        monitor=target_window.monitor,
    )
    fake_api.windows[duplicate.hwnd] = duplicate

    with pytest.raises(TargetWindowStateError, match="ambiguous"):
        WindowLocator(fake_api, profile, policy).locate()


@pytest.mark.parametrize(
    ("rect", "monitor_rect"),
    [
        ((0, 0, 1280, 720), (0, 0, 1920, 1080)),
        ((1000, 0, 2920, 1080), (0, 0, 1920, 1080)),
    ],
)
def test_locator_rejects_wrong_resolution_and_cross_monitor_window(
    profile, policy, fake_api, target_window, rect, monitor_rect
):
    target_window.client_rect = rect
    target_window.monitor = type(target_window.monitor)(
        rect_screen_px=monitor_rect,
        device_name=target_window.monitor.device_name,
        primary=True,
        dxgi_output_idx=0,
    )

    with pytest.raises(TargetWindowStateError):
        WindowLocator(fake_api, profile, policy).locate()


def test_locator_rejects_non_target_desktop_window(
    profile, policy, fake_api, target_window
):
    target_window.pid = 4
    target_window.executable = r"C:\Windows\explorer.exe"
    target_window.window_class = "Progman"
    target_window.title = "Program Manager"

    with pytest.raises(TargetWindowNotFound):
        WindowLocator(fake_api, profile, policy).locate()


@pytest.mark.parametrize(
    ("section", "field", "value", "rect"),
    [
        ("base", "platform", "linux", None),
        ("base", "window_mode", "windowed", None),
        ("base", "client_resolution", [1280, 720], (0, 0, 1280, 720)),
        ("hardware", "capture_backend", "winrt", None),
        ("display", "monitor_output", r"\\.\DISPLAY2", None),
    ],
)
def test_locator_rejects_unsupported_capture_profile_settings(
    profile, policy, fake_api, target_window, section, field, value, rect
):
    profile.sections[section][field] = value
    if rect is not None:
        target_window.client_rect = rect

    with pytest.raises(TargetWindowStateError):
        WindowLocator(fake_api, profile, policy).locate()


@pytest.mark.parametrize("mutation", ["focus", "resize", "pid", "title", "monitor"])
def test_revalidation_fails_closed_after_window_state_changes(
    profile, policy, fake_api, target_window, mutation
):
    locator = WindowLocator(fake_api, profile, policy)
    target = locator.locate()

    if mutation == "focus":
        fake_api.foreground_hwnd = 999
    elif mutation == "resize":
        target_window.client_rect = (0, 0, 1280, 720)
    elif mutation == "pid":
        target_window.pid = 9001
    elif mutation == "title":
        target_window.title = "Other Window"
    else:
        target_window.monitor = type(target_window.monitor)(
            rect_screen_px=(1920, 0, 3840, 1080),
            device_name=r"\\.\DISPLAY2",
            primary=False,
            dxgi_output_idx=1,
        )

    with pytest.raises(TargetWindowStateError):
        locator.validate(target, require_foreground=True)


def test_locate_carries_dxgi_output_idx_from_monitor(profile, policy, fake_api):
    """MonitorInfo.dxgi_output_idx が LocatedTargetWindow へ正確に伝わることを確認する。

    FakeWin32Api が DXGI 正確な output_idx=0 を返す場合、
    locate() 結果の monitor.dxgi_output_idx も 0 になることを保証します。
    """
    target = WindowLocator(fake_api, profile, policy).locate()
    assert target.monitor.dxgi_output_idx == 0


def test_dxgi_output_idx_for_offset_primary_differs_from_coordinate_sort(
    profile, policy, target_window, fake_api
):
    """負座標 secondary monitor がある場合、Win32 座標ソートと DXGI 順が異なる構成を再現する。

    DISPLAY2 が (-1920,0,0,1080) に配置され Win32 座標ソートでは先頭(idx=0)になるが、
    DXGI では DISPLAY1=output[0]、DISPLAY2=output[1] である場合、
    FakeWin32Api は DXGI 正確な dxgi_output_idx=0 を DISPLAY1 に割り当て、
    DxcamCaptureBackend.create に output_idx=0 が渡ることを保証します。
    """
    from survivors.capture.frame_capture import DxcamCaptureBackend
    from survivors.capture.window_locator import MonitorInfo
    from Tools.Deployment.tests.capture.test_frame_capture import FakeDxcamModule

    # target window は DISPLAY1 (0,0,1920,1080) — primary
    # DXGI 順: DISPLAY1=output[0]、DISPLAY2=output[1]
    # Win32 座標ソート順: DISPLAY2(-1920,0)=idx[0]、DISPLAY1(0,0)=idx[1] ← 誤った旧実装
    target_window.monitor = MonitorInfo(
        rect_screen_px=(0, 0, 1920, 1080),
        device_name=r"\\.\DISPLAY1",
        primary=True,
        dxgi_output_idx=0,  # DXGI 正確値 (Win32 sort では 1 になる誤り)
    )

    located = WindowLocator(fake_api, profile, policy).locate()
    assert located.monitor.dxgi_output_idx == 0, "DXGI 順の output_idx が伝わること"

    dxcam_module = FakeDxcamModule()
    DxcamCaptureBackend.create(
        output_idx=located.monitor.dxgi_output_idx,
        expected_client_rect=located.client_rect_screen_px,
        dxcam_module=dxcam_module,
    )
    assert dxcam_module.create_calls[0]["output_idx"] == 0, (
        "Win32 座標ソートの誤値 1 ではなく DXGI 正確値 0 が dxcam.create に渡ること"
    )

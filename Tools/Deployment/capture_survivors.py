#!/usr/bin/env python3
"""Survivors の実画面を版管理された外部データセットへ収録する。

synthetic または live のフレームを受け取り、安全な一時領域を
経由して完成セッションだけを外部ストアへ公開するコマンドである。
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time

import numpy as np

from survivors.capture import (
    CaptureSession,
    CapturedFrame,
    CtypesWin32Api,
    DxcamCaptureBackend,
    LocatedTargetWindow,
    TargetWindowForegroundLost,
    TargetWindowPolicy,
    WindowLocator,
)
from survivors.capture_dataset import DatasetWriter, ThreadedFrameWriter
from survivors.target_profile import load_runtime_profile


_AUTO_SESSION_ID_PATTERN = re.compile(r"^session-(\d{4,})$")


SYNTHETIC_PROFILE_HASH = "0" * 64
SYNTHETIC_BUILD_ID = "synthetic-disposable-v1"


class FakeCaptureBackend:
    """Disposable synthetic CapturedFrame source used only by smoke tests."""

    def frames(self, duration_sec: float):
        count = max(1, int(duration_sec * 2))
        started = time.monotonic_ns()
        for frame_id in range(count):
            pixels = np.zeros((1080, 1920, 4), dtype=np.uint8)
            pixels[..., 3] = 255
            pixels[0, 0, :3] = frame_id % 256
            yield CapturedFrame(
                pixels,
                started + frame_id + 1,
                frame_id,
                (0, 0, 1920, 1080),
                True,
                SYNTHETIC_PROFILE_HASH,
                SYNTHETIC_BUILD_ID,
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage=(
            "capture_survivors.py [--store-root STORE_ROOT] [--session-id SESSION_ID|auto] "
            "[--duration-sec N] [--synthetic] [--dry-run]"
        )
    )
    parser.add_argument("--store-root")
    parser.add_argument(
        "--session-id",
        default=datetime.now(timezone.utc).strftime("survivors-%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--duration-sec", type=float, default=1.0, metavar="N")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _declare_process_dpi_awareness() -> None:
    """起動直後にPer-Monitor DPI awarenessを宣言し、GetClientRect/GetWindowRectの
    仮想化ずれ(M8と同種の問題)を防ぐ。古いWindowsでは未対応のため、その場合は
    window_locatorのclient resolution検証がfail-closedする(survivors/input/helper.pyと同じ方針)。
    """
    if os.name != "nt":
        return
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4 (winuser.h)
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        pass


def _start_live_session() -> CaptureSession:
    _declare_process_dpi_awareness()
    profile = load_runtime_profile()
    policy = TargetWindowPolicy(
        process_executable="VampireSurvivors.exe",
        window_class="UnityWndClass",
        window_title="Vampire Survivors",
    )
    locator = WindowLocator(CtypesWin32Api(), profile, policy)
    target = locator.locate()
    backend = DxcamCaptureBackend.create(
        output_idx=target.monitor.dxgi_output_idx,
        device_idx=target.monitor.dxgi_device_idx,
        expected_client_rect=target.client_rect_screen_px,
    )
    return CaptureSession(locator, target, backend)


@contextlib.contextmanager
def _graceful_interrupt(flag: threading.Event):
    """Ctrl+Cを例外ではなくフラグ通知に変換し、収録ループを協調的に止める。

    ThreadedFrameWriterのワーカー例外保持はBaseExceptionごと記録して以後
    再送出し続ける仕組みのため、KeyboardInterruptをそのまま伝播させると
    無関係な保存失敗と誤認識される。1回目のCtrl+Cはフラグを立てるだけに
    留め、2回目は標準ハンドラへ委譲して本物の中断を発生させる。
    """
    previous_handler = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum, frame_):
        """SIGINTを1回目はフラグ設定、2回目は既定ハンドラへ委譲する。

        signal.signalに渡すハンドラそのものであり、OSから
        (シグナル番号, 割り込まれたフレーム)を受け取る決まった形を
        している。flagが既に立っていれば「もう一度押した」とみなし
        本来のKeyboardInterruptへ進ませる。
        """
        if flag.is_set():
            signal.default_int_handler(signum, frame_)
        else:
            flag.set()

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        yield flag
    finally:
        signal.signal(signal.SIGINT, previous_handler)


def _wait_for_foreground(
    locator: WindowLocator,
    target: LocatedTargetWindow,
    interrupted: threading.Event,
    *,
    poll_interval_sec: float = 0.5,
) -> bool:
    """対象ウィンドウがフォアグラウンドに来るまで軽量検証をポーリングする。

    起動直後にオペレーターが手動でウィンドウを前面へ出すために行っていた
    Start-Sleep運用を不要にする。フォアグラウンド消失以外の異常はそのまま
    伝播させ、中断された場合は例外を投げずFalseを返す。
    """
    while not interrupted.is_set():
        try:
            locator.validate_lightweight(target, require_foreground=True)
            return True
        except TargetWindowForegroundLost:
            interrupted.wait(poll_interval_sec)
    return False


def _next_auto_session_id(store_root: Path) -> str:
    """capture_sessions配下の既存連番から次のsession-NNNNを採番する。

    毎回一意な --session-id を手入力する手間をなくすため、既存の
    最大連番+1(存在しなければ1)を4桁ゼロ埋めで返す採番専用の関数である。
    """
    sessions_dir = store_root / "capture_sessions"
    max_n = 0
    if sessions_dir.is_dir():
        for entry in sessions_dir.iterdir():
            match = _AUTO_SESSION_ID_PATTERN.match(entry.name)
            if match:
                max_n = max(max_n, int(match.group(1)))
    return f"session-{max_n + 1:04d}"


def _capture_live(
    session: CaptureSession,
    duration_sec: float,
    interrupted: threading.Event,
    stats: dict,
):
    """live収録ループ本体。起動前のフォアグラウンド待機と経過時間の起点を管理する。

    フォアグラウンド待機の時間はduration_secに含めないため、
    session.start()直後を起点としてstatsへ書き込み、呼び出し側が
    正確なelapsed_sec/ended_reasonを算出できるようにする。alt-tab等で
    一時停止・再開したことがoperatorにもコンソール上で分かるよう、
    session.pausedの遷移をstderrへログ出力する。
    """
    try:
        if not _wait_for_foreground(session.locator, session.target, interrupted):
            stats["ended_reason"] = "interrupted"
            return
        session.start()
        stats["started_at"] = time.monotonic()
        deadline = stats["started_at"] + duration_sec
        was_paused = False
        while time.monotonic() < deadline:
            # frameが来ないtick(foreground一時停止中を含む)でも必ずここで
            # interruptedを見る。yieldは実フレームがあるときにしか起きないため、
            # ここで確認しないと一時停止中はCtrl+Cがdeadlineまで無視され続ける。
            if interrupted.is_set():
                stats["ended_reason"] = "interrupted"
                return
            frame = session.capture_next()
            if session.paused != was_paused:
                was_paused = session.paused
                message = (
                    "capture paused: target window lost foreground"
                    if was_paused
                    else "capture resumed: target window regained foreground"
                )
                print(message, file=sys.stderr)
            if frame is None:
                time.sleep(1 / 60)  # ponytail: 60fps上限ポーリング — busy loopを防ぐ
                continue
            yield frame
        stats["ended_reason"] = "duration_elapsed"
    finally:
        session.close()


def main(argv: list[str] | None = None) -> int:
    """CLI引数に従ってフレームを収録し、公開結果をJSONで返す。

    synthetic は従来の同期保存、live は重いPNG保存だけを並列化する。
    Ctrl+C(1回目)は収録済み分だけを安全に公開するか、0枚ならABORTED_EMPTY
    として空セッションを残さず終了し、2回目のCtrl+Cだけ本来のKeyboardInterrupt
    (未公開tempのrmtree)へエスケープする。
    """
    args = _parser().parse_args(argv)
    if args.store_root is None or not args.store_root.strip():
        print("error: --store-root is required; workspace fallback is forbidden", file=sys.stderr)
        return 1
    if args.duration_sec <= 0:
        print("error: --duration-sec must be positive", file=sys.stderr)
        return 1
    if not args.session_id.strip():
        print("error: --session-id must be non-empty", file=sys.stderr)
        return 1
    store_root = Path(args.store_root)
    interrupted = threading.Event()  # I6: main()呼び出しごとに毎回新規に作る
    stats: dict = {}
    loop_started_at = time.monotonic()
    try:
        with _graceful_interrupt(interrupted):
            session_id = args.session_id
            if session_id == "auto":
                session_id = _next_auto_session_id(store_root)
            if args.synthetic:
                frames = FakeCaptureBackend().frames(args.duration_sec)
                identity = SYNTHETIC_PROFILE_HASH, SYNTHETIC_BUILD_ID
            else:
                live_session = _start_live_session()
                frames = _capture_live(live_session, args.duration_sec, interrupted, stats)
                identity = (
                    live_session.target.target_profile_hash,
                    live_session.target.game_build_id,
                )
            if args.dry_run:
                first = next(iter(frames), None)
                close = getattr(frames, "close", None)
                if close is not None:
                    close()
                if first is None:
                    raise ValueError("capture produced no frame")
                print(json.dumps({"status": "DRY_RUN_OK", "session_id": session_id}))
                return 0
            with DatasetWriter(store_root, session_id, *identity) as writer:
                if args.synthetic:
                    for frame in frames:
                        writer.write_frame(frame)
                        if interrupted.is_set():  # 1フレーム分の書き込み完了直後にだけ確認する
                            break
                    active_writer = writer
                    checkpoint = "synthetic"
                else:
                    # withで囲むことで、submit_frame中の例外(fail-closedな
                    # TargetWindowStateErrorや2回目Ctrl+CのKeyboardInterrupt含む)
                    # でも必ずclose()が呼ばれexecutorが止まる。裸のthreaded.close()
                    # 呼び出しだと例外時にスキップされ、ワーカーがPNG書き込み中の
                    # ままDatasetWriter.__exit__のrmtreeと競合する(過去の回帰)。
                    with ThreadedFrameWriter(writer) as threaded:
                        for frame in frames:
                            threaded.submit_frame(frame)
                            if interrupted.is_set():  # I4: submit_frame内部では割り込まない
                                break
                    active_writer = threaded
                    checkpoint = "live-pilot"

                close = getattr(frames, "close", None)
                if close is not None:
                    close()  # _capture_liveのfinally(session.close())を即時実行させる

                ended_reason = (
                    "interrupted"
                    if interrupted.is_set()
                    else "source_exhausted"
                    if args.synthetic
                    else stats.get("ended_reason", "duration_elapsed")
                )
                elapsed_sec = time.monotonic() - stats.get("started_at", loop_started_at)

                if active_writer.frame_count == 0 and interrupted.is_set():
                    print(
                        json.dumps(
                            {
                                "status": "ABORTED_EMPTY",
                                "session_id": session_id,
                                "ended_reason": ended_reason,
                                "requested_duration_sec": args.duration_sec,
                                "elapsed_sec": elapsed_sec,
                            }
                        )
                    )
                    return 0
                manifest = active_writer.publish(operator_checkpoint=checkpoint)
        print(
            json.dumps(
                {
                    "status": "PUBLISHED",
                    "session_id": manifest.session_id,
                    "frame_count": manifest.frame_count,
                    "formal_dataset_eligible": manifest.formal_dataset_eligible,
                    "ended_reason": ended_reason,
                    "requested_duration_sec": args.duration_sec,
                    "elapsed_sec": elapsed_sec,
                }
            )
        )
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

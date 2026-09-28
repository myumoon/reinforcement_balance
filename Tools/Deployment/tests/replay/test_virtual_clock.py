"""virtual clock と recorded frame source の決定性テスト(06-01 タスク1)。

同じ記録からは同じ tick・同じ frame 列が出ること、duplicate/drop/timeout/focus_lost を再現できること、
実時計と実入力デバイスに触れないこと、async queue の到着順に結果が左右されないこと、
PyTorch 決定性設定の食い違いを拒否することを確かめます。
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import random
import sys
import time

import numpy as np
import pytest

import survivors
from survivors.replay import recorded_frame_source as rfs
from survivors.replay.recorded_frame_source import (
    DeterminismManifest,
    DeterminismMismatchError,
    RecordedEvent,
    RecordedFrameSource,
    RecordedSession,
)
from survivors.replay.virtual_clock import VirtualClock, schedule_ticks

TICK_NS = 66_666_667
SESSION = "sess-1"
PROFILE_HASH = "a" * 64
PIXELS = np.zeros((1080, 1920, 4), dtype=np.uint8)
MANIFEST = DeterminismManifest(
    deterministic_algorithms=True, device="cpu", nms_backend="torchvision.ops.nms",
    num_threads=1, num_interop_threads=1,
)
REPLAY_DIR = Path(survivors.__file__).parent / "replay"
WALL_CLOCK_APIS = (
    "sleep", "perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns",
    "time", "time_ns", "process_time", "process_time_ns", "thread_time", "thread_time_ns",
)


def _frame(seq: int, index: int, ts: int, kind: str = "frame") -> dict:
    """frame / duplicate event の dict を作る。"""
    return {
        "completion_seq": seq, "kind": kind, "timestamp_ns": ts,
        "session_frame_index": index, "correlation_id": f"{SESSION}:{index}",
    }


def _empty(seq: int, kind: str, ts: int) -> dict:
    """timeout / focus_lost event の dict を作る。"""
    return {"completion_seq": seq, "kind": kind, "timestamp_ns": ts,
            "session_frame_index": None, "correlation_id": None}


# frame0, frame1, timeout, duplicate1, (frame2 drop) frame3, focus_lost, timeout, frame4
EVENTS = [
    _frame(0, 0, 1 * TICK_NS),
    _frame(1, 1, 2 * TICK_NS),
    _empty(2, "timeout", 3 * TICK_NS),
    _frame(3, 1, 4 * TICK_NS, kind="duplicate"),
    _frame(4, 3, 5 * TICK_NS),
    _empty(5, "focus_lost", 6 * TICK_NS),
    _empty(6, "timeout", 7 * TICK_NS),
    _frame(7, 4, 8 * TICK_NS),
]


def _session_dict(events: list[dict]) -> dict:
    """RecordedSession.from_dict へ渡す session dict を作る。"""
    return {
        "session_id": SESSION, "target_profile_hash": PROFILE_HASH, "game_build_id": "build-1",
        "client_rect_screen_px": [0, 0, 1920, 1080], "determinism": MANIFEST.to_dict(),
        "events": events,
    }


def _replay(session: RecordedSession, module=rfs) -> list[tuple]:
    """source を最後まで poll し、(時計, 戻り値, queue の中身, paused) の trace を返す。

    controller と同じく1 poll ごとに 1ms の idle sleep を仮想時計で挟みます。
    """
    clock = VirtualClock()
    source = module.RecordedFrameSource(session, clock, lambda _index: PIXELS, replay_determinism=MANIFEST)
    source.start()
    trace = []
    while not source.exhausted:
        frame = source.capture_next()
        queued = source.frames.get_latest_nowait() if frame is not None else None
        assert queued is frame
        trace.append((
            clock(),
            None if frame is None else (frame.session_frame_index, frame.captured_monotonic_ns),
            source.paused,
        ))
        clock.sleep(0.001)
    assert source.capture_next() is None
    source.close()
    return trace


def test_same_timestamps_give_same_scheduled_ticks() -> None:
    """同じ timestamp 列からは常に同じ tick 列が返り、重複・飛びもそのまま表れる。"""
    stamps = [100, 100 + TICK_NS, 100 + TICK_NS + 5, 100 + 3 * TICK_NS]
    first = schedule_ticks(stamps, period_ns=TICK_NS)
    assert first == schedule_ticks(list(stamps), period_ns=TICK_NS) == (0, 1, 1, 3)
    assert schedule_ticks(stamps, period_ns=TICK_NS, origin_ns=0) == (0, 1, 1, 3)
    assert schedule_ticks([], period_ns=TICK_NS) == ()
    with pytest.raises(ValueError):
        schedule_ticks([200, 100], period_ns=TICK_NS)
    with pytest.raises(ValueError):
        schedule_ticks([1.5], period_ns=TICK_NS)


def test_virtual_clock_moves_only_when_told() -> None:
    """仮想時計は advance/advance_to/sleep でだけ進み、逆行や float ns を拒否する。"""
    clock = VirtualClock(start_ns=10)
    assert clock() == clock.now_ns() == 10
    assert clock.advance(5) == 15
    assert clock.advance_to(3) == 15  # 過去へは戻らない
    assert clock.advance_to(40) == 40
    clock.sleep(0.001)
    assert clock() == 1_000_040
    for bad in (lambda: clock.advance(-1), lambda: clock.advance(1.0),
                lambda: clock.sleep(-0.1), lambda: VirtualClock(True)):
        with pytest.raises(ValueError):
            bad()


def test_replay_trace_is_reproducible() -> None:
    """同じ session を2回再生すると、時計・frame・paused の trace が完全に一致する。"""
    session = RecordedSession.from_dict(_session_dict(EVENTS))
    assert _replay(session) == _replay(RecordedSession.from_dict(_session_dict(EVENTS)))


def test_duplicate_drop_timeout_and_focus_loss_are_reproduced() -> None:
    """duplicate は元の frame を元の時刻で再送し、drop は番号の飛び、timeout/focus_lost は None になる。"""
    session = RecordedSession.from_dict(_session_dict(EVENTS))
    trace = _replay(session)
    ms = 1_000_000
    assert trace == [
        (1 * TICK_NS, (0, 1 * TICK_NS), False),
        (2 * TICK_NS, (1, 2 * TICK_NS), False),
        (3 * TICK_NS, None, False),
        (4 * TICK_NS, (1, 2 * TICK_NS), False),  # duplicate: 記録済み時刻のまま
        (5 * TICK_NS, (3, 5 * TICK_NS), False),
        (6 * TICK_NS, None, True),  # focus_lost で pause
        (7 * TICK_NS, None, True),  # pause 中の timeout
        (8 * TICK_NS, (4, 8 * TICK_NS), False),  # frame 到着で pause 解除
    ]
    assert session.dropped_indices == (2,)
    # 記録時刻より先に時計が進んでいても逆行しない(sleep 分だけ先行)
    late = RecordedSession.from_dict(_session_dict([_frame(0, 0, 0), _frame(1, 1, ms // 2)]))
    assert [row[0] for row in _replay(late)] == [0, ms]


def test_replay_never_touches_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """実時計 API を全て例外に差し替えても再生が完走し、replay module は time を import しない。"""
    for path in REPLAY_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        assert not imported & {"time", "datetime"}, path.name
    baseline = _replay(RecordedSession.from_dict(_session_dict(EVENTS)))

    def _blocked(*_args, **_kwargs):
        raise AssertionError("replay path called a wall clock API")

    for name in WALL_CLOCK_APIS:
        monkeypatch.setattr(time, name, _blocked)
    assert _replay(RecordedSession.from_dict(_session_dict(EVENTS))) == baseline


def test_live_input_backend_unavailable_does_not_change_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """survivors.input を import 不能にしても replay module を読み込めて、結果も変わらない。"""
    baseline = _replay(RecordedSession.from_dict(_session_dict(EVENTS)))
    for name in list(sys.modules):
        if name.startswith(("survivors.input", "survivors.replay")):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.delattr(survivors, "replay", raising=False)
    monkeypatch.delattr(survivors, "input", raising=False)
    monkeypatch.setitem(sys.modules, "survivors.input", None)  # import すると ImportError

    fresh = importlib.import_module("survivors.replay.recorded_frame_source")
    importlib.import_module("survivors.replay.virtual_clock")
    session = fresh.RecordedSession.from_dict(_session_dict(EVENTS))
    assert _replay(session, module=fresh) == baseline
    assert sys.modules["survivors.input"] is None
    assert not any(name.startswith("survivors.input.") for name in sys.modules)


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_async_arrival_order_does_not_change_result(seed: int) -> None:
    """event の到着順を入れ替えても completion_seq 順に並べ直され、trace が変わらない。"""
    baseline = _replay(RecordedSession.from_dict(_session_dict(EVENTS)))
    shuffled = list(EVENTS)
    random.Random(seed).shuffle(shuffled)
    assert shuffled != EVENTS
    session = RecordedSession.from_dict(_session_dict(shuffled))
    assert [event.completion_seq for event in session.events] == list(range(len(EVENTS)))
    assert _replay(session) == baseline


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([_frame(0, 0, 10), _frame(0, 1, 20)], "completion_seq"),
        ([_frame(0, 0, 20), _frame(1, 1, 10)], "non-decreasing"),
        ([_frame(0, 1, 10), _frame(1, 0, 20)], "strictly increase"),
        ([_frame(0, 0, 10), _frame(1, 1, 20, kind="duplicate")], "undelivered"),
        ([{**_frame(0, 0, 10), "correlation_id": "other:0"}], "does not match"),
        ([{**_empty(0, "timeout", 10), "session_frame_index": 3}], "must not carry"),
        ([{**_frame(0, 0, 10), "extra": 1}], "keys mismatch"),
    ],
)
def test_corrupt_recordings_are_rejected(events: list[dict], message: str) -> None:
    """completion 順の重複・時刻逆行・番号逆行・未配信 duplicate・相関 id 不一致・未知キーを拒否する。"""
    with pytest.raises(ValueError, match=message):
        RecordedSession.from_dict(_session_dict(events))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deterministic_algorithms", False),
        ("device", "cuda:0"),
        ("nms_backend", "custom_nms"),
        ("num_threads", 4),
        ("num_interop_threads", 2),
    ],
)
def test_determinism_mismatch_is_rejected(field: str, value: object) -> None:
    """決定性設定が1項目でも記録と違えば、frame source の構築時点で拒否する。"""
    replay = DeterminismManifest.from_dict({**MANIFEST.to_dict(), field: value})
    with pytest.raises(DeterminismMismatchError, match=field):
        MANIFEST.require_match(replay)
    session = RecordedSession.from_dict(_session_dict(EVENTS))
    with pytest.raises(DeterminismMismatchError, match=field):
        RecordedFrameSource(session, VirtualClock(), lambda _index: PIXELS, replay_determinism=replay)
    MANIFEST.require_match(DeterminismManifest.from_dict(MANIFEST.to_dict()))


def test_determinism_manifest_reads_torch_settings() -> None:
    """from_torch は現在の torch の決定性設定と thread 数を読み取り、不正値・未知キーは拒否する。"""
    torch = pytest.importorskip("torch")
    current = DeterminismManifest.from_torch(device="cpu", nms_backend="torchvision.ops.nms")
    assert current.deterministic_algorithms is torch.are_deterministic_algorithms_enabled()
    assert current.num_threads == torch.get_num_threads()
    assert current.num_interop_threads == torch.get_num_interop_threads()
    with pytest.raises(ValueError):
        DeterminismManifest.from_dict({**MANIFEST.to_dict(), "num_threads": 0})
    with pytest.raises(ValueError, match="keys mismatch"):
        DeterminismManifest.from_dict({**MANIFEST.to_dict(), "cudnn_benchmark": False})


def test_recorded_event_rejects_frame_without_index() -> None:
    """frame event に番号や相関 id が無ければ拒否する。"""
    with pytest.raises(ValueError):
        RecordedEvent(completion_seq=0, kind="frame", timestamp_ns=0)
    with pytest.raises(ValueError):
        RecordedEvent(completion_seq=0, kind="bogus", timestamp_ns=0)

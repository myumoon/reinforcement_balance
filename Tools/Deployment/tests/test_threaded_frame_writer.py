"""並列フレーム永続化の順序・流量制御・失敗伝播を検証する。

初心者向けには、保存処理が並列でもメタデータの順番と失敗時の安全性が
同期版と変わらないことを確認するテスト群である。
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import capture_survivors
from survivors.capture.captured_frame import CapturedFrame
from survivors import capture_dataset
from survivors.capture_dataset import DatasetWriter, ThreadedFrameWriter


PROFILE_HASH = "a" * 64
BUILD_ID = "survivors-test-build"


def _frame(frame_id: int, timestamp_ns: int) -> CapturedFrame:
    """決定的な CapturedFrame を生成する。

    初心者向けには、実入力と同じ画面サイズを保ちながら、フレーム番号だけを
    先頭画素へ埋め込んだ入力を作る関数である。
    """
    pixels = np.zeros((1080, 1920, 4), dtype=np.uint8)
    pixels[0, 0] = [frame_id, 17, 31, 255]
    return CapturedFrame(
        frame_bgra=pixels,
        captured_monotonic_ns=timestamp_ns,
        session_frame_index=frame_id,
        client_rect_screen_px=(0, 0, 1920, 1080),
        foreground=True,
        target_profile_hash=PROFILE_HASH,
        game_build_id=BUILD_ID,
    )


@pytest.fixture
def fake_encode(monkeypatch) -> None:
    """PNG エンコードを決定的な軽量バイト列へ置換する。

    初心者向けには、圧縮速度に左右されず writer の順序と内容だけを比較する
    ためのテスト用差し替えである。
    """
    monkeypatch.setattr(
        capture_dataset,
        "_encode_lossless_png",
        lambda pixels: b"fake-png-" + bytes([int(pixels[0, 0, 0])]),
    )


def test_records_are_appended_in_submission_order(tmp_path, monkeypatch, fake_encode):
    """後続ワーカーが先に完了しても投入順で確定することを検証する。

    初心者向けには、2枚目の保存を先に終わらせても frames.jsonl が
    0番、1番の順から崩れないことを確認する。
    """
    monkeypatch.setattr(capture_dataset.os, "cpu_count", lambda: 2)
    with DatasetWriter(tmp_path, "ordered", PROFILE_HASH, BUILD_ID) as writer:
        persist = writer._persist_frame
        release_first = threading.Event()
        second_finished = threading.Event()

        def delayed_persist(frame):
            """先頭フレームだけ待機させ、完了順を意図的に逆転する。

            初心者向けには、並列処理で起こる追い越しを再現するテスト用関数である。
            """
            if frame.session_frame_index == 0:
                assert release_first.wait(2)
            else:
                second_finished.set()
            return persist(frame)

        monkeypatch.setattr(writer, "_persist_frame", delayed_persist)
        threaded = ThreadedFrameWriter(writer)
        threaded.submit_frame(_frame(0, 100))
        threaded.submit_frame(_frame(1, 200))
        assert second_finished.wait(2)
        release_first.set()
        threaded.close()

        assert [record.frame_id for record in writer._frame_records] == [0, 1]


def test_backpressure_blocks_the_capture_caller(tmp_path, monkeypatch, fake_encode):
    """有限 pending 上限で submit_frame 呼び出し元が待機することを検証する。

    初心者向けには、保存待ちが上限に達したとき、3枚目を捨てずに
    先頭の保存完了までキャプチャ側を止めることを確認する。
    """
    monkeypatch.setattr(capture_dataset.os, "cpu_count", lambda: 1)
    with DatasetWriter(tmp_path, "backpressure", PROFILE_HASH, BUILD_ID) as writer:
        persist = writer._persist_frame
        release = threading.Event()

        def blocked_persist(frame):
            """永続化を明示的に停止して pending 上限を再現する。

            初心者向けには、遅いディスクをイベント待機で模擬する関数である。
            """
            assert release.wait(2)
            return persist(frame)

        monkeypatch.setattr(writer, "_persist_frame", blocked_persist)
        threaded = ThreadedFrameWriter(writer)
        threaded.submit_frame(_frame(0, 100))
        threaded.submit_frame(_frame(1, 200))

        caller = threading.Thread(
            target=threaded.submit_frame,
            args=(_frame(2, 300),),
        )
        caller.start()
        time.sleep(0.05)
        assert caller.is_alive()

        release.set()
        caller.join(2)
        assert not caller.is_alive()
        threaded.close()
        assert [record.frame_id for record in writer._frame_records] == [0, 1, 2]


def test_worker_failure_is_rethrown_and_stops_appends(tmp_path, monkeypatch, fake_encode):
    """最初のワーカー例外を全APIで再送出し後続確定を止めることを検証する。

    初心者向けには、保存失敗を見逃して不完全なデータセットを公開せず、
    同じ失敗を submit・close・publish のどこからでも確認できることを試す。
    """
    monkeypatch.setattr(capture_dataset.os, "cpu_count", lambda: 1)
    failure = RuntimeError("injected worker failure")
    failed = threading.Event()
    with DatasetWriter(tmp_path, "failed", PROFILE_HASH, BUILD_ID) as writer:

        def fail_persist(_frame):
            """ワーカースレッドで同一例外を発生させる。

            初心者向けには、ディスク保存失敗を決定的に再現するテスト用関数である。
            """
            failed.set()
            raise failure

        monkeypatch.setattr(writer, "_persist_frame", fail_persist)
        threaded = ThreadedFrameWriter(writer)
        threaded.submit_frame(_frame(0, 100))
        assert failed.wait(2)

        with pytest.raises(RuntimeError) as submit_error:
            threaded.submit_frame(_frame(1, 200))
        with pytest.raises(RuntimeError) as close_error:
            threaded.close()
        with pytest.raises(RuntimeError) as publish_error:
            threaded.publish()

        assert submit_error.value is failure
        assert close_error.value is failure
        assert publish_error.value is failure
        assert writer._frame_records == []


def test_threaded_publish_matches_synchronous_writer(tmp_path, fake_encode):
    """決定的エンコード時に同期版と並列版の成果物が一致することを検証する。

    初心者向けには、処理方法だけを並列化しても PNG・frames.jsonl・
    FrameRecord の内容が変わらないことを比較する。
    """
    frames = [_frame(0, 100), _frame(1, 200), _frame(2, 300)]
    sync_writer = DatasetWriter(tmp_path / "sync", "session", PROFILE_HASH, BUILD_ID)
    for frame in frames:
        sync_writer.write_frame(frame)
    sync_manifest = sync_writer.publish("test")

    parallel_writer = DatasetWriter(
        tmp_path / "parallel", "session", PROFILE_HASH, BUILD_ID
    )
    threaded = ThreadedFrameWriter(parallel_writer)
    for frame in frames:
        threaded.submit_frame(frame)
    parallel_manifest = threaded.publish("test")

    assert parallel_manifest.frame_records == sync_manifest.frame_records
    assert (parallel_manifest.session_path / "frames.jsonl").read_bytes() == (
        sync_manifest.session_path / "frames.jsonl"
    ).read_bytes()
    for record in sync_manifest.frame_records:
        assert (parallel_manifest.session_path / record.object_path).read_bytes() == (
            sync_manifest.session_path / record.object_path
        ).read_bytes()


def test_live_capture_persists_on_worker_and_preserves_output(
    tmp_path, monkeypatch, fake_encode, capsys
):
    """live 収録だけがワーカー永続化を使い公開JSONを維持することを検証する。

    初心者向けには、実機の代わりに1フレームを渡し、PNG保存がメインスレッドを
    外れても従来と同じ PUBLISHED 応答になることを確認する。
    """
    live_session = SimpleNamespace(
        target=SimpleNamespace(
            target_profile_hash=PROFILE_HASH,
            game_build_id=BUILD_ID,
        )
    )
    monkeypatch.setattr(capture_survivors, "_start_live_session", lambda: live_session)
    monkeypatch.setattr(
        capture_survivors,
        "_capture_live",
        lambda _session, _duration: iter([_frame(0, 100)]),
    )
    persist = DatasetWriter._persist_frame
    worker_threads: list[int] = []

    def record_thread(self, frame):
        """永続化を実行したスレッドIDを記録する。

        初心者向けには、live 経路が本当にワーカースレッドへ移ったかを成果物と
        合わせて確認するテスト用ラッパーである。
        """
        worker_threads.append(threading.get_ident())
        return persist(self, frame)

    monkeypatch.setattr(DatasetWriter, "_persist_frame", record_thread)
    main_thread = threading.get_ident()

    result = capture_survivors.main(
        [
            "--store-root",
            str(tmp_path),
            "--session-id",
            "live-session",
            "--duration-sec",
            "0.1",
        ]
    )

    assert result == 0
    assert worker_threads and worker_threads[0] != main_thread
    assert json.loads(capsys.readouterr().out) == {
        "status": "PUBLISHED",
        "session_id": "live-session",
        "frame_count": 1,
        "formal_dataset_eligible": False,
    }

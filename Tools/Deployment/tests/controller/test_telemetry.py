"""controller telemetry の JSONL streaming 契約を検証する。

やさしい説明: session の固定情報と各 stage の完了情報が、終了を待たずに
1行ずつ保存され、並列 stage でも記録順を追えることを確認します。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

import pytest


def _header():
    """テスト用の session header を返す。

    やさしい説明: profile・build・artifact と実行環境の最小情報を固定します。
    """
    from survivors.controller.telemetry import TelemetrySessionHeader

    return TelemetrySessionHeader(
        session_id="session-1",
        mode="shadow",
        target_profile_hash="a" * 64,
        game_build_id="game-build-1",
        controller_build_id="controller-build-1",
        artifact_hashes={"combat_model": "b" * 64, "parser": "c" * 64},
        host={"os": "windows", "cpu": "test-cpu"},
        device={"capture": "test-device", "inference": "cpu"},
        dependency_versions={"python": "3.11.9"},
        deterministic_replay={"device": "cpu", "deterministic_algorithms": True},
        structured_telemetry_config={"retention_days": 7},
    )


def test_writer_streams_header_and_stage_before_close(tmp_path) -> None:
    """session header と stage event を即時 flush する。

    やさしい説明: process が途中停止しても、直前までの2行をファイルから読めます。
    """
    from survivors.controller.telemetry import TELEMETRY_SCHEMA_VERSION, TelemetryWriter

    path = tmp_path / "controller.jsonl"
    writer = TelemetryWriter(path, _header())
    sequence = writer.write_stage(
        "policy",
        correlation_id="session-1:frame-4",
        timestamp_ns=1_234,
        latency_ns=56,
        queue_depth=2,
        payload={"frame_id": "frame-4", "action_index": 3},
    )
    writer.flush()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert sequence == 1
    assert rows == [
        {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "event": "session_header",
            "sequence": 0,
            "session_id": "session-1",
            "mode": "shadow",
            "target_profile_hash": "a" * 64,
            "game_build_id": "game-build-1",
            "controller_build_id": "controller-build-1",
            "artifact_hashes": {"combat_model": "b" * 64, "parser": "c" * 64},
            "host": {"os": "windows", "cpu": "test-cpu"},
            "device": {"capture": "test-device", "inference": "cpu"},
            "dependency_versions": {"python": "3.11.9"},
            "deterministic_replay": {
                "device": "cpu",
                "deterministic_algorithms": True,
            },
            "structured_telemetry_config": {"retention_days": 7},
        },
        {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "event": "stage",
            "sequence": 1,
            "session_id": "session-1",
            "mode": "shadow",
            "stage": "policy",
            "correlation_id": "session-1:frame-4",
            "timestamp_ns": 1_234,
            "latency_ns": 56,
            "queue_depth": 2,
            "payload": {"frame_id": "frame-4", "action_index": 3},
        },
    ]
    writer.close()


def test_writer_records_parallel_completion_order_without_losing_lines(tmp_path) -> None:
    """並列 stage の書込みへ一意で単調な sequence を付ける。

    やさしい説明: 完了順が非決定でも、JSONL 上の実際の完了順を後から再現できます。
    """
    from survivors.controller.telemetry import TelemetryWriter

    path = tmp_path / "parallel.jsonl"
    writer = TelemetryWriter(path, _header())
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                writer.write_stage,
                "detector",
                correlation_id=f"frame-{index}",
                timestamp_ns=10_000 + index,
                latency_ns=index,
                queue_depth=index % 2,
                payload={"frame_id": f"frame-{index}"},
            )
            for index in range(32)
        ]
    returned_sequences = [future.result() for future in futures]
    writer.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()][1:]
    assert sorted(returned_sequences) == list(range(1, 33))
    assert [row["sequence"] for row in rows] == list(range(1, 33))
    assert len({row["correlation_id"] for row in rows}) == 32


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"target_profile_hash": "not-a-hash"}, "target_profile_hash"),
        ({"artifact_hashes": {"combat_model": "not-a-hash"}}, "artifact_hashes"),
        ({"mode": "dry-run"}, "mode"),
    ],
)
def test_header_rejects_invalid_session_identity(override, message) -> None:
    """壊れた session identity を file 作成前に拒否する。

    やさしい説明: profile・artifact・実行 mode の取り違えを記録開始時に止めます。
    """
    from survivors.controller.telemetry import TelemetrySessionHeader

    values = {
        "session_id": "session-1",
        "mode": "shadow",
        "target_profile_hash": "a" * 64,
        "game_build_id": "game-build-1",
        "controller_build_id": "controller-build-1",
        "artifact_hashes": {"combat_model": "b" * 64},
        "host": {"os": "windows"},
        "device": {"inference": "cpu"},
        "dependency_versions": {"python": "3.11.9"},
    }
    values.update(override)
    with pytest.raises(ValueError, match=message):
        TelemetrySessionHeader(**values)


def test_writer_refuses_to_overwrite_existing_telemetry(tmp_path) -> None:
    """既存 telemetry file の上書きを拒否する。

    やさしい説明: session id や出力先を誤って再利用しても過去の監査ログを失いません。
    """
    from survivors.controller.telemetry import TelemetryWriter

    path = tmp_path / "existing.jsonl"
    path.write_text("existing\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        TelemetryWriter(path, _header())
    assert path.read_text(encoding="utf-8") == "existing\n"

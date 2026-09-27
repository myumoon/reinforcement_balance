"""``survivors.controller.gate`` と ``compute_controller_gate.py`` の M11 gate 判定を検証する。

controller が書く telemetry JSONL と同じ形の行を組み立て、6つの閾値
(process crash / health stop / wrong-state / level-up candidate invalid /
p99 end-to-end / memory growth)がそれぞれ単独で FAIL を起こせることと、
全て満たしたときだけ PASS になることを確かめます。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import compute_controller_gate as cli
from survivors.controller.controller import EXIT_ERROR, EXIT_HEALTH_STOP, EXIT_TERMINAL_FAILURE
from survivors.controller.gate import (
    issue_gate_verdict,
    memory_growth_bytes_per_hour,
    read_telemetry,
)
from survivors.controller.telemetry import TELEMETRY_SCHEMA_VERSION, TelemetrySessionHeader, TelemetryWriter

_MS = 1_000_000
_HOUR_NS = 3_600_000_000_000
_HEADER = {"schema_version": TELEMETRY_SCHEMA_VERSION, "event": "session_header", "session_id": "s", "mode": "shadow"}
_OK_MEMORY = [(0, 500_000_000), (_HOUR_NS, 550_000_000)]


def _row(stage: str, cid: str, ts: int, payload: dict) -> dict:
    """controller の ``TelemetryWriter.write_stage`` と同じキーを持つ stage 行を作る。

    gate が読むのは stage/correlation_id/timestamp_ns/payload だけなので、
    それ以外の列は固定値で埋めます。
    """
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION, "event": "stage", "stage": stage,
        "correlation_id": cid, "timestamp_ns": ts, "latency_ns": 0, "queue_depth": 0, "payload": payload,
    }


def _tick(
    i: int, *, from_state: str = "gameplay", to_state: str | None = None, decision: str = "move",
    intent: str | None = None, effects: tuple[str, ...] = ("move",), latency_ms: int = 50,
) -> list[dict]:
    """1 frame 分の capture → policy → state_machine → effect 行を controller と同じ順で作る。

    capture の captured_ns から最後の effect(無ければ policy)までが end-to-end 遅延になります。
    """
    cid = f"s:{i}"
    t0 = 1_000 * _MS + i * 100 * _MS
    end = t0 + latency_ms * _MS
    rows = [
        _row("capture", cid, t0 + _MS, {"frame_index": i, "captured_ns": t0, "dropped_frames": 0}),
        _row("policy", cid, end, {"kind": decision, "ui_intent": {"kind": intent} if intent else None}),
        _row("state_machine", cid, end, {
            "from_state": from_state, "to_state": to_state or from_state, "effects": list(effects),
        }),
    ]
    rows += [_row("effect", cid, end, {"kind": kind, "disposition": "proposed"}) for kind in effects]
    return rows


def _shutdown() -> list[dict]:
    """正常終了時に controller が最後に書く shutdown 行を作る。"""
    return [_row("shutdown", "s:control", 10**15, {"exit_code": 0, "reason": "max_frames"})]


def _clean_rows() -> list[dict]:
    """全 gate を満たす session(gameplay 移動・level-up 1回選択・release_all)を作る。"""
    rows = [_HEADER]
    for i in range(20):
        rows += _tick(i)
    rows += _tick(20, to_state="level_up", effects=("release_all",))
    rows += _tick(21, from_state="level_up", decision="ui", intent="choose_card", effects=("release_all", "ui_click"))
    rows += _tick(22, from_state="level_up", to_state="gameplay", decision="no_op", effects=())
    return rows + _shutdown()


def test_clean_session_passes_all_gates_and_is_never_formal_eligible():
    verdict = issue_gate_verdict(_clean_rows(), _OK_MEMORY)
    assert verdict["status"] == "PASS", verdict["fail_reasons"]
    assert verdict["fail_reasons"] == []
    assert verdict["formal_shadow_eligible"] is False
    assert verdict["session_id"] == "s" and verdict["mode"] == "shadow"
    metrics = verdict["metrics"]
    assert metrics["level_up_candidate_eligible"] == 1 and metrics["level_up_candidate_invalid"] == 0
    assert metrics["end_to_end_samples"] == 23


def test_controller_exception_counts_as_process_crash():
    rows = _clean_rows()
    rows.insert(-1, _row("error", "s:control", 10**14, {"stage": "controller", "type": "RuntimeError", "message": "x"}))
    verdict = issue_gate_verdict(rows, _OK_MEMORY)
    assert verdict["status"] == "FAIL" and verdict["fail_reasons"] == ["process_crash"]
    assert verdict["metrics"]["process_crash_count"] == 1


def test_stage_error_is_not_a_process_crash():
    rows = _clean_rows()
    rows.insert(-1, _row("error", "s:3", 10**14, {"stage": "detector", "type": "RuntimeError", "message": "x"}))
    assert issue_gate_verdict(rows, _OK_MEMORY)["status"] == "PASS"


def _with_exit(exit_code: int, reason: str) -> list[dict]:
    """clean session の shutdown 行だけを指定 exit_code/reason に差し替える(error 行は足さない)。"""
    rows = _clean_rows()
    rows[-1] = _row("shutdown", "s:control", 10**15, {"exit_code": exit_code, "reason": reason})
    return rows


@pytest.mark.parametrize(
    ("exit_code", "reason"),
    [
        (EXIT_ERROR, "input_ack_failed:move"),  # error 行を書かない EXIT_ERROR 経路
        (EXIT_ERROR, "interrupted"),
        (EXIT_TERMINAL_FAILURE, "terminal:failed"),
        (99, "unknown"),
    ],
)
def test_abnormal_shutdown_exit_code_counts_as_process_crash(exit_code, reason):
    verdict = issue_gate_verdict(_with_exit(exit_code, reason), _OK_MEMORY)
    assert verdict["status"] == "FAIL" and verdict["fail_reasons"] == ["process_crash"]
    assert verdict["metrics"]["process_crash_count"] == 1


def test_health_stop_exit_code_is_not_a_process_crash():
    rows = _with_exit(EXIT_HEALTH_STOP, "health_stop:capture_gap")
    rows.insert(-1, _row("health", "s:control", 10**14, {"verdict": "stop"}))
    verdict = issue_gate_verdict(rows, _OK_MEMORY)
    assert verdict["fail_reasons"] == ["health_stop"]
    assert verdict["metrics"]["process_crash_count"] == 0


def test_missing_shutdown_row_counts_as_process_crash():
    verdict = issue_gate_verdict(_clean_rows()[:-1], _OK_MEMORY)
    assert verdict["fail_reasons"] == ["process_crash"]
    assert verdict["metrics"]["shutdown_row_present"] is False


def test_health_stop_row_fails_gate():
    rows = _clean_rows()
    rows.insert(-1, _row("health", "s:control", 10**14, {"verdict": "stop"}))
    verdict = issue_gate_verdict(rows, _OK_MEMORY)
    assert verdict["fail_reasons"] == ["health_stop"] and verdict["metrics"]["health_stop_count"] == 1


@pytest.mark.parametrize(
    ("from_state", "effects"),
    [
        ("level_up", ("move",)),
        ("paused", ("move",)),
        ("gameplay", ("release_all", "ui_click")),
        ("unknown", ("ui_key",)),
    ],
)
def test_input_action_in_wrong_state_fails_gate(from_state, effects):
    rows = _clean_rows()
    rows[-1:-1] = _tick(50, from_state=from_state, decision="ui", effects=effects)
    verdict = issue_gate_verdict(rows, _OK_MEMORY)
    assert "wrong_state_action_proposal" in verdict["fail_reasons"]
    assert verdict["metrics"]["wrong_state_action_proposal_count"] == 1


def test_ui_actions_in_every_modal_ui_state_and_release_all_anywhere_are_allowed():
    rows = _clean_rows()
    rows[-1:-1] = _tick(50, from_state="chest", decision="ui", intent="ack_chest", effects=("release_all", "ui_click"))
    rows[-1:-1] = _tick(51, from_state="target_reached_pending_transition", decision="ui", intent="confirm",
                        effects=("release_all", "ui_key"))
    rows[-1:-1] = _tick(52, from_state="unknown", decision="no_op", effects=("release_all",))
    assert issue_gate_verdict(rows, _OK_MEMORY)["status"] == "PASS"


def test_effect_without_state_machine_row_for_same_frame_is_wrong_state():
    rows = _clean_rows()
    rows.insert(-1, _row("effect", "s:99", 10**14, {"kind": "move", "disposition": "proposed"}))
    assert issue_gate_verdict(rows, _OK_MEMORY)["metrics"]["wrong_state_action_proposal_count"] == 1


def _level_up_visit(start: int, rejected: int) -> list[dict]:
    """level-up 画面1訪問分: ``rejected`` 回候補を解決できず、その後1回選択を送る。

    送信後の tick(ack 待ち)は判定対象外であることも確かめるため、送信後に無効果の tick を1つ足します。
    """
    rows = _tick(start, to_state="level_up", effects=("release_all",))
    for k in range(rejected):
        rows += _tick(start + 1 + k, from_state="level_up", decision="ui", intent="choose_card", effects=())
    n = start + 1 + rejected
    rows += _tick(n, from_state="level_up", decision="ui", intent="choose_card", effects=("release_all", "ui_click"))
    rows += _tick(n + 1, from_state="level_up", decision="ui", intent="choose_card", effects=())
    rows += _tick(n + 2, from_state="level_up", to_state="gameplay", decision="no_op", effects=())
    return rows


def _level_up_session(visits: int, rejected_total: int) -> list[dict]:
    """``visits`` 回の level-up 訪問のうち先頭訪問だけで ``rejected_total`` 回拒否される session を作る。"""
    rows = [_HEADER]
    for v in range(visits):
        rows += _level_up_visit(v * 10, rejected_total if v == 0 else 0)
    return rows + _shutdown()


def test_level_up_candidate_invalid_rate_at_threshold_passes():
    # 1 invalid / (1 invalid + 199 successful first sends) = 0.5%
    verdict = issue_gate_verdict(_level_up_session(199, 1), _OK_MEMORY)
    assert verdict["metrics"]["level_up_candidate_eligible"] == 200
    assert verdict["metrics"]["level_up_candidate_invalid"] == 1
    assert verdict["status"] == "PASS", verdict["fail_reasons"]


def test_level_up_candidate_invalid_rate_above_threshold_fails():
    verdict = issue_gate_verdict(_level_up_session(199, 2), _OK_MEMORY)
    assert verdict["metrics"]["level_up_candidate_invalid"] == 2
    assert verdict["fail_reasons"] == ["level_up_candidate_invalid_rate_exceeded"]


def test_level_up_button_intents_are_not_candidate_proposals():
    rows = [_HEADER] + _tick(0, to_state="level_up", effects=("release_all",))
    rows += _tick(1, from_state="level_up", decision="ui", intent="reroll", effects=())
    rows += _shutdown()
    assert issue_gate_verdict(rows, _OK_MEMORY)["metrics"]["level_up_candidate_eligible"] == 0


def test_p99_end_to_end_uses_nearest_rank_and_fails_above_110ms():
    def session(slow: int) -> list[dict]:
        rows = [_HEADER]
        for i in range(100):
            rows += _tick(i, latency_ms=200 if i < slow else 100)
        return rows + _shutdown()

    one_slow = issue_gate_verdict(session(1), _OK_MEMORY)
    assert one_slow["metrics"]["end_to_end_p99_ns"] == 100 * _MS and one_slow["status"] == "PASS"
    two_slow = issue_gate_verdict(session(2), _OK_MEMORY)
    assert two_slow["metrics"]["end_to_end_p99_ns"] == 200 * _MS
    assert two_slow["fail_reasons"] == ["end_to_end_p99_exceeded"]


def test_frame_without_policy_is_not_an_end_to_end_sample_and_no_samples_fails_closed():
    rows = [_HEADER, _row("capture", "s:0", 2 * _MS, {"captured_ns": _MS}),
            _row("obs", "s:0", 3 * _MS, {"emitted": False})] + _shutdown()
    verdict = issue_gate_verdict(rows, _OK_MEMORY)
    assert verdict["metrics"]["end_to_end_samples"] == 0
    assert verdict["fail_reasons"] == ["end_to_end_samples_unavailable"]


def test_memory_growth_gate():
    assert memory_growth_bytes_per_hour(_OK_MEMORY) == pytest.approx(50_000_000)
    leaking = [(0, 0), (_HOUR_NS // 2, 125_000_000), (_HOUR_NS, 250_000_000)]
    assert issue_gate_verdict(_clean_rows(), leaking)["fail_reasons"] == ["memory_growth_exceeded"]
    missing = issue_gate_verdict(_clean_rows())
    assert missing["fail_reasons"] == ["memory_samples_unavailable"]
    assert missing["metrics"]["memory_growth_bytes_per_hour"] is None


@pytest.mark.parametrize("samples", [[(0, 1)], [(5, 1), (5, 2)], [(5, 1), (1, 2)]])
def test_memory_samples_need_two_strictly_increasing_timestamps(samples):
    with pytest.raises(ValueError):
        memory_growth_bytes_per_hour(samples)


def _write_real_telemetry(path: Path) -> None:
    """実物の ``TelemetryWriter`` で clean session を書き出す(JSONL 形式の往復確認用)。"""
    header = TelemetrySessionHeader(
        session_id="s", mode="shadow", target_profile_hash="a" * 64, game_build_id="b",
        controller_build_id="c", artifact_hashes={"m": "d" * 64}, host={"h": 1}, device={"d": 1},
        dependency_versions={"x": "1"},
    )
    writer = TelemetryWriter(path, header)
    for row in _clean_rows()[1:]:
        writer.write_stage(row["stage"], correlation_id=row["correlation_id"], timestamp_ns=row["timestamp_ns"],
                           latency_ns=0, queue_depth=0, payload=row["payload"])
    writer.close()


def test_read_telemetry_round_trips_real_writer_output(tmp_path):
    path = tmp_path / "t.jsonl"
    _write_real_telemetry(path)
    assert issue_gate_verdict(read_telemetry(path), _OK_MEMORY)["status"] == "PASS"


@pytest.mark.parametrize(
    "text",
    ["", "{broken\n", json.dumps({**_HEADER, "schema_version": "other"}) + "\n",
     json.dumps(_row("capture", "s:0", 1, {"captured_ns": 0})) + "\n"],
)
def test_read_telemetry_rejects_broken_input(tmp_path, text):
    path = tmp_path / "t.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        issue_gate_verdict(read_telemetry(path))


def test_cli_writes_json_and_markdown_and_exits_zero_even_on_fail(tmp_path):
    telemetry = tmp_path / "t.jsonl"
    _write_real_telemetry(telemetry)
    out_json, out_md = tmp_path / "gate.json", tmp_path / "gate.md"
    assert cli.main(["--telemetry", str(telemetry), "--output-json", str(out_json),
                     "--output-markdown", str(out_md)]) == 0
    verdict = json.loads(out_json.read_text(encoding="utf-8"))
    assert verdict["status"] == "FAIL" and verdict["fail_reasons"] == ["memory_samples_unavailable"]
    assert verdict["formal_shadow_eligible"] is False and len(verdict["telemetry_sha256"]) == 64
    assert "formal_shadow_eligible: false" in out_md.read_text(encoding="utf-8")

    memory = tmp_path / "memory.json"
    memory.write_text(json.dumps(_OK_MEMORY), encoding="utf-8")
    assert cli.main(["--telemetry", str(telemetry), "--memory-samples", str(memory),
                     "--output-json", str(out_json), "--output-markdown", str(out_md)]) == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["status"] == "PASS"


def test_cli_rejects_missing_input_with_nonzero_exit(tmp_path):
    assert cli.main(["--telemetry", str(tmp_path / "missing.jsonl"), "--output-json", str(tmp_path / "g.json"),
                     "--output-markdown", str(tmp_path / "g.md")]) == 1
    assert not (tmp_path / "g.json").exists()

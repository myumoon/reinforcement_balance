"""controller telemetry JSONL から M11 shadow gate(6閾値)の PASS/FAIL を判定する。

controller が書いた telemetry を先頭から1回だけ読み、process crash・health stop・
状態違反の入力提案・level-up 候補の解決失敗率・end-to-end 遅延 p99・memory 増加率を
閾値と比べます。1つでも外れるか測れなければ FAIL にします(fail-closed)。
結果には必ず ``formal_shadow_eligible=False`` を入れ、正式 shadow verdict の代わりに使わせません(I4)。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any
import uuid

from .controller import EXIT_HEALTH_STOP, EXIT_OK
from .telemetry import TELEMETRY_SCHEMA_VERSION

GATE_SCHEMA_VERSION = "survivors.controller_gate.v1"
MAX_LEVEL_UP_CANDIDATE_INVALID_RATE = 0.005
MAX_END_TO_END_P99_NS = 110_000_000
# 200MB は10進(200,000,000 bytes)で解釈する。MiB より厳しい側。
MAX_MEMORY_GROWTH_BYTES_PER_HOUR = 200_000_000
_NS_PER_HOUR = 3_600_000_000_000

# state_machine.py の実装から読み取った「どの状態でどの入力 effect が出せるか」。
# move は _reduce_gameplay(GAMEPLAY)だけ、ui_click/ui_key は _dispatch_ui_click を呼ぶ
# LEVEL_UP/CHEST(_reduce_modal_ui_screen)と TARGET_REACHED_PENDING_TRANSITION だけが出す。
# release_all は全状態の安全解放なので検査対象外。
_ALLOWED_FROM_STATES = {
    "move": frozenset({"gameplay"}),
    "ui_click": frozenset({"level_up", "chest", "target_reached_pending_transition"}),
    "ui_key": frozenset({"level_up", "chest", "target_reached_pending_transition"}),
}
_UI_EFFECTS = frozenset({"ui_click", "ui_key"})
_CANDIDATE_INTENTS = frozenset({"choose_card", "choose_fallback"})


def read_telemetry(path: Path | str) -> Iterator[dict[str, Any]]:
    """telemetry JSONL を1行ずつ検証しながら dict として返す。

    全行を memory へ載せず逐次読みます。先頭行が session_header でない、
    JSON が壊れている、schema_version が違う、といった入力は ValueError で拒否します。
    """
    number = 0
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"telemetry line {number}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict) or row.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
                raise ValueError(f"telemetry line {number}: schema_version must be {TELEMETRY_SCHEMA_VERSION}")
            if (number == 1) != (row.get("event") == "session_header"):
                raise ValueError(f"telemetry line {number}: session_header must be exactly the first row")
            yield row
    if number == 0:
        raise ValueError("telemetry is empty")


def memory_growth_bytes_per_hour(samples: Sequence[Sequence[int]]) -> float:
    """(timestamp_ns, rss_bytes) の列から最小二乗の傾きで memory 増加率(bytes/hour)を返す。

    telemetry JSONL には memory 情報が無いため、sample の収集は呼び出し側
    (fixture replay tool 等の別タスク)の責務です。ここは計算だけを行います。
    2点以上で timestamp が狭義単調増加していないと ValueError を返します。
    """
    if len(samples) < 2:
        raise ValueError("memory samples need at least 2 points")
    times = [float(t) for t, _ in samples]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("memory sample timestamps must strictly increase")
    return statistics.linear_regression(times, [float(rss) for _, rss in samples]).slope * _NS_PER_HOUR


def issue_gate_verdict(
    rows: Iterable[Mapping[str, Any]], memory_samples: Sequence[Sequence[int]] | None = None
) -> dict[str, Any]:
    """telemetry 行を1回走査し、M11 の6閾値を判定した verdict dict を返す。

    行は controller の書き込み順(policy → state_machine → effect)を前提に読みます。
    ``memory_samples`` が None なら memory gate は測定不能として FAIL 理由に残します。
    どの閾値も満たし、全て測定できたときだけ ``status="PASS"`` になります。

    - process crash: ``run()`` が捕捉した例外(error 行の payload.stage=="controller")、
      shutdown 行が無い(プロセスが途中で死んだ)こと、shutdown 行の exit_code が 0(正常)でも
      2(health STOP。health_stop 側で FAIL する)でもないことを数える。exit_code 3(EXIT_ERROR:
      例外・``interrupted``・``input_ack_failed``・入力解放失敗)と 4(EXIT_TERMINAL_FAILURE:
      terminal failure による停止)と未知の値はすべて crash として扱う。
    - wrong-state: 入力 effect の直前の同一 frame の state_machine 行の from_state が許可外、
      または同一 frame の state_machine 行が無いものを数える。
    - level-up candidate invalid: LEVEL_UP 訪問内で最初の UI 送信までの choose_card/choose_fallback
      提案のうち、state machine が候補を解決できず UI effect を出さなかった割合。送信後の tick は
      ack 待ちと区別できないため数えない。候補 validity 自体は telemetry に無く、resolver が
      validity=False・信頼度不足・0件/複数件の候補を拒否した結果として観測する。
    - p99 end-to-end: capture の captured_ns から同一 frame の最後の policy/effect 行までの nearest-rank p99。
      policy まで進まなかった frame は標本にしない。
    """
    header: Mapping[str, Any] = {}
    controller_errors = health_stops = wrong_state = eligible = invalid = 0
    shutdown_seen = level_up_sent = False
    shutdown_exit_code: Any = None
    last_policy: tuple[str | None, str | None] = (None, None)
    last_state: tuple[str | None, str | None] = (None, None)
    captured: dict[str, int] = {}
    finished: dict[str, int] = {}

    for row in rows:
        if row.get("event") == "session_header":
            header = row
            continue
        stage, cid, payload = row["stage"], row["correlation_id"], row["payload"]
        if stage == "error" and payload.get("stage") == "controller":
            controller_errors += 1
        elif stage == "health":
            health_stops += 1
        elif stage == "shutdown":
            shutdown_seen = True
            shutdown_exit_code = payload.get("exit_code")
        elif stage == "capture":
            captured[cid] = payload["captured_ns"]
        elif stage == "policy":
            finished[cid] = row["timestamp_ns"]
            last_policy = (cid, (payload.get("ui_intent") or {}).get("kind"))
        elif stage == "state_machine":
            from_state, to_state = payload["from_state"], payload["to_state"]
            sent_ui = bool(_UI_EFFECTS.intersection(payload["effects"]))
            last_state = (cid, from_state)
            if from_state != "level_up":
                level_up_sent = False
            else:
                if (to_state == "level_up" and not level_up_sent and last_policy[0] == cid
                        and last_policy[1] in _CANDIDATE_INTENTS):
                    eligible += 1
                    invalid += not sent_ui
                level_up_sent = level_up_sent or sent_ui
        elif stage == "effect":
            finished[cid] = row["timestamp_ns"]
            allowed = _ALLOWED_FROM_STATES.get(payload["kind"])
            if allowed is not None and (last_state[0] != cid or last_state[1] not in allowed):
                wrong_state += 1

    latencies = sorted(finished[cid] - captured[cid] for cid in finished if cid in captured)
    p99 = latencies[math.ceil(0.99 * len(latencies)) - 1] if latencies else None
    invalid_rate = invalid / eligible if eligible else 0.0
    growth = memory_growth_bytes_per_hour(memory_samples) if memory_samples is not None else None
    abnormal_exit = shutdown_seen and shutdown_exit_code not in (EXIT_OK, EXIT_HEALTH_STOP)
    crashes = controller_errors + (not shutdown_seen) + abnormal_exit

    fail_reasons = [
        reason for reason, failed in (
            ("process_crash", crashes > 0),
            ("health_stop", health_stops > 0),
            ("wrong_state_action_proposal", wrong_state > 0),
            ("level_up_candidate_invalid_rate_exceeded", invalid_rate > MAX_LEVEL_UP_CANDIDATE_INVALID_RATE),
            ("end_to_end_samples_unavailable", p99 is None),
            ("end_to_end_p99_exceeded", p99 is not None and p99 > MAX_END_TO_END_P99_NS),
            ("memory_samples_unavailable", growth is None),
            ("memory_growth_exceeded", growth is not None and growth >= MAX_MEMORY_GROWTH_BYTES_PER_HOUR),
        ) if failed
    ]
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "status": "FAIL" if fail_reasons else "PASS",
        "session_id": header.get("session_id"),
        "mode": header.get("mode"),
        "formal_shadow_eligible": False,
        "fail_reasons": fail_reasons,
        "metrics": {
            "process_crash_count": crashes,
            "controller_error_rows": controller_errors,
            "shutdown_row_present": shutdown_seen,
            "health_stop_count": health_stops,
            "wrong_state_action_proposal_count": wrong_state,
            "level_up_candidate_eligible": eligible,
            "level_up_candidate_invalid": invalid,
            "level_up_candidate_invalid_rate": invalid_rate,
            "end_to_end_samples": len(latencies),
            "end_to_end_p99_ns": p99,
            "memory_samples": 0 if memory_samples is None else len(memory_samples),
            "memory_growth_bytes_per_hour": growth,
        },
        "thresholds": {
            "process_crash_count": 0,
            "health_stop_count": 0,
            "wrong_state_action_proposal_count": 0,
            "level_up_candidate_invalid_rate_max": MAX_LEVEL_UP_CANDIDATE_INVALID_RATE,
            "end_to_end_p99_ns_max": MAX_END_TO_END_P99_NS,
            "memory_growth_bytes_per_hour_below": MAX_MEMORY_GROWTH_BYTES_PER_HOUR,
        },
    }


def render_gate_markdown(verdict: Mapping[str, Any]) -> str:
    """verdict を人が読める Markdown へ整形する。

    状態・失敗理由・各指標と閾値を並べ、正式 verdict ではないことを明記します。
    """
    lines = [
        "# Survivors controller shadow gate (M11)",
        "",
        f"Status: **{verdict['status']}**",
        "",
        f"- session_id: `{verdict['session_id']}` / mode: `{verdict['mode']}`",
        f"- telemetry_sha256: `{verdict.get('telemetry_sha256')}`",
        f"- fail_reasons: {', '.join(verdict['fail_reasons']) or 'none'}",
        "- formal_shadow_eligible: false",
        "",
        "| metric | value |",
        "|---|---|",
        *(f"| {name} | {value} |" for name, value in verdict["metrics"].items()),
        "",
        f"Thresholds: `{json.dumps(verdict['thresholds'], sort_keys=True)}`",
        "",
        "This gate result is not a formal shadow verdict and must not replace a live canary.",
        "",
    ]
    return "\n".join(lines)


def write_gate_verdict(verdict: Mapping[str, Any], json_path: Path | str, markdown_path: Path | str) -> None:
    """verdict を JSON と Markdown へ一時ファイル経由で書き出す。

    両方の一時ファイルを書き終えてから置き換えるので、書き込み途中の壊れた
    ファイルが残りません。
    """
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    if json_path == markdown_path:
        raise ValueError("JSON and Markdown verdict paths must differ")
    token = uuid.uuid4().hex
    pairs = (
        (json_path, json.dumps(verdict, indent=2, sort_keys=True) + "\n"),
        (markdown_path, render_gate_markdown(verdict)),
    )
    temps = [path.with_name(f".{path.name}.{token}.tmp") for path, _ in pairs]
    try:
        for (_, text), temp in zip(pairs, temps):
            temp.parent.mkdir(parents=True, exist_ok=True)
            temp.write_text(text, encoding="utf-8")
        # ponytail: 2回の os.replace の間で落ちると JSON だけ新しくなる。backup/rollback が要るなら
        # spikes/survivors_vertical_feasibility.write_verdict の .bak 方式へ上げる。
        for (path, _), temp in zip(pairs, temps):
            os.replace(temp, path)
    finally:
        for temp in temps:
            temp.unlink(missing_ok=True)

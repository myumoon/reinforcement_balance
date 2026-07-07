from __future__ import annotations

import json
from pathlib import Path

from games.survivors.run_event_logger import JsonlEventLogger


def test_jsonl_event_logger_writes_one_line(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(path)

    logger.write("bootstrap_gate.result", step=123, payload={"weapon_key": "garlic", "p10": 456.0})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "bootstrap_gate.result"
    assert event["step"] == 123
    assert event["payload"] == {"weapon_key": "garlic", "p10": 456.0}
    assert isinstance(event["time_unix"], float)


def test_jsonl_event_logger_none_path_is_noop(tmp_path: Path):
    logger = JsonlEventLogger(None)
    logger.write("anything", step=1, payload={"ok": True})
    assert list(tmp_path.iterdir()) == []


def test_jsonl_event_logger_child_prefix(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(path).child("bootstrap_gate")

    logger.write("skip", step=10, payload={"reason": "in_flight"})

    event = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert event["event"] == "bootstrap_gate.skip"

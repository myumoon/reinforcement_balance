from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JsonlEventLogger:
    path: Path | None
    prefix: str = ""

    def __init__(self, path: Path | str | None, prefix: str = "") -> None:
        object.__setattr__(self, "path", Path(path) if path is not None else None)
        object.__setattr__(self, "prefix", prefix.strip("."))

    def child(self, prefix: str) -> "JsonlEventLogger":
        clean = prefix.strip(".")
        full = clean if not self.prefix else f"{self.prefix}.{clean}"
        return JsonlEventLogger(self.path, full)

    def write(self, event: str, step: int, payload: dict[str, Any]) -> None:
        if self.path is None:
            return
        clean_event = event.strip(".")
        full_event = clean_event if not self.prefix else f"{self.prefix}.{clean_event}"
        record = {
            "event": full_event,
            "step": int(step),
            "time_unix": time.time(),
            "payload": payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

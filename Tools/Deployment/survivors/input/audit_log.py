"""入力 lease の append-only JSONL audit writer。
受信・ack・release の時刻と理由を1行ずつ即時 flush し、事故後の追跡情報を残します。
"""
from __future__ import annotations
import json
from pathlib import Path
import threading
from typing import Any
class AuditLog:
    """process 内で直列化された durable JSONL logger。
    1 record を1回の書込みで追記し、複数 thread の行混在を lock で防ぎます。
    """
    def __init__(self, path: Path | str) -> None:
        """audit file の親 directory を作成する。
        helper 起動直後から記録可能な状態にします。
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    def write(self, event: str, **fields: Any) -> None:
        """event と timestamp fields を JSONL へ追記して flush する。
        compact・key sort 済み JSON にして、機械検査と差分確認を安定させます。
        """
        row = {"event": event, **fields}
        encoded = json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded + "\n")
            stream.flush()

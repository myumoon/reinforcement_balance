"""controller pipeline の structured telemetry を JSONL へ逐次保存する。

やさしい説明: session の固定情報を先頭へ1行だけ書き、その後は各 stage の
完了情報を受け取るたびに追記・flushします。全eventをmemoryへ保持しません。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import threading
from typing import Any, Literal, TextIO


TELEMETRY_SCHEMA_VERSION = "survivors.controller_telemetry.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class TelemetrySessionHeader:
    """controller session 全体で不変な実行identityと環境情報。

    やさしい説明: どのprofile・build・artifact・hostで実行した記録かを
    JSONLの先頭へ固定し、後続eventの出所を1つのsessionへ結び付けます。
    """

    session_id: str
    mode: Literal["shadow", "live"]
    target_profile_hash: str
    game_build_id: str
    controller_build_id: str
    artifact_hashes: Mapping[str, str]
    host: Mapping[str, Any]
    device: Mapping[str, Any]
    dependency_versions: Mapping[str, str]
    deterministic_replay: Mapping[str, Any] = field(default_factory=dict)
    structured_telemetry_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """session identityとhash形式を記録開始前に検証する。

        やさしい説明: 不明なmodeや壊れたprofile/artifact hashを早期に拒否し、
        別sessionの記録を同じ証跡として扱う事故を防ぎます。
        """
        for name in ("session_id", "game_build_id", "controller_build_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.mode not in ("shadow", "live"):
            raise ValueError("mode must be shadow or live")
        if not isinstance(self.target_profile_hash, str) or _SHA256.fullmatch(
            self.target_profile_hash
        ) is None:
            raise ValueError("target_profile_hash must be a sha256 hex digest")
        if not isinstance(self.artifact_hashes, Mapping) or not self.artifact_hashes:
            raise ValueError("artifact_hashes must be a non-empty mapping")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or _SHA256.fullmatch(value) is None
            for name, value in self.artifact_hashes.items()
        ):
            raise ValueError("artifact_hashes must contain named sha256 hex digests")
        for name in ("host", "device", "dependency_versions"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or not value:
                raise ValueError(f"{name} must be a non-empty mapping")
        for name in ("deterministic_replay", "structured_telemetry_config"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"{name} must be a mapping")

    def to_wire(self) -> dict[str, Any]:
        """JSON headerへ変換できる独立したdictを返す。

        やさしい説明: 呼び出し側のdictを保持せず、その時点の固定情報だけを
        コピーして先頭recordへ書き込める形にします。
        """
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "target_profile_hash": self.target_profile_hash,
            "game_build_id": self.game_build_id,
            "controller_build_id": self.controller_build_id,
            "artifact_hashes": dict(self.artifact_hashes),
            "host": dict(self.host),
            "device": dict(self.device),
            "dependency_versions": dict(self.dependency_versions),
            "deterministic_replay": dict(self.deterministic_replay),
            "structured_telemetry_config": dict(self.structured_telemetry_config),
        }


class TelemetryWriter:
    """thread-safeなappend-only JSONL streaming writer。

    やさしい説明: 非同期stageが同時に完了しても1行を混ぜず、実際に記録した
    順番をsequenceへ残します。各writeは即時flushし、event列は保存しません。
    """

    __slots__ = ("path", "_stream", "_lock", "_sequence", "_session_id", "_mode")

    def __init__(self, path: Path | str, header: TelemetrySessionHeader) -> None:
        """新規fileを作りsession headerを直ちに書き込む。

        やさしい説明: 既存fileは上書きせず、親directoryだけを作成してから
        先頭recordを保存します。初回write失敗時はhandleを閉じて例外を返します。
        """
        if not isinstance(header, TelemetrySessionHeader):
            raise ValueError("header must be TelemetrySessionHeader")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._sequence = 0
        self._session_id = header.session_id
        self._mode = header.mode
        row = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "event": "session_header",
            "sequence": 0,
            **header.to_wire(),
        }
        encoded = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self._stream: TextIO = self.path.open("x", encoding="utf-8", newline="\n")
        try:
            self._stream.write(encoded + "\n")
            self._stream.flush()
        except BaseException:
            self._stream.close()
            raise

    def write_stage(
        self,
        stage: str,
        *,
        correlation_id: str,
        timestamp_ns: int,
        latency_ns: int,
        queue_depth: int,
        payload: Mapping[str, Any] | None = None,
    ) -> int:
        """stage完了eventを1行追記し、割り当てたsequenceを返す。

        やさしい説明: frame/sessionをcorrelation_idで追跡し、処理時刻・所要時間・
        queue深さとstage固有情報を同じrecordへ保存してすぐ読める状態にします。
        """
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a non-empty string")
        if not isinstance(correlation_id, str) or not correlation_id:
            raise ValueError("correlation_id must be a non-empty string")
        for name, value in (
            ("timestamp_ns", timestamp_ns),
            ("latency_ns", latency_ns),
            ("queue_depth", queue_depth),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if payload is not None and not isinstance(payload, Mapping):
            raise ValueError("payload must be a mapping or None")

        with self._lock:
            sequence = self._sequence + 1
            row = {
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "event": "stage",
                "sequence": sequence,
                "session_id": self._session_id,
                "mode": self._mode,
                "stage": stage,
                "correlation_id": correlation_id,
                "timestamp_ns": timestamp_ns,
                "latency_ns": latency_ns,
                "queue_depth": queue_depth,
                "payload": dict(payload) if payload is not None else {},
            }
            encoded = json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            self._stream.write(encoded + "\n")
            self._stream.flush()
            self._sequence = sequence
            return sequence

    def flush(self) -> None:
        """現在までのJSONLをOSのfile bufferへ反映する。

        やさしい説明: 通常は各eventで自動flushされますが、shutdown処理からも
        明示的に同じ保証を要求できるようにします。
        """
        with self._lock:
            self._stream.flush()

    def close(self) -> None:
        """残ったbufferをflushしてfile handleを閉じる。

        やさしい説明: 複数のshutdown経路から呼ばれても、二重closeを安全に無視します。
        """
        with self._lock:
            if self._stream.closed:
                return
            self._stream.flush()
            self._stream.close()

"""固定 15 Hz のデシジョンスケジューラ。

combat tick を一定間隔に保ち、バックログ時は最新 snapshot だけを使う。
screen / UI 遷移評価は capture cadence (30 Hz) 側で行う。
"""
from __future__ import annotations

import time
from dataclasses import dataclass


DECISION_HZ: int = 15
_DECISION_INTERVAL_NS: int = 1_000_000_000 // DECISION_HZ  # 66.67 ms


@dataclass(frozen=True)
class DecisionTiming:
    """decision 1 回分のタイミングメタデータ。

    inference latency 計測と JSONL telemetry への出力で使う。
    """

    decision_id: str
    scheduled_ns: int
    inference_started_ns: int
    inference_finished_ns: int

    @property
    def inference_latency_ms(self) -> float:
        """inference にかかった時間をミリ秒で返す。"""
        return (self.inference_finished_ns - self.inference_started_ns) / 1_000_000.0

    @property
    def wall_latency_ms(self) -> float:
        """スケジュール時刻から inference 完了までの時間をミリ秒で返す。"""
        return (self.inference_finished_ns - self.scheduled_ns) / 1_000_000.0


class DecisionScheduler:
    """15 Hz 固定 cadence の decision タイミングを管理する。

    バックログが溜まった場合は最新 snapshot を使い、古い tick はスキップする。
    wall-clock を直接持たず、呼び出し側から timestamp を受け取る。
    """

    def __init__(self, *, hz: int = DECISION_HZ) -> None:
        """cadence と次回 scheduled_ns を初期化する。

        hz は 1 以上の整数のみ受理する。
        """
        if type(hz) is not int or hz <= 0:
            raise ValueError("hz must be a positive integer")
        self._hz = hz
        self._interval_ns: int = 1_000_000_000 // hz
        self._next_scheduled_ns: int | None = None

    @property
    def hz(self) -> int:
        """設定されている decision レートを返す。"""
        return self._hz

    @property
    def interval_ns(self) -> int:
        """tick 間隔をナノ秒で返す。"""
        return self._interval_ns

    def should_decide(self, now_ns: int) -> bool:
        """現在時刻に decision が必要かどうかを返す。

        first call では常に True を返し、以降は interval に従う。
        バックログの場合も True を返し、古い tick をスキップする。
        """
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative int")
        if self._next_scheduled_ns is None:
            return True
        return now_ns >= self._next_scheduled_ns

    def advance(self, now_ns: int) -> int:
        """decision 後に次の scheduled_ns を進めて現在の scheduled_ns を返す。

        バックログが溜まっていた場合は now_ns を基点に次回を設定する。
        """
        if self._next_scheduled_ns is None:
            scheduled = now_ns
        else:
            scheduled = self._next_scheduled_ns
        # バックログ消化: now より未来になるまで interval を加算
        next_ns = scheduled + self._interval_ns
        while next_ns <= now_ns:
            next_ns += self._interval_ns
        self._next_scheduled_ns = next_ns
        return scheduled

    def reset(self) -> None:
        """スケジュールをリセットする。

        新 run / episode 開始時に first-tick 動作へ戻す。
        """
        self._next_scheduled_ns = None

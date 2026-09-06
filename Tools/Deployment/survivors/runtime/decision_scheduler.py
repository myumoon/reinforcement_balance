"""固定 15 Hz のデシジョンスケジューラと時間ゲート。

combat tick を一定間隔に保ち、バックログ時は最新 snapshot だけを使って古い tick を
捨てる。あわせて snapshot age (captured_ns の鮮度) と inference timeout を、capture 層と
同じ単調高分解能時計 (`time.perf_counter_ns`) 上の値で判定する。Windows の
`time.monotonic_ns` は分解能が約 15.6 ms しかなく、1 tick ぶんの推論時間を測れない。
screen / UI 遷移評価は capture cadence (30 Hz) 側で行う。
これらの gate を runtime が必ず通ることで、古い画面に基づく移動指示や、
締切超過した推論結果がそのまま入力へ流れることを防ぐ。
"""
from __future__ import annotations

from dataclasses import dataclass


DECISION_HZ: int = 15
_DECISION_INTERVAL_NS: int = 1_000_000_000 // DECISION_HZ  # 66.67 ms

# capture は 30 Hz。3 frame 相当を超えた snapshot は移動判断に使わない。
DEFAULT_MAX_SNAPSHOT_AGE_NS: int = 100_000_000  # 100 ms
# 1 tick の推論に許す上限。超過した decision は結果を使わず安全側へ倒す。
DEFAULT_INFERENCE_TIMEOUT_NS: int = 20_000_000  # 20 ms


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
    """15 Hz 固定 cadence と時間ゲートを管理する。

    バックログが溜まった場合は最新 snapshot を使い、古い tick はスキップする。
    wall-clock を直接持たず、呼び出し側から perf_counter 系の timestamp を受け取る。
    """

    def __init__(
        self,
        *,
        hz: int = DECISION_HZ,
        max_snapshot_age_ns: int = DEFAULT_MAX_SNAPSHOT_AGE_NS,
        inference_timeout_ns: int = DEFAULT_INFERENCE_TIMEOUT_NS,
    ) -> None:
        """cadence・age 上限・inference 締切を初期化する。

        すべて正の整数のみ受理する。0 や負値は gate を無効化してしまうため拒否する。
        """
        if type(hz) is not int or hz <= 0:
            raise ValueError("hz must be a positive integer")
        if type(max_snapshot_age_ns) is not int or max_snapshot_age_ns <= 0:
            raise ValueError("max_snapshot_age_ns must be a positive integer")
        if type(inference_timeout_ns) is not int or inference_timeout_ns <= 0:
            raise ValueError("inference_timeout_ns must be a positive integer")
        self._hz = hz
        self._interval_ns: int = 1_000_000_000 // hz
        self._max_snapshot_age_ns = max_snapshot_age_ns
        self._inference_timeout_ns = inference_timeout_ns
        self._next_scheduled_ns: int | None = None
        self._decision_count = 0
        self._skipped_tick_count = 0

    @property
    def hz(self) -> int:
        """設定されている decision レートを返す。"""
        return self._hz

    @property
    def interval_ns(self) -> int:
        """tick 間隔をナノ秒で返す。"""
        return self._interval_ns

    @property
    def max_snapshot_age_ns(self) -> int:
        """移動判断に使える snapshot の最大 age をナノ秒で返す。"""
        return self._max_snapshot_age_ns

    @property
    def inference_timeout_ns(self) -> int:
        """1 decision の推論に許す最大時間をナノ秒で返す。"""
        return self._inference_timeout_ns

    @property
    def decision_count(self) -> int:
        """これまでに発行した decision 数を返す。"""
        return self._decision_count

    @property
    def skipped_tick_count(self) -> int:
        """バックログで飛ばした tick の総数を返す。

        30 分 run の schedule 健全性を telemetry で確認するために使う。
        """
        return self._skipped_tick_count

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

        バックログが溜まっていた場合は now_ns を基点に次回を設定し、
        飛ばした tick 数を skipped_tick_count へ積算する。
        """
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be a non-negative int")
        if self._next_scheduled_ns is None:
            scheduled = now_ns
        else:
            scheduled = self._next_scheduled_ns
        # バックログ消化: now より未来になるまで interval を加算
        next_ns = scheduled + self._interval_ns
        while next_ns <= now_ns:
            next_ns += self._interval_ns
            self._skipped_tick_count += 1
        self._next_scheduled_ns = next_ns
        self._decision_count += 1
        return scheduled

    def snapshot_age_ns(self, captured_ns: int, now_ns: int) -> int:
        """snapshot の age をナノ秒で返す。

        captured_ns は capture 層と同じ perf_counter 時計上の取得時刻。負の age
        (未来の timestamp) は時計不整合なので、そのまま負値として返し
        caller が拒否できるようにする。
        """
        return int(now_ns) - int(captured_ns)

    def is_snapshot_fresh(self, captured_ns: int, now_ns: int) -> bool:
        """snapshot が移動判断に使えるだけ新しいかを返す。

        captured_ns が 0 以下、未来、または age 上限超過なら False。
        assembler が timestamp を埋めなかった snapshot をそのまま使わせない。
        """
        if type(captured_ns) is not int or captured_ns <= 0:
            return False
        age = self.snapshot_age_ns(captured_ns, now_ns)
        return 0 <= age <= self._max_snapshot_age_ns

    def exceeded_inference_timeout(self, started_ns: int, finished_ns: int) -> bool:
        """1 decision の推論が締切を超えたかを返す。"""
        return (int(finished_ns) - int(started_ns)) > self._inference_timeout_ns

    def reset(self) -> None:
        """スケジュールをリセットする。

        新 run / episode 開始時に first-tick 動作へ戻す。統計値は保持する。
        """
        self._next_scheduled_ns = None

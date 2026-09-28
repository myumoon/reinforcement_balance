"""replay 専用の完全仮想時計と、timestamp 列から tick 列を決める純粋関数。

実時計(time.perf_counter_ns / time.sleep / time.monotonic など)を一切呼ばず、
advance()/advance_to()/sleep() で明示的に進めたときだけ時刻が動きます。
controller の ``clock_ns`` と ``sleep`` へそのまま渡せるので、同じ記録からは
何度再生しても同じ時刻列・同じ遅延値が得られます。
"""

from __future__ import annotations

from collections.abc import Sequence

NS_PER_SECOND = 1_000_000_000


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    """value が minimum 以上の int(bool 以外)であることを確かめて返す。

    時刻に float や bool が混ざると丸め方で結果が変わるため、入口で拒否します。
    """
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class VirtualClock:
    """明示的に進めたときだけ動く単調増加の仮想時計(ns)。

    ``clock()`` / ``now_ns()`` で現在時刻を返し、``advance(ns)`` で ns だけ進めます。
    ``sleep(seconds)`` は実際には待たず、その秒数ぶん時刻を進めるだけです。
    過去へ戻す操作は存在しないので、controller から見て時刻が逆行することはありません。
    """

    def __init__(self, start_ns: int = 0) -> None:
        """開始時刻 start_ns(非負の int)で時計を作る。"""
        self._now_ns = _require_int("start_ns", start_ns)

    def now_ns(self) -> int:
        """現在の仮想時刻(ns)を返す。"""
        return self._now_ns

    def __call__(self) -> int:
        """controller の ``clock_ns`` として使えるよう now_ns() と同じ値を返す。"""
        return self._now_ns

    def advance(self, ns: int) -> int:
        """時刻を ns(非負の int)だけ進め、進めた後の時刻を返す。"""
        self._now_ns += _require_int("ns", ns)
        return self._now_ns

    def advance_to(self, target_ns: int) -> int:
        """target_ns が未来ならそこまで進め、進めた後の時刻を返す(過去へは戻らない)。"""
        self._now_ns = max(self._now_ns, _require_int("target_ns", target_ns))
        return self._now_ns

    def sleep(self, seconds: float) -> None:
        """実時間は待たず、seconds 秒ぶん仮想時刻を進める(controller の ``sleep`` 注入用)。

        秒は ns へ四捨五入するので、同じ秒数からは常に同じ ns だけ進みます。
        """
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not seconds >= 0:
            raise ValueError("seconds must be a non-negative number")
        self._now_ns += round(seconds * NS_PER_SECOND)


def schedule_ticks(
    timestamps_ns: Sequence[int], *, period_ns: int, origin_ns: int | None = None
) -> tuple[int, ...]:
    """各 timestamp が origin から何番目の period tick に属するかを返す。

    例えば 15 Hz(period_ns≈66.7ms)の記録なら、各 frame が何 tick 目に届いたかが分かります。
    同じ tick に2つあれば重複、番号が飛べば取りこぼしです。origin を省くと先頭 timestamp を使います。
    timestamp が逆行していたり origin より前だったりすると ValueError にします(並べ替えて隠さない)。
    """
    period = _require_int("period_ns", period_ns, minimum=1)
    stamps = [_require_int("timestamp_ns", value) for value in timestamps_ns]
    if not stamps:
        return ()
    origin = stamps[0] if origin_ns is None else _require_int("origin_ns", origin_ns)
    ticks: list[int] = []
    previous = origin
    for stamp in stamps:
        if stamp < previous:
            raise ValueError("timestamps_ns must be non-decreasing and >= origin_ns")
        previous = stamp
        ticks.append((stamp - origin) // period)
    return tuple(ticks)

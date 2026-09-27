"""controller pipeline の health/watchdog 判定器(M5)。

やさしい説明: 毎フレームの計測値(capture時刻・focus・perception遅延・
観測の有効性・state machine の状態)と推論エラーを受け取り、閾値を超えたら
``STOP`` を返して以後ずっと ``STOP`` のまま固定(latch)します。
input release・nonzero exit・telemetry 書き込みは呼び出し側(controller.py)の
責務で、このモジュールは入力装置や TelemetryWriter に一切触れません。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import enum
import math
from typing import Any

from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema, DeployObservation
from reinbalance_survivors_contracts.ui_intent import ContractValidationError

from .state_machine import ControllerState

_MS = 1_000_000


class HealthVerdict(str, enum.Enum):
    """1回の判定結果。

    やさしい説明: ``OK`` は問題なし、``WARNING`` は異常の兆候(継続時間が
    まだ閾値未満)、``STOP`` は即座に input release + nonzero exit すべき状態です。
    """

    OK = "ok"
    WARNING = "warning"
    STOP = "stop"


class HealthReason(str, enum.Enum):
    """warning/stop の原因となった閾値の種類。

    やさしい説明: counters と first-failure context のキーとして使う名札です。
    """

    FOCUS_LOST = "focus_lost"
    CAPTURE_GAP = "capture_gap"
    PERCEPTION_P99 = "perception_p99"
    OBS_INVALID_STREAK = "obs_invalid_streak"
    UNKNOWN_STREAK = "unknown_streak"
    INFERENCE_ERROR = "inference_error"


def _require_int(name: str, value: Any, *, positive: bool = False) -> None:
    """``value`` が bool でない非負(または正)の int かを検証する。

    やさしい説明: 時刻や閾値に float や True が紛れ込むと比較が静かに
    狂うため、入口で ValueError にします。
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < (1 if positive else 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'non-negative'} int")


@dataclass(frozen=True, slots=True)
class HealthThresholds:
    """M5 の stop 閾値(単位は ns)。

    やさしい説明: 既定値は review-contract M5 の値そのものです。
    どれか1つでも「超えた」時点で STOP になります(等しい値は許容)。
    perception p99 は直近 ``perception_window`` 件で計算し、
    ``perception_min_samples`` 件たまるまでは評価しません(起動直後の1件の
    スパイクだけで p99=max となり停止する誤検知を避けるため)。
    """

    capture_gap_ns: int = 200 * _MS
    perception_p99_ns: int = 110 * _MS
    obs_invalid_streak_ns: int = 500 * _MS
    unknown_streak_ns: int = 1_000 * _MS
    perception_window: int = 300
    perception_min_samples: int = 100

    def __post_init__(self) -> None:
        """全閾値が正の int で、min_samples が window 以下かを検証する。

        やさしい説明: window より大きい min_samples は p99 を永遠に評価しない
        設定になり監視が黙って無効化されるため拒否します。
        """
        for name in (
            "capture_gap_ns",
            "perception_p99_ns",
            "obs_invalid_streak_ns",
            "unknown_streak_ns",
            "perception_window",
            "perception_min_samples",
        ):
            _require_int(name, getattr(self, name), positive=True)
        if self.perception_min_samples > self.perception_window:
            raise ValueError("perception_min_samples must not exceed perception_window")


@dataclass(frozen=True, slots=True)
class FailureContext:
    """最初の STOP がいつ・どの閾値で・どんな値で起きたか。

    やさしい説明: 事後調査で「何が最初に壊れたか」を1つに特定するための記録です。
    ``observed`` と ``threshold`` は数値比較系なら ns、focus/inference は None。
    """

    reason: HealthReason
    now_ns: int
    observed: int | None
    threshold: int | None
    detail: str

    def to_wire(self) -> dict[str, Any]:
        """telemetry payload へそのまま渡せる dict を返す。

        やさしい説明: enum を文字列へ直した JSON 化可能な形にします。
        """
        return {
            "reason": self.reason.value,
            "now_ns": self.now_ns,
            "observed": self.observed,
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class HealthReport:
    """現時点の verdict・counters・first-failure context の不変スナップショット。

    やさしい説明: controller.py はこれを ``to_wire()`` して telemetry の
    ``stage="health"`` payload として書き込みます。
    """

    verdict: HealthVerdict
    warning_counts: dict[str, int]
    stop_counts: dict[str, int]
    first_failure: FailureContext | None

    def to_wire(self) -> dict[str, Any]:
        """telemetry payload へそのまま渡せる dict を返す。

        やさしい説明: counters はコピーを返し、呼び出し側の変更が混ざらないようにします。
        """
        return {
            "verdict": self.verdict.value,
            "warning_counts": dict(self.warning_counts),
            "stop_counts": dict(self.stop_counts),
            "first_failure": None if self.first_failure is None else self.first_failure.to_wire(),
        }


def observation_is_valid(observation: DeployObservation | None, schema: DeployObsSchema) -> bool:
    """DeployObservation が policy へ渡せる状態かを既存契約で判定する。

    やさしい説明: 観測が作れなかった(None)か、``validate_for(schema)`` が
    ``ContractValidationError`` を出した場合を「無効」とします。判定ロジックは
    再実装せず、既存契約の検証をそのまま使います。
    """
    if observation is None:
        return False
    try:
        observation.validate_for(schema)
    except ContractValidationError:
        return False
    return True


def _p99(samples: deque[int]) -> int:
    """nearest-rank 法で p99 を返す。

    やさしい説明: 小さい順に並べて上位1%の境目の値を取ります(補間しない)。
    """
    ordered = sorted(samples)
    return ordered[math.ceil(0.99 * len(ordered)) - 1]


class HealthMonitor:
    """M5 の閾値を監視し、STOP を latch する状態付き watchdog。

    やさしい説明: controller.py からの使い方は次の3つだけです。

    - 毎フレーム ``ingest(...)`` を呼ぶ(perception 完了ごと)。
    - フレームが来なくても定期的に ``poll(now_ns)`` を呼ぶ(capture 停止の検知)。
    - 推論が例外を出したら ``record_inference_error(...)`` を呼ぶ(1回で STOP)。

    どれも ``HealthVerdict`` を返し、``STOP`` なら呼び出し側が input release と
    nonzero exit を行います。詳細は ``report()`` で取得します。一度 STOP に
    なると以後は常に STOP を返し、first-failure は最初の1件から変わりません。
    スレッド安全ではないため、呼び出しは controller の1スレッドに限定します。
    """

    def __init__(self, thresholds: HealthThresholds | None = None) -> None:
        """閾値を設定し、カウンタと streak を初期化する。

        やさしい説明: thresholds 省略時は M5 の既定値を使います。
        """
        self.thresholds = thresholds or HealthThresholds()
        self._latencies: deque[int] = deque(maxlen=self.thresholds.perception_window)
        self._last_now_ns: int | None = None
        self._last_capture_ns: int | None = None
        self._obs_invalid_since_ns: int | None = None
        self._unknown_since_ns: int | None = None
        self._warning_counts = {reason.value: 0 for reason in HealthReason}
        self._stop_counts = {reason.value: 0 for reason in HealthReason}
        self._first_failure: FailureContext | None = None

    @property
    def stopped(self) -> bool:
        """STOP が一度でも発生したかどうか。

        やさしい説明: True なら以後の判定は全て STOP です。
        """
        return self._first_failure is not None

    def _advance_clock(self, now_ns: int) -> None:
        """``now_ns`` を検証し、単調非減少であることを保証する。

        やさしい説明: 時計が巻き戻ると streak や gap が負になり異常を見逃すため、
        握り潰さず ValueError にします(呼び出し側は例外経路で release します)。
        """
        _require_int("now_ns", now_ns)
        if self._last_now_ns is not None and now_ns < self._last_now_ns:
            raise ValueError("now_ns must be monotonic non-decreasing")
        self._last_now_ns = now_ns

    def _stop(
        self, reason: HealthReason, now_ns: int, observed: int | None, threshold: int | None, detail: str
    ) -> None:
        """stop counter を増やし、最初の1件だけ first-failure として保存する。

        やさしい説明: 同じ tick で複数の閾値を超えても、最初に判定したものを
        first-failure にし、残りは counter にだけ反映します。
        """
        self._stop_counts[reason.value] += 1
        if self._first_failure is None:
            self._first_failure = FailureContext(reason, now_ns, observed, threshold, detail)

    def _verdict(self) -> HealthVerdict:
        """latch 状態と継続中の streak から verdict を決める。

        やさしい説明: STOP は WARNING より常に優先します。観測無効または
        UNKNOWN が閾値内で続いている間は WARNING です。
        """
        if self.stopped:
            return HealthVerdict.STOP
        if self._obs_invalid_since_ns is not None or self._unknown_since_ns is not None:
            return HealthVerdict.WARNING
        return HealthVerdict.OK

    def _check_capture_gap(self, now_ns: int, capture_ns: int) -> None:
        """直前 capture からの経過が閾値を超えていれば STOP にする。

        やさしい説明: 最初のフレームより前は比較対象がないので評価しません。
        """
        if self._last_capture_ns is None:
            return
        gap = capture_ns - self._last_capture_ns
        if gap > self.thresholds.capture_gap_ns:
            self._stop(HealthReason.CAPTURE_GAP, now_ns, gap, self.thresholds.capture_gap_ns, "capture gap exceeded")

    def _check_streak(
        self, reason: HealthReason, active: bool, since_ns: int | None, now_ns: int, limit_ns: int
    ) -> int | None:
        """継続系の条件を評価し、新しい streak 開始時刻を返す。

        やさしい説明: 条件が続いている時間が閾値を超えたら STOP、
        まだ閾値内なら warning counter を増やし、条件が消えたら None(リセット)を返します。
        """
        if not active:
            return None
        start = now_ns if since_ns is None else since_ns
        duration = now_ns - start
        if duration > limit_ns:
            self._stop(reason, now_ns, duration, limit_ns, f"{reason.value} exceeded")
        else:
            self._warning_counts[reason.value] += 1
        return start

    def ingest(
        self,
        *,
        now_ns: int,
        capture_timestamp_ns: int,
        window_focused: bool,
        perception_latency_ns: int,
        observation_valid: bool,
        controller_state: ControllerState,
    ) -> HealthVerdict:
        """1フレーム分の計測値を取り込み、全閾値を評価する。

        やさしい説明: 引数の意味は次のとおりです。

        - ``now_ns``: 判定時刻(単調時計)。
        - ``capture_timestamp_ns``: このフレームの capture 時刻(前フレームとの差が capture gap)。
        - ``window_focused``: PerceptionSnapshot の focus。False なら即 STOP。
        - ``perception_latency_ns``: capture から perception 完了までの遅延(p99 の母集団)。
        - ``observation_valid``: ``observation_is_valid()`` の結果。False が継続すると STOP。
        - ``controller_state``: ``StateMachine.context.state``。UNKNOWN が継続すると STOP。

        型不正や時計・capture 時刻の巻き戻りは ValueError(黙って無視しない)。
        """
        self._advance_clock(now_ns)
        _require_int("capture_timestamp_ns", capture_timestamp_ns)
        _require_int("perception_latency_ns", perception_latency_ns)
        for name, flag in (("window_focused", window_focused), ("observation_valid", observation_valid)):
            if not isinstance(flag, bool):
                raise ValueError(f"{name} must be a bool")
        if not isinstance(controller_state, ControllerState):
            raise ValueError("controller_state must be a ControllerState")
        if self._last_capture_ns is not None and capture_timestamp_ns < self._last_capture_ns:
            raise ValueError("capture_timestamp_ns must be monotonic non-decreasing")

        if not window_focused:
            self._stop(HealthReason.FOCUS_LOST, now_ns, None, None, "target window lost focus")

        self._check_capture_gap(now_ns, capture_timestamp_ns)
        self._last_capture_ns = capture_timestamp_ns

        self._latencies.append(perception_latency_ns)
        if len(self._latencies) >= self.thresholds.perception_min_samples:
            p99 = _p99(self._latencies)
            if p99 > self.thresholds.perception_p99_ns:
                self._stop(
                    HealthReason.PERCEPTION_P99, now_ns, p99, self.thresholds.perception_p99_ns, "perception p99 exceeded"
                )

        self._obs_invalid_since_ns = self._check_streak(
            HealthReason.OBS_INVALID_STREAK,
            not observation_valid,
            self._obs_invalid_since_ns,
            now_ns,
            self.thresholds.obs_invalid_streak_ns,
        )
        self._unknown_since_ns = self._check_streak(
            HealthReason.UNKNOWN_STREAK,
            controller_state is ControllerState.UNKNOWN,
            self._unknown_since_ns,
            now_ns,
            self.thresholds.unknown_streak_ns,
        )
        return self._verdict()

    def poll(self, now_ns: int) -> HealthVerdict:
        """フレームが届かない間の watchdog 判定を行う。

        やさしい説明: capture が止まると ``ingest`` が呼ばれず gap を検知できないため、
        controller のループから定期的に呼び、最後の capture から ``now_ns`` までの
        経過で capture gap を評価します。まだ1フレームも来ていなければ評価しません。
        """
        self._advance_clock(now_ns)
        if self._last_capture_ns is not None:
            gap = now_ns - self._last_capture_ns
            if gap > self.thresholds.capture_gap_ns:
                self._stop(
                    HealthReason.CAPTURE_GAP, now_ns, gap, self.thresholds.capture_gap_ns, "no capture within gap"
                )
        return self._verdict()

    def record_focus_lost(self, *, now_ns: int, detail: str = "target window lost focus") -> HealthVerdict:
        """capture 側で検出した focus/ウィンドウ状態の喪失を記録し、即 STOP にする。

        実 CaptureSession は focus を失うと frame を返さず ``TargetWindowStateError`` を送出するため、
        ``ingest(window_focused=False)`` には到達しません。その例外を受けた controller が呼びます。
        前面喪失以外(ウィンドウ消失・geometry 変化)も同じ例外なので、具体的な原因は ``detail`` に残します。
        """
        self._advance_clock(now_ns)
        self._stop(HealthReason.FOCUS_LOST, now_ns, None, None, detail or "target window lost focus")
        return HealthVerdict.STOP

    def record_inference_error(self, *, now_ns: int, detail: str) -> HealthVerdict:
        """推論エラーを記録し、1回で STOP にする。

        やさしい説明: policy/detector の推論が例外を出したら、その内容を
        ``detail`` に入れて呼びます。再試行はせず即 STOP です。
        """
        self._advance_clock(now_ns)
        if not isinstance(detail, str) or not detail:
            raise ValueError("detail must be a non-empty string")
        self._stop(HealthReason.INFERENCE_ERROR, now_ns, None, None, detail)
        return HealthVerdict.STOP

    def report(self) -> HealthReport:
        """現時点の verdict・counters・first-failure を返す。

        やさしい説明: counters は内部 dict のコピーなので、返り値を変更しても
        監視状態には影響しません。
        """
        return HealthReport(
            verdict=self._verdict(),
            warning_counts=dict(self._warning_counts),
            stop_counts=dict(self._stop_counts),
            first_failure=self._first_failure,
        )

"""非同期 HUD/world 結果を 15 Hz policy tick 用に結合する。

進行状態を後退させず、skew・stale・画面状態から用途別 validity を算出します。
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from reinbalance_survivors_contracts.canonical_json import canonical_hash
from .vision.entity_tracker import TrackedWorldStateV1
from .vision.hud_parser import HudStateV1
_ITEM_STATES = frozenset({"level_up_items", "level_up_fallback", "chest"})
_SCREEN_CONFIDENCE_THRESHOLD = 0.5
@dataclass(frozen=True)
class TemporalJoin:
    """一回の policy tick で束縛した HUD/world と validity を保持する。

    値を保持することと policy が利用可能かを分け、stale 時は fail-closed にします。
    """
    hud: HudStateV1; world: TrackedWorldStateV1; captured_ns: int
    hud_validity: float; world_validity: float
    combat_validity: float; item_validity: float
class TemporalAssembler:
    """最新 HUD/world を monotonic filter 後に join する stateful assembler。

    async producer は observe 後に tick を呼び、同期利用者は join で同じ規則を適用できます。
    """

    def __init__(
        self, *, policy_hz: float = 15., max_skew_ms: float = 50.,
        max_stale_age_ms: dict[str, float] | None = None, filter_alpha: float = .5,
    ) -> None:
        """cadence・skew・stale・平滑化係数を設定する。

        不正な時間設定を初期化時に拒否し、既定値は screen-space config と一致させます。
        """
        stale = dict(max_stale_age_ms or {"hud": 200., "world": 200.})
        if policy_hz <= 0 or max_skew_ms < 0 or set(stale) != {"hud", "world"} or min(stale.values()) < 0 or not 0 <= filter_alpha <= 1:
            raise ValueError("invalid temporal assembler config")
        self.tick_interval_ns = round(1_000_000_000 / policy_hz)
        self.max_skew_ns = round(max_skew_ms * 1_000_000)
        self.max_stale_ns = {key: round(value * 1_000_000) for key, value in stale.items()}
        self.filter_alpha = filter_alpha
        self._hud: HudStateV1 | None = None
        self._world: TrackedWorldStateV1 | None = None
        self._last_tick_ns: int | None = None
        self._session_id: str | None = None
        self._session_start_ns: int | None = None  # 現在sessionの最初のHUD timestamp

    def observe_hud(self, hud: HudStateV1) -> None:
        """最新 HUD を monotonic・bounded filter へ取り込む。

        OCR の timer/level/inventory 後退を止め、HP/XP の frame 揺れを平滑化します。
        """
        if not isinstance(hud, HudStateV1):
            raise TypeError("hud must be HudStateV1")
        if self._session_id is not None and hud.session_id != self._session_id:
            if self._hud is not None and hud.captured_monotonic_ns <= self._hud.captured_monotonic_ns:
                return  # 旧sessionの残存フレームは保持中の新sessionを上書きしない
            self._hud = None
            self._world = None
            self._last_tick_ns = None
            self._session_start_ns = hud.captured_monotonic_ns  # 新session開始境界を記録
        self._session_id = hud.session_id
        previous = self._hud
        if previous is None:
            self._hud = hud
            return
        if hud.captured_monotonic_ns < previous.captured_monotonic_ns:
            return
        timer = max(value for value in (previous.timer_seconds, hud.timer_seconds) if value is not None) if previous.timer_seconds is not None or hud.timer_seconds is not None else None
        level = max(value for value in (previous.level, hud.level) if value is not None) if previous.level is not None or hud.level is not None else None
        inventory = tuple(current if current is not None else old for old, current in zip(previous.inventory, hud.inventory, strict=True))
        hp = self._bounded_filter(previous.hp_ratio, hud.hp_ratio)
        xp = self._bounded_filter(previous.xp_ratio, hud.xp_ratio)
        self._hud = replace(
            hud, timer_seconds=timer, level=level, inventory=inventory, hp_ratio=hp, xp_ratio=xp,
            inventory_hash=canonical_hash({"slots": list(inventory)}),
        )

    def _bounded_filter(self, old: float | None, new: float | None) -> float | None:
        """欠損を保ちながら [0,1] 内で EMA filter を適用する。

        一時的な bar 検出揺れを抑え、結果は必ず共有 ratio 範囲へ clamp します。
        """
        if new is None:
            return old
        if old is None:
            return max(0., min(1., new))
        return max(0., min(1., old + self.filter_alpha * (new - old)))

    def observe_world(self, world: TrackedWorldStateV1) -> None:
        """最新 world state を時刻順に取り込む。

        遅れて届いた tracker 結果で新しい画面状態を上書きしません。
        """
        if not isinstance(world, TrackedWorldStateV1):
            raise TypeError("world must be TrackedWorldStateV1")
        if self._session_start_ns is not None and world.timestamp_ns < self._session_start_ns:
            return  # 現在session開始より前のworldは旧sessionの残存として拒否
        if self._world is None or world.timestamp_ns >= self._world.timestamp_ns:
            self._world = world

    def join(self, hud: HudStateV1, world: TrackedWorldStateV1, *, now_ns: int | None = None) -> TemporalJoin:
        """同期 HUD/world を取り込み一回分の用途別 validity を返す。

        skew または stale が閾値を超える場合、保持値を tensor へ入れても validity はゼロになります。
        """
        self.observe_hud(hud)
        self.observe_world(world)
        return self._current_join(max(self._hud.captured_monotonic_ns, self._world.timestamp_ns) if now_ns is None else now_ns)

    def tick(self, now_ns: int) -> TemporalJoin | None:
        """15 Hz cadence を満たす時だけ最新 producer 結果を join する。

        入力未到着または前 tick から 66.7ms 未満なら新しい policy frame を発行しません。
        """
        if type(now_ns) is not int or now_ns < 0 or self._hud is None or self._world is None:
            return None
        if self._last_tick_ns is not None and now_ns - self._last_tick_ns < self.tick_interval_ns:
            return None
        self._last_tick_ns = now_ns
        return self._current_join(now_ns)

    def _current_join(self, now_ns: int) -> TemporalJoin:
        """保持中 producer の skew・stale・screen routing を評価する。

        gameplay と item UI の validity を排他的にし、停止系 state は双方ゼロにします。
        """
        if self._hud is None or self._world is None or type(now_ns) is not int or now_ns < 0:
            raise ValueError("temporal sources and valid now_ns are required")
        hud_age = now_ns - self._hud.captured_monotonic_ns
        world_age = now_ns - self._world.timestamp_ns
        hud_valid = float(0 <= hud_age <= self.max_stale_ns["hud"])
        world_valid = float(0 <= world_age <= self.max_stale_ns["world"])
        joined_valid = float(hud_valid == world_valid == 1. and abs(self._hud.captured_monotonic_ns - self._world.timestamp_ns) <= self.max_skew_ns)
        confident = self._hud.screen_state_confidence >= _SCREEN_CONFIDENCE_THRESHOLD
        combat = joined_valid if self._hud.screen_state == "gameplay" and confident else 0.
        item = joined_valid if self._hud.screen_state in _ITEM_STATES and confident else 0.
        return TemporalJoin(self._hud, self._world, now_ns, hud_valid, world_valid, combat, item)

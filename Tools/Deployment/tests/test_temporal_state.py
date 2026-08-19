"""非同期 HUD/world 結合と temporal filter を検証する。

時刻ずれ、古い frame、画面状態による validity の切替を確認します。
"""

from dataclasses import replace

from survivors.temporal_state import TemporalAssembler
from survivors.vision.entity_tracker import PlayerAnchorState, TrackedWorldStateV1
from survivors.vision.hud_parser import HudStateV1

def _hud(timestamp: int, state: str = "gameplay", **changes) -> HudStateV1:
    """temporal test 用 HUD state を作る。

    値と時刻だけを差し替え、parser の画像処理からテストを切り離します。
    """
    base = HudStateV1(
        "hud_state.v1", "session", 1, timestamp, "a" * 64, state, .9, "ok",
        10., .9, "ok", False, .8, .9, "ok", .4, .9, "ok", 3, .9, "ok",
        ("whip",) + (None,) * 11, .9, "b" * 64, (), "c" * 64, (),
        False, False, False, .9, "ok",
    )
    return replace(base, **changes)

def _world(timestamp: int) -> TrackedWorldStateV1:
    """track なしの world state を作る。

    join の鮮度と skew だけを検証するため最小の状態を返します。
    """
    return TrackedWorldStateV1(1, timestamp, [], PlayerAnchorState(.5, .5, .9, False))

def test_join_applies_monotonic_and_bounded_filters() -> None:
    """timer・level・inventory の後退を防ぎ HP/XP を範囲内に保つ。

    OCR が一時的に古い値を返しても、確定済みの進行状態へ戻しません。
    """
    assembler = TemporalAssembler()
    assembler.join(_hud(1_000, timer_seconds=20., level=5), _world(1_000))
    joined = assembler.join(
        _hud(2_000, timer_seconds=19., level=4, inventory=(None,) * 12, hp_ratio=.95, xp_ratio=.05),
        _world(2_000),
    )
    assert joined.hud.timer_seconds == 20.
    assert joined.hud.level == 5
    assert joined.hud.inventory[0] == "whip"
    assert joined.hud.hp_ratio == .875
    assert joined.hud.xp_ratio == .225
    assert assembler.join(_hud(1_500), _world(1_500)).hud.captured_monotonic_ns == 2_000

def test_skew_and_staleness_fail_closed() -> None:
    """50ms skew と source stale age を超えた combat を無効化する。

    新旧フレームを無理に混ぜず、値を保持しても validity はゼロへ落とします。
    """
    assembler = TemporalAssembler()
    skewed = assembler.join(_hud(1_000_000_000), _world(1_060_000_001), now_ns=1_060_000_001)
    assert skewed.combat_validity == 0.
    stale = assembler.join(_hud(1_000_000_000), _world(1_000_000_000), now_ns=1_300_000_002)
    assert stale.hud_validity == stale.world_validity == stale.combat_validity == 0.

def test_screen_state_routes_combat_and_item_validity() -> None:
    """gameplay・item UI・停止画面の用途別 validity を切り替える。

    戦闘 policy と選択 policy が同じ frame を同時に有効扱いしないようにします。
    """
    assembler = TemporalAssembler()
    gameplay = assembler.join(_hud(1_000, "gameplay"), _world(1_000))
    level_up = assembler.join(_hud(2_000, "level_up_items"), _world(2_000))
    paused = assembler.join(_hud(3_000, "paused"), _world(3_000))
    assert (gameplay.combat_validity, gameplay.item_validity) == (1., 0.)
    assert (level_up.combat_validity, level_up.item_validity) == (0., 1.)
    assert (paused.combat_validity, paused.item_validity) == (0., 0.)

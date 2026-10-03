"""HUD のパッシブスロットから持続時間倍率を求める Common 関数を検証する。

実機（04-13）は Spellbinder などのパッシブとレベルから倍率を求めてビルダーへ渡します。
分からないパッシブやレベルがあるときに推測せず None（不明）を返すことを確かめます。
"""

from __future__ import annotations

import pytest

from reinbalance_survivors_contracts.deploy_obs_v2_features import HudSlot, duration_mult_from_hud_slots


def _passives(slots: dict[int, tuple[str | None, int | None]], *, omit: tuple[int, ...] = ()) -> list[HudSlot]:
    """パッシブ6スロットの HudSlot 列を作る。

    指定の無いスロットは空スロット確定、omit のスロットは「読めなかった」として渡しません。
    """
    out = []
    for index in range(6):
        if index in omit:
            continue
        name, level = slots.get(index, (None, None))
        out.append(HudSlot("passive", index, name, level))
    return out


def test_no_duration_passive_gives_one():
    """持続時間に効くパッシブが無いと確定していれば倍率は 1.0。

    空スロットと関係ないパッシブは倍率を変えません。
    """
    assert duration_mult_from_hud_slots(_passives({0: ("Spinach", 3)})) == 1.0


def test_spellbinder_and_torronas_add_bonus():
    """Spellbinder は 0.10×Lv、TorronasBox は 0.04+0.03×(Lv-1) を加算する。

    C++ ComputePassiveEffects と同じく加算値の合計に 1 を足した値になります。
    """
    assert duration_mult_from_hud_slots(_passives({1: ("Spellbinder", 3)})) == pytest.approx(1.3)
    both = _passives({1: ("Spellbinder", 1), 4: ("TorronasBox", 2)})
    assert duration_mult_from_hud_slots(both) == pytest.approx(1.17)


@pytest.mark.parametrize("hud", [
    None,
    _passives({}, omit=(5,)),
    _passives({1: ("Spellbinder", None)}),
    _passives({1: ("Spellbinder", 6)}),
    _passives({2: ("NotAPassive", 1)}),
])
def test_unknown_inputs_give_none(hud):
    """スロット不明・レベル不明・表に無いレベル・語彙外の名前は None（不明）。

    倍率に効くパッシブが隠れている可能性がある入力では値を推測しません。
    """
    assert duration_mult_from_hud_slots(hud) is None


def test_weapon_slots_are_ignored():
    """武器スロットは倍率に影響しない。

    パッシブ6スロットがそろっていれば、武器スロットの有無に関係なく値を返します。
    """
    hud = _passives({0: ("Spellbinder", 2)}) + [HudSlot("weapon", 0, "Whip", None)]
    assert duration_mult_from_hud_slots(hud) == pytest.approx(1.2)

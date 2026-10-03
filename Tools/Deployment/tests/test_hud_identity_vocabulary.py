"""HUD identity → Common 語彙の対応表（hud_identity_vocabulary_v1.yaml）を検証する。

全 identity が Common の語彙へ対応し、Common の全武器・全パッシブが表に現れること、
不正な表を fail-closed で拒否することを確かめます。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from reinbalance_survivors_contracts.deploy_obs_v2_features import load_deploy_obs_v2_feature_params

from survivors.hud_identity_vocabulary import DEFAULT_PATH, _parse, load_hud_identity_vocabulary

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _raw() -> dict:
    """既定の対応表 YAML を dict として読む。

    mutation テストの元データに使います。
    """
    return yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))


def test_every_mapping_target_is_in_common_vocabulary_and_covers_it():
    """対応先が全て Common 語彙にあり、逆に Common の全武器・全パッシブが表に現れる。

    sim の enum に武器が増えたのに表を直し忘れると、その武器のスロットが常に不明になるため検出します。
    """
    params = load_deploy_obs_v2_feature_params()
    vocab = load_hud_identity_vocabulary()
    assert set(vocab.weapons.values()) == set(params["weapon_vocabulary"][1:-1])
    assert set(vocab.passives.values()) == set(params["passive_vocabulary"][1:-1])


def test_profile_and_replay_identities_are_all_mapped():
    """標準 profile の候補語彙と replay suite に出る identity が全て表で扱われる。

    atlas が返しうる名前（whip, gold, chicken）と replay fixture の在庫名が、
    武器・パッシブ・非アイテムのどれかに必ず分類されることを確かめます。
    """
    vocab = load_hud_identity_vocabulary()
    known = set(vocab.weapons) | set(vocab.passives) | vocab.non_items
    profile = yaml.safe_load((CONFIGS / "mad_forest_standard_v1.yaml").read_text(encoding="utf-8"))
    assert set(profile["choice_taxonomy"]["candidate_vocabulary"]) <= known
    suite = yaml.safe_load((CONFIGS / "e2e_replay_suite_v1.yaml").read_text(encoding="utf-8"))
    inventories = [suite["session"]["default_inventory"]] + [f["inventory"] for f in suite["fixtures"] if "inventory" in f]
    assert len(inventories) >= 2 and all(set(inv) <= known for inv in inventories)


def test_type_name_rejects_wrong_kind_and_unknown_identity():
    """種別違い・表に無い identity は None（不明）になる。

    武器 id がパッシブ枠に出たような誤認識を、別の語彙名へ寄せずに不明として扱います。
    """
    vocab = load_hud_identity_vocabulary()
    assert vocab.type_name("whip", "weapon") == "Whip"
    assert vocab.type_name("spellbinder", "passive") == "Spellbinder"
    assert vocab.type_name("whip", "passive") is None
    assert vocab.type_name("gold", "weapon") is None
    assert vocab.type_name("mystery", "weapon") is None
    with pytest.raises(ValueError):
        vocab.type_name("whip", "relic")


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(extra=1),
    lambda d: d.update(schema_version="hud_identity_vocabulary.v0"),
    lambda d: d["weapons"].update(foo="NotAWeapon"),
    lambda d: d["weapons"].update(foo="Spinach"),
    lambda d: d["passives"].update(whip="Armor"),
    lambda d: d["weapons"].update(Whip2="Whip"),
    lambda d: d["weapons"].update(whip_alias="Whip"),
    lambda d: d["non_items"].append("whip"),
    lambda d: d["non_items"].append("gold"),
    lambda d: d.update(weapons={}),
    lambda d: d.pop("empty_slot"),
    lambda d: d.update(empty_slot=None),
    lambda d: d.update(empty_slot="whip"),
    lambda d: d.update(empty_slot="gold"),
    lambda d: d.update(empty_slot="Empty"),
])
def test_invalid_tables_are_rejected(mutate):
    """未知キー・版違い・語彙外の対応先・重複を読み込み時に拒否する。

    黙って補完すると誤ったスロットの種類がビルダーへ渡るため、全て ValueError で止めます。
    """
    data = _raw()
    mutate(data)
    with pytest.raises(ValueError):
        _parse(data)

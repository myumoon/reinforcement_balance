"""HUD item identity → Common 武器・パッシブ語彙の対応表を fail-closed で読み込む。

icon_matcher が返す小文字 id（例: whip）を、DeployObs v2 ビルダーが受け取る
C++ 由来の名前（例: Whip）へ写すための表です。未知キー・語彙に無い対応先・
重複した identity や対応先があれば読み込み時点で止め、黙って補完しません。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml
from reinbalance_survivors_contracts.deploy_obs_v2_features import load_deploy_obs_v2_feature_params

HUD_IDENTITY_VOCABULARY_VERSION = "hud_identity_vocabulary.v1"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "configs" / "hud_identity_vocabulary_v1.yaml"
_KEYS = frozenset({"schema_version", "weapons", "passives", "non_items", "empty_slot"})


@dataclass(frozen=True)
class HudIdentityVocabulary:
    """武器・パッシブ・非アイテム・空スロットの identity 対応表。

    weapons / passives は identity → Common 語彙名、non_items はスロットに入らない identity の集合です。
    empty_slot は空スロット用テンプレートの identity で、在庫の None（読めなかった枠）とは区別します。
    """

    weapons: Mapping[str, str]
    passives: Mapping[str, str]
    non_items: frozenset[str]
    empty_slot: str

    def type_name(self, identity: str, kind: str) -> str | None:
        """identity を指定スロット種別の Common 語彙名へ写す。

        対応表に無い、または種別が違う（武器 id がパッシブ枠にある等）ときは None（不明）を返します。
        """
        table = self.weapons if kind == "weapon" else self.passives if kind == "passive" else None
        if table is None:
            raise ValueError(f"unknown slot kind: {kind!r}")
        return table.get(identity)


def _parse(data: Any) -> HudIdentityVocabulary:
    """YAML から読んだ dict を検証して対応表にする。

    対応先が Common の語彙（先頭 None・末尾 unknown を除く）に無い場合や、
    同じ identity が複数の節に現れる場合は ValueError で止めます。
    """
    if not isinstance(data, dict) or set(data) != _KEYS:
        raise ValueError("hud identity vocabulary keys mismatch")
    if data["schema_version"] != HUD_IDENTITY_VOCABULARY_VERSION:
        raise ValueError("unsupported hud identity vocabulary version")
    params = load_deploy_obs_v2_feature_params()
    seen: set[str] = set()
    tables = {}
    for section, vocabulary in (("weapons", params["weapon_vocabulary"]), ("passives", params["passive_vocabulary"])):
        table = data[section]
        if not isinstance(table, dict) or not table:
            raise ValueError(f"{section} must be a non-empty mapping")
        allowed = set(vocabulary[1:-1])
        for identity, name in table.items():
            if not isinstance(identity, str) or not identity or identity != identity.lower() or identity in seen:
                raise ValueError(f"{section}: invalid or duplicated identity {identity!r}")
            if name not in allowed:
                raise ValueError(f"{section}: {identity!r} maps to unknown Common name {name!r}")
            seen.add(identity)
        if len(set(table.values())) != len(table):
            raise ValueError(f"{section}: Common names must be mapped at most once")
        tables[section] = MappingProxyType(dict(table))
    non_items = data["non_items"]
    if not isinstance(non_items, list) or not all(isinstance(v, str) and v and v == v.lower() for v in non_items):
        raise ValueError("non_items must be a list of lowercase identities")
    if seen & set(non_items) or len(set(non_items)) != len(non_items):
        raise ValueError("non_items overlaps items or is duplicated")
    empty_slot = data["empty_slot"]
    if not isinstance(empty_slot, str) or not empty_slot or empty_slot != empty_slot.lower() or empty_slot in seen | set(non_items):
        raise ValueError("empty_slot must be a lowercase identity distinct from items and non_items")
    return HudIdentityVocabulary(tables["weapons"], tables["passives"], frozenset(non_items), empty_slot)


def load_hud_identity_vocabulary(path: str | Path = DEFAULT_PATH) -> HudIdentityVocabulary:
    """対応表 YAML を読み込んで検証済みの HudIdentityVocabulary を返す。

    既定パスの読み込みはキャッシュし、毎フレームのファイル読み込みを避けます。
    """
    resolved = Path(path).resolve()
    return _load_cached(resolved) if resolved == DEFAULT_PATH else _parse(yaml.safe_load(resolved.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def _load_cached(path: Path) -> HudIdentityVocabulary:
    """既定の対応表を1回だけ読み込む。

    結果は読み取り専用なので全呼び出しで共有しても安全です。
    """
    return _parse(yaml.safe_load(path.read_text(encoding="utf-8")))

"""weapon_bootstrap_gate_policy.py

武器別 bootstrap gate policy。
通常武器は score gate、Peachone / Ebony Wings は trait exposure gate を使う。
"""
from __future__ import annotations

from dataclasses import dataclass

from games.survivors.survivors_weapon_table import WeaponEntry


@dataclass(frozen=True)
class WeaponBootstrapGatePolicy:
    gate_goal: str  # "score" または "trait_exposure"
    solo_target_p10: float
    solo_min_ep_len_p10: float
    solo_max_short_episode_rate: float
    integration_target_p10: float


def _key_set(keys: str | tuple[str, ...] | list[str]) -> set[str]:
    if isinstance(keys, str):
        return {part.strip() for part in keys.split(",") if part.strip()}
    return {str(part).strip() for part in keys if str(part).strip()}


def build_weapon_bootstrap_gate_policies(
    *,
    weapon_unlock_order: list[WeaponEntry],
    score_solo_target_p10: float,
    score_solo_min_ep_len_p10: float,
    score_solo_max_short_episode_rate: float,
    integration_target_p10: float,
    trait_weapon_keys: str | tuple[str, ...] | list[str],
    trait_target_p10: float,
    trait_min_ep_len_p10: float,
    trait_max_short_episode_rate: float,
) -> dict[int, WeaponBootstrapGatePolicy]:
    """武器別 gate policy を構築する。

    trait_weapon_keys に含まれる武器は trait exposure gate（低スコア閾値 + 生存品質重視）、
    それ以外は従来の score gate を使う。
    """
    trait_keys = _key_set(trait_weapon_keys)
    policies: dict[int, WeaponBootstrapGatePolicy] = {}
    for entry in weapon_unlock_order:
        if entry.key in trait_keys:
            policies[entry.weapon_id] = WeaponBootstrapGatePolicy(
                gate_goal="trait_exposure",
                solo_target_p10=trait_target_p10,
                solo_min_ep_len_p10=trait_min_ep_len_p10,
                solo_max_short_episode_rate=trait_max_short_episode_rate,
                integration_target_p10=integration_target_p10,
            )
        else:
            policies[entry.weapon_id] = WeaponBootstrapGatePolicy(
                gate_goal="score",
                solo_target_p10=score_solo_target_p10,
                solo_min_ep_len_p10=score_solo_min_ep_len_p10,
                solo_max_short_episode_rate=score_solo_max_short_episode_rate,
                integration_target_p10=integration_target_p10,
            )
    return policies

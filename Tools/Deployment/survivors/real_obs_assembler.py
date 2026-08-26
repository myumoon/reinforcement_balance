"""実 HUD/world state を deployable PerceptionSnapshot へ組み立てる。

モデル特徴と操作 ROI を別経路で生成し、最後に同一 identity へ atomic に束縛します。
"""
from __future__ import annotations
from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.item_decision import CandidateFeatures, ItemDecisionFeatures
from .deploy_obs_adapter import NamedEstimate, build_deploy_observation, normalized_category
from .perception_snapshot import PerceptionSnapshot, build_ui_presentation_from_hud
from .screen_space_features import build_screen_space_estimates
from .temporal_state import TemporalAssembler, TemporalJoin
from .vision.entity_tracker import TrackedEntityV1, TrackedWorldStateV1
from .vision.hud_parser import HudStateV1, ParsedCard
def _weapon_category(item_id: str | None) -> str:
    """先頭 weapon identity を DeployObs の粗いカテゴリへ写す。

    語彙外アイテムは adapter の予約 unknown へ送り、推測した新カテゴリを作りません。
    """
    if item_id is None:
        return "unknown"
    key = item_id.casefold()
    if any(token in key for token in ("garlic", "bible", "water", "song", "aura")):
        return "aura"
    if any(token in key for token in ("whip", "knife", "melee")):
        return "melee"
    if any(token in key for token in ("wand", "axe", "cross", "projectile")):
        return "projectile"
    return "unknown"

def _candidate(card: ParsedCard, used_ids: set[str], inventory: tuple[str | None, ...]) -> CandidateFeatures:
    """画面 card 一件を ROI なしの model candidate へ変換する。

    重複 identity は slot suffix で分離し、操作座標は一切受け取りません。
    """
    identity = card.item_id or f"unknown:{card.slot_index}"
    if identity in used_ids:
        identity = f"{identity}@{card.slot_index}"
    used_ids.add(identity)
    owned = card.item_id is not None and card.item_id in inventory
    return CandidateFeatures(
        kind=card.kind, item_id=identity, new_level=card.level or 0,
        owned=owned, is_new=not owned, is_evolve=card.kind == "evolved",
        is_union=False, has_prerequisite=False, slot_capacity=6,
    )

def _item_context(
    joined: TemporalJoin, snapshot_id: str, screen: dict[str, NamedEstimate],
    gameplay_world: TrackedWorldStateV1 | None, last_gameplay_ns: int | None, now_ns: int | None,
) -> tuple[ItemDecisionFeatures | None, tuple[CandidateFeatures, ...]]:
    """item UI 用 context と候補を visible feature だけから構築する。

    gameplay/停止画面では None を返し、fallback と chest も semantic choice として保持します。
    danger 特徴は直前の gameplay snapshot から取得し、item UI 中も正確な鮮度を算出します。
    """
    if joined.item_validity == 0:
        return None, ()
    used: set[str] = set()
    cards = sorted(joined.hud.cards, key=lambda value: value.slot_index)
    raw_ids = [card.item_id or f"unknown:{card.slot_index}" for card in cards]
    if len(raw_ids) != len(set(raw_ids)):
        return None, ()
    choices = tuple(_candidate(card, used, joined.hud.inventory) for card in cards)
    if not choices and joined.hud.screen_state == "chest":
        choices = (CandidateFeatures("chest", "ack_chest", 0, False, False, False, False, False, 0),)
    if not choices:
        return None, ()
    inventory_levels = tuple(1 if item is not None else 0 for item in joined.hud.inventory)
    nearest = screen.get("nearest_enemy_offset")
    radius = screen.get("nearest_enemy_screen_radius")
    enemy_density = screen["enemy_density"].value[0]
    gem_density = screen["gem_density"].value[0]
    effective_world = gameplay_world if gameplay_world is not None else joined.world
    visible = [track for track in effective_world.tracks if track.on_screen and not track.clipped and track.confidence >= .35]
    fallback = next(
        (choice.item_id for choice in choices if choice.kind == "fallback" or "gold" in choice.item_id.casefold() or "chicken" in choice.item_id.casefold()),
        "none",
    )
    max_cards = max(3, len(choices))
    world_age = (now_ns - gameplay_world.timestamp_ns) / 1_000_000_000 if now_ns is not None and gameplay_world is not None else 0.
    snapshot_age = (now_ns - last_gameplay_ns) / 1_000_000_000 if now_ns is not None and last_gameplay_ns is not None else 0.
    ui_state_age = max(0., (now_ns - joined.hud.captured_monotonic_ns) / 1_000_000_000) if now_ns is not None else 0.
    context = ItemDecisionFeatures(
        decision_id=snapshot_id, feature_schema="context_danger_v1",
        elapsed_time=joined.hud.timer_seconds or 0., level=joined.hud.level or 1,
        hp_ratio=joined.hud.hp_ratio or 0., xp_ratio=joined.hud.xp_ratio or 0.,
        weapon_slots=inventory_levels[:6], passive_slots=inventory_levels[6:],
        empty_slot_count=inventory_levels.count(0), evolution_readiness=0.,
        choice_count=len(choices), card_mask=(True,) * len(choices) + (False,) * (max_cards - len(choices)),
        fallback_kind=fallback, ui_state_validity=joined.item_validity, ui_state_age=ui_state_age,
        candidates=choices, max_item_cards=max_cards, enemy_density=enemy_density,
        gem_density=gem_density, nearest_enemy_screen_dist=radius.value[0] if radius else 1.,
        nearest_enemy_screen_dir=nearest.value if nearest else (0., 0.),
        boss_flag=any(track.coarse_class == "enemy" and "boss" in track.class_name for track in visible),
        hazard_flag=any(track.coarse_class == "hazard" for track in visible),
        world_validity=joined.world_validity, world_age=world_age, last_gameplay_snapshot_age=snapshot_age,
    )
    return context, choices

class RealObsAssembler:
    """実 parser/tracker 出力から一貫した policy snapshot を生成する。

    TemporalAssembler を所有し、複数 frame にまたがる単調性と鮮度を維持します。
    """

    def __init__(self, temporal: TemporalAssembler | None = None) -> None:
        """利用する temporal state を注入または既定生成する。

        test と runtime が同じ assembly 経路のまま cadence 設定だけを差し替えられます。
        """
        self.temporal = temporal or TemporalAssembler()
        self._last_gameplay_world: TrackedWorldStateV1 | None = None
        self._last_gameplay_screen: dict[str, NamedEstimate] | None = None
        self._last_gameplay_ns: int | None = None
        self._last_session_id: str | None = None

    def assemble(
        self, hud: HudStateV1, world: TrackedWorldStateV1,
        schema: DeployObsSchema, viewport: tuple[int, int],
    ) -> PerceptionSnapshot | None:
        """15 Hz cadence を守りながら HUD/world から PerceptionSnapshot を構築する。

        tick_interval 未満の入力は observe のみ受け付け、snapshot の発行を抑制します。
        UI presentation は tensor builder に渡さず、完成後の snapshot へ一度だけ格納します。
        """
        if not isinstance(schema, DeployObsSchema):
            raise TypeError("schema must be DeployObsSchema")
        if not isinstance(viewport, tuple) or len(viewport) != 2 or any(type(value) is not int or value <= 0 for value in viewport):
            raise ValueError("viewport must be a positive integer pair")
        self.temporal.observe_hud(hud)
        self.temporal.observe_world(world)
        now_ns = max(hud.captured_monotonic_ns, world.timestamp_ns)
        joined = self.temporal.tick(now_ns)
        if joined is None:
            return None
        if self._last_session_id != joined.hud.session_id:  # session変更時にgameplayキャッシュを破棄
            self._last_gameplay_world = None
            self._last_gameplay_screen = None
            self._last_gameplay_ns = None
            self._last_session_id = joined.hud.session_id
        frame_id = f"{joined.hud.session_id}:{joined.hud.frame_index}:{joined.world.frame_index}"
        snapshot_id = canonical_hash({
            "session_id": joined.hud.session_id, "frame_id": frame_id,
            "hud_timestamp_ns": joined.hud.captured_monotonic_ns, "world_timestamp_ns": joined.world.timestamp_ns,
            "parser_artifact_hash": joined.hud.parser_artifact_hash,
        })
        screen = build_screen_space_estimates(joined.world)
        if joined.combat_validity > 0:  # combat有効フレームのみdangerキャッシュを更新
            self._last_gameplay_world = joined.world
            self._last_gameplay_screen = screen
            self._last_gameplay_ns = joined.captured_ns
        estimates: dict[str, NamedEstimate] = {}
        combat = joined.combat_validity
        if joined.hud.hp_ratio is not None:
            estimates["player_hp"] = NamedEstimate((joined.hud.hp_ratio,), joined.hud.captured_monotonic_ns, joined.hud.hp_confidence * combat)
        if joined.hud.level is not None:
            estimates["level"] = NamedEstimate((min(joined.hud.level / 99., 1.),), joined.hud.captured_monotonic_ns, joined.hud.level_confidence * combat)
        estimates["weapon_category"] = NamedEstimate(
            (normalized_category(_weapon_category(joined.hud.inventory[0])),), joined.hud.captured_monotonic_ns,
            joined.hud.inventory_confidence * combat,
        )
        for name in ("player_screen_pos", "nearest_enemy_offset", "visible_enemy_count"):
            if name in screen:
                value = screen[name]
                estimates[name] = NamedEstimate(value.value, value.timestamp_ns, value.validity * combat)
        deploy_obs = build_deploy_observation(schema, estimates, joined.captured_ns)
        ui = build_ui_presentation_from_hud(joined.hud, viewport, snapshot_id=snapshot_id, frame_id=frame_id)
        if self._last_gameplay_screen is not None:
            item_context, choices = _item_context(
                joined, snapshot_id, self._last_gameplay_screen,
                self._last_gameplay_world, self._last_gameplay_ns, joined.captured_ns,
            )
        else:
            item_context, choices = None, ()  # キャッシュなし: item overlay をgameplayとして流さない
        diagnostics = {
            "hud_world_skew_ms": abs(hud.captured_monotonic_ns - world.timestamp_ns) / 1_000_000,
            "hud_validity": joined.hud_validity, "world_validity": joined.world_validity,
            "combat_validity": joined.combat_validity, "item_validity": joined.item_validity,
        }
        return PerceptionSnapshot(
            snapshot_id, frame_id, joined.captured_ns, joined.hud.parser_artifact_hash,
            ui.source_content_hash, ui.ui_state_key, joined.hud.screen_state, deploy_obs,
            item_context, choices, ui, diagnostics,
        )

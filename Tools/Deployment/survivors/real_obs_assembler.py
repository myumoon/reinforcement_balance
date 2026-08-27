"""実 HUD/world state を deployable PerceptionSnapshot へ組み立てる。

モデル特徴と操作 ROI を別経路で生成し、最後に同一 identity へ atomic に束縛します。
"""
from __future__ import annotations
from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.item_decision import CandidateFeatures, ItemDecisionFeatures
from reinbalance_survivors_contracts.ui_policy import (
    FallbackSemantic, FallbackTarget, ScreenState, UiPolicyInputV1,
)
from .deploy_obs_adapter import NamedEstimate, build_deploy_observation, normalized_category
from .perception_snapshot import PerceptionSnapshot, UiPresentationSnapshotV1, build_ui_presentation_from_hud
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

_HUD_TO_SCREEN_STATE: dict[str, ScreenState] = {
    "gameplay": ScreenState.GAMEPLAY,
    "level_up_items": ScreenState.LEVEL_UP,
    "level_up_fallback": ScreenState.FALLBACK,
    "chest": ScreenState.CHEST,
}
_KNOWN_FALLBACK_IDS = {FallbackSemantic.CHICKEN.value, FallbackSemantic.GOLD.value}


def _build_ui_policy_input(
    snapshot_id: str,
    frame_id: str,
    hud: HudStateV1,
    ui: UiPresentationSnapshotV1,
) -> UiPolicyInputV1:
    """HUD と UI presentation から UiPolicyInputV1 を構築する。

    combat_validity に依存せず hud.hp_ratio（画面由来）を直接使うことで、
    fallback 画面でも fallback_heuristic_v1 が HP 閾値を評価できる。
    """
    screen_state = _HUD_TO_SCREEN_STATE.get(hud.screen_state, ScreenState.UNKNOWN)
    hp_fraction = float(hud.hp_ratio) if hud.hp_ratio is not None else 0.
    fallback_targets = []
    for cand in ui.candidates:
        if cand.semantic_kind != "fallback_reward":
            continue
        cid = cand.choice_id.casefold()
        semantic = (
            FallbackSemantic.CHICKEN.value if "chicken" in cid
            else FallbackSemantic.GOLD.value if "gold" in cid
            else "unknown_fallback"
        )
        fallback_targets.append(FallbackTarget(
            target_id=cand.choice_id,
            target_index=cand.choice_index,
            semantic=semantic,
            valid=cand.validity,
        ))
    return UiPolicyInputV1(
        source_snapshot_hash=snapshot_id,
        source_frame_hash=frame_id,
        source_content_hash=ui.source_content_hash,
        ui_state_key=ui.ui_state_key,
        screen_state=screen_state,
        hp_fraction=hp_fraction,
        candidate_set_hash=ui.candidate_set_hash,
        inventory_hash=ui.inventory_hash,
        fallback_targets=tuple(fallback_targets),
    )


def _item_context(
    joined: TemporalJoin, snapshot_id: str, screen: dict[str, NamedEstimate],
    gameplay_world: TrackedWorldStateV1 | None, last_gameplay_ns: int | None, now_ns: int | None,
    valid_ui_ids: frozenset[str],
) -> tuple[ItemDecisionFeatures | None, tuple[CandidateFeatures, ...]]:
    """item UI 用 context と候補を visible feature だけから構築する。

    gameplay/停止画面では None を返し、item_card だけを model 候補とします。
    danger 特徴は直前の gameplay snapshot から取得し、item UI 中も正確な鮮度を算出します。
    """
    if joined.item_validity == 0:
        return None, ()
    used: set[str] = set()
    cards = sorted(joined.hud.cards, key=lambda value: value.slot_index)
    raw_ids = [card.item_id or f"unknown:{card.slot_index}" for card in cards]
    if len(raw_ids) != len(set(raw_ids)):
        return None, ()
    # fallback/gold/chicken は fallback_kind フィールドへ。chest は ack_chest ボタンのみで表現。
    fallback = next(
        (card.item_id or "" for card in cards
         if card.kind == "fallback" or "gold" in (card.item_id or "").casefold()
         or "chicken" in (card.item_id or "").casefold()),
        "none",
    )
    item_cards = [
        card for card in cards
        if card.kind not in ("fallback", "chest")
        and "gold" not in (card.item_id or "").casefold()
        and "chicken" not in (card.item_id or "").casefold()
    ]
    choices = tuple(_candidate(card, used, joined.hud.inventory) for card in item_cards)
    # UI validity 同期: 対応する UI target が valid でない候補をモデルから除く
    choices = tuple(c for c in choices if c.item_id in valid_ui_ids)
    if not choices:
        return None, ()
    # ponytail: HudStateV1.inventory は identity のみ保持。slot level・evolution_readiness・is_union・
    # has_prerequisite は画面から観測不可。context_danger_occupancy_v1 スキーマで
    # occupancy (0=空, 1=占有) を使う。simulator の context_danger_v1 とは別スキーマのため
    # 対応する専用モデルが必要。parser が per-slot level を提供すれば context_danger_v1 へ移行可。
    inventory_levels = tuple(1 if item is not None else 0 for item in joined.hud.inventory)
    nearest = screen.get("nearest_enemy_offset")
    radius = screen.get("nearest_enemy_screen_radius")
    enemy_density = screen["enemy_density"].value[0]
    gem_density = screen["gem_density"].value[0]
    effective_world = gameplay_world if gameplay_world is not None else joined.world
    visible = [track for track in effective_world.tracks if track.on_screen and not track.clipped and track.confidence >= .35]
    max_cards = max(3, len(choices))
    world_age = (now_ns - gameplay_world.timestamp_ns) / 1_000_000_000 if now_ns is not None and gameplay_world is not None else 0.
    snapshot_age = (now_ns - last_gameplay_ns) / 1_000_000_000 if now_ns is not None and last_gameplay_ns is not None else 0.
    ui_state_age = max(0., (now_ns - joined.hud.captured_monotonic_ns) / 1_000_000_000) if now_ns is not None else 0.
    # P2: fallback/missing anchor 由来の estimate (validity=0) を中立値へ戻し world_validity に反映する。
    # nearest が None の場合 (敵なし) は制約を課さない (near_val=1.0)。
    near_val = nearest.validity if nearest is not None else 1.0
    nearest_enemy_screen_dist = radius.value[0] if (radius is not None and near_val > 0) else 0.
    nearest_enemy_screen_dir = nearest.value if (nearest is not None and near_val > 0) else (0., 0.)
    world_validity = min(joined.world_validity, near_val)
    context = ItemDecisionFeatures(
        decision_id=snapshot_id, feature_schema="context_danger_occupancy_v1",
        elapsed_time=joined.hud.timer_seconds or 0., level=joined.hud.level or 1,
        hp_ratio=joined.hud.hp_ratio or 0., xp_ratio=joined.hud.xp_ratio or 0.,
        weapon_slots=inventory_levels[:6], passive_slots=inventory_levels[6:],
        empty_slot_count=inventory_levels.count(0), evolution_readiness=0.,
        choice_count=len(choices), card_mask=(True,) * len(choices) + (False,) * (max_cards - len(choices)),
        fallback_kind=fallback, ui_state_validity=joined.item_validity, ui_state_age=ui_state_age,
        candidates=choices, max_item_cards=max_cards, enemy_density=enemy_density,
        gem_density=gem_density, nearest_enemy_screen_dist=nearest_enemy_screen_dist,
        nearest_enemy_screen_dir=nearest_enemy_screen_dir,
        boss_flag=any(track.coarse_class == "enemy" and "boss" in track.class_name for track in visible),
        hazard_flag=any(track.coarse_class == "hazard" for track in visible),
        world_validity=world_validity, world_age=world_age, last_gameplay_snapshot_age=snapshot_age,
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
        valid_ui_ids = frozenset(c.choice_id for c in ui.candidates if c.validity and c.semantic_kind == "item_card")
        if self._last_gameplay_screen is not None:
            item_context, choices = _item_context(
                joined, snapshot_id, self._last_gameplay_screen,
                self._last_gameplay_world, self._last_gameplay_ns, joined.captured_ns,
                valid_ui_ids,
            )
        else:
            item_context, choices = None, ()  # キャッシュなし: item overlay をgameplayとして流さない
        ui_policy_input = _build_ui_policy_input(snapshot_id, frame_id, joined.hud, ui)
        diagnostics = {
            "hud_world_skew_ms": abs(joined.hud.captured_monotonic_ns - joined.world.timestamp_ns) / 1_000_000,
            "hud_validity": joined.hud_validity, "world_validity": joined.world_validity,
            "combat_validity": joined.combat_validity, "item_validity": joined.item_validity,
        }
        return PerceptionSnapshot(
            snapshot_id, frame_id, joined.captured_ns, joined.hud.parser_artifact_hash,
            ui.source_content_hash, ui.ui_state_key, joined.hud.screen_state, deploy_obs,
            item_context, choices, ui, diagnostics,
            ui_policy_input=ui_policy_input,
        )

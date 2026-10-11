"""formal annotation と runner が共有する GT snapshot 構築規則。

HUD truth、COCO label、capture metadata から一 frame 分の snapshot を作る。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from survivors.perception_benchmark import _FORMAL_FOREGROUND_CLASSES
from survivors.perception_snapshot import PerceptionSnapshot, build_ui_presentation_from_hud
from survivors.real_obs_assembler import RealObsAssembler, _build_ui_policy_input
from survivors.temporal_state import TemporalAssembler
from survivors.capture_dataset import FrameRecord
from survivors.vision.entity_tracker import (
    PlayerAnchorState,
    TrackedEntityV2,
    TrackedWorldStateV2,
)
from survivors.vision.hud_parser import (
    HUD_STATE_SCHEMA_VERSION,
    INV_SLOT_COUNT,
    HudStateV1,
    ParsedButton,
    ParsedCard,
    _compute_candidate_set_hash,
)
from survivors.vision.hud_truth import EMPTY_SLOT, HudTruthRecord
from survivors.vision.roi_layout import (
    CHEST_ACK_BUTTON_ROI,
    DEATH_CONFIRM_ROI,
    RESULT_CONFIRM_ROI,
    REROLL_BUTTON_ROI,
    card_roi,
)

_SHA256 = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class LabeledBox:
    """COCO annotation を pixel xyxy へ変換した一 box。

    category_id は source COCO の検証用 identity として任意で保持する。
    """

    class_name: str
    xyxy: tuple[float, float, float, float]
    category_id: int | None = None


@dataclass(frozen=True, slots=True)
class CocoImageEntry:
    """一 capture frame とその COCO box 群を対応付ける。

    object_sha256 は画像 label が同じ capture object を指すことの検証に使う。
    """

    image_id: int
    object_sha256: str
    boxes: tuple[LabeledBox, ...]


@dataclass(frozen=True, slots=True)
class CaptureGroundTruthFiles:
    """descriptor から検証済みの session ground-truth file bytes。

    runner が logical id と SHA-256 を照合した結果だけを loader へ渡す。
    """

    hud_truth: bytes
    world_coco: bytes
    frames_jsonl: bytes


def coco_image_entries_from_wire(wire: object) -> dict[int, CocoImageEntry]:
    """COCO JSON を image id ごとの box entry に変換する。

    image・category・annotation の参照切れや重複は label の取り違えとして拒否する。
    """
    if not isinstance(wire, dict) or not all(key in wire for key in ("images", "annotations", "categories")):
        raise ValueError("COCO payload must contain images, annotations, and categories")
    if not all(isinstance(wire[key], list) for key in ("images", "annotations", "categories")):
        raise ValueError("COCO collections must be arrays")
    categories: dict[int, str] = {}
    for row in wire["categories"]:
        if not isinstance(row, dict) or type(row.get("id")) is not int or type(row.get("name")) is not str:
            raise ValueError("COCO category has an invalid schema")
        if row["id"] in categories:
            raise ValueError("duplicate COCO category id")
        categories[row["id"]] = row["name"]
    images: dict[int, tuple[str, list[LabeledBox]]] = {}
    for row in wire["images"]:
        if not isinstance(row, dict) or type(row.get("id")) is not int:
            raise ValueError("COCO image has an invalid schema")
        image_id = row["id"]
        digest = row.get("object_sha256")
        if image_id in images or type(digest) is not str or len(digest) != 64 or not set(digest) <= _SHA256:
            raise ValueError("COCO image identity is invalid")
        images[image_id] = (digest, [])
    for row in wire["annotations"]:
        if not isinstance(row, dict):
            raise ValueError("COCO annotation must be an object")
        image_id, category_id = row.get("image_id"), row.get("category_id")
        bbox = row.get("bbox")
        if type(image_id) is not int or image_id not in images or type(category_id) is not int or category_id not in categories:
            raise ValueError("COCO annotation references an unknown image or category")
        if not isinstance(bbox, list) or len(bbox) != 4 or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in bbox
        ):
            raise ValueError("COCO bbox must contain four finite numbers")
        x, y, width, height = (float(value) for value in bbox)
        if width <= 0 or height <= 0:
            raise ValueError("COCO bbox must have positive area")
        images[image_id][1].append(
            LabeledBox(categories[category_id], (x, y, x + width, y + height), category_id)
        )
    return {
        image_id: CocoImageEntry(image_id, digest, tuple(boxes))
        for image_id, (digest, boxes) in images.items()
    }


def hud_state_from_truth(
    row: HudTruthRecord,
    *,
    session_id: str,
    frame_index: int,
    captured_ns: int,
    parser_artifact_hash: str,
    inventory_size: int = INV_SLOT_COUNT,
) -> HudStateV1:
    """confirmed HUD truth を HudStateV1 の完全な入力へ写す。

    未知値は値なし・confidence 0 にし、部分的に null な inventory は拒否する。
    """
    if not isinstance(row, HudTruthRecord) or row.confirmed is not True:
        raise ValueError("confirmed HUD truth is required")
    if row.session_id != session_id or row.frame_id != frame_index:
        raise ValueError("HUD truth frame identity does not match capture")
    if inventory_size != INV_SLOT_COUNT:
        raise ValueError("inventory_size must match HudStateV1")
    if type(captured_ns) is not int or captured_ns < 0:
        raise ValueError("captured_ns must be a non-negative integer")
    if type(parser_artifact_hash) is not str or len(parser_artifact_hash) != 64 or not set(parser_artifact_hash) <= _SHA256:
        raise ValueError("parser_artifact_hash must be a SHA-256")
    items = row.expected_items
    if items is None:
        inventory = (None,) * inventory_size
        inventory_confidence = 0.0
    else:
        if len(items) != inventory_size:
            raise ValueError("expected_items has an invalid slot count")
        if any(item is None for item in items) and not all(item is None for item in items):
            raise ValueError("null inventory slot is only valid when every slot is unknown")
        inventory = tuple(items)
        inventory_confidence = 0.0 if all(item is None for item in items) else 1.0
    levels = row.expected_slot_levels
    if levels is not None and (len(levels) != inventory_size or items is None):
        raise ValueError("expected_slot_levels require a complete inventory-sized truth")
    cards = tuple(
        ParsedCard(index, item_id, "unknown", None, 1.0, "ground_truth", card_roi(index).as_xyxy())
        for index, item_id in enumerate(row.expected_choice or ())
    )
    buttons: tuple[ParsedButton, ...]
    if row.expected_screen_state in {"level_up_items", "level_up_fallback"}:
        buttons = (ParsedButton("reroll", 1.0, "ground_truth", REROLL_BUTTON_ROI.as_xyxy()),)
    elif row.expected_screen_state == "chest":
        buttons = (ParsedButton("ack_chest", 1.0, "ground_truth", CHEST_ACK_BUTTON_ROI.as_xyxy()),)
    elif row.expected_screen_state == "death":
        buttons = (ParsedButton("confirm", 1.0, "ground_truth", DEATH_CONFIRM_ROI.as_xyxy()),)
    elif row.expected_screen_state in {"result", "target_reached_transition"}:
        buttons = (ParsedButton("confirm", 1.0, "ground_truth", RESULT_CONFIRM_ROI.as_xyxy()),)
    else:
        buttons = ()
    values = {
        "timer_seconds": row.expected_timer_seconds,
        "level": row.expected_level,
        "hp_ratio": row.expected_hp_ratio,
        "xp_ratio": row.expected_xp_ratio,
    }
    confidence = {name: 1.0 if value is not None else 0.0 for name, value in values.items()}
    reasons = {name: "ground_truth" if value is not None else "ground_truth_unknown" for name, value in values.items()}
    inventory_levels = tuple(levels) if levels is not None else (None,) * inventory_size
    return HudStateV1(
        HUD_STATE_SCHEMA_VERSION, session_id, frame_index, captured_ns, parser_artifact_hash,
        row.expected_screen_state, 1.0, "ground_truth",
        values["timer_seconds"], confidence["timer_seconds"], reasons["timer_seconds"],
        bool(row.expected_timer_seconds is not None and row.expected_timer_seconds >= 1800.0),
        values["hp_ratio"], confidence["hp_ratio"], reasons["hp_ratio"],
        values["xp_ratio"], confidence["xp_ratio"], reasons["xp_ratio"],
        values["level"], confidence["level"], reasons["level"],
        inventory, inventory_confidence, canonical_hash({"slots": list(inventory)}),
        cards, _compute_candidate_set_hash(row.expected_screen_state, cards), buttons,
        False, False, False, 0.0, "ground_truth_unknown",
        inventory_levels, 1.0 if levels is not None else 0.0,
    )


def world_state_from_labels(
    boxes: Sequence[LabeledBox],
    *,
    frame_index: int,
    timestamp_ns: int,
    class_map: Any,
    viewport: tuple[int, int],
) -> TrackedWorldStateV2:
    """COCO label を assembler が消費する tracked world state にする。

    class map の formal foreground だけを保ち、座標・可視性・anchor を同じ規則で埋める。
    """
    if not isinstance(viewport, tuple) or len(viewport) != 2 or any(type(value) is not int or value <= 0 for value in viewport):
        raise ValueError("viewport must be a positive integer pair")
    if type(frame_index) is not int or frame_index < 0 or type(timestamp_ns) is not int or timestamp_ns < 0:
        raise ValueError("frame and timestamp must be non-negative integers")
    width, height = viewport
    accepted: list[tuple[LabeledBox, int, str]] = []
    for box in boxes:
        if not isinstance(box, LabeledBox) or box.class_name not in _FORMAL_FOREGROUND_CLASSES:
            continue
        class_id = class_map.name_to_id(box.class_name)
        coarse = class_map.coarse_for(box.class_name)
        if len(box.xyxy) != 4 or any(not math.isfinite(value) for value in box.xyxy):
            raise ValueError("labeled box must contain finite xyxy coordinates")
        x1, y1, x2, y2 = box.xyxy
        if x1 >= x2 or y1 >= y2:
            raise ValueError("labeled box must have positive area")
        accepted.append((box, class_id, coarse))
    player_box = next((box for box, _class_id, _coarse in accepted if box.class_name == "player_anchor"), None)
    if player_box is None:
        anchor_x, anchor_y = width / 2.0, height / 2.0
        player_anchor = PlayerAnchorState(0.5, 0.5, 0.0, True)
    else:
        anchor_x = (player_box.xyxy[0] + player_box.xyxy[2]) / 2.0
        anchor_y = (player_box.xyxy[1] + player_box.xyxy[3]) / 2.0
        player_anchor = PlayerAnchorState(anchor_x / width, anchor_y / height, 1.0, False)
    entities: list[TrackedEntityV2] = []
    for index, (box, class_id, coarse) in enumerate(accepted):
        x1, y1, x2, y2 = box.xyxy
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        clipped = x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height
        entities.append(TrackedEntityV2(
            index, class_id, box.class_name, coarse, 1.0, 1, frame_index,
            cx / width, cy / height, (cx - anchor_x) / width, (cy - anchor_y) / height,
            0.0, 0.0, True, clipped, (x2 - x1) / width, (y2 - y1) / height,
            timestamp_ns,
        ))
    return TrackedWorldStateV2(frame_index, timestamp_ns, entities, player_anchor)


def normalize_snapshot_ids(
    snapshot: PerceptionSnapshot,
    *,
    session_id: str,
    frame_index: int,
    viewport: tuple[int, int],
    hud: HudStateV1,
) -> PerceptionSnapshot:
    """assembler の snapshot identity を session/frame 単位に正規化する。

    UI content hash と policy input も新 identity と時系列 filter 済み HUD から再構築する。
    """
    frame_id = f"{session_id}:{frame_index}"
    snapshot_id = canonical_hash({"session_id": session_id, "frame_id": frame_id})
    ui = build_ui_presentation_from_hud(
        hud, viewport, snapshot_id=snapshot_id, frame_id=frame_id,
        hud_valid=float(snapshot.diagnostics.get("hud_validity", 0.0)) > 0,
    )
    policy = _build_ui_policy_input(snapshot_id, frame_id, hud, ui, meta_priority=())
    return replace(
        snapshot, snapshot_id=snapshot_id, frame_id=frame_id,
        parser_artifact_hash=hud.parser_artifact_hash,
        source_content_hash=ui.source_content_hash, ui_state_key=ui.ui_state_key,
        screen_state=hud.screen_state, ui_presentation=ui, ui_policy_input=policy,
    )


def ground_truth_snapshot(
    truth_row: HudTruthRecord | None,
    coco_image_entry: CocoImageEntry | None,
    frame_record: FrameRecord,
    *,
    parser_artifact_hash: str,
    session_id: str,
    class_map: Any,
    viewport: tuple[int, int] = (1920, 1080),
    schema: DeployObsSchema | None = None,
) -> PerceptionSnapshot | None:
    """truth・label・capture bytes を assembler 経由の GT snapshot にする。

    label hash を先に照合し、non-gameplay 画面は同 frame の GT gameplay world を prefill する。
    """
    if truth_row is None or truth_row.confirmed is not True or coco_image_entry is None:
        return None
    if frame_record.object_sha256 != coco_image_entry.object_sha256:
        raise ValueError("frame object_sha256 does not match COCO image")
    if truth_row.frame_id != frame_record.frame_id or truth_row.session_id != session_id:
        raise ValueError("HUD truth identity does not match frame record")
    schema = schema or DeployObsSchema.default_v2()
    hud = hud_state_from_truth(
        truth_row, session_id=session_id, frame_index=frame_record.frame_id,
        captured_ns=frame_record.captured_monotonic_ns,
        parser_artifact_hash=parser_artifact_hash,
    )
    world = world_state_from_labels(
        coco_image_entry.boxes, frame_index=frame_record.frame_id,
        timestamp_ns=frame_record.captured_monotonic_ns,
        class_map=class_map, viewport=viewport,
    )
    temporal = TemporalAssembler(policy_hz=1_000_000)
    assembler = RealObsAssembler(temporal)
    if hud.screen_state != "gameplay":
        prefill_ns = frame_record.captured_monotonic_ns - temporal.tick_interval_ns - 1
        if prefill_ns < 0:
            raise ValueError("capture timestamp is too small for GT gameplay prefill")
        prefill_hud = replace(hud, captured_monotonic_ns=prefill_ns, screen_state="gameplay")
        prefill_world = replace(
            world, timestamp_ns=prefill_ns,
            tracks=[replace(track, first_seen_timestamp_ns=prefill_ns) for track in world.tracks],
        )
        if assembler.assemble(prefill_hud, prefill_world, schema, viewport) is None:
            raise RuntimeError("GT gameplay prefill was suppressed")
    snapshot = assembler.assemble(hud, world, schema, viewport)
    if snapshot is None:
        raise RuntimeError("GT snapshot was suppressed by temporal cadence")
    filtered_hud = temporal.last_hud
    if filtered_hud is None:
        raise RuntimeError("temporal assembler did not retain GT HUD")
    snapshot = normalize_snapshot_ids(
        snapshot, session_id=session_id, frame_index=frame_record.frame_id,
        viewport=viewport, hud=filtered_hud,
    )
    classes = {
        f"{session_id}:{frame_record.frame_id}:{index}": track.class_name
        for index, track in enumerate(world.tracks)
    }
    boxes = {}
    for index, box in enumerate(
        box for box in coco_image_entry.boxes
        if box.class_name in _FORMAL_FOREGROUND_CLASSES
    ):
        boxes[f"{session_id}:{frame_record.frame_id}:{index}"] = list(box.xyxy)
    counts: dict[str, int] = {}
    for class_name in classes.values():
        counts[class_name] = counts.get(class_name, 0) + 1
    diagnostics = dict(snapshot.diagnostics)
    diagnostics.update(
        foreground_entity_classes=classes,
        foreground_entity_boxes=boxes,
        foreground_entity_counts=counts,
        slice_boss_present=any(
            track.class_name == "enemy_boss" and track.on_screen and not track.clipped and track.confidence >= 0.35
            for track in world.tracks
        ),
        slice_hazard_present=any(
            track.coarse_class == "hazard" and track.on_screen and not track.clipped and track.confidence >= 0.35
            for track in world.tracks
        ),
    )
    if truth_row.expected_timer_seconds is not None:
        diagnostics["slice_elapsed_time"] = truth_row.expected_timer_seconds
    return replace(snapshot, diagnostics=diagnostics)


def semantic_hash(snapshot: PerceptionSnapshot) -> str:
    """runner の ground truth hash と同じ semantic fields を SHA-256 にする。

    pixel observation を含めず、formal annotation が検証する HUD・UI・entity 意味を束縛する。
    """
    context = snapshot.item_context
    context_payload = None if context is None else {
        "elapsed_time": context.elapsed_time,
        "level": context.level,
        "hp_ratio": context.hp_ratio,
        "xp_ratio": context.xp_ratio,
        "enemy_density": context.enemy_density,
        "nearest_enemy_screen_dist": context.nearest_enemy_screen_dist,
        "boss_flag": context.boss_flag,
        "hazard_flag": context.hazard_flag,
    }
    ui = snapshot.ui_presentation
    payload = {
        "frame_id": snapshot.frame_id,
        "captured_ns": snapshot.captured_ns,
        "screen_state": snapshot.screen_state,
        "item_context": context_payload,
        "inventory_hash": ui.inventory_hash,
        "candidate_set_hash": ui.candidate_set_hash,
        "candidates": [
            {
                "choice_id": target.choice_id,
                "choice_index": target.choice_index,
                "semantic_kind": target.semantic_kind,
                "validity": target.validity,
                "roi": [target.roi.left, target.roi.top, target.roi.right, target.roi.bottom],
            }
            for target in ui.candidates
        ],
        "buttons": [
            {
                "semantic_action": target.semantic_action,
                "capability": target.capability,
                "validity": target.validity,
                "roi": [target.roi.left, target.roi.top, target.roi.right, target.roi.bottom],
            }
            for target in ui.buttons
        ],
        "foreground_entity_counts": dict(snapshot.diagnostics.get("foreground_entity_counts", {})),
        "foreground_entity_classes": dict(snapshot.diagnostics.get("foreground_entity_classes", {})),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def formal_annotation_row(snapshot: PerceptionSnapshot, truth: HudTruthRecord) -> dict[str, Any]:
    """一 snapshot から immutable formal annotation wire row を作る。

    raw UI evidence を snapshot と同じ candidate 順で出力し、hash 検証へ渡す。
    """
    if (
        not isinstance(truth, HudTruthRecord)
        or snapshot.frame_id != f"{truth.session_id}:{truth.frame_id}"
    ):
        raise ValueError("annotation truth does not match snapshot")
    raw_inventory = list(truth.expected_items or (None,) * INV_SLOT_COUNT)
    if len(raw_inventory) != INV_SLOT_COUNT:
        raise ValueError("annotation inventory has an invalid slot count")
    return {
        "schema_version": "perception_formal_annotation.v1",
        "session_id": truth.session_id,
        "frame_index": truth.frame_id,
        "raw_card_ids": [candidate.choice_id for candidate in snapshot.ui_presentation.candidates],
        "raw_inventory": raw_inventory,
        "ground_truth_semantic_hash": semantic_hash(snapshot),
    }

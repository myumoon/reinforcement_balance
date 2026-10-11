"""formal GT snapshot の再現可能な構築を検証する。

合成した HUD truth と world label を通し、benchmark が読む hash 契約を守る。
"""

from __future__ import annotations

import hashlib

import pytest

from benchmark_survivors_perception import _ground_truth_semantic_hash
from reinbalance_survivors_contracts.canonical_json import canonical_hash
from survivors.capture_dataset import FrameRecord
from survivors.formal_ground_truth import (
    CocoImageEntry,
    LabeledBox,
    formal_annotation_row,
    ground_truth_snapshot,
)
from survivors.perception_benchmark import _FORMAL_FOREGROUND_CLASSES
from survivors.vision.entity_tracker import default_class_map
from survivors.vision.hud_truth import HudTruthRecord


def _truth(screen_state: str, *, items: tuple[str | None, ...] | None = None) -> HudTruthRecord:
    """指定画面の最小 truth 行を作る。

    inventory を全欠損にし、screen state ごとの assembler 経路だけを確認する。
    """
    return HudTruthRecord(
        "hud_truth.v1", "session-formal", 0, "annotator", True,
        screen_state, 20.0, 2, 0.75, 0.5, items, None,
        ("whip",) if screen_state == "level_up_items" else None,
        "hud", None, "2026-10-11T00:00:00Z", {},
    )


def _snapshot(screen_state: str):
    """固定 COCO label と frame metadata から一つの GT snapshot を構築する。

    player anchor と foreground box の両方を含め、entity と hash の写像を確かめる。
    """
    object_sha = hashlib.sha256(b"frame").hexdigest()
    frame = FrameRecord(0, 2_000_000_000, "frame.png", object_sha, (0, 0, 1920, 1080), True, "a" * 64, "build")
    image = CocoImageEntry(
        image_id=0,
        object_sha256=object_sha,
        boxes=(
            LabeledBox("player_anchor", (900, 500, 1020, 620)),
            LabeledBox("enemy_boss", (1100, 500, 1200, 600)),
            LabeledBox("hazard_area", (0, 300, 80, 380)),
        ),
    )
    return ground_truth_snapshot(
        _truth(screen_state), image, frame,
        parser_artifact_hash="b" * 64, session_id="session-formal",
        class_map=default_class_map(),
    )


def test_ground_truth_snapshot_replays_all_formal_screens_and_hashes() -> None:
    """全 formal 画面で GT snapshot と annotation hash が同じ入力から決まる。

    assembler の gameplay prefill を含む各画面の tick 抑制を避け、writer と runner の hash を照合する。
    """
    for screen_state in (
        "gameplay", "level_up_items", "level_up_fallback", "chest", "death", "result",
    ):
        first = _snapshot(screen_state)
        second = _snapshot(screen_state)
        assert first is not None and second is not None
        assert _ground_truth_semantic_hash(first) == _ground_truth_semantic_hash(second)
        assert first.diagnostics["foreground_entity_classes"] == {
            "session-formal:0:0": "player_anchor",
            "session-formal:0:1": "enemy_boss",
            "session-formal:0:2": "hazard_area",
        }
        assert set(first.diagnostics["foreground_entity_classes"].values()) <= set(_FORMAL_FOREGROUND_CLASSES)
        row = formal_annotation_row(first, _truth(screen_state))
        assert row["ground_truth_semantic_hash"] == _ground_truth_semantic_hash(first)
        assert row["raw_inventory"] == [None] * 12
        assert first.ui_presentation.inventory_hash == canonical_hash({"slots": [None] * 12})
        expected_cards = [candidate.choice_id for candidate in first.ui_presentation.candidates]
        assert row["raw_card_ids"] == expected_cards


def test_ground_truth_snapshot_rejects_partial_null_inventory() -> None:
    """一部だけ読めない inventory を confirmed truth として扱わない。

    全スロット欠損は許可し、部分欠損は後段 hash の意味を曖昧にしないよう拒否する。
    """
    with pytest.raises(ValueError, match="null inventory slot"):
        _snapshot_with_truth(_truth("gameplay", items=("whip", None) + ("empty_slot",) * 10))


def test_ground_truth_snapshot_rejects_frame_object_hash_mismatch() -> None:
    """COCO image と captured frame の object identity が違えば拒否する。

    同じ session 内の別画像 label が GT に混ざるのを防ぐ。
    """
    object_sha = hashlib.sha256(b"frame").hexdigest()
    frame = FrameRecord(0, 2_000_000_000, "frame.png", object_sha, (0, 0, 1920, 1080), True, "a" * 64, "build")
    image = CocoImageEntry(0, "0" * 64, ())
    with pytest.raises(ValueError, match="object_sha256"):
        ground_truth_snapshot(
            _truth("gameplay"), image, frame,
            parser_artifact_hash="b" * 64, session_id="session-formal",
            class_map=default_class_map(),
        )


def _snapshot_with_truth(truth: HudTruthRecord):
    """helper の固定 frame と label を受け取り truth を差し替える。

    inventory の validation だけを実際の GT builder に通す。
    """
    object_sha = hashlib.sha256(b"frame").hexdigest()
    frame = FrameRecord(0, 2_000_000_000, "frame.png", object_sha, (0, 0, 1920, 1080), True, "a" * 64, "build")
    image = CocoImageEntry(0, object_sha, ())
    return ground_truth_snapshot(
        truth, image, frame,
        parser_artifact_hash="b" * 64, session_id="session-formal",
        class_map=default_class_map(),
    )

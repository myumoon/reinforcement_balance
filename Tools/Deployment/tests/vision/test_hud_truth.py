"""hud_truth.v1（HUD 正解値 record）の読み書き・検査・行コマンドの test。

synthetic frame と合成 session だけで動かし、実ゲーム映像は使わない。
下書きが HudStateV1 を正しく写すこと、行コマンドの受理と拒否、
jsonl の往復、PNG の sha256 照合、hud_calibration との互換を確かめる。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from survivors.capture.captured_frame import CapturedFrame
from survivors.capture_dataset import DatasetWriter
from survivors.vision.hud_calibration import _validate_annotation
from survivors.vision.hud_parser import INV_SLOT_COUNT, HudParser, ParsedCard
from survivors.vision.hud_truth import (
    EMPTY_SLOT,
    HUD_TRUTH_FILENAME,
    HudTruthRecord,
    apply_command,
    draft_from_parser,
    expected_roi_for_state,
    load_frame_pixels,
    read_hud_truth,
    validate_record,
    write_hud_truth,
)
from survivors.vision.roi_layout import SLOT_LEVEL_VISIBLE_STATES

PROFILE_HASH = "a" * 64
BUILD_ID = "survivors-test-build"
NOW = "2026-10-09T00:00:00+00:00"
ITEMS = ("whip", "wand", EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT,
         "spinach", EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT, EMPTY_SLOT)
LEVELS = (3, 1, None, None, None, None, 2, None, None, None, None, None)


def _record(state: str = "gameplay", **overrides) -> HudTruthRecord:
    """妥当な未確定 record を作り、overrides で field を差し替える。"""
    base = HudTruthRecord(
        schema_version="hud_truth.v1",
        session_id="session-001",
        frame_id=5,
        annotator_id="tester",
        confirmed=False,
        expected_screen_state=state,
        expected_timer_seconds=61.0,
        expected_level=4,
        expected_hp_ratio=0.5,
        expected_xp_ratio=0.25,
        expected_items=ITEMS,
        expected_slot_levels=None,
        expected_choice=None,
        roi_name="hud",
        expected_roi=expected_roi_for_state(state),
        confirmed_at=None,
        draft_source={"parser_artifact_hash": "c" * 64, "atlas_content_hash": "none"},
    )
    return replace(base, **overrides)


class _FixedParser:
    """parse で決まった HudStateV1 を返す parser の代役。"""

    def __init__(self, state) -> None:
        """返す HudStateV1 を受け取る。"""
        self.state = state

    def parse(self, frame_bgra, **_kwargs):
        """受け取った frame を無視して固定の state を返す。"""
        return self.state


def _draft(state, frame) -> HudTruthRecord:
    """固定 state の parser で下書きを作る。"""
    return draft_from_parser(
        frame, _FixedParser(state), session_id="session-001", frame_index=3,
        captured_monotonic_ns=100, annotator_id="tester", atlas_content_hash="none",
    )


def test_draft_copies_hud_state_and_nulls_partial_inventory(gameplay_frame, dummy_parser_artifact_hash):
    """下書きは HudStateV1 を写し、読めない slot があれば items 全体を null にする。"""
    state = HudParser(parser_artifact_hash=dummy_parser_artifact_hash).parse(
        gameplay_frame, session_id="session-001", frame_index=3, captured_monotonic_ns=100
    )
    partial = replace(state, inventory=("whip",) + (None,) * (INV_SLOT_COUNT - 1))
    draft = _draft(partial, gameplay_frame)
    assert draft.expected_screen_state == state.screen_state
    assert draft.expected_timer_seconds == state.timer_seconds
    assert draft.expected_level == state.level
    assert draft.expected_hp_ratio == state.hp_ratio
    assert draft.expected_xp_ratio == state.xp_ratio
    assert draft.expected_items is None
    assert draft.expected_slot_levels is None
    assert draft.expected_roi == expected_roi_for_state(state.screen_state)
    assert draft.confirmed is False and draft.confirmed_at is None
    assert draft.draft_source["parser_artifact_hash"] == dummy_parser_artifact_hash
    assert validate_record(draft) == ()

    cards = (
        ParsedCard(0, "whip", "weapon", 2, 0.9, "ok", None),
        ParsedCard(1, None, "unknown", None, 0.1, "low", None),
        ParsedCard(2, "spinach", "passive", 1, 0.9, "ok", None),
    )
    full = replace(state, screen_state="level_up_items", inventory=ITEMS, cards=cards)
    draft = _draft(full, gameplay_frame)
    assert draft.expected_items == ITEMS
    assert draft.expected_slot_levels is None
    assert draft.expected_choice == ("whip", "spinach")
    assert draft.expected_roi == expected_roi_for_state("level_up_items")


def test_expected_roi_for_state_hud_and_negative_states():
    """HUD あり状態は HP〜XP バーの pixel 矩形、HUD なし状態は None。"""
    roi = expected_roi_for_state("gameplay")
    assert roi == (0, 32, 1920, 75)
    for state in ("level_up_items", "level_up_fallback", "chest"):
        assert expected_roi_for_state(state) == roi
    for state in ("paused", "target_reached_transition", "death", "result", "unknown"):
        assert expected_roi_for_state(state) is None


@pytest.mark.parametrize(
    ("line", "field", "value"),
    [
        ("set timer 123.0", "expected_timer_seconds", 123.0),
        ("set level 7", "expected_level", 7),
        ("set hp 0.85", "expected_hp_ratio", 0.85),
        ("set xp 0.2", "expected_xp_ratio", 0.2),
        ("set timer null", "expected_timer_seconds", None),
        ("set item 2 axe", "expected_items", ITEMS[:2] + ("axe",) + ITEMS[3:]),
        ("set items null", "expected_items", None),
        ("set choice whip|axe|spinach", "expected_choice", ("whip", "axe", "spinach")),
        ("set choice null", "expected_choice", None),
    ],
)
def test_apply_command_updates_fields(line, field, value):
    """set コマンドで対応する field が更新される。"""
    updated, error = apply_command(_record(expected_choice=("x",)), line)
    assert error is None
    assert getattr(updated, field) == value


def test_set_item_on_null_items_starts_partial_draft():
    """items が null でも set item で 1 slot ずつ入れられる（未確定のうちは他 slot は null）。"""
    updated, error = apply_command(_record(expected_items=None), "set item 0 whip")
    assert error is None
    assert updated.expected_items == ("whip",) + (None,) * (INV_SLOT_COUNT - 1)
    _, error = apply_command(updated, "ok", now=NOW)
    assert error is not None and "slot 1" in error


def test_set_state_updates_roi_and_drops_levels():
    """set state は ROI を自動更新し、段階マークが出ない state では levels を null にする。"""
    record = _record("level_up_items", expected_slot_levels=LEVELS)
    updated, error = apply_command(record, "set state death")
    assert error is None
    assert updated.expected_roi is None
    assert updated.expected_slot_levels is None
    updated, error = apply_command(record, "set state gameplay")
    assert error is None
    assert updated.expected_roi == expected_roi_for_state("gameplay")
    assert updated.expected_slot_levels is None
    updated, error = apply_command(record, "set state level_up_fallback")
    assert updated.expected_slot_levels == LEVELS
    _, error = apply_command(record, "set state flying")
    assert error is not None


def test_set_levels_accepts_valid_levels():
    """level-up 画面では set levels と set level-of で slot level を入れられる。"""
    record = _record("level_up_items")
    updated, error = apply_command(record, "set levels 3 1 - - - - 2 - - - - -")
    assert error is None
    assert updated.expected_slot_levels == LEVELS
    updated, error = apply_command(updated, "set level-of 1 4")
    assert error is None
    assert updated.expected_slot_levels[1] == 4
    confirmed, error = apply_command(updated, "ok", now=NOW)
    assert error is None and confirmed.confirmed and confirmed.confirmed_at == NOW
    updated, error = apply_command(updated, "set levels null")
    assert error is None and updated.expected_slot_levels is None


@pytest.mark.parametrize(
    ("state", "line", "reason"),
    [
        ("level_up_items", "set levels 3 1 - - - - 2 - - - -", "12"),
        ("level_up_items", "set levels 10 1 - - - - 2 - - - - -", "1..9"),
        ("level_up_items", "set levels 0 1 - - - - 2 - - - - -", "1..9"),
        ("level_up_items", "set levels 3 1 2 - - - 2 - - - - -", EMPTY_SLOT),
        ("gameplay", "set levels 3 1 - - - - 2 - - - - -", "gameplay"),
        ("level_up_items", "set level-of 0 3", "set levels"),
        ("gameplay", "set nothing 1", "unknown field"),
        ("gameplay", "jump", "unknown command"),
        ("gameplay", "set hp 1.5", "expected_hp_ratio"),
        ("gameplay", "set level abc", "invalid literal"),
    ],
)
def test_apply_command_rejects_without_change(state, line, reason):
    """不正なコマンドは record を変えず理由を返す。"""
    record = _record(state)
    updated, error = apply_command(record, line)
    assert updated is record
    assert error is not None and reason in error


def test_set_levels_rejected_when_items_null():
    """expected_items が null の frame には slot level を入れられない。"""
    record = _record("level_up_items", expected_items=None)
    _, error = apply_command(record, "set levels 3 1 - - - - 2 - - - - -")
    assert error is not None and "expected_items is null" in error


def test_set_items_null_also_drops_levels():
    """set items null は expected_slot_levels も null にする（items null なら levels も null）。"""
    record = _record("level_up_items", expected_slot_levels=LEVELS)
    updated, error = apply_command(record, "set items null")
    assert error is None
    assert updated.expected_items is None and updated.expected_slot_levels is None


def test_validate_is_lenient_for_unconfirmed_drafts():
    """未確定の下書きは slot 単位の null を許し、確定行では拒否する。"""
    partial = ("whip",) + (None,) * (INV_SLOT_COUNT - 1)
    draft = _record("level_up_items", expected_items=partial,
                    expected_slot_levels=(2,) + (None,) * (INV_SLOT_COUNT - 1))
    assert validate_record(draft) == ()
    confirmed = replace(draft, confirmed=True, confirmed_at=NOW)
    errors = validate_record(confirmed)
    assert any("expected_items slot 1" in error for error in errors)
    missing_level = _record("level_up_items", confirmed=True, confirmed_at=NOW,
                            expected_slot_levels=(3, None) + LEVELS[2:])
    assert any("expected_slot_levels slot 1" in error for error in validate_record(missing_level))


def test_validate_rejects_roi_mismatch_and_bad_state():
    """expected_roi が state から決まる値と違う、または state が未知なら違反。"""
    assert validate_record(_record(expected_roi=None))
    assert validate_record(_record("death", expected_roi=(0, 0, 1, 1)))
    assert validate_record(_record(expected_screen_state="flying", expected_roi=None))
    assert validate_record(_record(expected_items=ITEMS[:11]))
    assert SLOT_LEVEL_VISIBLE_STATES == frozenset({"level_up_items", "level_up_fallback"})


def test_hud_truth_jsonl_round_trip_preserves_unknown_fields(tmp_path):
    """書いて読むと frame_id 昇順になり、未知 field も保持される。"""
    first = _record(frame_id=9, extra={"note": "keep me"})
    second = _record("level_up_items", frame_id=2, expected_slot_levels=LEVELS,
                     expected_choice=("whip", "axe"), confirmed=True, confirmed_at=NOW)
    path = write_hud_truth(tmp_path, [first, second])
    assert path == tmp_path / HUD_TRUTH_FILENAME
    restored = read_hud_truth(tmp_path)
    assert [record.frame_id for record in restored] == [2, 9]
    assert restored == [second, first]
    assert restored[1].extra == {"note": "keep me"}
    wire = json.loads(path.read_text(encoding="utf-8").splitlines()[1])
    assert wire["note"] == "keep me"
    write_hud_truth(tmp_path, restored)
    assert read_hud_truth(tmp_path) == restored
    assert not (tmp_path / f".{HUD_TRUTH_FILENAME}.tmp").exists()


def test_write_rejects_duplicates_and_invalid_confirmed_rows(tmp_path):
    """重複 frame_id と、規則違反の確定行は書かずに ValueError。"""
    with pytest.raises(ValueError, match="duplicate"):
        write_hud_truth(tmp_path, [_record(), _record()])
    bad = _record(confirmed=True, confirmed_at=NOW, expected_slot_levels=LEVELS)
    with pytest.raises(ValueError, match="expected_slot_levels"):
        write_hud_truth(tmp_path, [bad])
    assert not (tmp_path / HUD_TRUTH_FILENAME).exists()
    (tmp_path / HUD_TRUTH_FILENAME).write_text(
        "\n".join(json.dumps(_record().to_wire()) for _ in range(2)) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_hud_truth(tmp_path)


def test_load_frame_pixels_verifies_sha256(tmp_path, gameplay_frame):
    """PNG を BGRA で読み、object_sha256 が違えば ValueError。"""
    writer = DatasetWriter(tmp_path, "session-001", PROFILE_HASH, BUILD_ID)
    writer.write_frame(CapturedFrame(
        frame_bgra=gameplay_frame, captured_monotonic_ns=100, session_frame_index=0,
        client_rect_screen_px=(0, 0, 1920, 1080), foreground=True,
        target_profile_hash=PROFILE_HASH, game_build_id=BUILD_ID,
    ))
    writer.publish(operator_checkpoint="synthetic")
    manifest = DatasetWriter.restore(tmp_path, "session-001", metadata_only=True)
    record = manifest.frame_records[0]
    pixels = load_frame_pixels(manifest.session_path, record)
    assert pixels.dtype == np.uint8 and pixels.shape == (1080, 1920, 4)
    assert np.array_equal(pixels, gameplay_frame)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frame_pixels(manifest.session_path, replace(record, object_sha256="0" * 64))


def test_hud_calibration_ignores_slot_levels_as_unknown_field():
    """hud_calibration._validate_annotation は expected_slot_levels を未知 field として無視する。"""
    record = _record("level_up_items", expected_slot_levels=LEVELS, confirmed=True, confirmed_at=NOW)
    wire = record.to_wire()
    assert wire["expected_slot_levels"] == list(LEVELS)
    assert wire["expected_choice"] is None
    _validate_annotation(wire)


def _write_session(store_root: Path, frame) -> None:
    """同じ合成 frame を 3 枚持つ capture session を store_root に作る。"""
    writer = DatasetWriter(store_root, "session-001", PROFILE_HASH, BUILD_ID)
    for frame_id in range(3):
        writer.write_frame(CapturedFrame(
            frame_bgra=frame, captured_monotonic_ns=100 + frame_id, session_frame_index=frame_id,
            client_rect_screen_px=(0, 0, 1920, 1080), foreground=True,
            target_profile_hash=PROFILE_HASH, game_build_id=BUILD_ID,
        ))
    writer.publish(operator_checkpoint="synthetic")


def _write_label(path: Path, *, checked: bool, labels: tuple[str, ...] = ()) -> None:
    """X-AnyLabeling 形式の label JSON を書く。"""
    shapes = [{"label": label, "shape_type": "rectangle", "points": [[10, 10], [100, 100]]} for label in labels]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"shapes": shapes, "imagePath": path.stem + ".png",
                                "imageData": None, "checked": checked}), encoding="utf-8")


def test_cli_annotates_checked_frames_with_labels_and_resume(tmp_path, gameplay_frame):
    """CLI は checked label の frame を下書きし、undo・ok・ok-range・--resume が働く。"""
    import io
    import annotate_survivors_hud_truth as cli

    store, work = tmp_path / "store", tmp_path / "work"
    _write_session(store, gameplay_frame)
    _write_label(work / "session-001" / "00000000.json", checked=True)
    _write_label(work / "session-001" / "00000001.json", checked=False, labels=("card",))
    _write_label(work / "session-001" / "00000002.json", checked=True)
    session_path = store / "capture_sessions" / "session-001"
    base_args = ["--store-root", str(store), "--session-id", "session-001", "--annotator-id", "tester"]

    with pytest.raises(SystemExit):
        cli.main(base_args)
    out = io.StringIO()
    code = cli.main(base_args + ["--work-root", str(work)],
                    stdin=io.StringIO("set level 7\nundo\nok\nok-range 2 2\n"), stdout=out)
    assert code == 0, out.getvalue()
    assert cli.STATE_NOTICE in out.getvalue()
    assert "ok-range: 1 frame を確定" in out.getvalue()
    records = read_hud_truth(session_path)
    assert [(r.frame_id, r.expected_screen_state, r.confirmed) for r in records] == [
        (0, "gameplay", True), (2, "gameplay", True)]
    assert records[0].expected_level != 7
    assert cli.main(base_args + ["--frame-ids", "0", "1"], stdin=io.StringIO(""), stdout=out) == 1

    out = io.StringIO()
    code = cli.main(base_args + ["--frame-ids", "0", "1", "2", "--resume", "--work-root", str(work), "--from-labels"],
                    stdin=io.StringIO("ok-range 1 1\nset level 9\n"), stdout=out)
    assert code == 0, out.getvalue()
    assert "停止: frame 1" in out.getvalue() and "expected_screen_state" in out.getvalue()
    records = read_hud_truth(session_path)
    assert [(r.frame_id, r.confirmed) for r in records] == [(0, True), (1, False), (2, True)]
    assert records[1].expected_screen_state == "level_up_items"
    assert records[1].expected_level == 9

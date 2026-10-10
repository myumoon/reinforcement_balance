"""HUD truth record（hud_truth.v1）の schema・読み書き・下書き生成・行コマンド適用。

capture session の各 frame について、人が確認した HUD の正解値
（画面状態、timer、level、HP・XP 比、所持 item、slot level、card、HUD 帯 ROI）を
session 配下の hud_truth.jsonl に 1 行 1 frame で保存する。
下書きは development parser の HudStateV1 から作り、人が行コマンドで直して確定する。
既存の annotations.jsonl や X-AnyLabeling の label は読み書きしない。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from survivors.capture_dataset import FrameRecord, _resolve_relative
from survivors.vision.hud_parser import INV_SLOT_COUNT, SCREEN_STATES, HudParser
from survivors.vision.roi_layout import (
    HP_BAR_ROI,
    SLOT_LEVEL_VISIBLE_STATES,
    XP_BAR_ROI,
    norm_to_pixels,
)

# hud_truth.jsonl の schema 版
HUD_TRUTH_SCHEMA_VERSION: Final[str] = "hud_truth.v1"
# session 配下の保存先 file 名
HUD_TRUTH_FILENAME: Final[str] = "hud_truth.jsonl"
# 空きスロットを表す item 値
EMPTY_SLOT: Final[str] = "empty_slot"
# slot level の上限（deploy_obs_v2_features.yaml の max_passive_level に合わせた緩い上限）
MAX_SLOT_LEVEL: Final[int] = 9
# HUD 帯（HP バー〜XP バー）が見える画面状態
HUD_VISIBLE_STATES: Final[frozenset[str]] = frozenset(
    {"gameplay", "level_up_items", "level_up_fallback", "chest"}
)
# card（expected_choice）が出る画面状態。確定行ではこれ以外の state の expected_choice は null
LEVEL_UP_STATES: Final[frozenset[str]] = frozenset({"level_up_items", "level_up_fallback"})
# HUD truth の ROI 名（hud_calibration の roi_name と同じ）
HUD_ROI_NAME: Final[str] = "hud"
# 1 画面の基準解像度
_FRAME_W, _FRAME_H = 1920, 1080


@dataclass(frozen=True)
class HudTruthRecord:
    """hud_truth.jsonl の 1 行（1 frame 分の HUD 正解値）。

    expected_* は人が確認した正解値で、読めないものは None（JSON では null）。
    expected_items と expected_slot_levels は 12 slot（武器 6 + パッシブ 6）の配列。
    extra には schema に無い key を入れ、書き戻すときにそのまま保存する。
    """

    schema_version: str
    session_id: str
    frame_id: int
    annotator_id: str
    confirmed: bool
    expected_screen_state: str
    expected_timer_seconds: float | None
    expected_level: int | None
    expected_hp_ratio: float | None
    expected_xp_ratio: float | None
    expected_items: tuple[str | None, ...] | None
    expected_slot_levels: tuple[int | None, ...] | None
    expected_choice: tuple[str, ...] | None
    roi_name: str
    expected_roi: tuple[int, int, int, int] | None
    confirmed_at: str | None
    draft_source: dict
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """extra が schema の key と重ならないことだけを確かめる。

        値の妥当性は validate_record がまとめて検査する（下書きの途中状態を許すため）。
        """
        overlap = set(self.extra) & _WIRE_KEYS
        if overlap:
            raise ValueError(f"extra must not contain schema keys: {sorted(overlap)}")

    def to_wire(self) -> dict:
        """JSON にそのまま書ける dict を返す。

        tuple は list に戻し、extra の未知 key も同じ階層に書き戻す。
        """
        wire: dict = dict(self.extra)
        for name in _WIRE_KEYS:
            value = getattr(self, name)
            wire[name] = list(value) if isinstance(value, tuple) else value
        wire["draft_source"] = dict(self.draft_source)
        return wire

    @classmethod
    def from_wire(cls, wire: dict) -> "HudTruthRecord":
        """JSON の dict から record を作る。

        schema の key がそろっていなければ ValueError。未知 key は extra に入れて保持する。
        配列 field は list か null だけを受け入れ、tuple に変換する。
        """
        if not isinstance(wire, dict):
            raise ValueError("hud truth record must be a JSON object")
        missing = sorted(_WIRE_KEYS - set(wire))
        if missing:
            raise ValueError(f"hud truth record is missing keys: {missing}")
        values = {name: wire[name] for name in _WIRE_KEYS}
        for name in ("expected_items", "expected_slot_levels", "expected_choice", "expected_roi"):
            value = values[name]
            if value is not None and not isinstance(value, list):
                raise ValueError(f"{name} must be a list or null")
            values[name] = None if value is None else tuple(value)
        if not isinstance(values["draft_source"], dict):
            raise ValueError("draft_source must be an object")
        extra = {key: value for key, value in wire.items() if key not in _WIRE_KEYS}
        return cls(**values, extra=extra)


# schema に含まれる key（extra 以外の dataclass field）
_WIRE_KEYS: Final[frozenset[str]] = frozenset(
    f.name for f in fields(HudTruthRecord) if f.name != "extra"
)


def expected_roi_for_state(state: str) -> tuple[int, int, int, int] | None:
    """画面状態から決まる HUD 帯の pixel 矩形を返す（HUD なし状態は None）。

    HUD あり状態（gameplay／level_up_items／level_up_fallback／chest）では
    HP バー上端から XP バー下端までを 1920×1080 pixel に戻した矩形 (l, t, r, b) を返す。
    それ以外の状態は HUD が見えない負例なので None を返す。
    """
    if state not in HUD_VISIBLE_STATES:
        return None
    hp = norm_to_pixels(HP_BAR_ROI, _FRAME_W, _FRAME_H)
    xp = norm_to_pixels(XP_BAR_ROI, _FRAME_W, _FRAME_H)
    return (min(hp.x0, xp.x0), min(hp.y0, xp.y0), max(hp.x1, xp.x1), max(hp.y1, xp.y1))


def _is_number(value: object) -> bool:
    """bool を除く有限の int／float なら True。

    True/False も int の仲間として扱われるので、ここで数値から外す。
    NaN や無限大も正解値としては使えないので False にする。
    """
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_record(record: HudTruthRecord) -> tuple[str, ...]:
    """record の違反理由を tuple で返す（空なら妥当）。

    confirmed=True の record では slot 単位の null を禁止する（読めない frame は配列全体を null）。
    confirmed=False の下書きでは slot 単位の null を許す。
    expected_slot_levels は段階マークが見える画面でだけ非 null にでき、
    item の slot は 1..9、empty_slot の slot は null でなければならない。
    確定行の expected_choice は level-up 画面（LEVEL_UP_STATES）でだけ非 null にできる。
    CLI の受理判定・write_hud_truth・test はすべてこの関数を通す。
    """
    errors: list[str] = []
    strict = record.confirmed is True
    if record.schema_version != HUD_TRUTH_SCHEMA_VERSION:
        errors.append(f"schema_version must be {HUD_TRUTH_SCHEMA_VERSION!r}")
    if not isinstance(record.session_id, str) or not record.session_id:
        errors.append("session_id must be a non-empty string")
    if isinstance(record.frame_id, bool) or not isinstance(record.frame_id, int) or record.frame_id < 0:
        errors.append("frame_id must be a non-negative integer")
    if not isinstance(record.annotator_id, str) or not record.annotator_id:
        errors.append("annotator_id must be a non-empty string")
    if not isinstance(record.confirmed, bool):
        errors.append("confirmed must be a bool")
    if strict and not isinstance(record.confirmed_at, str):
        errors.append("confirmed record needs confirmed_at")
    if record.confirmed_at is not None and not isinstance(record.confirmed_at, str):
        errors.append("confirmed_at must be a string or null")
    state = record.expected_screen_state
    if state not in SCREEN_STATES:
        errors.append(f"expected_screen_state must be one of {sorted(SCREEN_STATES)}, got {state!r}")

    timer = record.expected_timer_seconds
    if timer is not None and (not _is_number(timer) or timer < 0):
        errors.append(f"expected_timer_seconds must be a non-negative number, got {timer!r}")
    level = record.expected_level
    if level is not None and (isinstance(level, bool) or not isinstance(level, int) or level < 1):
        errors.append(f"expected_level must be a positive integer, got {level!r}")
    for name in ("expected_hp_ratio", "expected_xp_ratio"):
        value = getattr(record, name)
        if value is not None and (not _is_number(value) or not 0.0 <= value <= 1.0):
            errors.append(f"{name} must be a number in [0, 1], got {value!r}")

    items = record.expected_items
    if items is not None:
        if len(items) != INV_SLOT_COUNT:
            errors.append(f"expected_items must have {INV_SLOT_COUNT} slots, got {len(items)}")
        for slot, item in enumerate(items):
            if item is None:
                if strict:
                    errors.append(
                        f"expected_items slot {slot} is null; confirmed rows need every slot "
                        "(set items null if any slot is unreadable)"
                    )
            elif not isinstance(item, str) or not item:
                errors.append(f"expected_items slot {slot} must be an item_id or {EMPTY_SLOT!r}")

    levels = record.expected_slot_levels
    if levels is not None:
        if state not in SLOT_LEVEL_VISIBLE_STATES:
            errors.append(
                f"expected_slot_levels must be null for state {state!r} "
                f"(slot level marks are visible only in {sorted(SLOT_LEVEL_VISIBLE_STATES)})"
            )
        if items is None:
            errors.append("expected_slot_levels must be null when expected_items is null")
        if len(levels) != INV_SLOT_COUNT:
            errors.append(f"expected_slot_levels must have {INV_SLOT_COUNT} slots, got {len(levels)}")
        elif items is not None and len(items) == INV_SLOT_COUNT:
            for slot, (item, value) in enumerate(zip(items, levels)):
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int)
                    or not 1 <= value <= MAX_SLOT_LEVEL
                ):
                    errors.append(f"expected_slot_levels slot {slot} must be 1..{MAX_SLOT_LEVEL}, got {value!r}")
                elif item == EMPTY_SLOT and value is not None:
                    errors.append(f"expected_slot_levels slot {slot} must be null for {EMPTY_SLOT}")
                elif value is None and strict and item != EMPTY_SLOT:
                    errors.append(
                        f"expected_slot_levels slot {slot} is null for an item; confirmed rows need "
                        "every item level (set levels null if any mark is unreadable)"
                    )

    choice = record.expected_choice
    if choice is not None:
        if strict and state not in LEVEL_UP_STATES:
            errors.append(
                f"expected_choice must be null for state {state!r} on confirmed rows "
                f"(cards are shown only in {sorted(LEVEL_UP_STATES)}; set choice null)"
            )
        for index, item in enumerate(choice):
            if not isinstance(item, str) or not item:
                errors.append(f"expected_choice[{index}] must be a non-empty item_id")

    if record.roi_name != HUD_ROI_NAME:
        errors.append(f"roi_name must be {HUD_ROI_NAME!r}")
    if state in SCREEN_STATES and record.expected_roi != expected_roi_for_state(state):
        errors.append(
            f"expected_roi must be {expected_roi_for_state(state)!r} for state {state!r}, "
            f"got {record.expected_roi!r}"
        )
    source = record.draft_source
    if not isinstance(source, dict) or not all(
        isinstance(source.get(key), str) for key in ("parser_artifact_hash", "atlas_content_hash")
    ):
        errors.append("draft_source needs string parser_artifact_hash and atlas_content_hash")
    return tuple(errors)


def _hud_truth_path(session_path: os.PathLike[str] | str) -> Path:
    """session 配下の hud_truth.jsonl の path を返す。

    読み込みと書き込みで同じ場所を使うため、path の組み立てをここに集める。
    """
    return Path(session_path) / HUD_TRUTH_FILENAME


def _sorted_unique(records: list[HudTruthRecord]) -> list[HudTruthRecord]:
    """frame_id 昇順に並べ、重複 frame_id があれば ValueError。

    1 frame に正解値は 1 行だけなので、同じ frame_id が 2 行あれば壊れたデータとして扱う。
    並べた後なら隣同士を比べるだけで重複が見つかる。
    """
    ordered = sorted(records, key=lambda record: record.frame_id)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.frame_id == current.frame_id:
            raise ValueError(f"duplicate frame_id in hud truth: {current.frame_id}")
    return ordered


def read_hud_truth(session_path: os.PathLike[str] | str) -> list[HudTruthRecord]:
    """session の hud_truth.jsonl を frame_id 昇順で読む（file が無ければ空）。

    重複 frame_id や壊れた行は ValueError。未知 key は record.extra に保持される。
    """
    path = _hud_truth_path(session_path)
    if not path.is_file():
        return []
    records: list[HudTruthRecord] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(HudTruthRecord.from_wire(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid hud truth at line {line_number}: {exc}") from exc
    return _sorted_unique(records)


def write_hud_truth(session_path: os.PathLike[str] | str, records: list[HudTruthRecord]) -> Path:
    """records を frame_id 昇順で hud_truth.jsonl に原子的に書く。

    重複 frame_id、または confirmed 行の validate_record 違反があれば何も書かずに ValueError。
    書き込みは annotations.jsonl と同じく一時 file → fsync → replace の順で行う。
    """
    ordered = _sorted_unique(list(records))
    for record in ordered:
        if record.confirmed is True:
            errors = validate_record(record)
            if errors:
                raise ValueError(f"frame {record.frame_id}: {'; '.join(errors)}")
    path = _hud_truth_path(session_path)
    temp_path = path.with_name(f".{HUD_TRUTH_FILENAME}.tmp")
    with temp_path.open("wb") as stream:
        for record in ordered:
            line = json.dumps(record.to_wire(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write(line.encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)
    return path


def load_frame_pixels(session_path: os.PathLike[str] | str, frame_record: FrameRecord) -> NDArray[np.uint8]:
    """frame_record の PNG を 1 枚だけ読み、sha256 を照合して BGRA uint8 配列で返す。

    file の bytes が frame_record.object_sha256 と一致しなければ ValueError。
    戻り値は HudParser.parse の入力と同じ (1080, 1920, 4) の BGRA。
    """
    path = _resolve_relative(Path(session_path), frame_record.object_path, "object_path")
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != frame_record.object_sha256:
        raise ValueError(f"object hash mismatch: {frame_record.object_path}")
    pixels = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError(f"object is not an 8-bit RGBA PNG: {frame_record.object_path}")
    return pixels


def draft_from_parser(
    frame_bgra: NDArray[np.uint8],
    parser: HudParser,
    *,
    session_id: str,
    frame_index: int,
    captured_monotonic_ns: int,
    annotator_id: str,
    atlas_content_hash: str,
) -> HudTruthRecord:
    """parser の HudStateV1 を写した未確定の下書き record を返す。

    inventory に読めない slot（None）が 1 つでもあれば expected_items は配列全体を null にする。
    段階値の信頼度が .5 以上のパネルでは expected_slot_levels に写す（items が null なら null）。
    card の item_id（None を除く）を上から expected_choice に入れ、無ければ null。
    """
    state = parser.parse(
        frame_bgra,
        session_id=session_id,
        frame_index=frame_index,
        captured_monotonic_ns=captured_monotonic_ns,
    )
    inventory = tuple(state.inventory)
    choice = tuple(card.item_id for card in state.cards if card.item_id is not None)
    return HudTruthRecord(
        schema_version=HUD_TRUTH_SCHEMA_VERSION,
        session_id=session_id,
        frame_id=frame_index,
        annotator_id=annotator_id,
        confirmed=False,
        expected_screen_state=state.screen_state,
        expected_timer_seconds=state.timer_seconds,
        expected_level=state.level,
        expected_hp_ratio=state.hp_ratio,
        expected_xp_ratio=state.xp_ratio,
        expected_items=None if any(item is None for item in inventory) else inventory,
        expected_slot_levels=(state.inventory_levels if state.inventory_levels_confidence >= .5
                              and state.screen_state in SLOT_LEVEL_VISIBLE_STATES
                              and None not in inventory else None),
        expected_choice=choice or None,
        roi_name=HUD_ROI_NAME,
        expected_roi=expected_roi_for_state(state.screen_state),
        confirmed_at=None,
        draft_source={
            "parser_artifact_hash": state.parser_artifact_hash,
            "atlas_content_hash": atlas_content_hash,
        },
    )


def with_state(record: HudTruthRecord, state: str) -> HudTruthRecord:
    """画面状態を変え、expected_roi と expected_slot_levels を状態に合わせて直した record を返す。

    HUD 帯 ROI は状態から自動で決まる。段階マークが出ない状態では slot level を null にする。
    """
    levels = record.expected_slot_levels if state in SLOT_LEVEL_VISIBLE_STATES else None
    return replace(
        record,
        expected_screen_state=state,
        expected_roi=expected_roi_for_state(state),
        expected_slot_levels=levels,
    )


def _parse_null(token: str) -> bool:
    """null を表す token（null または -）なら True。

    行コマンドで「値が読めない」を入れるときの書き方を 1 か所で決める。
    """
    return token in {"null", "-"}


def _parse_slot(token: str) -> int:
    """slot 番号（0..11）を int で返す。範囲外は ValueError。

    0..5 が武器、6..11 がパッシブの slot。数字でない token も ValueError になる。
    """
    slot = int(token)
    if not 0 <= slot < INV_SLOT_COUNT:
        raise ValueError(f"slot must be 0..{INV_SLOT_COUNT - 1}, got {slot}")
    return slot


def _parse_level_token(token: str) -> int | None:
    """slot level の token（数字か -）を int か None に変換する。

    空き slot や読めない slot は - で入れる。範囲の検査は validate_record に任せる。
    """
    return None if _parse_null(token) else int(token)


def _apply_set(record: HudTruthRecord, args: list[str]) -> HudTruthRecord:
    """set サブコマンドを適用した record を返す（書式の誤りは ValueError）。

    args は「set」の後ろの token 列（先頭が field 名）。元の record は変えず、新しい record を返す。
    値として正しいかどうかは呼び出し側の apply_command が validate_record で確かめる。
    """
    if not args:
        raise ValueError("set needs a field name")
    name, rest = args[0], args[1:]

    def one() -> str:
        """引数が 1 個であることを確かめて返す。

        set timer 1 2 のように値が多すぎる・足りない入力を ValueError で断る。
        """
        if len(rest) != 1:
            raise ValueError(f"set {name} needs exactly one value")
        return rest[0]

    if name == "timer":
        value = one()
        return replace(record, expected_timer_seconds=None if _parse_null(value) else float(value))
    if name == "level":
        value = one()
        return replace(record, expected_level=None if _parse_null(value) else int(value))
    if name in {"hp", "xp"}:
        value = one()
        return replace(record, **{f"expected_{name}_ratio": None if _parse_null(value) else float(value)})
    if name == "state":
        return with_state(record, one())
    if name == "item":
        if len(rest) != 2:
            raise ValueError("usage: set item SLOT item_id|empty_slot")
        slot = _parse_slot(rest[0])
        items = list(record.expected_items or (None,) * INV_SLOT_COUNT)
        items[slot] = rest[1]
        return replace(record, expected_items=tuple(items))
    if name == "items":
        if one() != "null":
            raise ValueError("usage: set items null（slot ごとは set item SLOT ID）")
        return replace(record, expected_items=None, expected_slot_levels=None)
    if name == "choice":
        value = one()
        if value == "null":
            return replace(record, expected_choice=None)
        return replace(record, expected_choice=tuple(value.split("|")))
    if name == "levels":
        if rest == ["null"]:
            return replace(record, expected_slot_levels=None)
        if len(rest) != INV_SLOT_COUNT:
            raise ValueError(f"set levels needs {INV_SLOT_COUNT} values (use - for null), got {len(rest)}")
        return replace(record, expected_slot_levels=tuple(_parse_level_token(token) for token in rest))
    if name == "level-of":
        if len(rest) != 2:
            raise ValueError("usage: set level-of SLOT N")
        if record.expected_slot_levels is None:
            raise ValueError("expected_slot_levels is null; まず set levels で全体を入れる")
        slot = _parse_slot(rest[0])
        levels = list(record.expected_slot_levels)
        levels[slot] = _parse_level_token(rest[1])
        return replace(record, expected_slot_levels=tuple(levels))
    raise ValueError(f"unknown field: {name}")


def apply_command(
    record: HudTruthRecord, line: str, *, now: str | None = None
) -> tuple[HudTruthRecord, str | None]:
    """行コマンドを 1 つ適用し (新しい record, 拒否理由) を返す純関数。

    受理したら (更新後の record, None)、拒否したら (元の record, 理由) を返す。
    結果が validate_record に違反するコマンドも拒否する（規則は validate_record だけに置く）。
    ok は confirmed=True にして confirmed_at に now（省略時は現在の UTC 時刻）を入れる。
    """
    tokens = line.split()
    if not tokens:
        return record, "empty command"
    try:
        if tokens == ["ok"]:
            stamp = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
            updated = replace(record, confirmed=True, confirmed_at=stamp)
        elif tokens[0] == "set":
            updated = _apply_set(record, tokens[1:])
        else:
            return record, f"unknown command: {line.strip()}"
    except ValueError as exc:
        return record, str(exc)
    errors = validate_record(updated)
    if errors:
        return record, "; ".join(errors)
    return updated, None

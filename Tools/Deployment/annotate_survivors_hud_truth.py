"""capture session の frame に HUD の正解値（hud_truth.v1）を付ける対話 CLI。

development parser で各 frame の下書きを作り、人が stdin の行コマンドで直して確定する。
結果は session 配下の hud_truth.jsonl に保存し、既存の bbox annotation や label JSON は読むだけ。
画像は開かないので、表示された PNG の path を X-AnyLabeling などの別窓で見ながら作業する。
使い方は docs/deployment/capture_annotation_manual.md の「2-7. HUD の正解値を付ける」を参照。
"""

from __future__ import annotations

import argparse
import sys
from bisect import bisect_left
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from survivors.annotation_labels import iter_frame_files, read_label_file
from survivors.capture_dataset import DatasetWriter, FrameRecord
from survivors.vision.hud_parser import HudParser
from survivors.vision.hud_truth import (
    HudTruthRecord,
    apply_command,
    draft_from_parser,
    load_frame_pixels,
    read_hud_truth,
    validate_record,
    with_state,
    write_hud_truth,
)
from survivors.vision.icon_matcher import IconMatcher

# 下書きの前に遡って parse する frame 数（04-20 の段階マーク証拠が入るまでの暫定値）
BACKTRACK_FRAMES = 30
# 遡り parse の上限 frame 数
MAX_WARMUP_FRAMES = 400
# この frame 数を確定・skip するたびに途中保存する
SAVE_EVERY = 20
# atlas を使わないときに draft_source へ入れる値
NO_ATLAS_HASH = "none"
# ok-range で下書きと引き継ぎ値を同じとみなす許容差
# ponytail: 固定の許容差。timer が 1 秒以上進む範囲では毎回止まるので、そのときは 1 frame ずつ ok する
TIMER_TOLERANCE_S = 1.0
RATIO_TOLERANCE = 0.05
# 毎 frame の先頭に出す注意
STATE_NOTICE = (
    "注意: expected_screen_state は必ず画面を目視で確認すること"
    "（04-20 merge 前は level-up 画面の下書きが gameplay になる）"
)
# ok-range で比較する離散値の field
_DISCRETE_FIELDS = (
    "expected_screen_state", "expected_level", "expected_items",
    "expected_slot_levels", "expected_choice",
)
# ok-range で許容差つきで比較する連続値の field
_CONTINUOUS_FIELDS = (
    ("expected_timer_seconds", TIMER_TOLERANCE_S),
    ("expected_hp_ratio", RATIO_TOLERANCE),
    ("expected_xp_ratio", RATIO_TOLERANCE),
)


def _parser() -> argparse.ArgumentParser:
    """CLI の引数 parser を返す。

    対象 frame は --frame-ids か --work-root の checked label で決める。
    """
    parser = argparse.ArgumentParser(description="capture session の frame に HUD の正解値を付ける。")
    parser.add_argument("--store-root", type=Path, required=True, help="capture_sessions を含む store root")
    parser.add_argument("--session-id", required=True, help="対象 session（例: session-0004）")
    parser.add_argument("--annotator-id", required=True, help="確認者の ID")
    parser.add_argument("--work-root", type=Path, help="X-AnyLabeling の label がある work root（読むだけ）")
    parser.add_argument("--frame-ids", type=int, nargs="+", help="対象 frame_id（省略時は checked label の frame）")
    parser.add_argument("--resume", action="store_true", help="既存の hud_truth.jsonl を読み、確定済み frame を飛ばす")
    parser.add_argument("--atlas", type=Path, help="development atlas（省略時は icon 照合なし）")
    parser.add_argument("--from-labels", action="store_true",
                        help="label の card／death_result bbox を screen state の下書きに使う")
    return parser


def _checked_frame_ids(work_root: Path, session_id: str) -> list[int]:
    """work root の session で checked: true の label がある frame_id を昇順で返す。"""
    return [
        frame_id
        for frame_id, _png, label in iter_frame_files(work_root / session_id)
        if label is not None and read_label_file(label)[1]
    ]


def _label_state(label_path: Path, parser_state: str) -> tuple[str | None, str | None]:
    """label の bbox から screen state の下書きと表示用の注記を返す（使えなければ None）。

    card bbox が 1 つでもあれば level_up_items（parser の判定より優先）。
    death_result bbox があれば parser の death／result 判定を残し、それ以外なら death にする。
    death と result の区別は人が set state で確定する。
    """
    if not label_path.is_file():
        return None, None
    labels = {box.label for box in read_label_file(label_path)[0]}
    if "card" in labels:
        return "level_up_items", "from-labels: card bbox あり → level_up_items"
    if "death_result" in labels:
        state = parser_state if parser_state in {"death", "result"} else "death"
        return state, "from-labels: death_result bbox あり → set state death|result で確定すること"
    return None, None


def _format_slots(values: tuple | None) -> str:
    """12 slot の値を武器 6・パッシブ 6 に分けた 1 行にする（null は -）。"""
    if values is None:
        return "null"
    cells = ["-" if value is None else str(value) for value in values]
    return f"w: {' '.join(cells[:6])}  p: {' '.join(cells[6:])}"


def _carry_conflicts(draft: HudTruthRecord, carry: HudTruthRecord) -> list[str]:
    """下書きで読めている field のうち、引き継ぎ値と食い違うものを列挙する。"""
    conflicts: list[str] = []
    for name in _DISCRETE_FIELDS:
        value = getattr(draft, name)
        if value is not None and value != getattr(carry, name):
            conflicts.append(f"{name}: 下書き={value!r} 引き継ぎ={getattr(carry, name)!r}")
    for name, tolerance in _CONTINUOUS_FIELDS:
        value, carried = getattr(draft, name), getattr(carry, name)
        if value is not None and (carried is None or abs(value - carried) > tolerance):
            conflicts.append(f"{name}: 下書き={value!r} 引き継ぎ={carried!r}")
    return conflicts


class _Drafter:
    """遡り parse を挟みながら frame ごとの下書きを作る。

    下書きを作る frame の前の連続 frame を同じ parser で parse し、時間方向の状態をそろえる。
    前回 parse した frame と連続しない窓では reset_temporal_state() を呼んでから始める。
    """

    def __init__(self, args: argparse.Namespace, session_path: Path, frame_records: tuple[FrameRecord, ...]) -> None:
        """parser・atlas・frame 一覧を用意する。"""
        matcher = IconMatcher.load_development(args.atlas) if args.atlas else None
        self.atlas_hash = matcher.manifest.atlas_content_hash if matcher else NO_ATLAS_HASH
        self.parser = HudParser(parser_artifact_hash=f"development:{self.atlas_hash}", icon_matcher=matcher)
        self.parser.reset_temporal_state()
        self.args = args
        self.session_path = session_path
        self.records = sorted(frame_records, key=lambda record: record.frame_id)
        self.frame_ids = [record.frame_id for record in self.records]
        self.last_parsed: int | None = None
        self.hints: dict[int, str] = {}

    def _parse(self, position: int) -> None:
        """position の frame を parse して結果を捨てる（時間方向の状態だけを進める）。"""
        record = self.records[position]
        self.parser.parse(
            load_frame_pixels(self.session_path, record),
            session_id=self.args.session_id,
            frame_index=record.frame_id,
            captured_monotonic_ns=record.captured_monotonic_ns,
        )

    def draft(self, frame_id: int) -> HudTruthRecord:
        """frame_id の下書きを返す（--from-labels なら label の state を優先する）。"""
        position = bisect_left(self.frame_ids, frame_id)
        start = max(bisect_left(self.frame_ids, frame_id - BACKTRACK_FRAMES), position - MAX_WARMUP_FRAMES)
        if self.last_parsed is not None and start <= self.last_parsed + 1 <= position:
            start = self.last_parsed + 1
        else:
            self.parser.reset_temporal_state()
        for warmup in range(start, position):
            self._parse(warmup)
        record = self.records[position]
        draft = draft_from_parser(
            load_frame_pixels(self.session_path, record),
            self.parser,
            session_id=self.args.session_id,
            frame_index=frame_id,
            captured_monotonic_ns=record.captured_monotonic_ns,
            annotator_id=self.args.annotator_id,
            atlas_content_hash=self.atlas_hash,
        )
        self.last_parsed = position
        if self.args.from_labels:
            label_path = self.args.work_root / self.args.session_id / f"{frame_id:08d}.json"
            state, hint = _label_state(label_path, draft.expected_screen_state)
            if state is not None:
                draft = with_state(draft, state)
                self.hints[frame_id] = hint
        return draft

    def png_path(self, frame_id: int) -> Path:
        """frame_id の PNG の絶対 path を返す。"""
        record = self.records[bisect_left(self.frame_ids, frame_id)]
        return (self.session_path / record.object_path).resolve()


def _show(out: TextIO, drafter: _Drafter, record: HudTruthRecord, index: int, total: int) -> None:
    """1 frame 分の下書きを text で表示する。"""
    print(f"=== frame {record.frame_id:08d} ({index + 1}/{total}) {record.session_id} ===", file=out)
    print(STATE_NOTICE, file=out)
    print(f"png: {drafter.png_path(record.frame_id)}", file=out)
    if record.frame_id in drafter.hints:
        print(drafter.hints[record.frame_id], file=out)
    print(f"state: {record.expected_screen_state}  roi: {record.expected_roi}", file=out)
    print(
        f"timer: {record.expected_timer_seconds}  level: {record.expected_level}  "
        f"hp: {record.expected_hp_ratio}  xp: {record.expected_xp_ratio}",
        file=out,
    )
    print(f"items:  {_format_slots(record.expected_items)}", file=out)
    print(f"levels: {_format_slots(record.expected_slot_levels)}", file=out)
    choice = "null" if record.expected_choice is None else "|".join(record.expected_choice)
    print(f"choice: {choice}  confirmed: {record.confirmed}", file=out)


def run(args: argparse.Namespace, stdin: TextIO, out: TextIO) -> int:
    """対象 frame を順に表示し、行コマンドで確定した値を hud_truth.jsonl に保存する。

    quit・stdin の終端・例外のどれで終わっても、それまでの確定値を保存する。
    """
    manifest = DatasetWriter.restore(args.store_root, args.session_id, metadata_only=True)
    session_path = manifest.session_path
    existing = read_hud_truth(session_path)
    if existing and not args.resume:
        raise ValueError(f"{session_path / 'hud_truth.jsonl'} がすでにあります。続きは --resume を付ける")
    records = {record.frame_id: record for record in existing}
    known = {record.frame_id for record in manifest.frame_records}
    requested = sorted(set(args.frame_ids)) if args.frame_ids else _checked_frame_ids(args.work_root, args.session_id)
    targets: list[int] = []
    for frame_id in requested:
        if frame_id not in known:
            print(f"warning: frame {frame_id} は session に無いので飛ばす", file=out)
        elif not (frame_id in records and records[frame_id].confirmed):
            targets.append(frame_id)
    drafter = _Drafter(args, session_path, manifest.frame_records)

    position = 0
    current: HudTruthRecord | None = None
    base: HudTruthRecord | None = None
    history: list[tuple[int, HudTruthRecord | None, HudTruthRecord | None, dict]] = []
    since_save = 0

    def save() -> None:
        """現在の records を保存する。"""
        if records or existing:
            path = write_hud_truth(session_path, list(records.values()))
            print(f"saved: {path} ({len(records)} records)", file=out)

    def advance(count: int) -> None:
        """count frame 進んだことを数え、SAVE_EVERY ごとに保存する。"""
        nonlocal since_save
        since_save += count
        if since_save >= SAVE_EVERY:
            save()
            since_save = 0

    try:
        while position < len(targets):
            frame_id = targets[position]
            if current is None:
                base = drafter.draft(frame_id)
                current = records.get(frame_id, base)
                _show(out, drafter, current, position, len(targets))
            line = stdin.readline()
            if not line:
                break
            command = line.strip()
            tokens = command.split()
            if not tokens:
                continue
            if command == "quit":
                break
            if command == "undo":
                if not history:
                    print("拒否: 取り消せる操作がありません", file=out)
                    continue
                position, current, base, records = history.pop()
                _show(out, drafter, current, position, len(targets))
                continue
            snapshot = (position, current, base, dict(records))
            if command == "skip":
                history.append(snapshot)
                if current != base:
                    records[frame_id] = current  # 直した途中の値は未確定のまま残す
                position, current = position + 1, None
                advance(1)
                continue
            if tokens[0] == "ok-range":
                try:
                    first, last = (int(token) for token in tokens[1:])
                except ValueError:
                    print("拒否: usage: ok-range A B", file=out)
                    continue
                if not first <= frame_id <= last:
                    print(f"拒否: 表示中の frame {frame_id} が {first}..{last} に入っていません", file=out)
                    continue
                carries = [r for r in records.values() if r.confirmed and r.frame_id < frame_id]
                if not carries:
                    print("拒否: 引き継ぐ確認値（この frame より前の確定行）がありません", file=out)
                    continue
                carry = max(carries, key=lambda r: r.frame_id)
                draft, done = base, 0
                while True:
                    conflicts = _carry_conflicts(draft, carry)
                    if conflicts:
                        print(f"停止: frame {draft.frame_id} の下書きが引き継ぎ値と違います", file=out)
                        for conflict in conflicts:
                            print(f"  {conflict}", file=out)
                        break
                    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    confirmed = replace(
                        carry, frame_id=draft.frame_id, annotator_id=args.annotator_id,
                        draft_source=draft.draft_source, confirmed_at=stamp, extra={},
                    )
                    errors = validate_record(confirmed)
                    if errors:
                        print(f"停止: frame {draft.frame_id}: {'; '.join(errors)}", file=out)
                        break
                    records[draft.frame_id] = carry = confirmed
                    position, done = position + 1, done + 1
                    if position >= len(targets) or targets[position] > last:
                        draft = None
                        break
                    draft = drafter.draft(targets[position])
                if done:
                    history.append(snapshot)
                    print(f"ok-range: {done} frame を確定", file=out)
                    advance(done)
                    base = draft
                    current = None if draft is None else records.get(draft.frame_id, draft)
                    if current is not None:
                        _show(out, drafter, current, position, len(targets))
                continue
            updated, error = apply_command(current, command)
            if error is not None:
                print(f"拒否: {error}", file=out)
                continue
            history.append(snapshot)
            if updated.confirmed:
                records[frame_id] = updated
                position, current = position + 1, None
                advance(1)
            else:
                current = updated
                _show(out, drafter, current, position, len(targets))
    finally:
        if current is not None and not current.confirmed and current != base:
            records[current.frame_id] = current
        save()
    return 0


def main(argv: list[str] | None = None, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    """main(argv) -> int として CLI を実行する。

    --work-root と --frame-ids の両方が無ければ引数エラー。入力不正は stderr に理由を出して 1 を返す。
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.frame_ids and args.work_root is None:
        parser.error("--work-root か --frame-ids のどちらかが必要です")
    if args.from_labels and args.work_root is None:
        parser.error("--from-labels には --work-root が必要です")
    try:
        return run(args, stdin or sys.stdin, stdout or sys.stdout)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

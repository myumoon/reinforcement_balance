"""確認済み Survivors ラベルの付け替えを propose → 確認 → apply の2段階で行う CLI。

旧ラベル（既定 hazard_projectile / hazard_area）の shape を集めて対応表と一覧画像を作り（propose）、
ユーザーが対応表を確認・修正した後で、その内容どおりに JSON を書き換える（apply）。
apply は全行を書き込み前に検証し、1行でも不一致があれば何も書かずに終了コード 1 を返す。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from survivors.annotation_labels import (
    ALL_CLASSES,
    iter_frame_files,
    read_label_file,
)
from survivors.prelabel_detector import DEFAULT_CONFIG_PATH, load_config

DEFAULT_OLD_LABELS = ("hazard_projectile", "hazard_area")
DEFAULT_AURA_RADIUS_PX = 40.0
REVIEW = "review"
DELETE = "delete"
_ROW_KEYS = {"id", "file", "shape_index", "old_label", "bbox", "proposed_label", "reason"}
_TILE = 160
_SHEET_COLUMNS = 8


def _parser() -> argparse.ArgumentParser:
    """propose / apply サブコマンドの引数 parser を返す。

    propose は対応表と一覧画像を作るだけで、apply だけがラベル JSON を書き換える。
    """
    parser = argparse.ArgumentParser(description="確認済み Survivors ラベルを対応表で付け替える。")
    commands = parser.add_subparsers(dest="command", required=True)

    propose = commands.add_parser("propose", help="対応表 JSON と番号付き一覧画像を作る（データは書き換えない）")
    propose.add_argument("--work-root", required=True, type=Path)
    propose.add_argument("--output", required=True, type=Path, help="対応表 JSON の出力先（既存なら上書きしない）")
    propose.add_argument("--sheet", required=True, type=Path, help="番号付き一覧画像（PNG）の出力先")
    propose.add_argument("--old-labels", nargs="+", default=list(DEFAULT_OLD_LABELS))
    propose.add_argument(
        "--aura-radius-px", type=float, default=DEFAULT_AURA_RADIUS_PX,
        help="hazard_area の中心が player_anchor 中心からこの距離以内なら weapon_aura を提案する",
    )
    propose.add_argument("--prelabel-config", type=Path, default=DEFAULT_CONFIG_PATH,
                         help="JSON に player_anchor が無いときに使う固定矩形の設定")

    apply = commands.add_parser("apply", help="確認済みの対応表どおりにラベル JSON を書き換える")
    apply.add_argument("--work-root", required=True, type=Path)
    apply.add_argument("--mapping", required=True, type=Path)
    apply.add_argument("--backup-dir", type=Path, default=None,
                       help="既定は <work-root>/_relabel_backup/<YYYYmmdd-HHMMSS>")
    return parser


def _read_payload(path: Path) -> dict:
    """ラベル JSON を共有 reader で検証したうえで dict として読む。

    壊れた shape は read_label_file が ValueError にするので、ここでは補正しない。
    dict のまま持つことで checked・circle・flags などのキーを書き戻し時に保持できる。
    """
    read_label_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _shape_bbox(path: Path, index: int, shape: dict) -> list[float]:
    """shape の外接矩形 [left, top, right, bottom] を read_label_file と同じ規則で返す。

    rectangle は点の最小・最大、circle は中心点と円周上の点から求めた円の外接矩形。
    それ以外の shape_type は付け替え対象にできないので ValueError にする。
    """
    points = shape["points"]
    if shape["shape_type"] == "circle":
        (cx, cy), (ex, ey) = points
        radius = math.hypot(ex - cx, ey - cy)
        return [cx - radius, cy - radius, cx + radius, cy + radius]
    if shape["shape_type"] == "rectangle":
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
    raise ValueError(f"{path}: shape {index} has unsupported shape_type {shape['shape_type']!r} for relabel")


def _center(bbox: list[float]) -> tuple[float, float]:
    """外接矩形の中心座標を返す。

    player_anchor と hazard_area の距離計算に使う。
    """
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _session_dirs(work_root: Path) -> list[Path]:
    """work-root 直下の session ディレクトリを名前順に返す。

    `_relabel_backup` のように `_` / `.` で始まる作業用ディレクトリは除く。
    """
    return sorted(
        path for path in work_root.iterdir()
        if path.is_dir() and not path.name.startswith(("_", "."))
    )


def _propose_label(old_label: str, bbox: list[float], anchor: tuple[float, float], radius: float) -> tuple[str, str]:
    """1つの shape に対する提案ラベルと根拠を返す。

    hazard_projectile は weapon_projectile、プレイヤー中心付近の hazard_area は weapon_aura、
    それ以外は人が決める review にする。
    """
    if old_label == "hazard_projectile":
        return "weapon_projectile", "hazard_projectile は武器の弾として weapon_projectile を提案"
    if old_label == "hazard_area":
        cx, cy = _center(bbox)
        distance = math.hypot(cx - anchor[0], cy - anchor[1])
        if distance <= radius:
            return "weapon_aura", f"player_anchor 中心から {distance:.1f}px（閾値 {radius:g}px 以内）"
        return REVIEW, f"player_anchor 中心から {distance:.1f}px（閾値 {radius:g}px 超え）。人が判断する"
    return REVIEW, f"{old_label} は自動提案の対象外。人が判断する"


def _tile(image: np.ndarray | None, bbox: list[float], caption: str) -> np.ndarray:
    """bbox を切り出して正方形タイルに収め、番号と旧ラベルを描く。

    画像が読めない場合は黒地に caption だけを描く。
    """
    tile = np.zeros((_TILE, _TILE, 3), np.uint8)
    if image is not None:
        height, width = image.shape[:2]
        left, top = max(0, int(bbox[0])), max(0, int(bbox[1]))
        right, bottom = min(width, int(math.ceil(bbox[2]))), min(height, int(math.ceil(bbox[3])))
        if right > left and bottom > top:
            crop = image[top:bottom, left:right]
            scale = min((_TILE - 20) / crop.shape[1], (_TILE - 20) / crop.shape[0])
            size = (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale)))
            resized = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
            tile[20:20 + size[1], :size[0]] = resized
    cv2.putText(tile, caption, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return tile


def _write_sheet(path: Path, tiles: list[np.ndarray]) -> None:
    """タイルを横 _SHEET_COLUMNS 枚のグリッドに並べて PNG に保存する。

    足りないマスは黒で埋める。
    """
    rows = math.ceil(len(tiles) / _SHEET_COLUMNS)
    blank = np.zeros((_TILE, _TILE, 3), np.uint8)
    padded = tiles + [blank] * (rows * _SHEET_COLUMNS - len(tiles))
    sheet = np.vstack([np.hstack(padded[r * _SHEET_COLUMNS:(r + 1) * _SHEET_COLUMNS]) for r in range(rows)])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise OSError(f"{path}: cannot write sheet image")


def _to_bgr(image: np.ndarray | None) -> np.ndarray | None:
    """一覧画像用に BGR 3チャンネルへそろえる。

    キャプチャ PNG は BGRA のことがあるため。
    """
    if image is None or image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)


def propose(args: argparse.Namespace) -> int:
    """旧ラベルの shape を集め、対応表 JSON と一覧画像を書く。

    ラベル JSON は読むだけで書き換えない。対応表の id が一覧画像の番号になる。
    """
    if args.output.exists():
        raise ValueError(f"{args.output}: mapping already exists (確認済みの対応表を上書きしない)")
    anchor_boxes = [box for box in load_config(args.prelabel_config).fixed_boxes if box.label == "player_anchor"]
    if not anchor_boxes:
        raise ValueError(f"{args.prelabel_config}: fixed_boxes has no player_anchor")
    default_anchor = _center([anchor_boxes[0].left, anchor_boxes[0].top, anchor_boxes[0].right, anchor_boxes[0].bottom])
    old_labels = set(args.old_labels)
    unknown = sorted(old_labels - set(ALL_CLASSES))
    if unknown:
        raise ValueError(f"--old-labels has unknown annotation labels: {unknown}")

    rows: list[dict] = []
    tiles: list[np.ndarray] = []
    for session_dir in _session_dirs(args.work_root):
        for _, image_path, label_path in iter_frame_files(session_dir):
            if label_path is None:
                continue
            payload = _read_payload(label_path)
            shapes = payload["shapes"]
            anchor = next(
                (_center(_shape_bbox(label_path, i, s)) for i, s in enumerate(shapes) if s["label"] == "player_anchor"),
                default_anchor,
            )
            image = None
            for index, shape in enumerate(shapes):
                if shape["label"] not in old_labels:
                    continue
                if image is None and image_path is not None:
                    image = _to_bgr(cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED))
                bbox = _shape_bbox(label_path, index, shape)
                label, reason = _propose_label(shape["label"], bbox, anchor, args.aura_radius_px)
                row_id = len(rows) + 1
                rows.append({
                    "id": row_id,
                    "file": label_path.relative_to(args.work_root).as_posix(),
                    "shape_index": index,
                    "old_label": shape["label"],
                    "bbox": bbox,
                    "proposed_label": label,
                    "reason": reason,
                })
                tiles.append(_tile(image, bbox, f"{row_id} {shape['label']}"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    if tiles:
        _write_sheet(args.sheet, tiles)
    else:
        print("対象 shape が無いため一覧画像は作らない")
    print(f"対象 shape 数: {len(rows)}")
    for label in sorted({row["proposed_label"] for row in rows}):
        print(f"  {label}: {sum(row['proposed_label'] == label for row in rows)}")
    return 0


def _resolve_label_file(work_root: Path, relative: object) -> Path:
    """対応表の file を work-root 内の八桁ラベル JSON パスへ解決する。

    絶対パスや `..` で work-root の外を指す値は拒否する。
    """
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"mapping file must be a non-empty string: {relative!r}")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.suffix != ".json":
        raise ValueError(f"mapping file must be a relative .json path inside work-root: {relative!r}")
    return work_root / candidate


def _load_plan(work_root: Path, mapping_path: Path) -> dict[Path, tuple[dict, dict[int, str]]]:
    """対応表を全行検証し、書き換えが必要なファイルごとの変更内容を返す。

    review 残存・未知の提案ラベル・重複行・旧ラベルや bbox の不一致は ValueError。
    戻り値は {ラベル JSON: (payload, {shape_index: 新ラベル or delete})} で、変更なしの行は含めない。
    """
    rows = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{mapping_path}: mapping must be a JSON list")
    payloads: dict[Path, dict] = {}
    changes: dict[Path, dict[int, str]] = {}
    seen: set[tuple[Path, int]] = set()
    for number, row in enumerate(rows, start=1):
        where = f"{mapping_path}: row {number}"
        if not isinstance(row, dict) or set(row) != _ROW_KEYS:
            raise ValueError(f"{where} must contain exactly {sorted(_ROW_KEYS)}")
        proposed = row["proposed_label"]
        if proposed == REVIEW:
            raise ValueError(f"{where} (id={row['id']}) is still 'review'; decide a label before apply")
        if proposed != DELETE and proposed not in ALL_CLASSES:
            raise ValueError(f"{where} has unknown proposed_label {proposed!r}")
        if row["old_label"] not in ALL_CLASSES:
            raise ValueError(f"{where} has unknown old_label {row['old_label']!r}")
        index = row["shape_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f"{where} shape_index must be a non-negative integer")
        path = _resolve_label_file(work_root, row["file"])
        if (path, index) in seen:
            raise ValueError(f"{where} duplicates {row['file']} shape {index}")
        seen.add((path, index))
        if path not in payloads:
            payloads[path] = _read_payload(path)
        shapes = payloads[path]["shapes"]
        if index >= len(shapes):
            raise ValueError(f"{where}: {row['file']} has no shape {index}")
        shape = shapes[index]
        if shape["label"] != row["old_label"] or _shape_bbox(path, index, shape) != row["bbox"]:
            raise ValueError(f"{where}: {row['file']} shape {index} changed since propose (label/bbox mismatch)")
        if proposed != row["old_label"]:
            changes.setdefault(path, {})[index] = proposed
    return {path: (payloads[path], edits) for path, edits in changes.items()}


def _write_json_atomic(path: Path, payload: dict) -> None:
    """dict を一時ファイルへ書いてから os.replace で置き換える。

    途中で失敗しても元ファイルは壊れない。
    """
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def apply(args: argparse.Namespace) -> int:
    """検証済みの対応表どおりにラベル変更・shape 削除を行う。

    全行の検証とバックアップ先の確認が終わってからバックアップを取り、その後で書き換える。
    書き換え中に失敗した場合は、書き換え済みのファイルをバックアップから戻す。
    """
    work_root = args.work_root
    plan = _load_plan(work_root, args.mapping)
    backup_root = args.backup_dir or work_root / "_relabel_backup" / datetime.now().strftime("%Y%m%d-%H%M%S")
    backups = {path: backup_root / path.relative_to(work_root) for path in plan}
    existing = [str(target) for target in backups.values() if target.exists()]
    if existing:
        raise ValueError(f"backup already exists: {existing[0]}")

    for path, target in backups.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    written: list[Path] = []
    try:
        for path, (payload, edits) in plan.items():
            payload["shapes"] = [
                {**shape, "label": edits[index]} if index in edits else shape
                for index, shape in enumerate(payload["shapes"])
                if edits.get(index) != DELETE
            ]
            _write_json_atomic(path, payload)
            written.append(path)
    except BaseException:
        for path in written:
            shutil.copy2(backups[path], path)
        raise

    changed = sum(len(edits) for _, edits in plan.values())
    print(f"書き換えたファイル数: {len(plan)}")
    print(f"変更した shape 数: {changed}")
    print(f"バックアップ: {backup_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """main(argv) -> int として propose / apply を実行する。

    入力不正や不一致は stderr に理由を出して終了コード 1 を返す。
    """
    args = _parser().parse_args(argv)
    try:
        return propose(args) if args.command == "propose" else apply(args)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

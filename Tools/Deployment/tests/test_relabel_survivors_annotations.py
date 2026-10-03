"""ラベル付け替え CLI（propose / apply）の契約を検証する。

合成した work-root だけを使い、propose がデータを書き換えないこと、
apply がバックアップ・キー保持・fail-closed（不一致なら何も書かない）を守ることを確認する。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from relabel_survivors_annotations import main


def _shape(label: str, points: list, shape_type: str = "rectangle") -> dict:
    """X-AnyLabeling が保存する全キーを持つ shape を返す。

    flags / description などが apply 後も残ることを確かめるため、値を空にしない。
    """
    return {
        "label": label,
        "score": None,
        "points": points,
        "group_id": None,
        "description": f"note-{label}",
        "difficult": False,
        "shape_type": shape_type,
        "flags": {"keep": True},
        "attributes": {},
    }


def _write_frame(session_dir: Path, frame_id: int, shapes: list[dict]) -> Path:
    """確認済みラベル JSON と 1920x1080 の PNG を1フレーム分書き出す。

    トップレベルにも flags を入れ、書き換え後の保持を検証できるようにする。
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(session_dir / f"{frame_id:08d}.png"), np.full((1080, 1920, 3), 80, np.uint8))
    path = session_dir / f"{frame_id:08d}.json"
    path.write_text(
        json.dumps({
            "version": "4.0.6",
            "flags": {"frame": "x"},
            "shapes": shapes,
            "imagePath": f"{frame_id:08d}.png",
            "imageData": None,
            "imageHeight": 1080,
            "imageWidth": 1920,
            "checked": True,
        }),
        encoding="utf-8",
    )
    return path


def _work_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    """2 session・2 フレームの合成 work-root を作る。

    frame 1 は JSON 内の player_anchor を基準にし、frame 2 は設定の固定矩形（中心 954, 503）を基準にする。
    """
    root = tmp_path / "annotation_work"
    first = _write_frame(root / "session-a", 1, [
        _shape("player_anchor", [[100, 100], [140, 140]]),            # 中心 (120, 120)
        _shape("hazard_projectile", [[10, 10], [30, 20]]),
        _shape("hazard_area", [[120, 120], [150, 120]], "circle"),     # 中心 (120, 120) → aura
        _shape("gem_blue", [[1, 1], [5, 5]]),
    ])
    second = _write_frame(root / "session-b", 2, [
        _shape("hazard_area", [[1500, 800], [1600, 900]]),             # 固定矩形中心から遠い → review
        _shape("hazard_area", [[934, 483], [974, 523]]),               # 中心 (954, 503) → aura
    ])
    return root, first, second


def _propose(root: Path, tmp_path: Path) -> list[dict]:
    """propose を実行して対応表を読み込む。

    一覧画像が書かれたことも併せて確認する。
    """
    mapping = tmp_path / "mapping.json"
    sheet = tmp_path / "sheet.png"
    assert main(["propose", "--work-root", str(root), "--output", str(mapping), "--sheet", str(sheet)]) == 0
    assert cv2.imread(str(sheet)) is not None
    return json.loads(mapping.read_text(encoding="utf-8"))


def _snapshot(root: Path) -> dict[str, bytes]:
    """work-root 内の全 JSON の中身を相対パスごとに返す。

    書き換えが起きていないことをバイト単位で比べるのに使う。
    """
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*.json"))}


def _apply(root: Path, rows: list[dict], tmp_path: Path) -> int:
    """対応表を書き出して apply を実行し、終了コードを返す。

    バックアップ先は tmp_path/backup に固定する。
    """
    mapping = tmp_path / "decided.json"
    mapping.write_text(json.dumps(rows), encoding="utf-8")
    return main([
        "apply", "--work-root", str(root), "--mapping", str(mapping), "--backup-dir", str(tmp_path / "backup"),
    ])


def test_propose_suggests_by_rule_and_does_not_modify_data(tmp_path: Path) -> None:
    """propose は規則どおりに提案し、ラベル JSON を1バイトも変えない。

    projectile → weapon_projectile、プレイヤー中心付近の area → weapon_aura、遠い area → review。
    """
    root, _, _ = _work_root(tmp_path)
    before = _snapshot(root)

    rows = _propose(root, tmp_path)

    assert _snapshot(root) == before
    assert [(r["id"], r["file"], r["shape_index"], r["old_label"], r["proposed_label"]) for r in rows] == [
        (1, "session-a/00000001.json", 1, "hazard_projectile", "weapon_projectile"),
        (2, "session-a/00000001.json", 2, "hazard_area", "weapon_aura"),
        (3, "session-b/00000002.json", 0, "hazard_area", "review"),
        (4, "session-b/00000002.json", 1, "hazard_area", "weapon_aura"),
    ]
    assert rows[1]["bbox"] == [90.0, 90.0, 150.0, 150.0]
    assert all(row["reason"] for row in rows)


def test_propose_refuses_to_overwrite_existing_mapping(tmp_path: Path) -> None:
    """既存の対応表（ユーザーが確認中かもしれない）を上書きしない。

    終了コード 1 で、ファイル内容はそのまま残る。
    """
    root, _, _ = _work_root(tmp_path)
    mapping = tmp_path / "mapping.json"
    mapping.write_text("[]", encoding="utf-8")

    result = main(["propose", "--work-root", str(root), "--output", str(mapping), "--sheet", str(tmp_path / "s.png")])

    assert result == 1
    assert mapping.read_text(encoding="utf-8") == "[]"


def test_apply_backs_up_changes_deletes_and_preserves_other_keys(tmp_path: Path) -> None:
    """apply はバックアップを取ってから、ラベル変更と shape 削除だけを行う。

    checked・circle の shape_type・flags・description などは保持し、変更なしの行は触らない。
    """
    root, first, second = _work_root(tmp_path)
    rows = _propose(root, tmp_path)
    rows[2]["proposed_label"] = "delete"
    original_first, original_second = first.read_bytes(), second.read_bytes()

    assert _apply(root, rows, tmp_path) == 0

    assert (tmp_path / "backup" / "session-a" / "00000001.json").read_bytes() == original_first
    assert (tmp_path / "backup" / "session-b" / "00000002.json").read_bytes() == original_second
    first_payload = json.loads(first.read_text(encoding="utf-8"))
    assert first_payload["checked"] is True and first_payload["flags"] == {"frame": "x"}
    assert [s["label"] for s in first_payload["shapes"]] == [
        "player_anchor", "weapon_projectile", "weapon_aura", "gem_blue",
    ]
    aura = first_payload["shapes"][2]
    assert aura["shape_type"] == "circle" and aura["points"] == [[120, 120], [150, 120]]
    assert aura["flags"] == {"keep": True} and aura["description"] == "note-hazard_area"
    second_payload = json.loads(second.read_text(encoding="utf-8"))
    assert [s["label"] for s in second_payload["shapes"]] == ["weapon_aura"]
    assert second_payload["shapes"][0]["points"] == [[934, 483], [974, 523]]


def test_apply_with_remaining_review_writes_nothing(tmp_path: Path) -> None:
    """review が1行でも残っていれば終了コード 1 で、データもバックアップも書かない。

    ユーザー確認が済んでいない対応表で確認済みデータを書き換えない。
    """
    root, _, _ = _work_root(tmp_path)
    rows = _propose(root, tmp_path)
    before = _snapshot(root)

    assert _apply(root, rows, tmp_path) == 1

    assert _snapshot(root) == before
    assert not (tmp_path / "backup").exists()


def test_apply_with_changed_bbox_or_label_writes_nothing(tmp_path: Path) -> None:
    """propose 後に手編集された shape があれば、他の行も含めて何も書かない。

    bbox の不一致・旧ラベルの不一致のどちらでも終了コード 1。
    """
    root, first, _ = _work_root(tmp_path)
    rows = _propose(root, tmp_path)
    rows[2]["proposed_label"] = "weapon_zone"
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["shapes"][1]["points"] = [[11, 10], [30, 20]]
    first.write_text(json.dumps(payload), encoding="utf-8")
    before = _snapshot(root)

    assert _apply(root, rows, tmp_path) == 1
    assert _snapshot(root) == before

    rows_label = [dict(row) for row in rows]
    rows_label[0]["bbox"] = [11.0, 10.0, 30.0, 20.0]
    rows_label[0]["old_label"] = "hazard_area"
    assert _apply(root, rows_label, tmp_path) == 1
    assert _snapshot(root) == before
    assert not (tmp_path / "backup").exists()


def test_apply_rejects_unknown_keys_labels_and_paths_outside_work_root(tmp_path: Path) -> None:
    """対応表の未知キー・未知ラベル・work-root 外のパスは補正せず拒否する。

    いずれも終了コード 1 で、データは変わらない。
    """
    root, _, _ = _work_root(tmp_path)
    rows = _propose(root, tmp_path)
    rows[2]["proposed_label"] = "weapon_zone"
    before = _snapshot(root)

    for mutate in (
        lambda row: row.update(extra=1),
        lambda row: row.update(proposed_label="weapon_typo"),
        lambda row: row.update(file="../outside/00000001.json"),
        lambda row: row.update(shape_index=99),
    ):
        broken = [dict(row) for row in rows]
        mutate(broken[0])
        assert _apply(root, broken, tmp_path) == 1
    assert _apply(root, rows + [dict(rows[0])], tmp_path) == 1
    assert _snapshot(root) == before

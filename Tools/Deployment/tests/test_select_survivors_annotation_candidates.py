"""候補フレーム抽出処理の回帰テスト。

合成フレームだけを使って候補選択を確認する。
既存ファイルが保護されることも確かめる。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from survivors.annotation_labels import ALL_CLASSES
from survivors.capture.captured_frame import CapturedFrame
from survivors.capture_dataset import DatasetWriter
from select_survivors_annotation_candidates import main


PROFILE_HASH = "c" * 64


def _publish_session(tmp_path: Path, frames: list[tuple[int, bool]]):
    """合成フレームを含む公開セッションを作る。

    ピクセル値と foreground を指定して manifest を作成する。
    候補抽出 CLI から復元できる形式で保存する。
    """
    writer = DatasetWriter(tmp_path, "session-a", PROFILE_HASH, "build-1")
    for frame_id, (pixel, foreground) in enumerate(frames):
        pixels = np.zeros((1080, 1920, 4), dtype=np.uint8)
        pixels[..., 0] = pixel
        pixels[..., 3] = 255
        writer.write_frame(
            CapturedFrame(
                frame_bgra=pixels,
                captured_monotonic_ns=frame_id + 1,
                session_frame_index=frame_id,
                client_rect_screen_px=(0, 0, 1920, 1080),
                foreground=foreground,
                target_profile_hash=PROFILE_HASH,
                game_build_id="build-1",
            )
        )
    return writer.publish()


def _invoke(store_root: Path, work_root: Path, *options: str) -> int:
    """候補抽出 CLI を指定した引数で実行する。

    テストごとに同じ必須引数を使う。
    追加オプションだけを呼び出し元から受け取る。
    """
    return main(
        [
            "--store-root",
            str(store_root),
            "--session-id",
            "session-a",
            "--work-root",
            str(work_root),
            *options,
        ]
    )


def test_candidate_selection_filters_deduplicates_and_applies_stride(tmp_path, capsys):
    """foreground・hash・stride の順で候補を絞る。

    最小 frame_id の重複を残して候補を間引く。
    選ばれた PNG とクラス一覧を確かめる。
    """
    manifest = _publish_session(
        tmp_path,
        [(0, True), (0, True), (100, True), (200, False), (255, True)],
    )
    work_root = tmp_path / "work"

    assert _invoke(tmp_path, work_root, "--stride", "2") == 0

    output_dir = work_root / "session-a"
    assert sorted(path.name for path in output_dir.glob("*.png")) == [
        "00000000.png",
        "00000004.png",
    ]
    for frame_id in (0, 4):
        record = next(item for item in manifest.frame_records if item.frame_id == frame_id)
        source = manifest.session_path / record.object_path
        output = output_dir / f"{frame_id:08d}.png"
        assert hashlib.sha256(output.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest()
    class_lines = (work_root / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert class_lines == list(ALL_CLASSES)
    # 実データの classes.txt と同じく、labeled_region の後ろに武器エフェクト4クラスが並ぶ。
    assert class_lines[-5:] == [
        "labeled_region", "weapon_projectile", "weapon_zone", "weapon_orbit", "weapon_aura"
    ]
    output = capsys.readouterr().out
    assert "総数: 5" in output
    assert "foreground: 4" in output


def test_kmeans_selection_is_deterministic(tmp_path):
    """k-means は seed 固定で同じ候補集合を選ぶ。

    同一セッションを別々の work-root へ出力する。
    2回の選択結果を比較する。
    """
    _publish_session(tmp_path, [(0, True), (40, True), (120, True), (240, True)])
    work_roots = [tmp_path / "work-a", tmp_path / "work-b"]

    for work_root in work_roots:
        assert _invoke(tmp_path, work_root, "--count", "2", "--stride", "1") == 0

    selected = [
        sorted(path.name for path in (work_root / "session-a").glob("*.png"))
        for work_root in work_roots
    ]
    assert len(selected[0]) == 2
    assert selected[0] == selected[1]


def test_existing_png_and_json_are_preserved(tmp_path, capsys):
    """既存候補画像とラベル JSON を変更しない。

    8桁でない補助ファイルも候補として処理しない。
    新規 PNG だけが配置され、JSON が作られないことを確かめる。
    """
    manifest = _publish_session(tmp_path, [(10, True), (200, True)])
    work_root = tmp_path / "work"
    output_dir = work_root / "session-a"
    output_dir.mkdir(parents=True)
    existing_png = output_dir / "00000000.png"
    existing_json = output_dir / "00000000.json"
    readme_json = output_dir / "README.json"
    readme_png = output_dir / "notes.png"
    existing_png.write_bytes(b"keep png")
    existing_json.write_bytes(b'{"checked": true}\n')
    readme_json.write_bytes(b"not a label file")
    readme_png.write_bytes(b"not a frame")

    assert _invoke(tmp_path, work_root, "--count", "10", "--stride", "1") == 0

    assert existing_png.read_bytes() == b"keep png"
    assert existing_json.read_bytes() == b'{"checked": true}\n'
    assert readme_json.read_bytes() == b"not a label file"
    assert readme_png.read_bytes() == b"not a frame"
    assert not (output_dir / "00000001.json").exists()
    second = next(item for item in manifest.frame_records if item.frame_id == 1)
    assert (output_dir / "00000001.png").read_bytes() == (
        manifest.session_path / second.object_path
    ).read_bytes()
    assert "新規配置数: 1" in capsys.readouterr().out

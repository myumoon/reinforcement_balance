"""window plan 指定時の annotation 候補配置を検証するテスト。

登録済み窓の stride に従い、窓外 frame を候補へ混ぜない動作を確かめる。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from survivors.annotation_labels import ALL_CLASSES
from survivors.capture.captured_frame import CapturedFrame
from select_survivors_annotation_candidates import main
from survivors.capture_dataset import DatasetWriter
from survivors.evaluation_windows import EvaluationWindowPlan


PROFILE_HASH = "c" * 64


def _publish_session(tmp_path: Path, frames: list[tuple[int, bool]]):
    """合成フレームを含む公開セッションを作成する。
    pixel 値と foreground を固定し、legacy 候補選択を再現できる manifest を返す。
    """
    writer = DatasetWriter(tmp_path, "session-a", PROFILE_HASH, "build-1")
    for frame_id, (pixel, foreground) in enumerate(frames):
        pixels = np.zeros((1080, 1920, 4), dtype=np.uint8)
        pixels[..., 0] = pixel
        pixels[..., 3] = 255
        writer.write_frame(CapturedFrame(
            frame_bgra=pixels,
            captured_monotonic_ns=frame_id + 1,
            session_frame_index=frame_id,
            client_rect_screen_px=(0, 0, 1920, 1080),
            foreground=foreground,
            target_profile_hash=PROFILE_HASH,
            game_build_id="build-1",
        ))
    return writer.publish()


def _invoke(store_root: Path, work_root: Path, *options: str) -> int:
    """legacy 候補選択 CLI を任意 option 付きで呼び出す。
    共通の store/session/work-root 引数はここで固定する。
    """
    return main([
        "--store-root", str(store_root), "--session-id", "session-a",
        "--work-root", str(work_root), *options,
    ])


def test_candidate_selection_filters_deduplicates_and_applies_stride(tmp_path, capsys):
    """foreground・hash・stride の順で候補を絞る。
    選択 PNG と class list が従来どおり配置されることを確かめる。
    """
    manifest = _publish_session(
        tmp_path, [(0, True), (0, True), (100, True), (200, False), (255, True)],
    )
    work_root = tmp_path / "work"
    assert _invoke(tmp_path, work_root, "--stride", "2") == 0
    output_dir = work_root / "session-a"
    assert sorted(path.name for path in output_dir.glob("*.png")) == [
        "00000000.png", "00000004.png",
    ]
    for frame_id in (0, 4):
        record = next(item for item in manifest.frame_records if item.frame_id == frame_id)
        source = manifest.session_path / record.object_path
        output = output_dir / f"{frame_id:08d}.png"
        assert hashlib.sha256(output.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest()
    class_lines = (work_root / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert class_lines == list(ALL_CLASSES)
    assert class_lines[class_lines.index("hud_hp") - 4:class_lines.index("hud_hp")] == [
        "weapon_projectile", "weapon_zone", "weapon_orbit", "weapon_aura",
    ]
    assert class_lines[-2:] == ["labeled_region", "weapon_target"]
    output = capsys.readouterr().out
    assert "総数: 5" in output
    assert "foreground: 4" in output


def test_kmeans_selection_is_deterministic(tmp_path):
    """k-means は seed 固定で同じ候補集合を選ぶ。
    work-root を分けて 2 回出力した frame 名を比較する。
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
    """既存候補画像と label JSON を変更しない。
    新規 PNG だけを追加し、補助ファイルを frame と誤認しないことを確かめる。
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


def _rule() -> dict[str, object]:
    """候補選択に必要な evaluation_windows.v1 rule を返す。
    各 event は hash-bound で、内容依存の選択条件を置かない。
    """
    return {
        "fixed_offsets_per_time_band": {
            "window_frames": 5,
            "offsets_seconds": {"early": 0, "mid": 10, "late": 20},
        },
        "run_tail": {"seconds": 10},
        "event_windows": {
            "detector": "development_hud_parser",
            "states": ["level_up_items"],
            "pre_frames": 0,
            "post_frames": 0,
            "max_per_band": 4,
            "parser_artifact_hash": "a" * 64,
        },
        "boss_windows": {
            "detector": "development_world_detector",
            "checkpoint_hash": "b" * 64,
            "score_threshold": 0.5,
            "pre_frames": 0,
            "post_frames": 0,
            "max_per_session": 8,
        },
    }


def test_window_plan_places_only_stride_selected_frames(tmp_path: Path, monkeypatch) -> None:
    """window mode は plan session の範囲だけを指定 stride で配置する。
    窓外 frame と stride から外れた frame が配置されないことを確かめる。
    """
    store_root = tmp_path / "store"
    session_path = store_root / "capture_sessions" / "synthetic-session"
    records = tuple(
        SimpleNamespace(frame_id=index, object_path=f"frames/{index:08d}.png", object_sha256=f"{index:064x}", foreground=True)
        for index in range(10)
    )
    manifest = SimpleNamespace(
        session_id="synthetic-session", session_path=session_path,
        frame_count=len(records), frame_records=records,
    )
    monkeypatch.setattr(
        DatasetWriter, "restore",
        classmethod(lambda _cls, _root, _sid, **_kwargs: manifest),
    )
    placed: list[int] = []
    monkeypatch.setattr(
        "select_survivors_annotation_candidates._place_candidate",
        lambda _source, destination: placed.append(int(destination.stem)) is None,
    )
    plan = EvaluationWindowPlan.create(
        {manifest.session_id: [(1, 4), (7, 9)]}, rule=_rule(), stride=3,
        generated_at_utc="2026-10-11T00:00:00Z",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan.to_wire()), encoding="utf-8")

    assert main([
        "--store-root", str(store_root), "--session-id", manifest.session_id,
        "--work-root", str(tmp_path / "work"), "--window-plan", str(plan_path),
    ]) == 0
    assert placed == [1, 4, 7]
    assert (tmp_path / "work" / "classes.txt").is_file()

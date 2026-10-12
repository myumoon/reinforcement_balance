"""自己一致用 work-root 準備 CLI の契約を確認する。

画像選択の再現性と、確認済み画像だけを空の B 側へ置く条件を守る。
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _write_checked_frame(root: Path, session_id: str, frame_id: int, *, checked: bool = True) -> None:
    """テスト用に画像と確認状態つき label を作る。

    CLI が checked frame だけを選び、label を B へ複製しないことを確かめる。
    """
    session = root / session_id
    session.mkdir(parents=True, exist_ok=True)
    stem = f"{frame_id:08d}"
    (session / f"{stem}.png").write_bytes(b"png")
    (session / f"{stem}.json").write_text(json.dumps({
        "shapes": [], "imagePath": f"{stem}.png", "imageData": None, "checked": checked,
    }), encoding="utf-8")
    os.utime(session / f"{stem}.json", (1_700_000_000, 1_700_000_000))


def _run_prepare(
    work_root_a: Path,
    work_root_b: Path,
    *,
    count: int = 3,
    seed: int = 17,
    minimum_gap_hours: float | None = None,
):
    """準備 CLI を独立した Python プロセスで実行する。

    実際のコマンド引数と終了コードを通して public entrypoint を検証する。
    """
    script = Path(__file__).parents[1] / "prepare_survivors_self_agreement.py"
    command = [
        sys.executable, str(script), "--work-root-a", str(work_root_a),
        "--work-root-b", str(work_root_b), "--count", str(count), "--seed", str(seed),
    ]
    if minimum_gap_hours is not None:
        command.extend(["--minimum-gap-hours", str(minimum_gap_hours)])
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_prepare_selects_checked_frames_deterministically_and_leaves_labels_empty(tmp_path: Path) -> None:
    """同じ seed は同じ checked frame を選び、B には画像と manifest だけ置く。

    未確認 frame と label JSON が出力されず、指定した枚数だけを複製する。
    """
    source = tmp_path / "source"
    for frame_id in range(5):
        _write_checked_frame(source, "session-a", frame_id, checked=frame_id != 4)
    first, second = tmp_path / "first", tmp_path / "second"

    result = _run_prepare(source, first)
    repeated = _run_prepare(source, second)

    assert result.returncode == 0, result.stderr
    assert repeated.returncode == 0, repeated.stderr
    first_manifest = json.loads((first / "self_agreement_manifest.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "self_agreement_manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["selected_frames"] == second_manifest["selected_frames"]
    assert len(first_manifest["selected_frames"]) == 3
    assert first_manifest["blind"] is True
    assert all(frame["frame_id"] != 4 for frame in first_manifest["selected_frames"])
    assert sorted(path.name for path in first.rglob("*.json")) == ["self_agreement_manifest.json"]
    for frame in first_manifest["selected_frames"]:
        assert (first / frame["session_id"] / frame["image_file"]).is_file()


def test_prepare_rejects_nonempty_destination(tmp_path: Path) -> None:
    """既存ファイルのある B work-root を上書きしない。

    既存成果物を保持し、誤った再実行によるラベル・画像の混在を防ぐ。
    """
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_checked_frame(source, "session-a", 1)
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    result = _run_prepare(source, destination, count=1)

    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_prepare_rejects_source_labels_inside_minimum_gap(tmp_path: Path) -> None:
    """A の最終 label 更新が設定時間より新しければ B root を作らない。

    準備 CLI 自身が待機条件を fail-closed で強制することを確かめる。
    """
    source, destination = tmp_path / "source", tmp_path / "destination"
    _write_checked_frame(source, "session-a", 1)
    label = source / "session-a" / "00000001.json"
    recent = time.time() - 3600
    os.utime(label, (recent, recent))

    result = _run_prepare(source, destination, count=1, minimum_gap_hours=24)

    assert result.returncode != 0
    assert "minimum gap" in result.stderr
    assert not destination.exists()

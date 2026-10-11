"""artifact restore helper の contract test。

descriptor が参照する全 object を別 root へ復元し、共有 verdict と一致することを確かめます。
"""

from pathlib import Path

from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from survivors.artifact_restore_test import run_restore_test
from benchmark_survivors_perception import _validate_restore_verdict


def test_restore_test_copies_every_descriptor_file_and_publishes_verdict(tmp_path: Path) -> None:
    """複数 file を full restore し、verdict を primary に保存する。

正式 consumer と同じ validator が descriptor と restore verdict の binding を受け入れます。
    """
    primary = ArtifactStore(tmp_path / "primary")
    package = primary.put_bytes(
        logical_id="hud/package.json", data=b"package", media_type="application/json"
    )
    atlas = primary.put_bytes(
        logical_id="hud/atlas.json", data=b"atlas", media_type="application/json"
    )
    subject = ArtifactDescriptor(
        logical_id="hud/formal-release",
        node_kind="source_descriptor",
        producer_id="hud-formal-release",
        producer_version="1",
        identity_metadata={"manifest_logical_id": package.logical_id},
        parents=(),
        files=(package, atlas),
    )

    verdict, verdict_descriptor = run_restore_test(
        primary,
        subject,
        backup_root=tmp_path / "backup",
        restore_root=tmp_path / "restored",
        verdict_logical_id="hud/restore-verdict",
    )

    assert verdict.passed is True
    assert verdict.checked_object_count == 2
    assert primary.resolve("hud/restore-verdict") is not None
    restored = ArtifactStore(tmp_path / "restored")
    assert all(restored.verify(ref).ok for ref in subject.files)
    _validate_restore_verdict(subject, verdict_descriptor)


def test_restore_test_rejects_nonempty_restore_root(tmp_path: Path) -> None:
    """既存内容を消さず、空でない restore root は拒否する。

前回試行の object を混ぜて成功判定することを防ぎます。
    """
    primary = ArtifactStore(tmp_path / "primary")
    ref = primary.put_bytes(
        logical_id="hud/manifest.json", data=b"manifest", media_type="application/json"
    )
    subject = ArtifactDescriptor(
        logical_id="hud/formal-release",
        node_kind="source_descriptor",
        producer_id="hud-formal-release",
        producer_version="1",
        identity_metadata={"manifest_logical_id": ref.logical_id},
        files=(ref,),
    )
    occupied = tmp_path / "restored"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")

    try:
        run_restore_test(
            primary,
            subject,
            backup_root=tmp_path / "backup",
            restore_root=occupied,
            verdict_logical_id="hud/restore-verdict",
        )
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("non-empty restore root was accepted")
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "keep"

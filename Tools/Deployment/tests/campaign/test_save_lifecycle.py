"""save lifecycle の backup・canonical 差し替え・post hash・復元を検証します。

実 file と write-once artifact store を使い、game 実行中の拒否、検証失敗時に save が
変わらないこと、記録の上書き拒否、resume 時に backup を取り直さないことを固定します。
"""

from __future__ import annotations

import hashlib

import pytest

from survivors.campaign import save_lifecycle as sl
from survivors.campaign.campaign_runner import ArtifactConflict, ArtifactStore
from survivors.campaign.save_lifecycle import SaveLifecycle, SaveLifecycleError, atomic_replace

ORIGINAL = b"original-save"
CANONICAL = b"canonical-save"


def _sha(data: bytes) -> str:
    """test 用 SHA-256 です。

    実装と独立に期待値を計算します。
    """
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def env(tmp_path):
    """元 save・artifact store・停止 flag を用意します。

    flag を False にすると game 実行中を再現します。
    """
    save = tmp_path / "game" / "SaveData.sav"
    save.parent.mkdir()
    save.write_bytes(ORIGINAL)
    store = ArtifactStore(tmp_path / "artifacts")
    stopped = {"value": True}
    lifecycle = SaveLifecycle(save, store, processes_stopped=lambda: stopped["value"])
    lifecycle.register_canonical(CANONICAL)
    return save, store, stopped, lifecycle


def _ready(lifecycle: SaveLifecycle) -> None:
    """backup と cloud sync attestation を記録します。

    canonical 差し替えの前提条件です。
    """
    lifecycle.backup_original()
    lifecycle.attest_cloud_sync({"cloud_sync_disabled": True, "operator": "test"})


def test_backup_is_stored_and_not_retaken_on_resume(env):
    """backup は元 save を保存し、resume では現在の save を読み直しません。

    canonical 差し替え後に backup を呼んでも元 save の hash のままです。
    """
    save, store, _stopped, lifecycle = env
    digest = lifecycle.backup_original()
    assert digest == _sha(ORIGINAL)
    assert store.read(sl.BACKUP) == ORIGINAL
    lifecycle.attest_cloud_sync({"cloud_sync_disabled": True})
    lifecycle.install_canonical("a1", _sha(CANONICAL))
    assert save.read_bytes() == CANONICAL
    assert lifecycle.backup_original() == _sha(ORIGINAL)
    assert store.read(sl.BACKUP) == ORIGINAL


def test_backup_refuses_while_game_running_or_missing(env, tmp_path):
    """game 実行中や元 save が無い場合は backup しません。

    どちらも artifact を作らずに例外です。
    """
    _save, store, stopped, lifecycle = env
    stopped["value"] = False
    with pytest.raises(SaveLifecycleError, match="running"):
        lifecycle.backup_original()
    assert not store.exists(sl.BACKUP_RECORD)
    missing = SaveLifecycle(tmp_path / "none.sav", store, processes_stopped=lambda: True)
    with pytest.raises(SaveLifecycleError, match="unreadable"):
        missing.backup_original()


def test_install_requires_backup_and_cloud_sync_first(env):
    """backup と cloud sync attestation の前は save を変更しません。

    cloud sync attestation は disabled=true だけを受け付けます。
    """
    save, _store, _stopped, lifecycle = env
    with pytest.raises(SaveLifecycleError, match="backup and cloud sync"):
        lifecycle.install_canonical("a1", _sha(CANONICAL))
    lifecycle.backup_original()
    with pytest.raises(SaveLifecycleError, match="cloud_sync_disabled"):
        lifecycle.attest_cloud_sync({"cloud_sync_disabled": False})
    with pytest.raises(SaveLifecycleError, match="backup and cloud sync"):
        lifecycle.install_canonical("a1", _sha(CANONICAL))
    assert save.read_bytes() == ORIGINAL


def test_install_is_verified_atomic_and_records_pre_hash(env):
    """canonical save を検証付きで差し替え、pre hash を記録します。

    一時 file は残らず、同じ attempt の pre hash は write-once です。
    """
    save, store, _stopped, lifecycle = env
    _ready(lifecycle)
    assert lifecycle.install_canonical("a1", _sha(CANONICAL)) == _sha(CANONICAL)
    assert save.read_bytes() == CANONICAL
    assert lifecycle.slot_hash("pre", "a1") == _sha(CANONICAL)
    assert lifecycle.slot_hash("post", "a1") is None
    assert [path.name for path in save.parent.iterdir()] == [save.name]


def test_install_refuses_while_running_or_on_hash_mismatch(env):
    """game 実行中・canonical hash 不一致では save を変えません。

    pre hash も記録されません。
    """
    save, _store, stopped, lifecycle = env
    _ready(lifecycle)
    stopped["value"] = False
    with pytest.raises(SaveLifecycleError, match="running"):
        lifecycle.install_canonical("a1", _sha(CANONICAL))
    stopped["value"] = True
    with pytest.raises(SaveLifecycleError, match="expected save hash"):
        lifecycle.install_canonical("a1", "0" * 64)
    assert save.read_bytes() == ORIGINAL
    assert lifecycle.slot_hash("pre", "a1") is None


def test_atomic_replace_temp_verification_failure_keeps_original(tmp_path, monkeypatch):
    """一時 file の検証に失敗したら元 save は変わらず一時 file も残りません。

    書き込み途中の破損を fsync 後の読み戻しで検出します。
    """
    target = tmp_path / "save.sav"
    target.write_bytes(ORIGINAL)
    real_fsync = sl.os.fsync

    def corrupt(fd):
        """fsync の直前に一時 file を壊します。"""
        sl.os.write(fd, b"!")
        real_fsync(fd)

    monkeypatch.setattr(sl.os, "fsync", corrupt)
    with pytest.raises(SaveLifecycleError, match="temporary save failed verification"):
        atomic_replace(target, CANONICAL, _sha(CANONICAL))
    assert target.read_bytes() == ORIGINAL
    assert [path.name for path in tmp_path.iterdir()] == ["save.sav"]


def test_post_run_hash_requires_stop_and_is_write_once(env):
    """post hash は process 停止後にだけ記録し、上書きできません。

    同じ attempt で異なる save hash を記録しようとすると ArtifactConflict です。
    """
    save, _store, stopped, lifecycle = env
    _ready(lifecycle)
    lifecycle.install_canonical("a1", _sha(CANONICAL))
    save.write_bytes(b"played")
    stopped["value"] = False
    with pytest.raises(SaveLifecycleError, match="running"):
        lifecycle.post_run_hash("a1")
    stopped["value"] = True
    assert lifecycle.post_run_hash("a1") == _sha(b"played")
    save.write_bytes(b"tampered")
    with pytest.raises(ArtifactConflict):
        lifecycle.post_run_hash("a1")
    assert lifecycle.slot_hash("post", "a1") == _sha(b"played")


def test_restore_original_passes_and_is_idempotent(env):
    """campaign 終了時に元 save を復元し PASS verdict を記録します。

    save が元のままなら再呼び出しは同じ verdict を返し、evidence に restore 状態が載ります。
    """
    save, _store, _stopped, lifecycle = env
    _ready(lifecycle)
    lifecycle.install_canonical("a1", _sha(CANONICAL))
    verdict = lifecycle.restore_original()
    assert verdict["status"] == "PASS"
    assert verdict["restored_sha256"] == _sha(ORIGINAL)
    assert save.read_bytes() == ORIGINAL
    assert lifecycle.restore_original() == verdict
    evidence = lifecycle.evidence()
    assert evidence == {
        "backup_sha256": _sha(ORIGINAL),
        "cloud_sync_attested": True,
        "canonical_sha256": _sha(CANONICAL),
        "restore_status": "PASS",
    }


def test_restore_rechecks_save_after_recorded_pass(env):
    """記録済み PASS 後も再呼び出しのたびに現在の save を検証します。

    save が canonical にずれていれば backup から書き戻して DRIFT を残し、
    書き戻せなければ PASS を返さず例外にします。
    """
    save, store, _stopped, lifecycle = env
    _ready(lifecycle)
    lifecycle.install_canonical("a1", _sha(CANONICAL))
    verdict = lifecycle.restore_original()
    lifecycle.install_canonical("a2", _sha(CANONICAL))
    assert lifecycle.restore_original() == verdict
    assert save.read_bytes() == ORIGINAL
    assert store.stream(sl.RESTORE_ATTEMPTS)[0] == {"status": "DRIFT", "observed_sha256": _sha(CANONICAL)}
    lifecycle.install_canonical("a3", _sha(CANONICAL))
    store._path(sl.BACKUP).write_bytes(b"corrupted-backup")
    with pytest.raises(SaveLifecycleError, match="restore failed"):
        lifecycle.restore_original()
    assert save.read_bytes() == CANONICAL


def test_restore_failure_is_recorded_without_verdict(env):
    """backup が壊れていれば save を変えず、FAIL を attempt stream に残して例外にします。

    PASS verdict は書かれないので、是正後に再実行できます。
    """
    save, store, stopped, lifecycle = env
    _ready(lifecycle)
    lifecycle.install_canonical("a1", _sha(CANONICAL))
    stopped["value"] = False
    with pytest.raises(SaveLifecycleError, match="running"):
        lifecycle.restore_original()
    stopped["value"] = True
    store._path(sl.BACKUP).write_bytes(b"corrupted-backup")
    with pytest.raises(SaveLifecycleError, match="restore failed"):
        lifecycle.restore_original()
    assert save.read_bytes() == CANONICAL
    assert store.stream(sl.RESTORE_ATTEMPTS)[0]["status"] == "FAIL"
    assert lifecycle.evidence()["restore_status"] is None


def test_canonical_registration_is_write_once(env):
    """canonical save は別内容で再登録できません。

    同じ内容の再登録は idempotent です。
    """
    _save, _store, _stopped, lifecycle = env
    assert lifecycle.register_canonical(CANONICAL) == _sha(CANONICAL)
    with pytest.raises(ArtifactConflict):
        lifecycle.register_canonical(b"other")

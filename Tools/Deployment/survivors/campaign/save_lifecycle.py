"""campaign 用 Survivors save file の backup・差し替え・復元を管理します。

campaign 開始前に元の save を artifact store へ退避し、各 slot の前に canonical save を
一時 file で検証してから atomic に置き換えます。run 後の save hash と campaign 終了時の
復元 verdict も記録し、formal 証跡に必要な save 情報がそろっているかを返します。
game / launcher が動いている間は save に一切触れません。
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

BACKUP = "save/original_backup.bin"
BACKUP_RECORD = "save/original_backup.json"
CANONICAL = "save/canonical.bin"
CLOUD_SYNC = "save/cloud_sync_attestation.json"
RESTORE = "save/restore_verdict.json"
RESTORE_ATTEMPTS = "save_restore_attempts"


class SaveLifecycleError(RuntimeError):
    """save の退避・差し替え・復元を安全に行えないことを表します。

    呼び出し側は save が未変更または検証失敗の状態として扱い、run を正式証跡にしません。
    """


def _sha(data: bytes) -> str:
    """bytes の SHA-256 hex を返します。

    save の同一性はすべてこの hash で比較します。
    """
    return hashlib.sha256(data).hexdigest()


def atomic_replace(path: str | os.PathLike[str], data: bytes, expected_hash: str) -> None:
    """一時 file の検証後に save を atomic に置き換えます。

    同じ directory の一時 file へ書いて fsync し、読み戻した hash が期待値と一致した
    ときだけ `os.replace` します。置き換え後も読み戻して再検証します。
    一時 file の検証に失敗した場合、元の save は変更されません。
    """
    target = Path(path)
    if _sha(data) != expected_hash:
        raise SaveLifecycleError("source bytes do not match the expected save hash")
    temporary = target.with_name(target.name + ".reinbalance-tmp")
    try:
        with open(temporary, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if _sha(temporary.read_bytes()) != expected_hash:
            raise SaveLifecycleError("temporary save failed verification")
        # ponytail: directory entry の fsync は Windows で不可。rename の永続性は NTFS journal に任せる
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if _sha(target.read_bytes()) != expected_hash:
        raise SaveLifecycleError("replaced save failed read-back verification")


class SaveLifecycle:
    """1 campaign 分の save backup・canonical 差し替え・post hash・復元を記録します。

    artifact store には write-once で記録するので、backup や hash 記録を後から書き換えられません。
    `processes_stopped` が True を返さない限り save を読み書きしません。
    ponytail: save は単一 file 前提。複数 file の save になったら directory 単位の manifest にする。
    """

    def __init__(
        self,
        save_path: str | os.PathLike[str],
        artifacts: Any,
        *,
        processes_stopped: Callable[[], bool],
    ) -> None:
        """save path、artifact store、停止確認 callback を受け取ります。

        artifact store は `exists` / `read` / `put_immutable` / `put_json` / `read_json` /
        `append` を持つ write-once store です。
        """
        self._path = Path(save_path)
        self._artifacts = artifacts
        self._processes_stopped = processes_stopped

    def _require_stopped(self, action: str) -> None:
        """game / launcher の停止を確認します。

        停止を確認できない場合は save に触れず例外にします。
        """
        if self._processes_stopped() is not True:
            raise SaveLifecycleError(f"game or launcher is running; refusing to {action}")

    def _json(self, name: str) -> Mapping[str, Any] | None:
        """記録済み JSON を読みます。

        未記録なら None を返します。
        """
        return self._artifacts.read_json(name) if self._artifacts.exists(name) else None

    def backup_original(self) -> str:
        """campaign 開始前の元 save を artifact store へ退避します。

        記録済みなら現在の save を読み直さずに記録済み hash を返します(resume 時に
        canonical save を元 save と取り違えないため)。
        """
        record = self._json(BACKUP_RECORD)
        if record is not None:
            return str(record["sha256"])
        self._require_stopped("back up the original save")
        try:
            data = self._path.read_bytes()
        except OSError as exc:
            raise SaveLifecycleError(f"original save is unreadable: {exc}") from exc
        digest = self._artifacts.put_immutable(BACKUP, data)
        if _sha(self._artifacts.read(BACKUP)) != digest:
            raise SaveLifecycleError("original save backup failed read-back verification")
        self._artifacts.put_json(BACKUP_RECORD, {"sha256": digest, "size": len(data), "source": str(self._path)})
        return digest

    def register_canonical(self, data: bytes) -> str:
        """各 slot で使う canonical save を artifact store へ登録します。

        write-once なので、別内容の canonical save で上書きできません。
        """
        return self._artifacts.put_immutable(CANONICAL, bytes(data))

    def attest_cloud_sync(self, attestation: Mapping[str, Any]) -> None:
        """cloud sync 無効化の operator checkpoint を記録します。

        `cloud_sync_disabled` が True の attestation だけを受け付けます。
        """
        if not isinstance(attestation, Mapping) or attestation.get("cloud_sync_disabled") is not True:
            raise SaveLifecycleError("cloud sync disable attestation must confirm cloud_sync_disabled=true")
        self._artifacts.put_json(CLOUD_SYNC, dict(attestation))

    def install_canonical(self, attempt_id: str, expected_hash: str) -> str:
        """slot の preflight として canonical save を検証付きで差し替えます。

        backup と cloud sync attestation が無い、game が動いている、canonical save の
        hash が違う場合は save を変更せずに例外にします。成功時は pre-run hash を記録します。
        """
        evidence = self.evidence()
        if evidence["backup_sha256"] is None or not evidence["cloud_sync_attested"]:
            raise SaveLifecycleError("original backup and cloud sync attestation are required first")
        self._require_stopped("replace the canonical save")
        if not self._artifacts.exists(CANONICAL):
            raise SaveLifecycleError("canonical save is not registered")
        atomic_replace(self._path, self._artifacts.read(CANONICAL), expected_hash)
        self._artifacts.put_json(f"save/pre/{attempt_id}.json", {"attempt_id": attempt_id, "sha256": expected_hash})
        return expected_hash

    def post_run_hash(self, attempt_id: str) -> str:
        """run 後の save hash を記録します。

        process 停止を確認してから読み、attempt ごとに1回だけ記録します。
        """
        self._require_stopped("hash the post-run save")
        try:
            digest = _sha(self._path.read_bytes())
        except OSError as exc:
            raise SaveLifecycleError(f"post-run save is unreadable: {exc}") from exc
        self._artifacts.put_json(f"save/post/{attempt_id}.json", {"attempt_id": attempt_id, "sha256": digest})
        return digest

    def slot_hash(self, kind: str, attempt_id: str) -> str | None:
        """記録済みの pre / post save hash を返します。

        未記録なら None で、その run は formal 証跡になりません。
        """
        if kind not in {"pre", "post"}:
            raise ValueError("kind must be pre or post")
        record = self._json(f"save/{kind}/{attempt_id}.json")
        return None if record is None else str(record["sha256"])

    def restore_original(self) -> Mapping[str, Any]:
        """campaign 終了時に元 save を復元して verify します。

        呼び出すたびに現在の save を読み戻し、backup hash と一致しなければ backup から書き戻して
        再検証します(記録済み PASS verdict を検証なしで返さない)。PASS verdict は write-once です。
        記録済み verdict 後に save がずれていた場合は DRIFT を attempt stream に残します。
        失敗は attempt stream へ残して例外にするので、是正後に再実行できます。
        """
        self._require_stopped("restore the original save")
        record = self._json(BACKUP_RECORD)
        if record is None:
            raise SaveLifecycleError("original save backup is missing")
        expected = str(record["sha256"])
        try:
            current = _sha(self._path.read_bytes())
        except OSError:
            current = None
        if current != expected:
            if self._artifacts.exists(RESTORE):
                self._artifacts.append(RESTORE_ATTEMPTS, [{"status": "DRIFT", "observed_sha256": current}])
            try:
                atomic_replace(self._path, self._artifacts.read(BACKUP), expected)
            except (SaveLifecycleError, OSError) as exc:
                self._artifacts.append(RESTORE_ATTEMPTS, [{"status": "FAIL", "reason": str(exc)}])
                raise SaveLifecycleError(f"original save restore failed: {exc}") from exc
        verdict = {"status": "PASS", "backup_sha256": expected, "restored_sha256": _sha(self._path.read_bytes())}
        self._artifacts.put_json(RESTORE, verdict)
        return verdict

    def evidence(self) -> dict[str, Any]:
        """formal 証跡に必要な save 記録の有無を返します。

        backup hash、cloud sync attestation、canonical hash、restore verdict をまとめます。
        """
        backup = self._json(BACKUP_RECORD)
        restore = self._json(RESTORE)
        canonical = self._artifacts.read(CANONICAL) if self._artifacts.exists(CANONICAL) else None
        return {
            "backup_sha256": None if backup is None else backup["sha256"],
            "cloud_sync_attested": self._artifacts.exists(CLOUD_SYNC),
            "canonical_sha256": None if canonical is None else _sha(canonical),
            "restore_status": None if restore is None else restore["status"],
        }

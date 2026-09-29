"""durable launch ledger の順序・一意性・fail-closed・fault injection を検証します。

実際の NTFS 上に SQLite WAL/FULL ledger を作り、mutation の拒否と crash 後の分類を固定します。
disk full・commit 失敗・DB 破損・partial row・PID 再利用を再現し、どれも activation へ進まないことを確認します。
"""

from __future__ import annotations

import os
import sqlite3
import sys

import pytest

from survivors.campaign import durable_launch_store as dls
from survivors.campaign.campaign_schema import CampaignEvent, EventType, validate_campaign_events
from survivors.campaign.durable_launch_store import (
    Classification,
    DurableLaunchStore,
    LaunchIntent,
    LaunchStage,
    LedgerCommitError,
    LedgerIdentityError,
    LedgerIntegrityError,
    LedgerOrderError,
    ProbeResult,
    ProcessAttestation,
    ProcessIdentity,
    UnsupportedStorageError,
    acl_findings,
    check_storage,
    current_user_sid,
    new_launch_nonce,
    read_sddl,
)

MANIFEST = "a" * 64


@pytest.fixture
def store(tmp_path):
    """current-user ACL 付きの新規 ledger を開きます。

    Windows 以外では formal-ineligible verdict を確認した上で、この環境では検証不能として扱います。
    """
    ledger = tmp_path / "ledger"
    if os.name != "nt":
        with pytest.raises(UnsupportedStorageError, match="platform_not_windows"):
            DurableLaunchStore(ledger)
        pytest.skip("formal_ineligible: platform_not_windows (verdict asserted)")
    opened = DurableLaunchStore(ledger)
    yield opened
    opened.close()


def _intent(slot: int = 0, attempt: str | None = None, **overrides) -> LaunchIntent:
    """test 用の launch intent を作ります。

    slot ごとに異なる id と新しい nonce を割り当てます。
    """
    exe = sys.executable
    values = dict(
        campaign_manifest_hash=MANIFEST, slot_id=slot, attempt_id=attempt or f"attempt-{slot}",
        reserved_run_id=f"run-{attempt or slot}", gameplay_attempt_id=f"gameplay-{attempt or slot}",
        launch_nonce=new_launch_nonce(), executable_path=exe, executable_hash="b" * 64,
        build_hash="c" * 64, config_hash="d" * 64, argv=(exe, "-c", "pass"),
    )
    values.update(overrides)
    return LaunchIntent(**values)


def _attestation(pid: int = 4242, create_time: int = 133_000_000_000_000_000, job: str = "job-1") -> ProcessAttestation:
    """test 用の kernel identity attestation を作ります。

    PID と create time を変えて別 process を表現します。
    """
    identity = ProcessIdentity(pid, create_time, sys.executable, 7, 99)
    return ProcessAttestation(identity, job, "pid:1:ct:1")


def _probe(result: ProbeResult):
    """常に同じ結果を返す probe を作ります。

    kernel 照会の代わりに分類ロジックだけを検証します。
    """
    return lambda _identity: result


def _advance(store: DurableLaunchStore, attempt: str, stage: LaunchStage, slot: int = 0) -> ProcessAttestation:
    """attempt を指定段階まで正規順序で進めます。

    途中の attestation を返し、呼び出し側が process_ref を使えるようにします。
    """
    store.commit_intent(_intent(slot, attempt))
    attestation = _attestation(pid=1000 + slot, job=f"job-{attempt}")
    order = [LaunchStage.PROCESS_ATTESTED, LaunchStage.RESUME_INTENT, LaunchStage.PROCESS_LAUNCH_CONFIRMED,
             LaunchStage.FORMAL_RUN_ACTIVATED]
    for step in order[: order.index(stage) + 1] if stage in order else []:
        if step is LaunchStage.PROCESS_ATTESTED:
            store.commit_attestation(attempt, attestation)
        elif step is LaunchStage.RESUME_INTENT:
            store.commit_resume_intent(attempt, attestation.identity.process_ref)
        elif step is LaunchStage.PROCESS_LAUNCH_CONFIRMED:
            store.commit_confirmation(attempt, attestation.identity.process_ref)
        else:
            assert store.activate(attempt, "normal", _probe(ProbeResult.ALIVE)) is True
    return attestation


def _validated_0602(store: DurableLaunchStore, attempt: str, slot: int = 0) -> tuple[CampaignEvent, ...]:
    """ledger 由来 event 列を 06-02 lifecycle validator に通します。

    reserved/preflight を前置し、campaign_manifest_hash 束縛も含めて検証します。
    """
    prefix = [
        CampaignEvent(EventType.FORMAL_SLOT_RESERVED, slot, campaign_manifest_hash=MANIFEST),
        CampaignEvent(EventType.ATTEMPT_PREFLIGHT, slot, attempt_id=attempt, campaign_manifest_hash=MANIFEST),
    ]
    events = store.campaign_events(attempt)
    validate_campaign_events([*prefix, *events], expected_manifest_hash=MANIFEST)
    return events


def _count(store: DurableLaunchStore, stage: LaunchStage) -> int:
    """指定段階の row 数を数えます。

    二重 activation や replacement が 0 件であることの確認に使います。
    """
    return store._conn.execute("SELECT COUNT(*) FROM launch_rows WHERE stage=?", (stage.value,)).fetchone()[0]


def test_open_reads_back_wal_full_integrity_schema_and_current_user_acl(store) -> None:
    """ledger を開くと WAL/FULL・integrity・schema version・ACL が実値で揃います。

    directory DACL は protected かつ current user の ACE だけです。
    """
    assert store.verdict.eligible and store.verdict.facts["filesystem"] == "NTFS"
    assert store.verdict.facts["drive_type"] == "fixed"
    assert store.pragmas == {"journal_mode": "wal", "synchronous": "FULL", "integrity_check": "ok",
                             "sqlite_version": sqlite3.sqlite_version}
    sid = current_user_sid()
    assert acl_findings(read_sddl(str(store.directory)), sid, require_owner=True, require_protected=True) == []
    assert acl_findings(read_sddl(str(store.path)), sid, require_owner=False, require_protected=False) == []
    raw = sqlite3.connect(store.path)
    try:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == dls.LEDGER_SCHEMA_VERSION
        assert raw.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        raw.close()


@pytest.mark.parametrize("path", [r"\\server\share\ledger", r"\\?\UNC\server\share\ledger", "//server/share/x"])
def test_unc_paths_are_formal_ineligible(path: str) -> None:
    """UNC / device namespace path は OS 照会前に formal-ineligible です。

    ledger も開かずに UnsupportedStorageError で止まります。
    """
    verdict = check_storage(path)
    assert verdict.eligible is False and verdict.label == "formal_ineligible"
    with pytest.raises(UnsupportedStorageError):
        DurableLaunchStore(path)


@pytest.mark.parametrize(
    ("drive", "filesystem", "reason"),
    [("remote", "NTFS", "drive_remote"), ("removable", "NTFS", "drive_removable"),
     ("cdrom", "UDF", "drive_cdrom"), ("fixed", "FAT32", "filesystem_not_ntfs"),
     ("ramdisk", "NTFS", "drive_ramdisk")],
)
def test_non_local_or_non_ntfs_volume_is_formal_ineligible(tmp_path, monkeypatch, drive, filesystem, reason) -> None:
    """network・removable・non-NTFS の volume は formal-ineligible verdict になります。

    volume 照会だけを差し替え、ledger directory が作られないことも確認します。
    """
    if os.name != "nt":
        assert check_storage(tmp_path).reasons == ("platform_not_windows",)
        return
    monkeypatch.setattr(dls, "_volume_facts", lambda _path: {
        "volume_root": "Z:\\", "drive_type": drive, "filesystem": filesystem, "volume_serial": 1})
    verdict = check_storage(tmp_path / "ledger")
    assert verdict.eligible is False and reason in verdict.reasons
    with pytest.raises(UnsupportedStorageError, match=reason):
        DurableLaunchStore(tmp_path / "ledger")
    assert not (tmp_path / "ledger").exists()


def test_volume_query_failure_is_formal_ineligible(tmp_path, monkeypatch) -> None:
    """volume 情報を OS から読めない場合も fail closed です。

    照会失敗を eligible と誤認しないことを確認します。
    """
    def _fail(_path: str) -> dict:
        raise OSError(5, "denied")

    monkeypatch.setattr(dls, "_volume_facts", _fail)
    verdict = check_storage(tmp_path)
    assert verdict.eligible is False


def test_existing_directory_with_inherited_acl_is_rejected(tmp_path) -> None:
    """current-user 専用でない既存 directory は ACL read-back で拒否されます。

    既存 ACL を黙って書き換えず、formal-ineligible として止めます。
    """
    if os.name != "nt":
        assert check_storage(tmp_path).eligible is False
        return
    with pytest.raises(UnsupportedStorageError):
        DurableLaunchStore(tmp_path)


def test_acl_findings_reject_foreign_ace_null_dacl_and_unprotected() -> None:
    """ACL 検査は foreign ACE・NULL DACL・継承有効 DACL・owner 不一致を全部報告します。

    SDDL 文字列だけを入力にする純粋な検査です。
    """
    sid = "S-1-5-21-1-2-3-1001"
    assert acl_findings(f"O:{sid}D:P(A;OICI;FA;;;{sid})", sid, require_owner=True, require_protected=True) == []
    assert "foreign_ace:A;OICI;FA;;;WD" in acl_findings(
        f"O:{sid}D:P(A;OICI;FA;;;{sid})(A;OICI;FA;;;WD)", sid, require_owner=True, require_protected=True)
    assert "dacl_missing_or_empty" in acl_findings(f"O:{sid}D:NO_ACCESS_CONTROL", sid,
                                                   require_owner=True, require_protected=True)
    assert "dacl_not_protected" in acl_findings(f"O:{sid}D:AI(A;ID;FA;;;{sid})", sid,
                                                require_owner=False, require_protected=True)
    assert "owner_not_current_user" in acl_findings(f"O:BAD:P(A;;FA;;;{sid})", sid,
                                                    require_owner=True, require_protected=True)
    assert "foreign_ace:D;;FA;;;S-1-5-21-9" in acl_findings(f"D:P(A;;FA;;;{sid})(D;;FA;;;S-1-5-21-9)", sid,
                                                            require_owner=False, require_protected=True)


@pytest.mark.parametrize("statement", ["PRAGMA journal_mode=WAL", "PRAGMA synchronous"])
def test_pragma_read_back_failure_fails_closed(tmp_path, monkeypatch, statement) -> None:
    """SQLite が WAL / FULL を採用しなかった場合は ledger を開きません。

    read-back 値だけを差し替え、LedgerIntegrityError で止まることを確認します。
    """
    if os.name != "nt":
        assert check_storage(tmp_path).eligible is False
        return
    original = dls._read_pragma
    monkeypatch.setattr(dls, "_read_pragma", lambda conn, sql: (
        "delete" if sql == statement and "journal" in sql else 1 if sql == statement else original(conn, sql)))
    with pytest.raises(LedgerIntegrityError, match="PRAGMA read-back mismatch"):
        DurableLaunchStore(tmp_path / "ledger")


def test_schema_version_mismatch_fails_closed(store) -> None:
    """未知の schema version の ledger は開けません。

    version を書き換えた DB を再度開くと LedgerIntegrityError です。
    """
    store.close()
    raw = sqlite3.connect(store.path)
    raw.execute("PRAGMA user_version=2")
    raw.close()
    with pytest.raises(LedgerIntegrityError, match="unsupported ledger schema"):
        DurableLaunchStore(store.directory)
    store._conn = sqlite3.connect(":memory:")


def test_full_protocol_activates_once_and_matches_0602_events(store) -> None:
    """正規順序で confirm まで進んだ launch だけが activation でき、2回目は冪等です。

    ledger 由来 event 列は 06-02 validator を通り、activation row は1件だけです。
    """
    _advance(store, "attempt-0", LaunchStage.PROCESS_LAUNCH_CONFIRMED)
    assert store.activate("attempt-0", "normal", _probe(ProbeResult.ALIVE)) is True
    assert store.activate("attempt-0", "normal", _probe(ProbeResult.ALIVE)) is False
    assert _count(store, LaunchStage.FORMAL_RUN_ACTIVATED) == 1
    events = _validated_0602(store, "attempt-0")
    assert [event.event_type for event in events] == [
        EventType.LAUNCH_INTENT_COMMITTED, EventType.BROKER_PROCESS_ATTESTED,
        EventType.PROCESS_LAUNCH_CONFIRMED, EventType.FORMAL_RUN_ACTIVATED]
    assert all(event.campaign_manifest_hash == MANIFEST for event in events)


@pytest.mark.parametrize(
    ("reached", "mutation"),
    [
        (LaunchStage.LAUNCH_INTENT, "resume"),
        (LaunchStage.LAUNCH_INTENT, "confirm"),
        (LaunchStage.LAUNCH_INTENT, "intent"),
        (LaunchStage.PROCESS_ATTESTED, "attest"),
        (LaunchStage.PROCESS_ATTESTED, "confirm"),
        (LaunchStage.RESUME_INTENT, "resume"),
        (LaunchStage.RESUME_INTENT, "gate_failed"),
        (LaunchStage.RESUME_INTENT, "activate"),
        (LaunchStage.PROCESS_LAUNCH_CONFIRMED, "confirm"),
        (LaunchStage.PROCESS_LAUNCH_CONFIRMED, "gate_failed"),
        (LaunchStage.FORMAL_RUN_ACTIVATED, "uncertain"),
    ],
)
def test_duplicate_and_out_of_order_mutations_are_rejected(store, reached, mutation) -> None:
    """重複・順序外の mutation は LedgerOrderError で拒否され row が増えません。

    resume intent 以降の gate failure(no-process 主張)も順序違反として拒否します。
    """
    attestation = _advance(store, "attempt-0", reached)
    ref = attestation.identity.process_ref
    before = store.history("attempt-0").stages
    actions = {
        "intent": lambda: store.commit_intent(_intent(0, "attempt-0")),
        "attest": lambda: store.commit_attestation("attempt-0", attestation),
        "resume": lambda: store.commit_resume_intent("attempt-0", ref),
        "confirm": lambda: store.commit_confirmation("attempt-0", ref),
        "gate_failed": lambda: store.commit_gate_failure("attempt-0", "late"),
        "uncertain": lambda: store.commit_uncertain("attempt-0", "late"),
        "activate": lambda: store.activate("attempt-0", "normal", _probe(ProbeResult.ALIVE)),
    }
    with pytest.raises((LedgerOrderError, LedgerIdentityError)):
        actions[mutation]()
    assert store.history("attempt-0").stages == before


def test_terminal_rows_block_every_further_mutation(store) -> None:
    """gate failure / uncertain の後はどの段階も append できません。

    terminal 後の resume・confirm・activation を全部拒否します。
    """
    attestation = _advance(store, "attempt-0", LaunchStage.PROCESS_ATTESTED)
    store.commit_gate_failure("attempt-0", "attestation failed")
    for action in (
        lambda: store.commit_resume_intent("attempt-0", attestation.identity.process_ref),
        lambda: store.commit_uncertain("attempt-0", "x"),
        lambda: store.commit_gate_failure("attempt-0", "x"),
    ):
        with pytest.raises(LedgerOrderError):
            action()
    with pytest.raises(LedgerOrderError):
        store.activate("attempt-0", "normal", _probe(ProbeResult.ALIVE))


def test_resume_or_confirm_for_a_different_process_is_rejected(store) -> None:
    """attested process と違う process_ref の resume / confirm は束縛違反です。

    identity 未証明の process を confirm しないことを確認します。
    """
    _advance(store, "attempt-0", LaunchStage.PROCESS_ATTESTED)
    with pytest.raises(LedgerIdentityError):
        store.commit_resume_intent("attempt-0", "pid:1:ct:2")
    assert store.history("attempt-0").last_stage is LaunchStage.PROCESS_ATTESTED


@pytest.mark.parametrize("field", ["slot_id", "reserved_run_id", "gameplay_attempt_id", "launch_nonce"])
def test_replacement_launch_is_rejected(store, field) -> None:
    """同じ slot・予約 id・nonce を再利用する replacement launch は0件です。

    元 launch が terminal になった後でも、別 attempt の intent は拒否されます。
    """
    first = _intent(0, "attempt-0")
    store.commit_intent(first)
    store.commit_gate_failure("attempt-0", "gate")
    replacement = _intent(1, "attempt-1", **{field: getattr(first, field)})
    with pytest.raises(LedgerIdentityError, match="duplicate"):
        store.commit_intent(replacement)
    assert store.attempt_ids() == ("attempt-0",)


def test_pid_reuse_substitution_is_rejected(store) -> None:
    """同じ PID+create time を別 attempt へ attest する差し替えを拒否します。

    confirm 済みでも probe が PID_REUSED なら activation できず、uncertain に分類されます。
    """
    attestation = _advance(store, "attempt-0", LaunchStage.PROCESS_LAUNCH_CONFIRMED)
    store.commit_intent(_intent(1, "attempt-1"))
    with pytest.raises(LedgerIdentityError, match="duplicate process_ref"):
        store.commit_attestation("attempt-1", ProcessAttestation(attestation.identity, "job-other", "pid:1:ct:1"))
    with pytest.raises(LedgerIdentityError, match="pid_reused"):
        store.activate("attempt-0", "normal", _probe(ProbeResult.PID_REUSED))
    result = store.reconcile("attempt-0", _probe(ProbeResult.PID_REUSED))
    assert result.classification is Classification.UNCERTAIN and result.reason == "confirmed_process_pid_reused"
    assert _count(store, LaunchStage.FORMAL_RUN_ACTIVATED) == 0
    _validated_0602(store, "attempt-0")


@pytest.mark.parametrize(
    ("reached", "probe", "expected", "event"),
    [
        (LaunchStage.LAUNCH_INTENT, ProbeResult.UNKNOWN, Classification.PROVEN_NO_PROCESS, EventType.LAUNCH_GATE_FAILED),
        (LaunchStage.PROCESS_ATTESTED, ProbeResult.ALIVE, Classification.PROVEN_NO_PROCESS,
         EventType.LAUNCH_GATE_FAILED),
        (LaunchStage.RESUME_INTENT, ProbeResult.ALIVE, Classification.UNCERTAIN, EventType.LAUNCH_UNCERTAIN),
        (LaunchStage.PROCESS_LAUNCH_CONFIRMED, ProbeResult.ALIVE, Classification.CONFIRMED,
         EventType.PROCESS_LAUNCH_CONFIRMED),
        (LaunchStage.PROCESS_LAUNCH_CONFIRMED, ProbeResult.EXITED, Classification.UNCERTAIN,
         EventType.LAUNCH_UNCERTAIN),
        (LaunchStage.PROCESS_LAUNCH_CONFIRMED, ProbeResult.UNKNOWN, Classification.UNCERTAIN,
         EventType.LAUNCH_UNCERTAIN),
        (LaunchStage.FORMAL_RUN_ACTIVATED, ProbeResult.ALIVE, Classification.CONFIRMED,
         EventType.PROCESS_LAUNCH_CONFIRMED),
    ],
)
def test_reconcile_classifies_each_boundary_like_0602(store, reached, probe, expected, event) -> None:
    """各境界の crash を3分類し、再実行しても同じ結果で row が増えません。

    分類に対応する 06-02 event 種別と、ledger 由来 event 列の lifecycle 妥当性も確認します。
    """
    _advance(store, "attempt-0", reached)
    first = store.reconcile("attempt-0", _probe(probe))
    rows = store.history("attempt-0").stages
    again = store.reconcile("attempt-0", _probe(probe))
    assert first.classification is expected and again.classification is expected
    assert store.history("attempt-0").stages == rows
    assert first.event_type is event
    if expected is Classification.CONFIRMED and not first.activated:
        assert store.activate("attempt-0", "reconciliation", _probe(ProbeResult.ALIVE)) is True
        assert store.reconcile("attempt-0", _probe(ProbeResult.ALIVE)).activated is True
    if expected is not Classification.CONFIRMED:
        with pytest.raises(LedgerOrderError):
            store.activate("attempt-0", "reconciliation", _probe(ProbeResult.ALIVE))
    assert _count(store, LaunchStage.FORMAL_RUN_ACTIVATED) <= 1
    _validated_0602(store, "attempt-0")


def test_append_only_triggers_block_update_and_delete(store) -> None:
    """SQL で直接 UPDATE / DELETE しても trigger が拒否します。

    ledger row・identity claim の書換えで replacement を作れないことを確認します。
    """
    _advance(store, "attempt-0", LaunchStage.PROCESS_ATTESTED)
    for sql in ("UPDATE launch_rows SET stage='X'", "DELETE FROM launch_rows",
                "DELETE FROM identity_claims", "UPDATE ledger_meta SET value='x'"):
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            store._conn.execute(sql)


def test_disk_full_rolls_back_the_whole_transaction(store) -> None:
    """disk full(SQLITE_FULL)では commit 失敗になり、row も identity claim も残りません。

    max_page_count で DB を満杯にし、失敗した attempt の slot を後で再利用できることを確認します。
    """
    pages = store._conn.execute("PRAGMA page_count").fetchone()[0]
    store._conn.execute(f"PRAGMA max_page_count={pages}")
    failed = None
    for slot in range(500):
        try:
            store.commit_intent(_intent(slot, f"attempt-{slot}", argv=(sys.executable, "-c", "x" * 400)))
        except LedgerCommitError as exc:
            assert "full" in str(exc)
            failed = slot
            break
    assert failed is not None
    assert f"attempt-{failed}" not in store.attempt_ids()
    claims = store._conn.execute("SELECT COUNT(*) FROM identity_claims WHERE attempt_id=?",
                                 (f"attempt-{failed}",)).fetchone()[0]
    assert claims == 0
    store._conn.execute(f"PRAGMA max_page_count={pages * 100}")
    store.commit_intent(_intent(failed, f"attempt-{failed}"))


def test_commit_failure_leaves_no_row_after_reopen(store, monkeypatch) -> None:
    """COMMIT が失敗した段階は rollback され、再 open 後も存在しません。

    attestation commit を失敗させ、ledger が intent のまま残ることを確認します。
    """
    _advance(store, "attempt-0", LaunchStage.LAUNCH_INTENT)

    def _fail() -> None:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "_commit", _fail)
    with pytest.raises(LedgerCommitError, match="disk I/O error"):
        store.commit_attestation("attempt-0", _attestation())
    store.close()
    reopened = DurableLaunchStore(store.directory)
    try:
        assert reopened.history("attempt-0").stages == (LaunchStage.LAUNCH_INTENT,)
        assert reopened.reconcile("attempt-0", _probe(ProbeResult.ALIVE)).classification is (
            Classification.PROVEN_NO_PROCESS)
    finally:
        reopened.close()
    store._conn = sqlite3.connect(":memory:")


@pytest.mark.parametrize("target", ["header", "launch_rows_page"])
def test_database_corruption_fails_closed_on_open(store, target) -> None:
    """DB file の破損は open 時の read-back / integrity_check で拒否されます。

    header 破損と launch_rows の B-tree page 破損の両方を再現します。
    """
    _advance(store, "attempt-0", LaunchStage.PROCESS_LAUNCH_CONFIRMED)
    root = store._conn.execute("SELECT rootpage FROM sqlite_master WHERE name='launch_rows'").fetchone()[0]
    size = store._conn.execute("PRAGMA page_size").fetchone()[0]
    offset = 0 if target == "header" else (root - 1) * size
    store.close()
    with open(store.path, "r+b") as stream:
        stream.seek(offset)
        stream.write(os.urandom(512))
    with pytest.raises(LedgerIntegrityError):
        DurableLaunchStore(store.directory)
    store._conn = sqlite3.connect(":memory:")


@pytest.mark.parametrize("tamper", ["missing_key", "broken_chain", "non_canonical"])
def test_partial_or_tampered_row_fails_closed(store, tamper) -> None:
    """API を通らない partial row・chain 断絶・非 canonical payload を読み戻しで拒否します。

    history・reconcile・次の append がすべて LedgerIntegrityError で止まります。
    """
    attestation = _advance(store, "attempt-0", LaunchStage.PROCESS_ATTESTED)
    prev = store.history("attempt-0").row_hashes[-1]
    body = '{"process_ref":"%s"}' % attestation.identity.process_ref
    if tamper == "missing_key":
        body, row_hash = "{}", dls._row_hash("attempt-0", "RESUME_INTENT", "{}", prev)
    elif tamper == "broken_chain":
        row_hash = dls._row_hash("attempt-0", "RESUME_INTENT", body, "f" * 64)
        prev = "f" * 64
    else:
        body = '{ "process_ref": "%s" }' % attestation.identity.process_ref
        row_hash = dls._row_hash("attempt-0", "RESUME_INTENT", body, prev)
    store._conn.execute(
        "INSERT INTO launch_rows(attempt_id, stage, payload, prev_hash, row_hash) VALUES(?, ?, ?, ?, ?)",
        ("attempt-0", "RESUME_INTENT", body, prev, row_hash))
    with pytest.raises(LedgerIntegrityError):
        store.history("attempt-0")
    with pytest.raises(LedgerIntegrityError):
        store.reconcile("attempt-0", _probe(ProbeResult.ALIVE))
    with pytest.raises(LedgerIntegrityError):
        store.commit_confirmation("attempt-0", attestation.identity.process_ref)


def test_concurrent_stores_serialize_and_activate_once(store) -> None:
    """別接続の ledger から同時に進めても1段階は1回しか記録されません。

    2つ目の接続の重複 attestation は拒否され、activation も片方だけが True です。
    """
    other = DurableLaunchStore(store.directory)
    try:
        attestation = _advance(store, "attempt-0", LaunchStage.PROCESS_ATTESTED)
        with pytest.raises(LedgerOrderError):
            other.commit_attestation("attempt-0", attestation)
        other.commit_resume_intent("attempt-0", attestation.identity.process_ref)
        store.commit_confirmation("attempt-0", attestation.identity.process_ref)
        results = [store.activate("attempt-0", "normal", _probe(ProbeResult.ALIVE)),
                   other.activate("attempt-0", "normal", _probe(ProbeResult.ALIVE))]
        assert sorted(results) == [False, True]
        assert _count(store, LaunchStage.FORMAL_RUN_ACTIVATED) == 1
    finally:
        other.close()


def test_intent_rejects_malformed_fields() -> None:
    """intent の nonce 長・argv と executable の不一致・相対 path を拒否します。

    256-bit 未満の nonce は形式違反です。
    """
    with pytest.raises(ValueError):
        _intent(launch_nonce="ab" * 16)
    with pytest.raises(ValueError):
        _intent(argv=("other.exe",))
    with pytest.raises(ValueError):
        _intent(executable_path="python.exe", argv=("python.exe",))
    with pytest.raises(ValueError):
        ProcessIdentity(0, 1, "x", 0, 0)

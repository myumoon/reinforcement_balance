# Survivors campaign durable launch(Win64)

## 目的と範囲

06-03 は、process 起動の途中で controller や broker が crash したときに「起動したのか・していないのか」が分からなくなる隙間(process-start crash gap)を閉じる。SQLite の durable ledger と、別 process の Win32 launch broker で構成する。

- 実装: `Tools/Deployment/survivors/campaign/durable_launch_store.py`、`Tools/Deployment/survivors/campaign/win32_launch_broker.py`
- テスト: `Tools/Deployment/tests/campaign/test_durable_launch_store.py`、`test_win32_launch_broker.py`、`test_win32_launch_recovery_integration.py`
- 検証は harmless な target helper(`python.exe -c ...`。resume 後に marker file へ nonce を書いて sleep するだけ)で行う。本家 game の起動や formal campaign の開始・activation はこの PR の範囲外であり、06-04 が担当する。campaign の意味論・分母・report は 06-02 の担当で、この PR では変更しない。

## Support envelope(formal 適格条件)

ledger を開くたびに、次の条件をすべて OS / SQLite から読み戻して確認する。1つでも満たさなければ `UnsupportedStorageError` または `LedgerIntegrityError` を送出して止める(fail closed)。

| 検査 | 合格条件 | 不合格時の扱い |
|---|---|---|
| path 形式 | UNC・`\\?\`・`\\.\` ではない(symlink / junction は実体へ解決してから判定) | formal-ineligible(`unc_path`) |
| volume | `GetDriveTypeW` が fixed(network・removable・CD-ROM・RAM disk は不可) | formal-ineligible(`drive_<種別>`) |
| filesystem | `GetVolumeInformationW` が `NTFS` | formal-ineligible(`filesystem_not_ntfs`) |
| OS | Windows(`os.name == "nt"`) | formal-ineligible(`platform_not_windows`) |
| ACL | ledger directory の owner が current user、DACL が protected で ACE は current user の allow のみ。DB / WAL / SHM file も current user の ACE のみ | formal-ineligible(`owner_not_current_user` など) |
| SQLite | `journal_mode=WAL` と `synchronous=FULL(2)` を read-back で確認 | `LedgerIntegrityError` |
| integrity | `PRAGMA integrity_check` が `ok` | `LedgerIntegrityError` |
| schema | `user_version=1` かつ `ledger_meta.schema = survivors.durable_launch.v1` | `LedgerIntegrityError` |

新しい ledger directory は store が作成し、current user だけの protected DACL(`D:P(A;OICI;FA;;;<SID>)`)を付ける。既存 directory の ACL は書き換えずに検査だけ行う。support envelope 外の環境は、テストで skip せず formal-ineligible verdict として記録する(`StorageVerdict.label == "formal_ineligible"`)。

## Ledger 構造

- `launch_rows`: 1 row = 1 段階。`(attempt_id, stage)` は UNIQUE。payload は canonical JSON。`row_hash = sha256(canonical{attempt_id, stage, payload, prev_hash})` で attempt ごとの hash chain を作る。
- `identity_claims`: `(kind, value)` が PRIMARY KEY。intent 時に `slot`(`<manifest_hash>:<slot_id>`)・`reserved_run_id`・`gameplay_attempt_id`・`launch_nonce` を、attestation 時に `process_ref`・`job_ref` を claim する。
- 3 table すべてに UPDATE / DELETE を拒否する trigger を付けており、append-only である。
- 書込みは 1 段階につき 1 回の `BEGIN IMMEDIATE` transaction で行う。まず既存履歴を検証し、次に遷移を検査し、claim と row を insert して commit する。途中で失敗した場合は全体を rollback する。
- 読み込み時は全 row について、canonical payload・hash chain・遷移・key の完全一致・process_ref の束縛を再検証する。1つでも崩れていれば `LedgerIntegrityError` を送出する。

## Durable launch プロトコル

```text
controller                       ledger (SQLite WAL/FULL)            broker (別 process)
  | 1. storage / PRAGMA / ACL read-back
  | 2. LAUNCH_INTENT commit ------>|
  | 3. launch request (pipe) ---------------------------------------->| intent read-back + hash 照合
  |                                |                                  | executable を書込み拒否で保持し hash 照合
  |                                |                                  | CreateProcessW(CREATE_SUSPENDED, JOB_LIST)
  |                                |<----- PROCESS_ATTESTED commit ---| kernel identity attest
  |                                |<----- RESUME_INTENT commit ------|
  |                                |                                  | ResumeThread
  |                                |<----- PROCESS_LAUNCH_CONFIRMED --| 同じ identity の生存確認
  |<-------------------------------------------------- launch_result -|
  | 4. activate(confirmed のみ) -->| FORMAL_RUN_ACTIVATED(冪等)
```

段階遷移(`_NEXT`)は次のとおり。これ以外の遷移、同じ段階の重複、terminal 後の append はすべて `LedgerOrderError` になる。

| 現在の段階 | 許される次の段階 |
|---|---|
| (なし) | `LAUNCH_INTENT` |
| `LAUNCH_INTENT` | `PROCESS_ATTESTED` / `LAUNCH_GATE_FAILED` |
| `PROCESS_ATTESTED` | `RESUME_INTENT` / `LAUNCH_GATE_FAILED` |
| `RESUME_INTENT` | `PROCESS_LAUNCH_CONFIRMED` / `LAUNCH_UNCERTAIN` |
| `PROCESS_LAUNCH_CONFIRMED` | `FORMAL_RUN_ACTIVATED` / `LAUNCH_UNCERTAIN` |
| `FORMAL_RUN_ACTIVATED` / `LAUNCH_GATE_FAILED` / `LAUNCH_UNCERTAIN` | なし(以後の outcome は 06-04 の責務) |

中心となる不変条件は次の2つである。

- broker は `RESUME_INTENT` の commit が成功した後にしか `ResumeThread` を呼ばない。
- `LAUNCH_GATE_FAILED`(no-process)は `RESUME_INTENT` より前にしか書けない。

したがって ledger に `RESUME_INTENT` が無い attempt の target は、一度も user-mode code を実行していない。reconciler が先に terminal row を commit した場合、broker の resume intent commit は拒否され、broker は resume できない。ledger の直列化そのものが、この排他を保証する。

### Kernel identity attestation

process 環境変数の nonce(`REINBALANCE_LAUNCH_NONCE`)は target に渡すだけで、attestation には使わない。attestation には次の kernel 値を使う。

- PID と create time(`GetProcessTimes`)。`process_ref = "pid:<pid>:ct:<create_time>"` とする。
- image path(`QueryFullProcessImageNameW`)と image file identity(volume serial + file index)。これらは broker が書込み拒否(`FILE_SHARE_READ` のみ)で保持している executable の file identity と一致しなければならない。hash 照合から起動までの間に executable を差し替えることはできない。
- `IsProcessInJob` で broker の Job に所属していること。
- initial thread の suspend count が 1 であること(一度も実行されていない)。

Job は `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` 付きで作り、`PROC_THREAD_ATTRIBUTE_JOB_LIST` で process の生成と同時に所属させる。生成から Job 所属までの隙間は無い。broker が kill されると Job handle が閉じ、target も OS によって終了する。

### Fail-closed 境界(broker)

| 失敗箇所 | broker の動作 | ledger の結果 |
|---|---|---|
| request の key / schema / digest 不正 | 何もしない | 変化なし(`rejected`) |
| nonce / config / executable / command hash の read-back 不一致 | 何もしない | 変化なし(`rejected`) |
| attempt が `LAUNCH_INTENT` 以外(重複・replacement request) | 何もしない | 変化なし(`rejected`) |
| executable を開けない / disk 上の hash 不一致 | process を作らない | `LAUNCH_GATE_FAILED` |
| CreateProcess 失敗 / attestation 失敗 / attestation commit 失敗 | Job ごと terminate して終了を待つ | `LAUNCH_GATE_FAILED` |
| resume intent commit 失敗 | resume せずに terminate | `LAUNCH_GATE_FAILED`(commit 自体が曖昧で `RESUME_INTENT` が残った場合は `LAUNCH_UNCERTAIN`) |
| ResumeThread 失敗 / resume 後の生存確認失敗 / confirm commit 失敗 | Job ごと terminate | `LAUNCH_UNCERTAIN` |

terminal row の commit 自体が失敗した場合も、broker は `reconcile` で ledger 上の段階から分類し直し、その結果を返す。`confirmed` を返すのは confirm commit が成功したときだけである。

## Reconciliation と 06-02 event 契約との対応

`DurableLaunchStore.reconcile(attempt_id, probe)`(Win32 環境では `win32_launch_broker.reconcile_ledger(store)`)は、crash 後の attempt を次の3分類に確定する。必要な場合は terminal row を append する。

| ledger の最終段階 | probe | 分類 | 06-02 `EventType` | activation |
|---|---|---|---|---|
| `LAUNCH_INTENT` / `PROCESS_ATTESTED` | 不要 | proven-no-process | `LAUNCH_GATE_FAILED` | 不可(launch-gate failure) |
| `RESUME_INTENT` | 不要 | uncertain | `LAUNCH_UNCERTAIN` | 不可(campaign block) |
| `PROCESS_LAUNCH_CONFIRMED` | `ALIVE` | confirmed | `PROCESS_LAUNCH_CONFIRMED` | 可(`activation_source=reconciliation`) |
| `PROCESS_LAUNCH_CONFIRMED` | `EXITED` / `PID_REUSED` / `UNKNOWN` | uncertain | `LAUNCH_UNCERTAIN` | 不可 |
| `FORMAL_RUN_ACTIVATED` | 不要 | confirmed(activated) | `PROCESS_LAUNCH_CONFIRMED` | 済み(冪等) |
| `LAUNCH_GATE_FAILED` / `LAUNCH_UNCERTAIN` | 不要 | 記録済みの分類 | 同名 | 不可 |

`probe_identity` は PID を開き、create time・image path・file identity がすべて attestation と一致し、かつ未終了のときだけ `ALIVE` を返す。PID が再利用されて別の process を指している場合は `PID_REUSED` になる。

`store.campaign_events(attempt_id)` は ledger の履歴を 06-02 の `CampaignEvent` 列へ写す。対応は `LAUNCH_INTENT → LAUNCH_INTENT_COMMITTED`、`PROCESS_ATTESTED → BROKER_PROCESS_ATTESTED`、`PROCESS_LAUNCH_CONFIRMED`、`FORMAL_RUN_ACTIVATED`、`LAUNCH_GATE_FAILED`、`LAUNCH_UNCERTAIN` である。`RESUME_INTENT` は ledger 内部の境界なので event にしない。どの event にも intent に束縛された `campaign_manifest_hash` を付ける。テストでは `FORMAL_SLOT_RESERVED`・`ATTEMPT_PREFLIGHT` を前置し、`validate_campaign_events` を通ることを全分類で確認している。06-02 は confirmed の後の `LAUNCH_GATE_FAILED` も許すが、本 ledger は resume 後の no-process 主張を禁止している。つまり本 ledger が出す event 列は、06-02 が受理する列の部分集合である。

## 後続(06-04)向けの公開 API

### `durable_launch_store.py`

| API | 意味 |
|---|---|
| `DurableLaunchStore(directory)` | support envelope・ACL・PRAGMA・integrity・schema を検査して開く。失敗時は `UnsupportedStorageError` / `LedgerIntegrityError` |
| `LaunchIntent(...)` / `new_launch_nonce()` | intent を作る。nonce は 256-bit(64 桁 hex)。`argv[0] == executable_path`。`command_hash = canonical_hash(argv)` |
| `commit_intent(intent)` | `LAUNCH_INTENT`。slot・reserved run id・gameplay attempt id・nonce の再利用は `LedgerIdentityError` |
| `commit_attestation(attempt_id, ProcessAttestation)` | `PROCESS_ATTESTED`。同じ `process_ref` / `job_ref` を別の attempt で使うと `LedgerIdentityError` |
| `commit_resume_intent(attempt_id, process_ref)` | `RESUME_INTENT`。attested 以外の process_ref は `LedgerIdentityError` |
| `commit_confirmation(attempt_id, process_ref)` | `PROCESS_LAUNCH_CONFIRMED` |
| `commit_gate_failure(attempt_id, reason)` / `commit_uncertain(attempt_id, reason)` | terminal row |
| `activate(attempt_id, source, probe) -> bool` | confirmed かつ probe が `ALIVE` のときだけ append して `True`。activation 済みなら `False`(row は増えない)。それ以外は例外 |
| `reconcile(attempt_id, probe)` / `reconcile_all(probe)` | 上表の分類(`Reconciliation.classification` / `.event_type`)を返す |
| `history(attempt_id)` / `campaign_events(attempt_id)` | 検証済み履歴 / 06-02 event 列 |

例外の型: `LedgerOrderError`(重複・順序外)、`LedgerIdentityError`(一意性・束縛)、`LedgerCommitError`(disk full など。rollback 済み)、`LedgerIntegrityError`(破損)、`UnsupportedStorageError`(formal-ineligible)。どれも `LedgerError` のサブクラスで、捕まえた場合は activation してはならない。

### `win32_launch_broker.py`

- 起動: `python -m survivors.campaign.win32_launch_broker --ledger <dir> --pipe \\.\pipe\<name>`。ledger を開けない場合は非0で終了し、pipe も作らない。
- pipe: `FILE_FLAG_FIRST_PIPE_INSTANCE`・`PIPE_REJECT_REMOTE_CLIENTS`・protected DACL `D:P(A;;GA;;;<current SID>)` で作る。接続ごとに client process の token SID を照合し、一致しなければ request を読まずに拒否する。
- frame: 4 byte little-endian の長さ + UTF-8 JSON(上限 64 KiB)。1 接続につき 1 request。
- request(`build_launch_request(intent)`): `{"kind":"launch","schema":"survivors.launch_broker.v1","attempt_id","launch_nonce","config_hash","executable_hash","command_hash"}`。key は完全一致でなければならない。
- response: `{"kind":"launch_result","attempt_id","status","process_ref","job_ref","failure_reason"}`。`status` は `confirmed` / `proven_no_process` / `uncertain` / `rejected` のいずれか。不正な request には `{"kind":"rejected","error"}` を返す。
- shutdown: `{"kind":"shutdown"}` を受けると `{"kind":"shutdown_ack"}` を返して終了する。このとき confirm 済みの target も Job close で終了する。
- client: `request_broker(pipe, message, broker_pid=...)` は、pipe の server PID が起動した broker の PID と一致し、その SID が current user であることを確認してから送信する。
- `integration_environment(store)`: OS・filesystem facts・storage verdict・SQLite の read-back・Python version・build hash(broker / store / campaign_schema の SHA-256)を返す。
- `--test-fault-hold-at <境界> --test-fault-dir <dir>`: crash 境界テスト専用のオプション。指定した境界で `<境界>.reached` を書いて停止し、`release` file を待つ。formal run では指定しない。

## テストと PR 受け入れ条件

- `test_durable_launch_store.py` は次を確認する: 実 NTFS 上の WAL/FULL・integrity・schema・ACL の read-back。UNC / network / removable / non-NTFS / volume 照会失敗 / 既存 ACL の formal-ineligible 判定。PRAGMA read-back 失敗。重複・順序外 mutation。replacement launch。PID reuse による差し替え。各境界の reconcile と 06-02 event 検証。append-only trigger。fault injection(disk full = `max_page_count`、COMMIT 失敗、DB header / B-tree page の破損、partial row / chain 断絶 / 非 canonical payload)。
- `test_win32_launch_broker.py` は次を確認する: 厳格な request 形式。hash read-back 不一致時に ledger が変化しないこと。disk 上の executable 不一致。正常 launch の kernel identity・resume marker・Job close による終了。PID reuse の検出。attestation 失敗時と resume intent commit 失敗時に helper が terminate され resume されないこと。resume 後の失敗が uncertain になること。pipe の server PID / client SID の照合。
- `test_win32_launch_recovery_integration.py` は、controller・broker・target helper を実 process で動かす。intent commit 後、CreateProcess 前後、attestation 前後、resume 前後、confirm 前後の各境界で、controller と broker の同時 kill・broker のみの kill・controller のみの kill をテスト process から行う。そのうえで次を確認する: reconcile の分類、resume marker の有無、kill-on-job-close による helper の終了、replacement launch 0件、二重 activation 0件(activation row は最大1)、PID reuse の検出、再起動した broker への再 launch request の拒否、06-02 event の妥当性。結果は `integration_result.json` に OS・filesystem・SQLite・build hash と共に保存する。support envelope 外の環境では skip せず、formal-ineligible verdict と ledger を開けないことを記録する。
- 受け入れ条件: 本家 game なしで、intent から confirmation・reconciliation までの全 crash 境界を helper process で検証できること。分類が 06-02 event 契約と一致すること。unsupported storage・identity 未証明・同時 crash のいずれでも silent replacement や二重 activation が起きないこと。
- 実行コマンド: `bash Tools/run-pytest.sh Tools/Deployment/tests -q -rs`。Win32 のテストは Windows の Python で実行する。WSL native の Python では formal-ineligible(`platform_not_windows`)と判定される。

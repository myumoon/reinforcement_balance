# Survivors controller telemetry/shadow runbook (05-04)

`Tools/Deployment/run_survivors_controller.py`(controller 起動 CLI)、
`Tools/Deployment/compute_controller_gate.py`(M11 shadow gate CLI)、
`Tools/Deployment/replay_controller_fixture.py`(development fixture replay tool)を
使う運用手順です。shadow/live の起動、gate の判定、health thresholds、telemetry の
保管、成果物の backup/restore を扱います。閾値や終了コードはすべて実装側の定数が
唯一の正であり、本ドキュメントはそれを転記したものです(食い違った場合は実装を
優先してください)。

コマンドはいずれも `Tools/Deployment` をカレントディレクトリにして実行します
(`survivors.*` パッケージを import するため)。出力先はすべて引数で指定する必須
パスで、リポジトリ内に既定の出力先はありません。以下の例では出力先を
リポジトリ外のディレクトリ(`$RUN_DIR`、例: `D:/reinbalance_controller_runs/<session>`)
に置きます。リポジトリ内に出力すると Git 管理下に telemetry が紛れ込むため、
リポジトリ外を使ってください。

## 起動手順(shadow)

shadow mode は既定のモードで、OS へ入力を一切送らない観測専用の実行です。
`--live` を渡さない限り shadow になります。

```bash
cd Tools/Deployment
python run_survivors_controller.py \
  --telemetry "$RUN_DIR/telemetry.jsonl" \
  --combat-package <combat package> \
  --detector-config <detector config yaml> \
  --class-map <class map> \
  --detector-weights <detector weights> \
  --detector-manifest <checkpoint manifest>
```

| 引数 | 必須 | 意味 |
|---|---|---|
| `--telemetry` | 必須 | 新規作成する telemetry JSONL のパス。 |
| `--combat-package` | 必須 | combat policy package(hash 検証付きで読む)。 |
| `--detector-config` / `--class-map` / `--detector-weights` / `--detector-manifest` | 必須 | world detector 一式。weights の sha256 が manifest の `model_hash` と違えば読み込まずに止まる。 |
| `--score-threshold` | 任意 | detector の score 閾値(既定 0.5)。 |
| `--session-id` | 任意 | 既定は `controller-<UTC時刻>`。 |
| `--max-frames` | 任意 | 処理する frame 数の上限(既定は無制限)。 |
| `--campaign-run-mode` | 任意 | `CampaignRunMode` の値(既定 `formal_single_attempt`)。 |
| `--target-profile` | 任意 | target profile のパス。省略時は `.env/` の確定済み profile(`load_runtime_profile()`)。 |

対象ウィンドウは `VampireSurvivors.exe` / window class `YYGameMakerYY` /
title `Vampire Survivors` で特定します。

## 起動手順(live)

live mode は実入力を送るモードで、`--live --target-profile <path> --ack-risk` の
3点がそろったときだけ受け付けます。

```bash
cd Tools/Deployment
python run_survivors_controller.py \
  --telemetry "$RUN_DIR/telemetry.jsonl" \
  <shadow と同じ必須引数> \
  --live --target-profile <target profile yaml> --ack-risk
```

| 条件 | 結果 |
|---|---|
| `--live` に `--target-profile` か `--ack-risk` が欠けている | argparse エラー(exit 2)。 |
| `--ack-risk` を `--live` なしで渡した | 誤操作として argparse エラー(exit 2)。 |
| `RuntimeBundle.assert_live_eligible()` か `CheckpointManifest.assert_formal_eligible()` が失敗(どちらかが development artifact) | capture・入力 helper を作る前に `live_gate` 行を telemetry へ書き、`EXIT_LIVE_REJECTED=5` で拒否(M10、`_live_gate()`)。 |
| 上記を通過 | 入力 helper(`InputLeaseController`)を起動し、入力監査ログを telemetry の隣に `<telemetry stem>.input_audit.jsonl` として書く。 |

- live 中の arm/disarm は CLI 自身ではなく、別プロセスの input helper が
  `Ctrl+Shift+F12` の edge で切り替えます。CLI は hotkey を監視しません。
- hold-to-run の dead-man 操作は導入されていません。押している間だけ動く
  種類の安全装置は無いため、停止は hotkey による disarm・health STOP・
  プロセス終了に頼ります。
- 現状の CLI は combat package を development bundle
  (`RuntimeBundle.from_golden_fixture`、`development_only=true` /
  `live_eligible=false`)に包んで読むため、`--live` は現時点では常に exit 5 で
  拒否されます。正式 `RuntimeBundle` の読み込み経路は formal artifact
  (03-05/04-10)がそろった後に配線する予定で、それまでは live を運用できません。

## 終了コード

1920×1080 日本語 UI の画面判定を実測配置へ修正しました。宝箱演出は約21秒なので、既定 `chest_timeout` は60秒です。「開く」は操作候補に出さず、縮んだパネルの終了ボタンだけを押します。自動で開くかは未確認のため、次回録画で一度何も押さずに待ってください。手動が必要な仕様なら60秒で停止します。

atlas がない、または card surface の template がない level-up ではカードが invalid になり、入力せず2秒で `level_up_timeout` に停止します。一時停止は入力なし・timeout なしで、再開時に段階格子の保持を持ち越しません。death／result の confirm は観測のみでクリックしません。終了は青い内側が .90 以上になるまで待ち、白字入りの定常ボタンを confidence 1.0 で返します。リロールは白字の面積が大きいため .85 以上で同じ信頼度を返します。既存 retry gate は .99 のままで、終了が初回クリック後200 ms以上残れば一度だけ再送できます。

`run_survivors_controller.py` は `SurvivorsController.run()` の戻り値をそのまま
プロセス終了コードにします(`controller.py` の `EXIT_*` 定数と
`run_survivors_controller.py` の `EXIT_LIVE_REJECTED`)。

| コード | 意味 |
|---|---|
| `0` | 正常終了。 |
| `2` | health STOP(下記 health thresholds のいずれかを超えた)。argparse の引数エラーも exit 2 になるため、stderr で区別する。 |
| `3` | 例外(controller 起動前の例外は telemetry へ `error` 行を書いてから返す)。 |
| `4` | terminal failure(`FORMAL_RUN_TERMINAL_FAILURE`)。 |
| `5` | live 拒否(正式 verdict 不在)。 |

## gate の実行

controller が書いた telemetry から M11 shadow gate の判定 JSON/Markdown を発行します。
判定ロジックは `survivors.controller.gate.issue_gate_verdict` です。

```bash
cd Tools/Deployment
python compute_controller_gate.py \
  --telemetry "$RUN_DIR/telemetry.jsonl" \
  --memory-samples "$RUN_DIR/memory_samples.json" \
  --output-json "$RUN_DIR/gate_verdict.json" \
  --output-markdown "$RUN_DIR/gate_verdict.md"
```

`--memory-samples` は `[[timestamp_ns, rss_bytes], ...]` 形式の JSON です
(2点以上、timestamp は狭義単調増加)。memory 情報は telemetry に含まれないため
別途渡す必要があり、省略すると memory gate は測定不能として FAIL になります。
`run_survivors_controller.py` 自身は memory sample を収集しないため、実ウィンドウ
実行の memory sample は運用側で別途採取してください(収集 tool は本 PR の範囲外)。

判定基準(`gate.py` の定数から引用):

| 項目 | PASS 条件 | FAIL 理由 |
|---|---|---|
| process crash | 0件(controller 例外の `error` 行が無く、`shutdown` 行があり、その `exit_code` が `0` または `2`)。`exit_code` `3`(例外・`interrupted`・`input_ack_failed`・入力解放失敗。`error` 行が無い経路も含む)・`4`(terminal failure)・未知の値は crash として数える。`2` は `health_stop` 側で FAIL する | `process_crash` |
| health stop | 0件 | `health_stop` |
| wrong-state action proposal | 0件 | `wrong_state_action_proposal` |
| level-up candidate invalid rate | `MAX_LEVEL_UP_CANDIDATE_INVALID_RATE=0.005`(0.5%)以下 | `level_up_candidate_invalid_rate_exceeded` |
| end-to-end p99 | `MAX_END_TO_END_P99_NS=110_000_000`(110ms)以下。sample が無ければ FAIL | `end_to_end_p99_exceeded` / `end_to_end_samples_unavailable` |
| memory growth | `MAX_MEMORY_GROWTH_BYTES_PER_HOUR=200_000_000`(200MB/hour、10進)未満。最小二乗の傾きで計算 | `memory_growth_exceeded` / `memory_samples_unavailable` |

CLI の終了コードは、判定を発行できたら PASS/FAIL を問わず `0`、入力ファイルの
欠損・JSON 破損・schema 不一致のときだけ `1` です。PASS/FAIL は出力 JSON の
`status` と `fail_reasons` で確認します。出力 JSON には元 telemetry の
`telemetry_sha256` が入り、どの telemetry から作った判定かを後から照合できます。
gate の出力は常に `formal_shadow_eligible=false` です。

## health thresholds

`health_monitor.py` の `HealthThresholds`(M5)の既定値です。どれか1つでも閾値を
「超えた」時点で STOP になり(等しい値は許容)、以後は STOP のまま固定されます。
STOP になると controller は入力を解放して exit 2 で終わります。

| 閾値 | 既定値 | 意味 |
|---|---|---|
| `capture_gap_ns` | 200ms | 直前 capture からの経過時間。 |
| `perception_p99_ns` | 110ms | 直近 `perception_window` 件の perception 遅延 p99。 |
| `obs_invalid_streak_ns` | 500ms | 観測(DeployObs)が無効な状態の継続時間。 |
| `unknown_streak_ns` | 1秒 | state machine が `UNKNOWN` の継続時間。 |
| `perception_window` | 300件 | p99 を計算する直近 sample 数。 |
| `perception_min_samples` | 100件 | この件数に達するまで p99 を評価しない(起動直後の1件のスパイクで止まる誤検知を防ぐ)。 |

上表とは別経路で、次の2つは閾値なしで即 STOP します。

| 条件 | 扱い |
|---|---|
| focus loss(`window_focused=False`) | 即 STOP。 |
| inference error | 1回で STOP。 |

実 capture(`CaptureSession`)は一時的な focus 喪失を例外にせず内部 pause して frame None を返します。
controller は `paused` プロパティが立ち上がった tick を検知して `HealthMonitor.record_focus_lost()` を
呼び、`focus_lost` の STOP として記録して入力解放・exit 2 で終わります。ウィンドウ消失・geometry 変化
など回復不能な状態異常は `capture_next()` が `TargetWindowStateError` を送出し、その例外を受けた
controller も同じく `record_focus_lost()` を呼びます。具体的な原因は health 行の `first_failure.detail`
(例: `target window lost foreground`)で確認します。

## telemetry 項目と未記録項目

controller が stage ごとに書く主な payload です(`controller.py`)。

| stage | 記録する項目 |
|---|---|
| `capture` | frame index・captured_ns・foreground・dropped_frames・target_profile_hash・game_build_id |
| `hud_parser` | screen_state・parser_artifact_hash と、`HudStateV1` の全 confidence/reason(`screen_state_*`・`timer_*`・`hp_*`・`xp_*`・`level_*`・`inventory_confidence`・`capability_*`) |
| `obs` | snapshot_id・frame_id・screen_state・ui_state_key・source_content_hash・obs_hash・obs_valid・obs_timestamp_ns、DeployObs の要約(`obs_invalid_count`=validity<1 の要素数・`obs_validity_mean`・`obs_age_mean`・`obs_age_max`。生配列は書かない)、UiPresentation の `ui_schema_hash`・`ui_candidate_set_hash`・`ui_inventory_hash`(snapshot を出した tick は毎回) |
| `input_release` | released と、helper の release audit と同じ時計の `release_timestamp_ns`(`time.time_ns`)・`release_monotonic_ns`(`time.monotonic_ns`) |

次の項目は、I6 で凍結された既存モジュール(`agent_runtime.py`・`input/controller.py`・
`state_machine.py` 等)の public API に出ていないため、controller からは取得できず **未記録** です。
凍結 API を変更しない限り追加できません。

- item logits
- resolver rejection reason
- equivalence metrics
- input helper PID / lease sequence / lease expiry / helper 側 ack(helper 側の ack・release は `.input_audit.jsonl` にだけ残る)

### release audit と telemetry の相関手順

telemetry の時刻(`timestamp_ns`)は既定で `time.perf_counter_ns` で、helper の
`.input_audit.jsonl` が使う `time.monotonic_ns`/`time.time_ns` とは別の時計です
(Windows では数十 ms ずれるため直接比較しない)。相関は次の手順で行います。

1. audit ファイルは `<telemetry stem>.input_audit.jsonl`(同じディレクトリ)で対応付ける。
2. telemetry の `input_release` 行の `release_monotonic_ns` と、audit の `event=="release"`・`reason=="emergency"`
   行の `release_monotonic_ns` を比べる。helper が先に時刻を取るので差は 0 以上で、
   `time.monotonic_ns` の分解能(Windows で約 15.6ms)+ release 所要時間に収まる。
   `release_timestamp_ns`(wall clock)同士でも同様に確認できる。

## telemetry retention

| 項目 | 現状 |
|---|---|
| 書き込み方式 | `TelemetryWriter` が JSONL へ streaming write(先頭に session header 1行、以後 stage ごとに追記・flush)。全 event をメモリへ保持しない(M4)。 |
| raw frames/video との分離 | raw frames/video の retention/config(`Tools/Deployment/capture_store/` 関連)と、structured telemetry の retention/config は分離されている(I2)。telemetry は `--telemetry` で指定したパスにだけ書かれる。 |
| 自動ローテーション/削除 | **実装されていない。** telemetry ファイル(および live の `.input_audit.jsonl`)は運用者が手動、または外部ジョブで管理する。 |

長時間実行では telemetry が1ファイルに追記され続けるため、ディスク容量は運用側で
監視してください。保管が必要な telemetry は、次節の手順で `ArtifactStore` へ登録
してから元ファイルを整理します。

## backup/restore

`replay_controller_fixture.py` の `run_replay()` は、telemetry・`gate_verdict.json`・
`gate_verdict.md`・`memory_samples.json` を `ArtifactStore`
(`reinbalance_survivors_contracts.artifact_store.ArtifactStore`)へ `put_bytes` し、
登録直後に `verify` で検証します(I5)。store を別の場所へ退避して復元する手順は
次の4段階です(`Tools/Deployment/tests/controller/test_shadow_artifact_backup_restore.py`
が M13 として検証している手順を運用手順にしたもの)。

| 手順 | 操作 |
|---|---|
| 1. backup | store ディレクトリ全体を別ディレクトリへコピーする(`shutil.copytree(<store>, <backup>)`)。 |
| 2. 元 store の削除 | 元の store を削除する(復元が backup だけで成立することを確認するため。運用では backup の検証が済むまで元 store を残してもよい)。 |
| 3. 開き直し | コピー先を新しい `ArtifactStore(<backup>)` で開き直す。 |
| 4. 読み戻し | 各成果物の `ArtifactRef` について `store.verify(ref, expected_size_bytes=ref.size_bytes).ok` を確認し、`store.resolve(ref.logical_id) == ref` を確認してから、`store.object_path(ref.store_uri)` の中身を読み戻す。 |

各成果物の `ArtifactRef` は replay tool の `manifest.json` の `artifacts[].ref` に
記録されています。restore 後の telemetry は、元の値と同じ stage order・choice
support・helper safety refs を再計算できることが M13 test で確認済みです。

## development_only smoke tool

`replay_controller_fixture.py` は 30分相当(27,000 tick = 30分 x 15 Hz)の virtual
schedule を fixture で controller へ流す smoke/開発用 tool です。capture・detector・
HUD parser だけを台本どおりの fake に差し替え、それ以外(tracker・obs assembler・
AgentRuntime・state machine・health monitor・telemetry)は本物を使います。

```bash
cd Tools/Deployment
python replay_controller_fixture.py --output-dir "$RUN_DIR/fixture_replay"
```

| 引数 | 既定値 | 意味 |
|---|---|---|
| `--output-dir` | 必須 | telemetry・verdict・manifest の出力先。 |
| `--ticks` | 27,000 | 処理する frame 数。短い smoke には小さい値を渡す。 |
| `--memory-sample-every` | 300 | memory sample を採る frame 間隔(virtual 20秒ごと)。 |
| `--session-id` | 乱数 | session id。 |
| `--store-root` | `<output-dir>/artifact_store` | `ArtifactStore` の root。 |
| `--seed` | 0 | development combat policy の乱数 seed。 |

終了コードは gate が PASS・restore 検証が全件成功・controller が正常終了のときだけ
`0`、それ以外は `1` です。

この tool の結果は `manifest.json` に `development_only=true`・
`formal_shadow_eligible=false` として記録されます(I4)。**正式な 60分以上の
real-window shadow 実行の代わりにはなりません。** live canary や formal shadow
verdict の根拠にも使いません。正式な real-window shadow 実行(D1)は、ユーザーが
録画/正式実行を行うまで `WAITING_MANUAL` のままです。

## 非保証範囲

- 正式 artifact に紐づく real-window shadow 60分以上の実行そのもの(D1)は本 PR の
  code scope 外です。
- telemetry の自動ローテーション/削除、実ウィンドウ実行中の memory sample 収集、
  hold-to-run の dead-man 操作は実装されていません。
- live mode は正式 `RuntimeBundle` の読み込み経路が配線されるまで常に拒否されます。

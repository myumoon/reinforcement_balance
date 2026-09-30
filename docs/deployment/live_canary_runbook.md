# Survivors live canary campaign runbook(06-04)

## 目的と範囲

06-04 は、06-02 の campaign 契約と 06-03 の durable launch broker を使って、save の退避と復元、controller の起動、stage の進行、artifact の収集をまとめて行う runner と CLI を提供する。

- 実装: `Tools/Deployment/survivors/campaign/campaign_runner.py`、`Tools/Deployment/survivors/campaign/save_lifecycle.py`、`Tools/Deployment/run_survivors_campaign.py`
- 設定: `Tools/Deployment/configs/mad_forest_canary_v1.yaml`
- テスト: `Tools/Deployment/tests/campaign/test_campaign_runner.py`、`test_save_lifecycle.py`、`test_run_survivors_campaign.py`
- 実機での C0〜C4 実行(`D06-LIVE-CAMPAIGN`)は、この PR の merge 後に人が行う手動 gate であり、この PR の範囲外である。

runner は 06-02 の identity や report、06-03 の launch の意味論を複製も変更もしない。`CreateProcessW` の呼び出しや SQLite 行の操作も行わず、06-02 validator・reporter と 06-03 `DurableLaunchStore`・broker の public API だけを呼ぶ。

## Stage schedule

stage の秒数と slot 数は 06-02 の `STAGE_POLICIES` が唯一の正であり、config にもこの文書にも独自の値は持たない。

| stage | 1 run の秒数 | slot 数 |
|---|---|---|
| C0 | 1800 | 2 |
| C1 | 3600 | 4 |
| C2 | 7200 | 8 |
| C3 | 14400 | 16 |
| C4 | 28800 | 20 |

- 上位 stage は、直下の stage が `stage_passed` で確定した後にしか開始できない。
- run mode は `formal_single_attempt` で固定する。1 slot につき gameplay attempt を 1 件だけ予約し、process 内の restart や差し替えは行わない。
- code・model・config・profile・threshold を 1 つでも変えたら、新しい campaign id で C0 からやり直す。config の内容は `config_hash` として plan に束縛されており、同じ campaign id のまま変えると `PlanMismatch` で拒否される。

## CLI

```bash
cd Tools/Deployment
python run_survivors_campaign.py run --campaign-id <id> --artifacts <primary root> --ledger <ledger dir> \
    --save <game の save file> --canonical-save <canonical save> [--max-slots N] [--backup-root <backup store>]
python run_survivors_campaign.py restore --backup-root <backup store> --artifacts <空の root>
```

結果は 1 行の JSON として stdout へ出力する。

| 終了コード | 意味 |
|---|---|
| 0 | 全 stage が pass して finalize 済み、`--max-slots` による停止(再開可能)、または restore 成功 |
| 2 | fail closed による拒否(formal parents なし、finalize 済み、plan 不一致、未 reconcile の intent など) |
| 3 | stage が pass せずに campaign が確定した(finalize 済み)、または `preflight_blocked` |

### Development dry-run

`--live` を付けない `run` は development dry-run になる。

- 06-03 の実 `LaunchBroker` を process 内で動かし、harmless な target helper(`python.exe -c "import time; time.sleep(600)"`)を実際に起動する。helper は broker を閉じるときに Job ごと終了する。
- controller・helper・target・operator は fake(`DevelopmentHarness`)であり、checkpoint は自動で承認され、run は stage の秒数を待たずに合成値の成功を返す。
- plan は `mode=synthetic` になる。plan・stage manifest・per-run manifest・summary はすべて `development_only=true` かつ `formal_campaign_eligible=false` であり、`formal_evidence_eligible` は常に false になる。dry-run に `--formal-parents` を渡すと拒否される。
- durable ledger は Windows・NTFS・固定ディスクでしか開けない(06-03 の support envelope)。

### 停止・再開・finalize

- `--max-slots N` を付けると、現在の stage で N slot を処理した時点で停止する(`status=stopped`)。停止中は finalize せず、canonical save も置いたままにする。
- 再開は同じコマンドを `--max-slots` なしで(または付けて)もう一度実行するだけでよい。pass 済みの stage は飛ばし、途中の stage は続きの slot から進める。
- 全 stage が pass するか、stage が pass せずに確定すると、CLI は `finish()` を呼ぶ。`finish()` は元 save を復元して検証し、`campaign/summary.json` を write-once で作成する。finalize 後に同じ campaign id で `run` すると拒否される。
- `preflight_blocked` では finalize しない。原因を直してから同じコマンドを再実行すると、runner は新しい stage execution id(`x2`, `x3` …)で同じ stage を最初からやり直す。block された実行も証跡から除外しない。

### Backup store への複製と復元

- `--backup-root` を付けると、`run` の終了時に primary root の全 file を `<backup>/bundle/` へ複製し、各 file の SHA-256 を `<backup>/bundle_index.json` に記録する。artifact は write-once と append-only なので、毎回上書きで複製すれば最新の mirror になる。
- `restore` は、復元先が空であること、bundle の file の集合が index と一致すること、全 file の hash が一致することを確認してから書き込む。1 つでも満たさなければ、復元先に何も書かずに拒否する。
- 復元した root を `--artifacts` に指定すれば、そのまま再開できる。durable ledger は bundle に含まれないため、同じ ledger directory を使うこと(ledger の耐久性については `campaign_durability_win64.md` を参照)。

## Live(formal)campaign の前提

`--live` は次の順に検査し、1 つでも欠けていれば artifact・ledger・save に一切触れずに拒否する(fail closed)。

1. `--formal-parents <json>` があり、`prerequisites`(06-02 prerequisite wire、`development_only=false`)と `prerequisite_parent_hash` を含むこと。無い場合は formal manifest を発行しない。
2. config の `thresholds` と `threshold_parent_hash` が解決済みであること。metric floor は 03-04 の exact-target verdict と 04-10 の final perception verdict から決まり、`threshold_parent_hash` はその 2 つの evidence hash から導いた値と一致しなければならない。
3. live 用の controller・helper・target・operator adapter がそろっていること。この build にはまだ無いため、`--live` は常に拒否される(`D06-LIVE-CAMPAIGN` で接続する)。

runner 自体も、formal plan に `development_only=true` の port が渡されると `begin()` の時点で拒否する。

## 手動手順(live gate)

以下は `D06-LIVE-CAMPAIGN` で実機を動かすときの手順である。各 checkpoint は operator port の `checkpoint(name, context)` に対応しており、承認されなければ runner は先へ進まない。

### 1. Cloud sync の無効化(`cloud_sync_disabled`)

1. Steam の対象タイトルのプロパティで Steam Cloud の同期を無効にする。
2. game と launcher を完全に終了する。
3. checkpoint で無効化を確認したと答える。runner はこれを `save/cloud_sync_attestation.json` に記録する。この記録が無い campaign は正式な証跡にならない。

### 2. Save の退避と canonical save

1. `--save` には、game が実際に読む save file を指定する。runner は `begin()` で、元の save を `save/original_backup.bin` と `.json`(hash)へ 1 回だけ退避する。再開時には取り直さない。
2. `--canonical-save` には、各 slot の開始状態となる save を指定する。hash は plan の `canonical_save_hash` に束縛される。
3. 各 slot の開始前に、runner は game が停止していることを確認し、canonical save を一時 file へ書いて検証してから atomic に置き換える。その pre hash と、run 後の post hash を `save/pre|post/<attempt_id>.json` に記録する。
4. game が動いている間は save を置き換えない。置き換えに失敗した場合、その attempt は preflight failure になり、reserved identity は作られない。

### 3. Arm と focus(`arm` / `focus` / `manual_start`)

- `arm`: input helper の arm toggle(`Ctrl+Shift+F12` の edge で arm と disarm を切り替える)を arm にする。これはキーを押し続ける dead-man 方式ではない。toggle なので、止めたいときは同じ操作で disarm するか、controller を停止する。
- `focus`: 対象の game window を foreground にする。focus が外れると入力は送られない。
- `manual_start`: stage の開始操作は operator が手で行う。UI restart は無効であり、runner は menu への入力を送らない。menu への入力が観測された run は safety failure になる。
- timeout した run は、controller の terminate の後に、helper lease の release を外部の observer で確認する。確認できなければ safety failure になる。

### 4. 終了時の save 復元

- finalize(`finish()`)で元 save を書き戻し、hash を検証して `save/restore_verdict.json` に記録する。verdict が `PASS` でない campaign は正式な証跡にならない。
- 最後に Steam Cloud を元の設定へ戻すのは、復元の verdict を確認した後にする。

## Recovery

| 状況 | 対応 |
|---|---|
| runner / CLI が途中で落ちた | 同じコマンドを再実行する。activation 済みの slot は `runner_interrupted_after_activation` の safety failure として消費し、再 launch しない。broker へ送信中だった intent は ledger の reconcile で分類され、再送しない。 |
| `UnreconciledLaunch`(終了コード 2) | ledger に未 reconcile の launch intent が残っている。`campaign_durability_win64.md` の手順で reconcile してから再開する。reconcile するまで次の slot は開始しない。 |
| `preflight_blocked`(終了コード 3) | 原因(game の起動中、save の不一致など)を直して再実行する。新しい stage execution id で同じ stage を最初からやり直す。 |
| `launch_gate_blocked` / `campaign_blocked` / `stage_not_promoted` | campaign は確定済みで、finalize により元 save も復元済みである。新しい campaign id で C0 からやり直す。 |
| 停止中に PC を離れる | `--max-slots` で停止すると canonical save が置かれたままになる。元の save で遊ぶ場合は、先に campaign を finalize するか、`save/original_backup.bin` の hash を確認してから手で戻す。 |
| artifact root が失われた | `restore` で backup store から空の root へ戻し、その root で再開する。index や hash が合わない bundle は復元されない。 |
| save の復元に失敗した | `save/restore_verdict.json` が無い、または `PASS` でない。`save/original_backup.json` の SHA-256 と照合しながら `save/original_backup.bin` を手で書き戻す。この campaign は正式な証跡にしない。 |

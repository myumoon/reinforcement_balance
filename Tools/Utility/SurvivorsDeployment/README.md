# Survivors Deployment 運用ユーティリティ

このディレクトリは、Survivors の Deployment 用ローカル環境を準備し、外部 artifact store と対象環境の証跡を安全に確認するための Windows 用 workflow です。PowerShell の作業ディレクトリには依存しないため、Explorer から `.bat` をダブルクリックできます。

## 実行順

`.env/survivors_deployment.env` はリポジトリ直下に作成しますが、`.env/` 全体が Git 管理外です。まず次を実行し、テンプレートをコピーしてから実値をローカルで設定してください。

```text
00_ConfigureLocalEnvironment.bat
01_SetupDeploymentEnvironment.bat
02_PrepareArtifactStore.bat
03_CollectTargetEvidence.bat
```

PR #308 が `main` にマージされた後だけ、最後に次を実行します。

```text
04_VerifyDeploymentSmoke.bat
```

canonical save の運用バックアップが必要なときは、設定確認後に次を個別に実行します。これは artifact store の同期やrestoreではありません。

```text
05_BackupCanonicalSave.bat
```

各 workflow は失敗時に非ゼロ終了します。値そのものを PR、issue、チャット、コンソールログへ貼り付けないでください。`.env/survivors_deployment.env` はローカル専用であり、テンプレート・PowerShell・`.bat`・テスト・README だけが Git 管理対象です。

## 設定キー

| キー | 用途 |
|---|---|
| `REINBALANCE_PYTHON` | 固定済み依存関係の install と smoke test に使う、絶対パスの Python 実行ファイル。 |
| `SURVIVORS_ARTIFACT_PRIMARY_ROOT` | capture manifest、証跡 manifest、artifact object の主保存先。リポジトリ外に置く。 |
| `SURVIVORS_ARTIFACT_BACKUP_ROOT` | primary のバックアップ先。正式運用では primary と別ボリュームに置く。 |
| `SURVIVORS_EVIDENCE_ROOT` | MachineInfo、実行ファイル metadata、canonical hash の証跡保存先。リポジトリ外に置く。 |
| `VAMPIRE_SURVIVORS_EXE` | 対象 Vampire Survivors 実行ファイル。自動探索せず、設定値だけを使う。 |
| `SURVIVORS_CANONICAL_SAVE` | `03` では任意、`05` では必須。ユーザーが明示的に選んだ canonical save ファイル。 |
| `SURVIVORS_CANONICAL_SAVE_BACKUP_ROOT` | `05` 専用。canonical save の世代バックアップ先。artifact primary/backup root と兼用しない。 |
| `SURVIVORS_TARGET_PROFILE` | 任意。ユーザーが明示的に選んだ TargetProfile の証跡を取得する。空欄ならスキップする。 |

`03` はsaveとprofileの本体をコピーしません。設定されたファイルが存在することを確認し、外部evidenceにbasenameとcanonical hashだけを記録します。空欄のsave/profileを自動探索してcanonical扱いすることもありません。

MachineInfo JSON と executable metadata JSON は、収集後にそれぞれ primary artifact store へ登録します。target-evidence manifest には各ファイルの canonical hash と `artifact_ref`（content SHA-256、サイズ、`artifact://sha256/...` URI）を記録します。対象ゲーム exe 本体はコピーせず、basename と canonical hash のみを記録します。

## artifact store の安全境界

primary、backup、evidence はすべてリポジトリ外でなければなりません。`02` は通常実行でも次を必ず検証します。

- 正規化後の primary と backup が同一パスでないこと。
- Windows の volume serial/device が異なること。
- 保存先がファイルではなく、作成可能なディレクトリであること。
- `Tools/Artifacts/manage_artifacts.py check-backup-root` が成功すること。

primary と backup を別ボリュームにするのは、同じディスクや同じボリュームの障害・誤削除で主保存とバックアップが同時に失われるのを避けるためです。`02_PrepareArtifactStore.ps1 -DryRun` は同じ検証を行いますが、フォルダは作成しません。

## canonical save バックアップ

`05` は `SURVIVORS_CANONICAL_SAVE` を読み取り専用の入力として扱い、`SURVIVORS_CANONICAL_SAVE_BACKUP_ROOT` 配下のUTC時刻付き世代ディレクトリへコピーします。canonical save の編集・削除、Steam／Steam Cloud／Vampire Survivors の起動・停止・設定変更は行いません。artifact store の `SURVIVORS_ARTIFACT_PRIMARY_ROOT`／`SURVIVORS_ARTIFACT_BACKUP_ROOT` はこの運用バックアップ先ではありません。

コピー前に入力が通常ファイルであること、入力とバックアップrootが同一パスでないこと、バックアップrootがlocal fixed NTFS上でreparse pointを経由しないことを確認します。コピーは世代ディレクトリ内の一時ファイルへ行い、sourceと一時ファイルのSHA-256およびサイズが一致した場合だけ非上書きrenameで確定します。既存世代は上書きせず、失敗時は非ゼロ終了します。

成功した世代にはsave本体と `backup-result.json` が作られます。標準出力にも同じschemaのJSONを出力し、`source_path`、`backup_path`、`source_hash`、`backup_hash`、`size`、`collected_at`、`match`、`schema_version` を記録します。失敗JSONは `status=failure`、`match=false` と `blocking_reason` を含みます。save内容やenvの他の値は記録しません。

既定ではリポジトリrootから導出した `.env/survivors_deployment.env` を読みます。別のenv、または明示した2つのパスを使う場合は次のように実行できます。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File Tools/Utility/SurvivorsDeployment/05_BackupCanonicalSave.ps1 -EnvPath <ENV_FILE>
powershell.exe -NoProfile -ExecutionPolicy Bypass -File Tools/Utility/SurvivorsDeployment/05_BackupCanonicalSave.ps1 -CanonicalSavePath <SAVE_FILE> -BackupRoot <BACKUP_DIRECTORY>
```

ゲームやSteam Cloudがsaveを書き換えている可能性はツール側で判断しません。operatorが適切な時点を選び、成功JSONとsidecarの `match=true` を確認してください。restoreはこのツールの対象外です。

## PR #308 と smoke test

PR #308 のレビュー中は `00`〜`03` までを確認します。`04` は PR #308 の `Tools/Deployment/capture_survivors.py` が利用できる状態でだけ実行してください。CLI は現在の `--duration-sec` を使い、`--store-root` を primary store に明示します。

`04` が起動するのは `--synthetic` を付けた 1 秒の capture と dry-run だけです。成功した synthetic manifest は `formal_dataset_eligible=false` であり、実ゲーム、live capture、実行ファイルの自動探索、annotation の対話入力は行いません。live capture は D04 の手動ゲートを通過した後に、既存の Deployment CLI を明示的に実行してください。

## ログと再現コマンド

通常のログは `%LOCALAPPDATA%\ReinBalance\DeploymentLogs` に保存します。権限制限された環境では、リポジトリ外の一時ログへフォールバックします。ログには設定値を記録しません。

Explorer の代わりに、リポジトリ root から次のように実行できます。

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File Tools/Utility/SurvivorsDeployment/00_ConfigureLocalEnvironment.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File Tools/Utility/SurvivorsDeployment/01_SetupDeploymentEnvironment.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File Tools/Utility/SurvivorsDeployment/02_PrepareArtifactStore.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File Tools/Utility/SurvivorsDeployment/03_CollectTargetEvidence.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File Tools/Utility/SurvivorsDeployment/04_VerifyDeploymentSmoke.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File Tools/Utility/SurvivorsDeployment/05_BackupCanonicalSave.ps1
```

`00`〜`04` の失敗時はまず外部operator logのworkflow名とexit codeを確認してください。`05` の失敗時は標準出力JSONの `blocking_reason` を確認します。次に `.env` のキーが空欄でないこと、各保存先が意図した外部rootであること、artifact primary/backupが別ボリュームであること、canonical save backup rootがlocal fixed NTFSであることを確認してください。実値を含む `.env` やevidence payloadをGit管理領域へ移動しないでください。

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

各 workflow は失敗時に非ゼロ終了します。値そのものを PR、issue、チャット、コンソールログへ貼り付けないでください。`.env/survivors_deployment.env` はローカル専用であり、テンプレート・PowerShell・`.bat`・テスト・README だけが Git 管理対象です。

## 設定キー

| キー | 用途 |
|---|---|
| `REINBALANCE_PYTHON` | 固定済み依存関係の install と smoke test に使う、絶対パスの Python 実行ファイル。 |
| `SURVIVORS_ARTIFACT_PRIMARY_ROOT` | capture manifest、証跡 manifest、artifact object の主保存先。リポジトリ外に置く。 |
| `SURVIVORS_ARTIFACT_BACKUP_ROOT` | primary のバックアップ先。正式運用では primary と別ボリュームに置く。 |
| `SURVIVORS_EVIDENCE_ROOT` | MachineInfo、実行ファイル metadata、canonical hash の証跡保存先。リポジトリ外に置く。 |
| `VAMPIRE_SURVIVORS_EXE` | 対象 Vampire Survivors 実行ファイル。自動探索せず、設定値だけを使う。 |
| `SURVIVORS_CANONICAL_SAVE` | 任意。ユーザーが明示的に選んだ save の証跡を取得する。空欄ならスキップする。 |
| `SURVIVORS_TARGET_PROFILE` | 任意。ユーザーが明示的に選んだ TargetProfile の証跡を取得する。空欄ならスキップする。 |

save と profile の本体はコピーしません。設定されたファイルが存在することを確認し、外部 evidence に basename と canonical hash だけを記録します。空欄の save/profile を自動探索して canonical 扱いすることもありません。

## artifact store の安全境界

primary、backup、evidence はすべてリポジトリ外でなければなりません。`02` は通常実行でも次を必ず検証します。

- 正規化後の primary と backup が同一パスでないこと。
- Windows の volume serial/device が異なること。
- 保存先がファイルではなく、作成可能なディレクトリであること。
- `Tools/Artifacts/manage_artifacts.py check-backup-root` が成功すること。

primary と backup を別ボリュームにするのは、同じディスクや同じボリュームの障害・誤削除で主保存とバックアップが同時に失われるのを避けるためです。`02_PrepareArtifactStore.ps1 -DryRun` は同じ検証を行いますが、フォルダは作成しません。

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
```

失敗時はまず外部 operator log の workflow 名と exit code を確認し、次に `.env` のキーが空欄でないこと、保存先がリポジトリ外であること、primary/backup が別ボリュームであることを確認してください。実値を含む `.env` や evidence payload を Git 管理領域へ移動しないでください。

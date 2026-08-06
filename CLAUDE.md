# CLAUDE.md

このファイルはこのリポジトリのコードで作業するときのClaude Code (claude.ai/code) のための情報を提供します。

## プロジェクト概要

UE5 C++ ゲームプロジェクト（"ReinBalance"）。Unreal Engine 5.4 対象。ゲームバランスの強化学習を目的とし、Python (Stable-Baselines3) で訓練し ONNX モデルを UE5 (NNERuntimeORT) で推論する。

| ゲーム | ポート | 概要 |
|--------|--------|------|
| BalancePole | 8765 | 倒立振子。カートを左右に動かして2本のポールを立てる古典RL課題 |
| CoinGame | 8766 | コイン収集。フィールド上のコインを集めながら敵を回避する |
| Survivors | 8767 | ローグライクサバイバー。Aura攻撃でXP・レベルアップしながら敵を倒す |

```
Python (SB3 PPO) ↕ HTTP ↕ UE5 Editor (PIE)
  /reset /step /obs_schema /params → ONNX → NNERuntimeORT
```

ディレクトリ構成・モジュール依存関係は [`docs/project_structure.md`](docs/project_structure.md) を参照。

---

## ビルド

詳細は [`docs/ue5_build.md`](docs/ue5_build.md) を参照。

```powershell
"C:/Program Files/Epic Games/UE_5.4/Engine/Build/BatchFiles/Build.bat" ReinBalanceEditor Win64 Development -Project="<PROJECT_ROOT>/ReinBalance/ReinBalance.uproject"
```

---

## 訓練クイックスタート

```cmd
conda activate reinbalance
cd Tools\Training
python train.py --game survivors --total-steps 500000
```

スクリプト一覧・モジュール構成・実装上の注意は [`docs/train/overview.md`](docs/train/overview.md) を参照。

---

## ルール

- モジュールなどをインポートする場合はバージョンを固定してください

### テスト実行

Python テストは以下のコマンドを使用してください。

```bash
bash Tools/run-pytest.sh Tools/Training/tests -q -rs
```

`Tools/run-pytest.sh` は WSL native conda → Windows conda の順に Python 環境を自動検出します。
Codex / 自動化フローを含め環境を問わず同じコマンドで実行できます。

`/usr/bin/python3` には pytest / torch / sb3 がインストールされていないため、使用できません。

`review-contract.md` の M11/M12 を記載する際も上記コマンドを使用してください。

### ブランチ名のルール

- 機能実装 : feature/機能名
- バグ修正 : fix/バグ内容
- ドキュメント : docs/内容

### 作業ルール

- main ブランチ上で直接編集しないでください
- 編集する場合は git worktree を使用してブランチを切って作業してください
  ```bash
  git worktree add ../.worktrees/<ブランチ名> -b <ブランチ名>
  ```
- ソースコードを変更した場合は worktree 上でビルドが成功することを確認してください
- 実装が完了した場合はプルリクエストを作成して、レビュワーを指定してください
- AI作業量・コンテキスト予算は [`docs/agent_workflow/token_budget_policy.md`](docs/agent_workflow/token_budget_policy.md) を参照してください。

### プラン実装ルール

- プランを実装する際、明示的に「分割して」と指示がない限り **1つのworktreeブランチ・1つのPR** にまとめること
- 実装途中でPRを作成・マージしないこと（全フェーズ完了後に `gh pr create` を1回だけ実行する）
- 実装には常に `/worktree-dev-review` スキルを使用すること

### その他ルール

- 返答は日本語で返答してください

### モデル選択

| タスク種別 | モデル | 判断基準 |
|----------|--------|---------|
| 設計・長文生成・複雑な分析 | Opus | 複数ファイルを横断する変更、要件の矛盾検出 |
| 通常のコード編集・テスト追加 | Sonnet | 単一ファイル編集、既存パターンの適用 |
| 簡単な調査・結果の整形 | Haiku | grep結果の整形、1行修正 |

### コメント規約

- **対象:** `.cpp` / `.h` / `.py` / `.yaml` / `.toml` の全ファイル。**ファイル冒頭コメント**と、各**クラス・関数**（YAML/TOML は各**セクション**）にコメントを付ける。
- **書式（2段構成・日本語）:** 先頭行は専門用語を使ってよいので**簡潔に1行**で説明する。その下に、詳しくない人でも内容が把握できる**やさしい説明を数行**書く（必要なら補足も）。
- **Python:** module 先頭 / class 定義直後 / def 定義直後に docstring を置く。

  ```python
  """<1行説明>

  <初心者用説明>
  <補足>
  """
  ```

- **C++ / ヘッダ:** `///` または `/** ... */` で同じ2段構成にする。
- **YAML / TOML:** ファイル先頭に `#` の2段コメントを置き、各セクション（キー群）の上に `#` で役割コメントを付ける。
- コメントの付与・更新は**ロジックを変えない**こと（diff はコメント/docstring の追加・更新のみ）。既存の英語 docstring は日本語2段へ置き換える。

---

## 参照ドキュメント

| ドキュメント | 内容 |
|---|---|
| [`docs/goal.md`](docs/goal.md) | プロジェクト最終目標・中間目標・ロードマップ |
| [`docs/project_structure.md`](docs/project_structure.md) | ディレクトリ構成・モジュール依存関係・エンジン設定 |
| [`docs/ue5_build.md`](docs/ue5_build.md) | UE5 ビルドコマンド・注意事項 |
| [`docs/ue5_tests.md`](docs/ue5_tests.md) | UE5 テスト環境・注意事項 |
| [`docs/train/overview.md`](docs/train/overview.md) | 訓練スクリプト概要・モジュール構成 |
| [`docs/train/ue5_env.md`](docs/train/ue5_env.md) | UE5 HTTP API 仕様・`/params` の挙動 |
| [`docs/train/impl_notes.md`](docs/train/impl_notes.md) | 実装上の注意点・既知の問題（コールバック・多並列・Resume・SPALF） |
| [`docs/train/reward_fn_policy.md`](docs/train/reward_fn_policy.md) | Survivors reward_fn 設計ポリシー 12 項目（reward_fn を実装・修正する際の必須チェックリスト） |

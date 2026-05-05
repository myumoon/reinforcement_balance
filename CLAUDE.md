# CLAUDE.md

このファイルはこのリポジトリのコードで作業するときのClaude Code (claude.ai/code) のための情報を提供します。

## プロジェクト概要

UE5 C++ ゲームプロジェクト（"ReinBalance"）。Unreal Engine 5.4 対象。ゲームバランスの強化学習を目的としており、Python (Stable-Baselines3) で訓練し ONNX モデルを UE5 (NNERuntimeORT) で推論する。

### 実装済みゲームモード

| ゲーム | ポート | 概要 |
|--------|--------|------|
| BalancePole | 8765 | 倒立振子。カートを左右に動かして2本のポールを立てる古典RL課題 |
| CoinGame | 8766 | コイン収集。フィールド上のコインを集めながら敵を回避する |
| Survivors | 8767 | ローグライクサバイバー。Aura攻撃でXP・レベルアップしながら敵を倒す |

### 訓練アーキテクチャ

UE5 エディタ内でHTTPサーバーを起動し、Python から REST API でやり取りする。

```
Python (SB3 PPO)
  ↕ HTTP (localhost:port)
UE5 Editor (PIE)
  ├── /reset  → エピソードリセット
  ├── /step   → 行動送信・観測/報酬受信
  ├── /obs_schema → 観測空間定義（起動時1回取得）
  └── /params → カリキュラム用パラメータ更新（Survivors のみ）
        ↓ 訓練完了後
  ONNX エクスポート → NNERuntimeORT で推論
```

---

## ビルドコマンド

Unreal Build Tool (UBT) を使用するUnreal Engineプロジェクトです。UE 5.4 がインストールされている前提：

```powershell
# エディタビルド（通常はこちら）
"C:/Program Files/Epic Games/UE_5.4/Engine/Build/BatchFiles/Build.bat" ReinBalanceEditor Win64 Development -Project="<PROJECT_ROOT>/ReinBalance/ReinBalance.uproject"

# ゲームビルド
"C:/Program Files/Epic Games/UE_5.4/Engine/Build/BatchFiles/Build.bat" ReinBalance Win64 Development -Project="<PROJECT_ROOT>/ReinBalance/ReinBalance.uproject"
```

`ReinBalance.sln` を使って Visual Studio 2022（v14.38 ツールセット）からもビルド可能です。

> **注意**: UE5 エディタが起動中（Live Coding 有効）の状態では UBT ビルドが失敗します。ビルド前にエディタを閉じてください。

---

## ディレクトリ構成

```
ue5_reinforcement_balance/
├── ReinBalance/                           # UE5 プロジェクトルート
│   ├── Config/                            # エンジン設定 (DefaultEngine.ini 等)
│   ├── Content/Models/                    # 訓練済み ONNX モデル
│   └── Source/
│       ├── ReinBalance/                   # Runtime モジュール（ゲームロジック）
│       │   ├── Public/
│       │   │   ├── BalanceCart.h          # 倒立振子ゲーム
│       │   │   ├── CoinGame.h             # コインゲーム
│       │   │   ├── SurvivorsGame.h        # サバイバーゲーム（XP/レベル/Aura）
│       │   │   ├── *InferenceController.h # ONNX 推論コントローラ（各ゲーム）
│       │   │   └── *GameView.h            # UI ウィジェット管理
│       │   └── Private/
│       ├── ReinBalanceEditor/             # Editor モジュール（訓練用HTTPサーバー）
│       │   └── Private/Training/
│       │       ├── BalanceHttpEnvService.cpp
│       │       ├── CoinHttpEnvService.cpp
│       │       └── SurvivorsHttpEnvService.cpp
│       ├── ReinBalance.Target.cs
│       └── ReinBalanceEditor.Target.cs
├── Tools/
│   └── Training/                          # Python 訓練スクリプト
│       ├── envs/                          # gymnasium 環境ラッパー
│       │   ├── base_ue5_env.py            # HTTP通信基底クラス
│       │   ├── balance_env.py / coin_env.py / survivors_env.py
│       │   └── *_env_stub.py              # UE5 なしでの動作確認用スタブ
│       ├── games/                         # EUREKA ループ用ゲーム設定
│       │   ├── coin_eureka_config.py
│       │   └── survivors_eureka_config.py
│       ├── requirements.txt               # バージョン固定
│       ├── setup.bat / setup.sh           # 環境セットアップ
│       ├── train.py                       # 訓練エントリーポイント（SB3 PPO）
│       ├── eureka_loop.py                 # EUREKA型LLM報酬シェーピングループ
│       └── export_onnx.py                 # ONNX 変換スクリプト
├── Documents/                             # 設計ドキュメント
├── .claude/docs/                          # Claude Code 用設計メモ・ロードマップ
└── CLAUDE.md
```

---

## モジュール依存関係

| モジュール | 種別 | 依存 |
|-----------|------|------|
| ReinBalance | Runtime | Core, CoreUObject, Engine, InputCore, EnhancedInput |
| ReinBalanceEditor | Editor | ReinBalance, HTTPServer, HTTP, Json, JsonUtilities |

---

## エンジン設定

- グラフィックス: DX12 / Shader Model 6（Linux では Vulkan SM6）
- 入力: Enhanced Input System プラグイン有効
- エディタ: Modeling Tools Editor Mode プラグイン有効
- 推論: NNERuntimeORT プラグイン有効
- デフォルトマップ: `/Engine/Maps/Templates/OpenWorld`
- PCH モード: UseExplicitOrSharedPCHs

---

## 訓練の流れ

### 基本訓練

```cmd
# 1. UE5 エディタでテストレベルを開き PIE 起動
# 2. 別ターミナルで:
conda activate reinbalance
cd Tools\Training
python train.py --game survivors --total-steps 500000
```

主なオプション:

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--game` | ゲーム種別 (balance/coin/survivors) | balance |
| `--frame-skip N` | N ステップ分アクションを繰り返す（Action Repeat）| 1 |
| `--reward-fn path` | EUREKA生成の報酬シェーピング関数を適用 | なし |
| `--dry-run` | UE5 なしでスタブ環境で動作確認 | false |
| `--resume path` | 途中から再開 | なし |
| `--ent-coef` | PPO エントロピー係数 | 0.01 |

### EUREKA ループ（LLM による報酬シェーピング）

LLM が報酬シェーピング関数を反復生成・改良する。

```cmd
python eureka_loop.py --game-config games/survivors_eureka_config.py --iterations 5
python eureka_loop.py --game-config games/coin_eureka_config.py --llm openai --model gpt-4o
```

生成結果は `eureka_results/<run_name>/` に保存。ベスト関数を `--reward-fn` で train.py に渡せる。

### ONNX エクスポートと推論

```cmd
python export_onnx.py --game survivors --model models/survivors_model
# → ReinBalance/Content/Models/survivors_model.onnx に保存
```

---

## ゲームパラメータ（SurvivorsGame）

カリキュラム学習で使用するパラメータ（`/params` エンドポイントで動的変更可能）:

| パラメータ | 型 | デフォルト | 説明 |
|-----------|----|-----------|------|
| MaxActiveEnemies | int | 6 | 同時出現最大敵数 |
| EnemySpeedMult | float | 1.0 | 敵速度倍率 |
| EnemySpawnInterval | float | 5.0 | 敵スポーン間隔（秒） |
| MinAuraRadius | float | 2.0 | Aura最小半径（m）Lv0時 |
| MaxAuraRadius | float | 10.0 | Aura最大半径（m）MaxLevel時 |
| XPGrowth | float | 3.0 | レベルアップ必要XP増分 |

> **注意**: Python設定ファイル `games/survivors_eureka_config.py` の `_MIN_AURA_RADIUS = 2.5` は
> C++ の `MinAuraRadius = 2.0f` と不一致。報酬関数生成プロンプトで使う定数として修正が必要。

---

## ルール

- モジュールなどをインポートする場合はバージョンを固定してください

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

### その他ルール

- 返答は日本語で返答してください

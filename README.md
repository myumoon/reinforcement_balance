# ReinBalance

UE5 ゲームのバランス調整を強化学習で行うプロジェクト。  
Python (Stable-Baselines3) で PPO 訓練を行い、ONNX モデルを UE5 (NNERuntimeORT) で推論する。  
EUREKA 型 LLM ループによる報酬シェーピング関数の自動生成にも対応。

## ゲームモード

| ゲーム | ポート | 概要 |
|--------|--------|------|
| BalancePole | 8765 | 倒立振子。カートを左右に動かして2本のポールを立てる古典RL課題 |
| CoinGame | 8766 | コイン収集。フィールド上のコインを集めながら敵を回避する |
| Survivors | 8767 | ローグライクサバイバー。Aura攻撃でXP・レベルアップしながら敵を倒す |

## 環境要件

### UE5 / C++

| ツール | バージョン |
|--------|-----------|
| Unreal Engine | 5.4 |
| Visual Studio | 2022（v14.38 ツールセット） |
| Windows SDK | 22621 |
| .NET Framework | 4.6.2 |

### Python 訓練環境

| ツール | バージョン |
|--------|-----------|
| Python | 3.11 |
| Anaconda | 最新版（環境管理） |

## ディレクトリ構成

```
ue5_reinforcement_balance/
├── ReinBalance/
│   ├── Config/                        # エンジン設定
│   ├── Content/Models/                # 訓練済み ONNX モデル
│   └── Source/
│       ├── ReinBalance/               # Runtime モジュール（ゲームロジック）
│       ├── ReinBalanceEditor/         # Editor モジュール（訓練用HTTPサーバー）
│       │   └── Private/Training/
│       │       ├── BalanceHttpEnvService.cpp
│       │       ├── CoinHttpEnvService.cpp
│       │       └── SurvivorsHttpEnvService.cpp
│       ├── ReinBalance.Target.cs
│       └── ReinBalanceEditor.Target.cs
├── Tools/
│   └── Training/                      # Python 訓練スクリプト
│       ├── envs/                      # gymnasium 環境ラッパー
│       ├── games/                     # EUREKA ループ用ゲーム設定
│       ├── requirements.txt           # バージョン固定
│       ├── setup.bat                  # Windows セットアップ
│       ├── setup.sh                   # WSL/Linux セットアップ
│       ├── train.py                   # 訓練エントリーポイント
│       ├── eureka_loop.py             # EUREKA型LLM報酬シェーピングループ
│       └── export_onnx.py             # ONNX 変換スクリプト
├── Documents/                         # 設計ドキュメント
├── .claude/docs/                      # Claude Code 用設計メモ・ロードマップ
└── .gitignore
```

## セットアップ

### 1. リポジトリのクローン

```cmd
git clone <リポジトリURL>
cd ue5_reinforcement_balance
```

### 2. アセットの配置

Google Drive から最終リビジョンのアセットをダウンロードし、`ReinBalance/Content/` に配置してください。

https://drive.google.com/drive/folders/1K9mLsr1hXK57hIaL_xIgReT3TTyspMxE?usp=drive_link

### 3. UE5 ビルド

```powershell
# エディタビルド（通常はこちら）
"<UE5_INSTALL_DIR>/Engine/Build/BatchFiles/Build.bat" ReinBalanceEditor Win64 Development -Project="<PROJECT_ROOT>/ReinBalance/ReinBalance.uproject"

# ゲームビルド
"<UE5_INSTALL_DIR>/Engine/Build/BatchFiles/Build.bat" ReinBalance Win64 Development -Project="<PROJECT_ROOT>/ReinBalance/ReinBalance.uproject"
```

- `<UE5_INSTALL_DIR>`: 例 `C:\Program Files\Epic Games\UE_5.4`
- `<PROJECT_ROOT>`: このリポジトリのルートパス

Visual Studio からビルドする場合は `ReinBalance/ReinBalance.sln` を開いてください。

> **注意**: UE5 エディタが起動中（Live Coding が有効）のときは UBT ビルドが失敗します。ビルド前にエディタを閉じてください。

### 4. Python 訓練環境のセットアップ

#### Anaconda のインストール（初回のみ）

[Anaconda](https://www.anaconda.com/download) をインストールしてください。

```cmd
conda --version  # インストール確認
```

#### conda 環境の作成とパッケージインストール

```cmd
cd Tools\Training
setup.bat
conda activate reinbalance
```

#### 動作確認（UE5 なし）

```cmd
conda activate reinbalance
cd Tools\Training
python train.py --game survivors --dry-run
```

`--dry-run` はスタブ環境を使用するため UE5 の起動は不要です。

## 訓練の実行

### 基本訓練

1. UE5 エディタでテストレベルを開き、Play In Editor (PIE) を開始する
2. 別のコマンドプロンプトで訓練スクリプトを実行する

```cmd
conda activate reinbalance
cd Tools\Training

# Survivors ゲームで訓練
python train.py --game survivors --total-steps 500000

# フレームスキップ（1アクションを4ステップ繰り返す）
python train.py --game survivors --frame-skip 4 --total-steps 300000

# 報酬シェーピング関数を使用
python train.py --game survivors --reward-fn eureka_results/my_run/best/reward_fn.py
```

### EUREKA ループ（LLM による報酬シェーピング）

LLM が報酬シェーピング関数を反復生成・改良する。Anthropic / OpenAI どちらも使用可能。

```cmd
# Survivors で5イテレーション実行（Anthropic Claude）
python eureka_loop.py --game-config games/survivors_eureka_config.py --iterations 5

# CoinGame で OpenAI GPT-4o を使用
python eureka_loop.py --game-config games/coin_eureka_config.py --llm openai --model gpt-4o

# 前回のランを再開
python eureka_loop.py --game-config games/survivors_eureka_config.py --resume my_run
```

生成された報酬関数は `eureka_results/<run_name>/best/reward_fn.py` に保存される。

### ONNX エクスポートと推論

```cmd
# モデルを ONNX に変換
python export_onnx.py --game survivors --model models/survivors_model
```

変換されたモデルは `ReinBalance/Content/Models/survivors_model.onnx` に保存され、UE5 の NNERuntimeORT で推論に使用される。

## ソース管理

Git のみ使用。

| 管理対象 | ツール |
|----------|--------|
| ソースコード・設定ファイル・訓練スクリプト | Git |
| アセット（Content/） | Git 管理外（Google Drive からダウンロード） |
| 訓練済み ONNX モデル（Content/Models/） | Git |

## エンジン設定

- **グラフィックス**: DX12 / Shader Model 6
- **入力**: Enhanced Input System
- **プラグイン**: NNERuntimeORT（ONNX推論）
- **デフォルトマップ**: `/Engine/Maps/Templates/OpenWorld`

## モジュール依存関係

| モジュール | 種別 | 依存 |
|-----------|------|------|
| ReinBalance | Runtime | Core, CoreUObject, Engine, InputCore, EnhancedInput |
| ReinBalanceEditor | Editor | ReinBalance, HTTPServer, HTTP, Json, JsonUtilities |

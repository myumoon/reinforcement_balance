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
│       ├── base/                          # 共通基底クラス
│       │   ├── base_ue5_env.py            # HTTP通信基底クラス
│       │   ├── entity_attention_extractor.py  # エンティティアテンション基底
│       │   └── eureka_game_config.py      # EUREKA設定基底クラス
│       ├── common/                        # 共通ユーティリティ
│       │   ├── obs_schema.py              # obs_schema取得・解析
│       │   └── utils.py                   # _linear_schedule 等
│       ├── games/                         # ゲーム別モジュール
│       │   ├── balance/                   # BalancePole
│       │   │   ├── balance_env.py
│       │   │   └── balance_env_stub.py
│       │   ├── coin/                      # CoinGame
│       │   │   ├── coin_entity_attention_extractor.py
│       │   │   ├── coin_env.py
│       │   │   ├── coin_env_stub.py
│       │   │   └── coin_eureka_config.py
│       │   └── survivors/                 # Survivors
│       │       ├── survivors_entity_attention_extractor.py
│       │       ├── survivors_env.py
│       │       ├── survivors_env_stub.py
│       │       └── survivors_eureka_config.py
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
| `--resume path[@step]` | 再開する run のパス（`@step` で特定ステップ指定可。例: `runs/survivors/v06/train/run-base@2M`）| なし |
| `--ent-coef` | PPO エントロピー係数 | 0.01 |

### EUREKA ループ（LLM による報酬シェーピング）

LLM が報酬シェーピング関数を反復生成・改良する。

```cmd
python eureka_loop.py --game-config games/survivors/survivors_eureka_config.py --iterations 5
python eureka_loop.py --game-config games/coin/coin_eureka_config.py --llm openai --model gpt-4o
```

生成結果は `eureka_results/<run_name>/` に保存。ベスト関数を `--reward-fn` で train.py に渡せる。

### ONNX エクスポートと推論

```cmd
python export_onnx.py --game survivors --model models/survivors_model
# → ReinBalance/Content/Models/survivors_model.onnx に保存
```

---

## ゲームパラメータ（SurvivorsGame）

カリキュラム学習で使用するパラメータ（`/params` エンドポイントで動的変更可能）。
パラメータ一覧・通信設計・既知の注意事項は [`Tools/Training/docs/training_impl_notes.md`](Tools/Training/docs/training_impl_notes.md) を参照。

---

## Python 訓練スクリプト設計

### モジュール構成の思想

`Tools/Training/` は以下の 3 層で構成されている:

| ディレクトリ | 役割 | 変更頻度 |
|------------|------|---------|
| `base/` | 全ゲーム共通の抽象基底クラス | 低（設計変更時のみ） |
| `common/` | ゲーム非依存のユーティリティ関数 | 低 |
| `games/<game>/` | ゲーム固有の実装 | 高（新ゲーム追加・調整） |

### 新しいゲームを追加する手順

1. `games/<game>/` ディレクトリを作成し `__init__.py` を置く
2. `<game>_env.py`: `BaseUE5Env` を継承して gymnasium 環境を実装
3. `<game>_env_stub.py`: `gym.Env` を直接継承した dry-run 用スタブを実装
4. `<game>_entity_attention_extractor.py`（任意）: `EntityAttentionExtractor` を継承してゲーム固有設定を内包
5. `<game>_eureka_config.py`（任意）: `EurekaGameConfig` を継承して EUREKA ループ用設定を実装
6. `train.py` の `_GAME_DEFAULTS` と env 選択分岐にゲームを追加

### エンティティアテンション

`base/entity_attention_extractor.py` の `EntityAttentionExtractor` が基底クラス。

ゲーム固有サブクラスで `item_key` / `use_polar` / `enemy_scalar_keys` 等を事前設定する:
- `games/coin/coin_entity_attention_extractor.py`: `CoinEntityAttentionExtractor`
- `games/survivors/survivors_entity_attention_extractor.py`: `SurvivorsEntityAttentionExtractor`

`train.py --entity-attention` 指定時はゲームに対応するサブクラスが自動選択される。

> **注意（既存モデルの再開）**: リファクタリング前（PR #37 以前）に `--entity-attention` で
> 訓練したモデルを `--resume` する場合、cloudpickle が旧モジュールパスを参照するが、
> `train.py` 起動時に `sys.modules` へ互換エイリアスを自動登録するため **そのまま使用可能**。

### カリキュラム学習（Survivors 専用）

`curriculum_callback.py` の `CurriculumCallback` が実装済み。`--curriculum` フラグで有効化する。

```cmd
python train.py --game survivors --curriculum --total-steps 500000 \
    --curriculum-window 20 --curriculum-threshold 5.0
```

直近 `window` エピソードの `item_kill_score` 平均が `threshold` を超えると自動で敵数・速度・スポーン間隔が段階的に難化する。

実装上の注意事項（コールバック・多並列環境・Resume・SPALF）は [`Tools/Training/docs/training_impl_notes.md`](Tools/Training/docs/training_impl_notes.md) を参照。

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

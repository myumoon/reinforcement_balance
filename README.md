# ReinBalance

バランスゲームの強化学習をUE5で行うためのプロジェクト。
Python (Stable-Baselines3) で訓練し、ONNX モデルを UE5 (NNERuntimeORT) で推論する。

## 環境要件

### UE5 / C++

| ツール | バージョン |
|--------|-----------|
| Unreal Engine | 5.4 |
| Visual Studio | 2022（v14.38 ツールセット） |
| Windows SDK | 22621 |
| .NET Framework | 4.6.2 |
| Perforce (P4V) | DVCS版 |

### Python 訓練環境

| ツール | バージョン |
|--------|-----------|
| Python | 3.11 |
| Anaconda | 最新版（環境管理） |

## ディレクトリ構成

```
├── ReinBalance/
│   ├── Config/                        # エンジン設定
│   ├── Content/Models/                # 訓練済み ONNX モデル
│   ├── Plugins/
│   │   └── PythonTrainingComm/        # UE5↔Python HTTP 通信プラグイン
│   └── Source/
│       ├── ReinBalance/               # Runtime モジュール
│       ├── ReinBalanceEditor/         # Editor モジュール（訓練サービス）
│       ├── ReinBalance.Target.cs
│       └── ReinBalanceEditor.Target.cs
├── Tools/
│   └── Training/                      # Python 訓練スクリプト
│       ├── envs/                      # gymnasium 環境ラッパー
│       ├── requirements.txt           # バージョン固定
│       ├── setup.bat                  # Windows セットアップ
│       ├── setup.sh                   # WSL/Linux セットアップ
│       ├── train.py                   # 訓練エントリーポイント
│       └── export_onnx.py             # ONNX 変換スクリプト
├── Documents/                         # 設計ドキュメント
├── .p4ignore
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

### 4. Python 訓練環境のセットアップ

#### Anaconda のインストール（初回のみ）

[Anaconda](https://www.anaconda.com/download) をインストールしてください。
インストール後、コマンドプロンプトで以下を確認します。

```cmd
conda --version
```

#### conda 環境の作成とパッケージインストール

```cmd
cd Tools\Training
setup.bat
conda activate reinbalance
pip install -r requirements.txt
```

#### 動作確認（UE5 なし）

```cmd
conda activate reinbalance
python train.py --dry-run
```

`--dry-run` はスタブ環境を使用するため UE5 の起動は不要です。

## 訓練の実行

1. UE5 エディタでテストレベルを開き、Play In Editor (PIE) を開始する
2. 別のコマンドプロンプトで訓練スクリプトを実行する

```cmd
conda activate reinbalance
cd Tools\Training
python train.py
```

3. 訓練完了後、ONNX モデルに変換する

```cmd
python export_onnx.py
```

変換されたモデルは `ReinBalance/Content/Models/balance_model.onnx` に保存されます。

## ソース管理

Git と Perforce を併用しています。

| 管理対象 | ツール |
|----------|--------|
| ソースコード・設定ファイル・訓練スクリプト | Git |
| アセット（Content/） | Perforce（最終リビジョンを Google Drive に配置） |
| 訓練済み ONNX モデル（Content/Models/） | Git |

## エンジン設定

- **グラフィックス**: DX12 / Shader Model 6
- **入力**: Enhanced Input System
- **プラグイン**: NNERuntimeORT（推論）、PythonTrainingComm（訓練通信）
- **タイムステップ**: 固定 60fps（訓練安定化のため）
- **デフォルトマップ**: `/Engine/Maps/Templates/OpenWorld`

## モジュール依存関係

| モジュール | 種別 | 依存 |
|-----------|------|------|
| ReinBalance | Runtime | Core, CoreUObject, Engine, InputCore, EnhancedInput |
| ReinBalanceEditor | Editor | ReinBalance, PythonTrainingComm |
| PythonTrainingComm | Editor Plugin | HTTPServer, HTTP, Json |

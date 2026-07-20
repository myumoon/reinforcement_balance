# プロジェクト構成

## 主要ディレクトリ

| パス | 役割 |
|---|---|
| `ReinBalance/Source/ReinBalance/` | Runtime モジュール（ゲームロジック・推論） |
| `ReinBalance/Source/ReinBalanceEditor/Private/Training/` | Editor モジュール（訓練用 HTTP サーバー） |
| `ReinBalance/Content/Models/` | 訓練済み ONNX モデル |
| `Tools/Common/` | 共通契約 package `reinbalance-survivors-contracts`（型・canonical JSON/hash・UI policy） |
| `Tools/Training/` | Python 訓練スクリプト群 |
| `Tools/Deployment/` | 本家画面 capture/perception/control ランタイム（独立 lock・test root） |
| `docs/` | 参照ドキュメント |

詳細なファイル構成は `find` コマンドや IDE のファイルツリーで確認すること。

## モジュール依存関係

依存の詳細は以下を参照:
```
ReinBalance/Source/ReinBalance/ReinBalance.Build.cs
ReinBalance/Source/ReinBalanceEditor/ReinBalanceEditor.Build.cs
```

| モジュール | 種別 | 主な依存 |
|---|---|---|
| ReinBalance | Runtime | Core, Engine, EnhancedInput, NNERuntimeORT |
| ReinBalanceEditor | Editor | ReinBalance, HTTPServer, HTTP, Json |

## Python 環境

conda は Windows 側 (Anaconda) にインストールされており、WSL からは直接 `conda` コマンドは使えない。

**初回セットアップ（一度だけ実行）:**

```bash
bash Tools/setup_wsl.sh
source ~/.bashrc   # zsh の場合は source ~/.zshrc
```

セットアップ後は `python` / `python3` コマンドが reinbalance 環境を指す。

| 用途 | パス |
|------|------|
| セットアップスクリプト | `Tools/setup_wsl.sh` |
| Python 実行ファイル (Windows) | `%USERPROFILE%\anaconda3\envs\reinbalance\python.exe` |
| conda 本体 (Windows) | `%USERPROFILE%\anaconda3\Scripts\conda.exe` |

## Python パッケージ依存方向

Survivors の Training / Deployment は、共通契約 package `reinbalance-survivors-contracts`
（`Tools/Common`）だけを介して型・schema hash・UI policy を共有する。

```
reinbalance_survivors_contracts  <-  Tools/Training
reinbalance_survivors_contracts  <-  Tools/Deployment
```

- **依存は片方向のみ。** `Tools/Common` は Training / Deployment を import しない。
  Training と Deployment は互いを import せず、`sys.path` へ相互追加もしない。
- **`Tools/Common` が禁止する import:** `torch` / `stable_baselines3` / `sb3_contrib` /
  `gymnasium` / UE5 HTTP client / `cv2`(OpenCV) / `dxcam` / `onnxruntime` / `requests`、
  および Training・Deployment の module。NumPy は lazy array helper でのみ使用する。
- **schema 型・enum・canonical serializer は `Tools/Common` だけで定義する。** consumer 側で
  builder を複製したり独自 enum を作らない。
- **インストール順:** 先に `pip install -e Tools/Common`、その後 Training は
  `pip install -r Tools/Training/requirements.txt`、Deployment は
  `pip install -e Tools/Deployment`（独立 `requirements.lock` を持つ）。

## エンジン設定

`ReinBalance/Config/DefaultEngine.ini` および関連 `.ini` を参照。
有効プラグイン: Enhanced Input / Modeling Tools Editor Mode / NNERuntimeORT

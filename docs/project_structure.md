# プロジェクト構成

## 主要ディレクトリ

| パス | 役割 |
|---|---|
| `ReinBalance/Source/ReinBalance/` | Runtime モジュール（ゲームロジック・推論） |
| `ReinBalance/Source/ReinBalanceEditor/Private/Training/` | Editor モジュール（訓練用 HTTP サーバー） |
| `ReinBalance/Content/Models/` | 訓練済み ONNX モデル |
| `Tools/Training/` | Python 訓練スクリプト群 |
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

## エンジン設定

`ReinBalance/Config/DefaultEngine.ini` および関連 `.ini` を参照。
有効プラグイン: Enhanced Input / Modeling Tools Editor Mode / NNERuntimeORT

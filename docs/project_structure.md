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

WSL から Windows ホームディレクトリを取得するには `cmd.exe` と `wslpath` を組み合わせる:

```bash
WIN_HOME=$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')")
```

| 用途 | パス |
|------|------|
| Python 実行ファイル (WSL) | `$WIN_HOME/anaconda3/envs/reinbalance/python.exe` |
| Python 実行ファイル (Windows) | `%USERPROFILE%\anaconda3\envs\reinbalance\python.exe` |
| conda 本体 (WSL) | `$WIN_HOME/anaconda3/Scripts/conda.exe` |

WSL から Python スクリプトを実行するときは以下のように呼び出す:

```bash
WIN_HOME=$(wslpath "$(cmd.exe /c 'echo %USERPROFILE%' 2>/dev/null | tr -d '\r')")
$WIN_HOME/anaconda3/envs/reinbalance/python.exe <script>.py
```

## エンジン設定

`ReinBalance/Config/DefaultEngine.ini` および関連 `.ini` を参照。
有効プラグイン: Enhanced Input / Modeling Tools Editor Mode / NNERuntimeORT

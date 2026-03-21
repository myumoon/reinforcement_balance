# ReinBalance

バランスゲームの強化学習をUE5で行うためのプロジェクト。

## 環境要件

| ツール | バージョン |
|--------|-----------|
| Unreal Engine | 5.4 |
| Visual Studio | 2022（v14.38 ツールセット） |
| Windows SDK | 22621 |
| .NET Framework | 4.6.2 |
| Perforce (P4V) | DVCS版 |

## ディレクトリ構成

```
├── ReinBalance/              # UE5 プロジェクトルート
│   ├── Config/               # エンジン設定
│   ├── Content/              # ゲームアセット（P4で管理）
│   └── Source/               # C++ ソースコード
│       ├── ReinBalance/      # メインモジュール
│       ├── ReinBalance.Target.cs
│       └── ReinBalanceEditor.Target.cs
├── Documents/                # ドキュメント
├── .p4ignore                 # Perforce除外設定
└── .gitignore                # Git除外設定
```

## ソース管理

Git と Perforce を併用しています。

| 管理対象 | ツール                                    |
|----------|----------------------------------------|
| ソースコード・設定ファイル | Git                                    |
| アセット（Content/） | Perforce(最終リビジョン分をGoogleDriveに配置) |

## アセット

Google Drive に最終リビジョン分のアセットを配置しています。
Content/フォルダに配置してください。

https://drive.google.com/drive/folders/1K9mLsr1hXK57hIaL_xIgReT3TTyspMxE?usp=drive_link

## ビルド

### コマンドラインからビルド

```powershell
"<UE5_INSTALL_DIR>/Engine/Build/BatchFiles/Build.bat" ReinBalance Win64 Development -Project="<PROJECT_ROOT>/ReinBalance/ReinBalance.uproject"
```

- `<UE5_INSTALL_DIR>`: UE5 のインストールディレクトリ（例: `C:\Program Files\Epic Games\UE_5.4`）
- `<PROJECT_ROOT>`: このリポジトリのルートパス

### Visual Studio からビルド

`ReinBalance/ReinBalance.sln` を開いてビルドしてください。

## エンジン設定

- **グラフィックス**: DX12 / Shader Model 6
- **入力**: Enhanced Input System
- **プラグイン**: Modeling Tools Editor Mode
- **デフォルトマップ**: `/Engine/Maps/Templates/OpenWorld`
- **PCH**: UseExplicitOrSharedPCHs

## モジュール依存関係

`Core`, `CoreUObject`, `Engine`, `InputCore`, `EnhancedInput`

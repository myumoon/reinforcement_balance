# CLAUDE.md

このファイルはこのリポジトリのコードで作業するときのClaude Code (claude.ai/code) のための情報を提供します。

## プロジェクト概要

UE5 C++ ゲームプロジェクト（"ReinBalance"）。Unreal Engine 5.4 対象。ゲームバランスの強化学習を目的としており、現在はブランク C++ テンプレートをベースにした初期状態です。

## ビルドコマンド

Unreal Build Tool (UBT) を使用するUnreal Engineプロジェクトです。UE 5.4 がインストールされている前提：

```bash
# コマンドラインからビルド
"C:/Program Files/Epic Games/UE_5.4/Engine/Build/BatchFiles/Build.bat" ReinBalance Win64 Development -Project="%cd%/ReinBalance/ReinBalance.uproject"
```

`ReinBalance.sln` を使って Visual Studio 2022（v14.38 ツールセット）からもビルド可能です。

## ディレクトリ構成


| ディレクトリ | 説明 |
|-------------|------|
| `Documents/` | ドキュメントファイルをまとめるディレクトリ |
| `ReinBalance/` | UE5のプロジェクトルートディレクトリ |
| `ReinBalance/Source/` | UE5プロジェクトのソースコード |


- **モジュール**: `ReinBalance`（Runtime、デフォルトローディングフェーズ）
- **ソースルート**: `ReinBalance/Source/ReinBalance/`
- **ビルドターゲット**:
  - `ReinBalance.Target.cs` — ゲームターゲット
  - `ReinBalanceEditor.Target.cs` — エディタターゲット

### モジュール依存関係（ReinBalance.Build.cs）

Public: Core, CoreUObject, Engine, InputCore, EnhancedInput

### エンジン設定

- グラフィックス: DX12 / Shader Model 6（Linux では Vulkan SM6）
- 入力: Enhanced Input System プラグイン有効
- エディタ: Modeling Tools Editor Mode プラグイン有効
- デフォルトマップ: `/Engine/Maps/Templates/OpenWorld`
- PCH モード: UseExplicitOrSharedPCHs

### ルール

- モジュールなどをインポートする場合はバージョンを固定してください

### その他制約

- 返答は日本語で返答してください


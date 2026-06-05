# UE5 ビルドガイド

## 前提: エンジンルートの設定

ビルドスクリプトはエンジンルートを以下の優先順位で解決する。

1. `UE_ROOT` 環境変数
2. 共通インストールパス（`C:\UnrealEngine\UE_5.4` → `C:\Program Files\Epic Games\UE_5.4`）

**推奨**: `UE_ROOT` を設定しておくと環境差異を吸収できる。

```powershell
$env:UE_ROOT = "C:\UnrealEngine\UE_5.4"   # PowerShell セッション内
# または System 環境変数として永続設定
```

## ビルドコマンド

```powershell
# エディタビルド（通常はこちら）
"$env:UE_ROOT/Engine/Build/BatchFiles/Build.bat" ReinBalanceEditor Win64 Development -Project="<PROJECT_ROOT>/ReinBalance/ReinBalance.uproject"

# ゲームビルド
"$env:UE_ROOT/Engine/Build/BatchFiles/Build.bat" ReinBalance Win64 Development -Project="<PROJECT_ROOT>/ReinBalance/ReinBalance.uproject"
```

`ReinBalance.sln` から Visual Studio 2022（v14.38 ツールセット）でもビルド可能。

## Low Level Tests 実行

**注意:** Low Level Tests は **ソース版 UE5.4** が必要。Launcher 版では Program ターゲットがブロックされる。

UE Editor / PIE 非起動でロジック単体テストをビルド・実行する。詳細は [`docs/testing/low_level_tests.md`](testing/low_level_tests.md) を参照。

```powershell
# ビルド + 実行（すべての [unit] テスト）
Tools/Tests/RunLowLevelTests.ps1

# タグフィルタ指定
Tools/Tests/RunLowLevelTests.ps1 -Filter "[unit][survivors]"

# ビルドスキップ（既存 exe を再実行）
Tools/Tests/RunLowLevelTests.ps1 -SkipBuild

# エンジンルートを明示指定
Tools/Tests/RunLowLevelTests.ps1 -EngineRoot "D:\MyUE\UE_5.4"
```

## 注意事項

- UE5 エディタが起動中（Live Coding 有効）の状態では UBT ビルドが失敗する。ビルド前にエディタを閉じること。

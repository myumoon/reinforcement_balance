# UE5 ビルドガイド

## ビルドコマンド

UE 5.4 がインストールされている前提。

```powershell
# エディタビルド（通常はこちら）
"C:/Program Files/Epic Games/UE_5.4/Engine/Build/BatchFiles/Build.bat" ReinBalanceEditor Win64 Development -Project="<PROJECT_ROOT>/ReinBalance/ReinBalance.uproject"

# ゲームビルド
"C:/Program Files/Epic Games/UE_5.4/Engine/Build/BatchFiles/Build.bat" ReinBalance Win64 Development -Project="<PROJECT_ROOT>/ReinBalance/ReinBalance.uproject"
```

`ReinBalance.sln` から Visual Studio 2022（v14.38 ツールセット）でもビルド可能。

## Low Level Tests 実行

UE Editor / PIE 非起動でロジック単体テストをビルド・実行する。詳細は [`docs/testing/low_level_tests.md`](testing/low_level_tests.md) を参照。

```powershell
# ビルド + 実行（すべての [unit] テスト）
Tools/RunLowLevelTests.ps1

# タグフィルタ指定
Tools/RunLowLevelTests.ps1 -Filter "[unit][survivors]"

# ビルドスキップ（既存 exe を再実行）
Tools/RunLowLevelTests.ps1 -SkipBuild
```

## 注意事項

- UE5 エディタが起動中（Live Coding 有効）の状態では UBT ビルドが失敗する。ビルド前にエディタを閉じること。

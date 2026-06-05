# Low Level Tests ガイド

## 前提

**ソース版 UE5.4 が必要。** Launcher 版では Program ターゲットがブロックされビルドできない。

`UE_ROOT` 環境変数にエンジンルートを設定しておくことを推奨（省略時は共通パスを自動探索）。

```powershell
$env:UE_ROOT = "C:\UnrealEngine\UE_5.4"
```

## 目的

UE5 Editor / PIE / World 起動に依存せず、ゲームロジックだけを高速に単体テストする。
`ReinBalanceLogicTests` ターゲットを UBT で直接ビルドし、生成された exe を実行する。

## 実行コマンド

```powershell
# ビルド + 全 [unit] テスト実行
Tools/Tests/RunLowLevelTests.ps1

# タグフィルタ指定
Tools/Tests/RunLowLevelTests.ps1 -Filter "[unit][survivors]"

# ビルドスキップ（既存 exe を再実行）
Tools/Tests/RunLowLevelTests.ps1 -SkipBuild

# Debug ビルドで実行
Tools/Tests/RunLowLevelTests.ps1 -Configuration Debug

# エンジンルートを明示指定
Tools/Tests/RunLowLevelTests.ps1 -EngineRoot "D:\MyUE\UE_5.4"
```

> **Note:** `--timeout` の単位は LowLevelTestsRunner 側で「分」として解釈される。`-TimeoutMinutes 2`（デフォルト）はそのまま2分のタイムアウトを意味する。

## テストタグ運用

| タグ | 用途 |
|---|---|
| `[unit]` | 通常のロジック単体テスト（TDD で常に回す） |
| `[survivors]` | Survivors ゲーム固有のテスト |
| `[coin]` | CoinGame 固有のテスト |
| `[balance]` | BalancePole 固有のテスト |
| `[logic]` | 純ロジック（状態遷移・計算） |
| `[integration]` | 複数ロジックを組み合わせるテスト |
| `[slow]` | TDD の通常ループでは除外するテスト |

## テストファイル命名規則

対象クラス・機能に合わせて命名し、できるだけソース構造をミラーする。

```
Source/Programs/ReinBalanceLogicTests/Private/
  SanityTest.cpp
  Survivors/
    SurvivorsLevelingTests.cpp
    SurvivorsWeaponTests.cpp
    SurvivorsObsSchemaTests.cpp
  Coin/
    CoinGameTests.cpp
  Balance/
    BalancePoleTests.cpp
```

## テスト実装ルール

- 新しいロジック変更は先に失敗する Low Level Test を書く
- 1 テストケースは独立して実行できるようにする
- 実アセット・World・PIE・HTTP・乱数非固定に依存しない
- テストデータはコードで生成する
- 浮動小数は許容誤差を明示する（`Approx(expected)` または `CHECK(FMath::IsNearlyEqual(..., tol))`）

## テスト対象・非対象

### 対象（ReinBalanceLogic）

- XP / レベルアップ計算
- 武器クールダウン・攻撃範囲・ダメージ計算
- 敵・ジェム・衝突の座標計算
- Observation schema / feature layout
- CoinGame の報酬・衝突・収集判定
- BalancePole の数値更新・終了判定

### 非対象（Low Level Tests では扱わない）

- `AActor` / `UActorComponent` の動作
- View / Mesh / Material / UMG
- NNE モデルロード
- Editor HTTP service
- アセットロード

非対象は必要なら別途 Automation テストや統合テストで扱う。

## モジュール構成

```
ReinBalanceLogicTests  ← TestTargetRules / TestModuleRules
  └─ 依存 → ReinBalanceLogic  ← Core + CoreUObject のみ
               └─ 依存禁止: Engine / UMG / Slate / NNE / UnrealEd / HTTPServer
```

## BuildGraph について

ローカル UE 5.4（Launcher 版）では `Engine/Build/LowLevelTests.xml` が存在しないため、
BuildGraph 導線は使用しない。実行導線は `Tools/Tests/RunLowLevelTests.ps1` のみ。
`LowLevelTests.xml` が確認できる環境に移行した後の任意導線とする。

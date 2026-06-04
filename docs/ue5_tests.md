# UE5 ロジック低レベルテスト実装ルール

このドキュメントは、UE5 C++ 側のゲームロジックを Low Level Tests でテスト駆動開発するための作業ルールです。

対象は `ReinBalance/Source/ReinBalance/**/Logic/` 配下、または今後追加するロジック専用モジュール配下の低レベルなコードです。View、Editor、HTTP、NNE、アセットロード、World/PIE 起動が必要な処理は対象外です。

## 実行コマンド

Low Level Logic Tests は Editor を起動せず、コマンドラインから実行します。

推奨コマンド:

```powershell
Tools/RunLowLevelTests.ps1 -Filter "[unit]"
```

個別ゲーム・機能だけを実行する例:

```powershell
Tools/RunLowLevelTests.ps1 -Filter "[unit][survivors]"
Tools/RunLowLevelTests.ps1 -Filter "[unit][coin]"
Tools/RunLowLevelTests.ps1 -Filter "[unit][balance]"
```

スクリプト未整備時、またはトラブルシュート時は次の流れで直接実行します。

```powershell
"C:/Program Files/Epic Games/UE_5.4/Engine/Build/BatchFiles/Build.bat" ReinBalanceLogicTests Win64 Development -Project="<PROJECT_ROOT>/ReinBalance/ReinBalance.uproject"

<PROJECT_ROOT>/ReinBalance/Binaries/Win64/ReinBalanceLogicTests/ReinBalanceLogicTests.exe --log --debug --timeout=2 -r console "[unit]" --extra-args -stdout
```

ターゲット名や exe パスを変更した場合は、このドキュメントと `Tools/RunLowLevelTests.ps1` を同時に更新してください。

## 実装後の確認

ロジックを変更したら、必ず Low Level Logic Tests が通ることを確認します。

最低限実行するコマンド:

```powershell
Tools/RunLowLevelTests.ps1 -Filter "[unit]"
```

対象を絞って開発した場合でも、完了前には関連タグ全体を実行します。

```powershell
Tools/RunLowLevelTests.ps1 -Filter "[unit][survivors]"
```

テスト失敗を残したまま実装完了にしないでください。失敗が仕様変更によるものなら、先に期待仕様を明文化し、テストを更新してから実装を直します。

## TDD の進め方

基本フローは Red, Green, Refactor です。

1. 先に失敗するテストを書く。
2. 失敗理由が意図した仕様不足であることを確認する。
3. 最小のロジック実装でテストを通す。
4. テストが通った状態を保ったまま設計を整える。
5. 境界値、失敗パターン、組み合わせを追加して仕様を固める。

t_wada さんが納得できる説明ができるかを基準にします。つまり、テストが「実装の写し」ではなく、振る舞いの仕様として読めること、失敗したときに何が壊れたか分かること、Red/Green/Refactor の小さな歩幅で進めたことを重視します。

## 対象コードの配置

低レベルでテストすべきロジックは、原則として `Logic/` 以下へ配置します。

例:

```text
ReinBalance/Source/ReinBalance/Public/Survivors/Logic/
ReinBalance/Source/ReinBalance/Private/Survivors/Logic/
```

今後ロジック専用モジュールを追加する場合も、同じ考え方で View、Editor、HTTP、NNE、アセット依存から分離します。

低レベルロジックの例:

- 状態遷移
- 数値計算
- 報酬計算
- 衝突・距離・範囲判定
- クールダウン
- レベルアップ
- observation schema / feature layout
- 入力値の正規化
- 境界値の丸めや clamp

`AActor`、`UActorComponent`、Widget、Mesh、Material、World、PIE、HTTP service、ONNX/NNE ロードに依存する処理は、テスト対象ロジックから切り離します。必要なら、それらは薄いアダプタとして Logic 側のピュア関数や小さなサービスを呼ぶ形にしてください。

## テスト必須ルール

`Logic/` 以下に低レベルな処理を追加・変更する場合、対応する Low Level Logic Tests を必ず追加または更新します。

テストなしで許容されるのは、コメント修正、ログ文言修正、純粋なリネームなど、振る舞いが変わらないことが明確な変更だけです。

レビュー時に見ること:

- 先にテストが書かれているか。
- 実装対象の振る舞いがテスト名とアサーションから分かるか。
- 正常系だけでなく境界値と失敗パターンがあるか。
- 既存仕様を壊していないことがテストで説明できるか。
- テストが Editor / PIE / World / アセットに依存していないか。

## テストコードでやってはいけないこと

実装コードをテストコードへコピペしないでください。

悪い例:

- テスト内で本番ロジックと同じ計算式を丸ごと再実装する。
- private 実装の都合をそのまま期待値にする。
- 現在の実装結果を固定して、仕様として妥当か確認しない。
- World やアセットを起動しないと通らないテストを Low Level Logic Tests に入れる。

良い例:

- Logic 側にピュア関数や小さな計算クラスを作り、実装コードとテストコードの両方から使用する。
- 期待値は仕様、表、手計算できる小さな例から導く。
- 仕様が複雑な場合は、テストケース表を作って入力と期待結果を明示する。
- ランダム要素は seed 固定、または乱数源を注入する。

## テスト内容の基準

すべての状況を網羅するつもりで、少なくとも次を確認します。

正常系:

- 代表的な入力で期待どおり動く。
- 複数ステップの状態遷移が期待どおり続く。
- ゲームごとの典型ケースが仕様名として読める。

境界値:

- 最小値、最大値。
- 0、1、ちょうど閾値、閾値の直前、閾値の直後。
- 空配列、1 要素、上限要素数。
- 距離 0、半径ちょうど、半径外。
- HP 0、XP ちょうどレベルアップ、クールダウン 0。

失敗・異常系:

- 不正な enum。
- 負の値。
- NaN / Inf が入り得る箇所。
- null 相当の参照や空データ。
- 上限を超えた入力。
- 仕様上 clamp すべき値。

組み合わせ:

- 複数条件が同時に成立する場合。
- 優先順位がある場合。
- 同フレームで複数イベントが起きる場合。
- 状態がリセットされた後に古い値が残らないこと。

回帰:

- 過去に壊れたケースは、必ず回帰テストとして残す。
- バグ修正時は、最初にそのバグを再現する失敗テストを書く。

## テスト名とタグ

テスト名は振る舞いを説明する文にします。

例:

```cpp
TEST_CASE("Survivors leveling consumes XP and carries overflow", "[unit][survivors][logic]")
```

タグ:

- `[unit]`: 通常の低レベル単体テスト。
- `[logic]`: Logic 配下の純ロジック。
- `[survivors]`, `[coin]`, `[balance]`: ゲーム別。
- `[regression]`: バグ再発防止。
- `[integration]`: 低レベルロジック同士の組み合わせ。
- `[slow]`: 通常の TDD ループから外す重いテスト。

通常の TDD では `[unit]` を中心に回します。重いテストや統合寄りのテストを追加する場合も、低レベル単体テストで仕様を先に押さえてください。

## 実装設計の指針

テストしやすくするために、実装を不自然に曲げないでください。ただし、ロジックをピュアに保つ設計は歓迎します。

推奨:

- 入力と出力が明確な関数にする。
- 時間、乱数、外部状態を引数または依存として注入する。
- 小さな値オブジェクトや構造体で状態を表現する。
- Engine 依存の型が不要なら標準 C++ または Core の型に寄せる。
- Actor / Component は薄くし、判断や計算は Logic へ移す。

避ける:

- `Tick` 内に直接複雑な計算を書く。
- View 更新とゲーム判断を同じ関数に混ぜる。
- アセットロード結果をロジック判定に直接使う。
- グローバル状態や現在時刻に直接依存する。
- テストのためだけの public API を安易に増やす。

## 完了条件

Logic の変更は、次を満たしてから完了とします。

- 対応する Low Level Logic Tests が追加または更新されている。
- 先に失敗するテストを書いたことを説明できる。
- 正常系、境界値、失敗パターン、必要な組み合わせがある。
- テストコードに実装コードのコピペがない。
- `Tools/RunLowLevelTests.ps1 -Filter "[unit]"` が成功している。
- 必要に応じて通常の UE5 ビルドも成功している。

## 参考

- https://dev.epicgames.com/documentation/unreal-engine/build-and-run-low-level-tests-in-unreal-engine
- https://dev.epicgames.com/documentation/unreal-engine/write-low-level-tests-in-unreal-engine
- https://dev.epicgames.com/documentation/unreal-engine/types-of-low-level-tests-in-unreal-engine
- https://qiita.com/koorinonaka/items/c674d7e3140072ca5ad1

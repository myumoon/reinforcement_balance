# Verification budget

変更に対する検証をfocused、affected suite、full suiteの3段階に分けます。品質ゲートは維持し、同じ証跡を同一commitで重複取得しません。予算と停止条件は [`token_budget_policy.md`](token_budget_policy.md) を正本とします。

## 3段階の使い分け

1. **Focused:** 実装中に、変更した契約・CLI・挙動へ直接対応するtestだけを実行する。
2. **Affected suite:** 複数fileへ影響するinterface変更やfocused test完了時に、影響を受けるsuiteを1回実行する。
3. **Full suite:** PRの最終commitで、変更領域に対応するfull suiteを1回実行する。

同じcommandを再実行する前に、HEADが変わったか、前回結果が今回の主張を証明できない理由を確認します。

## 変更領域別matrix

| 変更領域 | Focused | Affected／full |
|---|---|---|
| Common契約 | 対象test file | Common suite |
| Training CLI | 対象CLI test | Training suite |
| ArtifactStore | 変更した挙動の対象test | Artifacts suite |
| UE C++ | 対象LLT filter | Editor buildとLLT |
| 文書・絶対パス | repository path hygiene test | 原則として追加suiteなし |

代表的なfocused commandは次の形式にします。

```powershell
python -m pytest Tools/Common/tests/<target_test>.py -q --tb=short --maxfail=1
python -m pytest Tools/Training/tests/<target_cli_test>.py -q --tb=short --maxfail=1
python -m pytest Tools/Artifacts/tests/<target_test>.py -q --tb=short --maxfail=1
Tools/Tests/RunLowLevelTests.ps1 -Filter "[unit][<area>]"
```

UE C++変更では [`docs/ue5_build.md`](../ue5_build.md) のEditor buildと、[`docs/testing/low_level_tests.md`](../testing/low_level_tests.md) のLLTを両方実行します。文書・絶対パス変更では、対象branchで利用可能なrepository path hygiene testを実行します。guardがbaseへ未統合の場合は、無関係な実装を取り込まず、baseとの差分で新規違反がないことを記録します。

## Pytest出力

- 通常は `-q --tb=short --maxfail=1` を使う。
- 失敗時は最初のfailureから原因候補を絞る。
- 原因を特定した後だけ、対象test 1件をverboseで再実行する。
- harness設定が原因のfull test失敗は2回で停止し、設定と失敗要約を引き継ぐ。
- basetempを指定する場合は親directoryの存在をpreflightで確認する。

## Full suite結果の共有

Common、Artifacts、Trainingのfull suiteは、同一commitで各1回だけ実行します。成功結果を別agentが再実行しません。commit hashと結果を次の形式で共有します。

| Commit | Suite／command | Passed | Failed | Skipped | 実行者 | 備考 |
|---|---|---:|---:|---:|---|---|
| `<sha>` |  |  |  |  |  |  |

HEAD変更後は、変更領域に影響する証跡だけを取り直します。文書だけの追補でcode suiteを再実行する場合は、その理由を記録します。

## Recursive scanとtool出力

recursive scanを実行する前に、次を固定します。

- 対象directory
- 最大depth
- 除外directory（test一時領域、build生成物、cacheを含む）
- 検索patternと必要な出力列

最初のcallでは一覧を表示せずcountだけを取得します。countが300件を超える場合はdirectory、pattern、depthを狭めます。diff、traceback、logもsizeまたはline countを先に確認し、1 call 300行または20,000文字以内で取得します。

失敗時の共有は、command、exit code、最初のfailure、原因候補、次の1 actionだけに限定します。

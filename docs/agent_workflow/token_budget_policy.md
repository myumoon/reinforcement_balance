# AI作業量・コンテキスト予算

この文書は、AI agentによる実装・検証・reviewの作業量とコンテキスト消費を制御するための正本です。作業開始時に受入条件と予算を固定し、品質ゲートを維持したまま、重複実行と無制限なscope拡大を防ぎます。

## 作業開始時のcheckpoint

実装前に、受入条件、対象・対象外のsubsystem、想定files／lines／commits、agent数、test計画、review回数、権限・network・UE buildのpreflight結果を記録します。上限を超える見込みがある場合は、実装を始める前にPRまたはsessionを分割します。

記録には [`task_budget_checklist.md`](task_budget_checklist.md) を使用します。

## 運用予算

| 指標 | 通常上限 | 超過時の処置 |
|---|---:|---|
| 独立subsystem | 2 | 別PRへ分割する |
| 変更files | 20 | 別PRへ分割する |
| 追加・変更lines | 1,000 | 別PRへ分割する |
| commits | 8 | checkpointを作り、継続承認を得る |
| implementer agent | 1 | 2人目を使う理由を提示する |
| reviewer agent | 1 | 追加reviewはユーザー承認制とする |
| review fix rounds | 2 | architectureとscopeを再評価して停止する |
| full test suite | 2 | baselineまたは最終のどちらかに限定する |
| 1 tool callの出力 | 300行または20,000文字 | 件数・サイズを確認して範囲を絞る |
| agent brief | 1,500語 | summaryと参照pathへ分離する |

変更が2 subsystem、20 files、1,000 linesのいずれかを超える見込みなら、実装前にPRを分割します。新しい依頼が現在の受入条件を変える場合も、進行中のPRへ追加しません。

## 標準運用

- 通常はinline実装とPR単位の独立review 1回を使います。taskごとのsubagent reviewは明示承認がある場合だけ使います。
- subagentへ会話全履歴を渡さず、[`scoped_agent_brief.md`](scoped_agent_brief.md) を使って対象file、受入条件、base/head、既知のtest結果だけを渡します。通常は `fork_turns="none"` を使います。
- reviewerは1 PRにつき1人とし、追加reviewはユーザー承認制とします。specialist調査には実装・commit権限を与えません。
- agentへdiff全文を貼らず、base/headを共有します。100 KBを超えるdiffはfile単位に分割し、sizeを確認してから渡します。
- focused testは実装中、full suiteは最終時に1回実行します。同一commitの成功suiteを別agentが再実行しません。
- review fixは2 roundsで停止し、新しいImportantが残る場合はscopeまたは設計をユーザーと再確認します。
- recursive scan、full traceback、diff全文は先に件数・サイズを確認し、1回300行または20,000文字を上限にします。
- 品質ゲートは削除せず、変更に応じて実行範囲と実行タイミングを段階化します。

## 消費量の代理指標

実測token値の代わりに、次の値をtaskごとに記録します。

- 変更files、追加・変更lines、commits
- implementer／reviewer agent数、review fix rounds
- 実行したtest case数とfull suite回数
- 最大tool出力行数

3件のpilot後に [`token_budget_retrospective.md`](token_budget_retrospective.md) へ記録し、上限の維持・変更を判断します。

## 停止条件

次のいずれかに該当した場合は作業を停止し、scope、設計、分割先、または必要な権限をユーザーと確認します。

- 新しい依頼が既存PRの受入条件を変えた。
- review round 2の後に新しいImportant findingが出た。
- sandbox／network／Git承認が、権限付きの再試行1回後も拒否された。
- full testがtest harness設定の問題で2回失敗した。
- 変更が20 filesまたは1,000 linesを超えた。
- 大きなlogやdiffを読む前に、そのサイズと対象範囲を説明できない。

停止時は、完了済み、未完了、次の1 action、必要な権限だけを200語以内で引き継ぎます。

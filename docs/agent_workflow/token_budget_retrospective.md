# Token budget retrospective

対象policy: [`token_budget_policy.md`](token_budget_policy.md)

## 3件の実測値

| Pilot | Task／PR | Files | Lines | Commits | Implementer agents | Reviewer agents | Review fix rounds | Full suites | 最大tool出力行数 | 品質低下 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 |  |  |  |  |  |  |  |  |  | [ ] 有 / [ ] 無 |
| 2 |  |  |  |  |  |  |  |  |  | [ ] 有 / [ ] 無 |
| 3 |  |  |  |  |  |  |  |  |  | [ ] 有 / [ ] 無 |

## 上限超過

超過ごとに原因を1つだけ選択する: `予測漏れ` / `scope追加` / `test失敗` / `review追加`

| Pilot | 超過指標 | 上限 | 実測 | 原因 |
|---:|---|---:|---:|---|
|  |  |  |  |  |
|  |  |  |  |  |
|  |  |  |  |  |

## 判定値

- 上限内の作業数: / 3
- 上限超過の作業数: / 3
- 品質低下なしの作業数: / 3
- merge後のCritical／Important件数: /

## 判断

- [ ] 3件すべてが上限内、かつ品質低下なし: 現行ルールを維持する。
- [ ] 2件以上が上限超過: files上限を15、review fix上限を1へ下げる案をユーザーへ提示する。
- [ ] 上記以外: 上限は据え置き、超過原因と品質結果をユーザーと再確認する。

決定: [ ] 維持 / [ ] 厳格化案を提示 / [ ] 再確認

# Task budget checkpoint checklist

実装を始める前にこのchecklistを埋め、[`token_budget_policy.md`](token_budget_policy.md) の上限内であることを確認します。受入条件が変わった場合は同じPRへ追加せず、新しいchecklistを作成します。

## 1. 依頼とscope

- [ ] 依頼名:
- [ ] 依頼の受入条件:
  -
- [ ] 対象subsystem:
- [ ] 対象filesまたはdirectory:
- [ ] 対象外:
- [ ] 新しい依頼を同じPRへ追加しないことを確認した:

## 2. 見積りと分割（想定files／lines／commits）

| 指標 | 見積り | policy上限内 |
|---|---:|---|
| subsystem数 |  | [ ] |
| files数 |  | [ ] |
| 追加・変更lines |  | [ ] |
| commits |  | [ ] |

- [ ] 上限超過時の分割先:
- [ ] 分割後も受入条件を独立して検証できる:

## 3. Agentとreview

- [ ] implementer agent数:
- [ ] reviewer agent数:
- [ ] fork方式（通常は会話履歴を渡さない）:
- [ ] scoped briefのpath:
- [ ] review予定回数:
- [ ] review fix上限到達時の停止先:

## 4. Verification計画

- [ ] focused test（実装中）:
- [ ] affected suite（必要な場合）:
- [ ] full suite（最終commitで1回）:
- [ ] 同一commitの成功suiteを再実行しない共有方法:
- [ ] 想定test case数:

## 5. Preflight

- [ ] filesystem／sandbox権限:
- [ ] Git branch／worktree／remote権限:
- [ ] network／外部service:
- [ ] Python環境と固定version:
- [ ] UE build環境（`UE_ROOT`、target、configuration）:
- [ ] 不足項目と必要な承認:

## 6. 開始判定

- [ ] すべての見積りがpolicy上限内である。
- [ ] 超過項目は実装前に別PRまたはsessionへ分割した。
- [ ] 受入条件、予算、停止条件をユーザーへ提示した。
- 判定: [ ] 開始可 / [ ] 分割後に再確認 / [ ] 権限待ち

## 元レビュー10件への適用確認

10件を単一PRとして開始せず、責務と検証単位が独立する少なくとも次の4 PRへ分割します。

| 分割先 | Scope | 主なverification |
|---|---|---|
| build | UBT action graph、Editor build、LLT導線 | Editor build、LLT |
| durability／fidelity | 共通contract、lifecycle、fidelity判定 | 対象Common tests |
| ArtifactStore | store、bundle、identity | Artifacts suite |
| behavior fixes | UE C++／Trainingの個別挙動修正 | 対象test file、必要なLLT |

- [ ] 各PRに独立した受入条件がある。
- [ ] 後続PRの変更を先行PRのreviewへ混ぜない。

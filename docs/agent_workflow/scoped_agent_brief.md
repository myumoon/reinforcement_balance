# Scoped agent brief

このtemplateは、agentへ会話履歴を渡さず `fork_turns="none"` で依頼するために使います。[`token_budget_policy.md`](token_budget_policy.md) の上限を確認し、briefを1,500語以内に収めます。

briefは次の7項目だけで構成します。会話の経緯や関連しない設計判断は追加せず、必要な正本文書をpathで参照します。

## 1. 目的

- 解決する問題:
- 完了時の結果:

## 2. 対象file

- 対象fileと必要なline／symbol:
  -
- 読み取り用の参照file:
  -
- diffは本文へ貼らず、base/headから取得する。diffが100 KBを超える場合はfile単位に分割し、各fileのsizeを先に共有する。

## 3. 対象外

- 変更してはいけないfile／subsystem:
  -
- 判断または実装を依頼していない事項:
  -

## 4. 受入条件

- [ ]
- [ ]
- [ ] scope外の変更がない。

## 5. Base／head

- Repository:
- Base SHA:
- Head SHA:
- Review対象commitまたはfile:

## 6. 実行test

- 実行してよいfocused test:
- 既知のtest結果（commit hash、command、結果）:
- 未実行testと理由:
- 同一commitで成功済みのsuiteは再実行しない。

## 7. 出力形式

- Findings: `Critical`、`Important`、`Minor`の順で、`file:line`、根拠、必要なactionを記載する。該当なしの場合は`なし`と明記する。
- Test: agent自身が実行したcommandと結果だけを記載する。
- Scope: 読んだfileと、未確認の領域を記載する。
- Specialist調査ではfileを変更せず、調査結果と推奨する次の1 actionだけを返す。

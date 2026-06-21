---
name: notion-apply-run-issue
description: Use when the user says "notionの最新の課題に反映", "notionで実装を記載", or wants to record implementation details, hypotheses, and adoption decisions for a training run in Notion's RL learning list.
---

# Notion RL学習リスト 実装内容記録スキル

チャット内容または指定された run の実装内容・仮説・採用判断をまとめ、「RL学習リスト」に新規ページとして追加する。

---

## Step 1: 対象 run を特定する

ユーザーの発言またはチャット履歴から以下を読み取る:
- 対象の run 名（例: `v09_03`、`05_fix_penalty`）
- バージョン（例: `v09`、`v10`）
- ページタイトルや URL が直接指定された場合はそちらを優先する

---

## Step 2: RL学習リストデータベースを検索する

### 2-1. データベース自体を notion-search で探す

```
notion-search(
  query="RL学習リスト",
  page_size=5
)
```

結果から「RL学習リスト」という名前のデータベースページを特定し、その URL または ID を取得する。

### 2-2. データベースの schema と data_source_url を取得する

```
notion-fetch(
  id="<RL学習リストのURL or ID>"
)
```

レスポンス内の `<data-source url="collection://...">` タグから `data_source_url` を取得する。
データベースのプロパティ名（タイトル列の名前など）もこのレスポンスから確認する。

### 2-3. 最新の課題ページを SQL で検索する

```
notion-query-data-sources(
  data={
    "data_source_urls": ["collection://<data_source_id>"],
    "query": "SELECT * FROM \"collection://<data_source_id>\" WHERE \"<タイトル列名>\" LIKE ? LIMIT 10",
    "params": ["%<run名>%"]
  }
)
```

`<タイトル列名>` は Step 2-2 で取得した schema のタイトルプロパティ名（多くの場合 `要約` や `名前`）を使う。
列名は必ず二重引用符 (`"`) で囲むこと（SQLite の識別子クォート）。
列名自体に `"` が含まれる場合は `""` にエスケープする（例: `"my""col"`）。

### 2-4. 候補の評価

- **1件だけ一致**: そのページを対象ページとして使用する
- **複数件一致**: 候補を列挙してユーザーに正式タイトルまたは URL を確認する
  ```
  以下のページが候補として見つかりました。どのページに反映しますか？
  1. [タイトル] (更新日: yyyy-mm-dd)
  2. [タイトル] (更新日: yyyy-mm-dd)
  または正式なタイトルか URL を教えてください。
  ```
- **0件（ページが存在しない）**: ユーザーに確認する
  ```
  「{キーワード}」に一致するページが見つかりませんでした。
  - 正式なタイトルまたは URL を教えていただくか
  - 新規ページとして作成しますか？
  ```

---

## Step 3: 既存ページの内容を確認する

```
notion-fetch(
  id="<対象ページのID or URL>"
)
```

レスポンスから以下を確認する:
- 既存の「課題」セクションの内容（前回課題との対応付けに使用）
- ページ本文の構成（どのセクションが既に存在するか）

---

## Step 4: ページ本文を組み立てる

以下のセクション構成でページ本文を作成する:

```markdown
## 前回runの課題

{チャット内容・データから観測された問題点を箇条書き}
- {観測結果や指定内容}
- データがあれば最低限の数値・グラフを引用

## 問題の仮説

{課題から導いた仮説・改善アイディア}
- {仮説1}
- {仮説2}

## 採用判断

{次回 run に取り入れるもの・取り入れないものをチェック形式で}
- ✅ 採用: {アイディアと採用理由}
- ❌ 却下: {アイディアと却下理由}

## 実装概要

{実装の詳細・方針}
- PR: {PRへのリンク（わかれば）}
- 修正方針: {変更内容}
- 参考資料: {論文・記事リンク（あれば）}
- 検証内容: {何を確認するか}

## 実行結果

- [ ] TODO: run 完了後に結果を追記する
```

**注意**: `content` 内にページタイトルを重複して書かないこと。

---

## Step 5: RL学習リストに新規ページを作成する

### 5-1. タイトルの決定

実装内容から **10文字程度** の簡潔なタイトルを生成する。

例:
- 「報酬係数を下げて方策崩壊を防ぐ実装」 → `報酬係数削減実装`
- 「attention機構を修正して安定性を向上」 → `attention安定化`
- 「カリキュラム境界条件を修正する」 → `カリキュラム修正`

### 5-2. データソース schema の確認

Step 2-2 で取得した schema でタイトルプロパティ名を確認する（例: `要約`）。

### 5-3. ページ作成

```
notion-create-pages(
  parent={
    "type": "data_source_id",
    "data_source_id": "<Step 2-2 で取得した data_source_id>"
  },
  pages=[
    {
      "properties": {
        "<タイトルプロパティ名>": "<10文字程度のタイトル>"
      },
      "content": "<Step 4 で組み立てた本文>"
    }
  ]
)
```

**注意**:
- `content` 内にページタイトルを重複して書かないこと
- `parent.data_source_id` は `collection://` プレフィックスを**除いた** UUID 部分を使用する
- タイトルプロパティ名は必ず Step 2-2 の schema から取得した実際の名前を使うこと

---

## Step 6: 完了後の判断

もしチャット内容からすでに run の実行結果が判明している場合は、
`notion-add-issue` スキルを呼び出して次回課題ページも作成する。

---

## Step 7: 完了報告

```
完了しました。

【RL学習リストへの追加】
  タイトル: {タイトル}
  URL: {URL}
  セクション: 前回runの課題 / 問題の仮説 / 採用判断 / 実装概要 / 実行結果(TODO)
```

---

## エラーハンドリング

| ケース | 対応 |
|---|---|
| `notion-search` でRL学習リストが見つからない | `notion-fetch` にNotionワークスペースのURLを直接渡してみるか、ユーザーにURLを確認する |
| 候補ページが複数ある | 候補を列挙してユーザーに選択を求める |
| 課題ページが見つからない | 新規でも作成するか確認する |
| run 情報が不十分 | 「実装内容の詳細を教えてください」とユーザーに確認する |
| `notion-create-pages` が `data_source_id` エラーを返す | `database_id` 形式（`parent.type="database_id"`）に切り替えてリトライする |
| Notion MCP が応答しない | エラー内容をそのまま報告し、手動対応を案内する |

---

## ツール呼び出し早見表

| 目的 | ツール | 主要パラメータ |
|---|---|---|
| RL学習リストDBを探す | `notion-search` | `query="RL学習リスト"` |
| DB schema/data_source_url 取得 | `notion-fetch` | `id=<DB URL>` |
| ページを SQL で絞り込む | `notion-query-data-sources` | SQL LIKE 検索、列名はダブルクォート |
| 既存ページ内容を読む | `notion-fetch` | `id=<ページ URL>` |
| 新規ページを作成する | `notion-create-pages` | `parent.data_source_id`, `properties`, `content` |

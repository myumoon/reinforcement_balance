---
name: notion-add-issue
description: Use when the user says "notionにrun結果を反映", "notionに課題を記載", "～のページに結果を反映", or wants to record training run results and next issues to Notion's RL learning list.
---

# Notion RL学習リスト 結果反映スキル

チャット内容または指定された run の情報から「RL学習リスト」の前回 run ページを特定し結果を追記する。
その後、次回 run 用の課題ページを作成する。

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

### 2-3. 前回 run ページを SQL で検索する

```
notion-query-data-sources(
  data={
    "data_source_urls": ["collection://<data_source_id>"],
    "query": "SELECT * FROM \"collection://<data_source_id>\" WHERE <タイトル列名> LIKE ? LIMIT 10",
    "params": ["%<run名>%"]
  }
)
```

`<タイトル列名>` は Step 2-2 で取得した schema のタイトルプロパティ名（多くの場合 `要約` や `名前`）を使う。

### 2-4. 候補の評価

- **1件だけ一致**: そのページを前回 run ページとして使用する
- **複数件一致**: 候補を列挙してユーザーに正式タイトルまたは URL を確認する
  ```
  以下のページが候補として見つかりました。どちらが前回の run ページですか？
  1. [タイトル] (更新日: yyyy-mm-dd)
  2. [タイトル] (更新日: yyyy-mm-dd)
  または正式なタイトルか URL を教えてください。
  ```
- **0件（ページが存在しない）**: ユーザーに確認する
  ```
  「{キーワード}」に一致するページが見つかりませんでした。
  - 正式なタイトルまたは URL を教えていただくか
  - 次回 run 用の新規ページのみ作成しますか？
  ```

---

## Step 3: 前回 run ページの既存内容を取得する

```
notion-fetch(
  id="<前回runページのID or URL>"
)
```

レスポンスから以下を確認する:
- 既存の「課題」セクションの内容（Step 3 で前回課題の検証に使う）
- ページ本文の末尾（追記位置の確認）

---

## Step 4: 前回 run ページに結果を追記する

`insert_content` + `position: {type: "end"}` で既存内容の末尾に追記する。

```
notion-update-page(
  page_id="<前回runページのID>",
  command="insert_content",
  position={"type": "end"},
  content="""
## 結果

{チャット内容から読み取ったrun結果を箇条書きで記載}
- スコア・報酬の推移
- 到達フェーズ・ステップ数
- 観察された挙動の変化
- その他特記事項

## 前回課題の検証

{Step 3 で取得した既存「課題」セクションに対する検証結果}
- ✅ 解決: {解決した課題}
- ❌ 未解決: {未解決の課題とその理由}
- ➖ 検証不可: {検証できなかった課題}

## 次回への課題

{次回 run で試みるべき課題・仮説を箇条書きで記載}
- {課題1}
- {課題2}
"""
)
```

**注意**: `content` 内でページタイトルは含めないこと（タイトルはプロパティとして別管理）。

---

## Step 5: 次回 run 用の新規ページを作成する

### 5-1. タイトルの決定

次回の課題内容から **10文字程度** の簡潔なタイトルを生成する。

例:
- 「報酬係数を下げて方策崩壊を防ぐ」 → `報酬係数削減検証`
- 「exploration を増やして停滞を打破する」 → `exploration強化`
- 「カリキュラム threshold を緩和する」 → `threshold緩和`

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
      "content": """
## 目的

{前回 run の結果を踏まえた、今回 run で検証したい仮説・目的}

## 課題

{次回 run で試みること・変更点の詳細}
- {変更点1}
- {変更点2}

## 参照

- 前回 run: [<前回ページタイトル>](<前回ページURL>)
"""
    }
  ]
)
```

**注意**:
- `content` 内にページタイトルを重複して書かないこと
- `parent.data_source_id` は `collection://` プレフィックスを**除いた** UUID 部分を使用する
- タイトルプロパティ名は必ず Step 2-2 の schema から取得した実際の名前を使うこと

---

## Step 6: 完了報告

```
完了しました。

【前回 run ページへの追記】
  ページ: {ページタイトル}
  URL: {URL}
  追記内容: run結果・課題検証・次回課題

【次回 run 用ページの作成】
  タイトル: {タイトル}
  URL: {URL}
```

---

## エラーハンドリング

| ケース | 対応 |
|---|---|
| `notion-search` でRL学習リストが見つからない | `notion-fetch` にNotionワークスペースのURLを直接渡してみるか、ユーザーにURLを確認する |
| 前回ページが複数候補ある | 候補を列挙してユーザーに選択を求める |
| 前回ページが見つからない | 「見つかりません」と伝え、新規ページのみ作成するか確認する |
| `notion-fetch` で `<data-source url>` が複数ある | `notion-query-data-sources` の `data_source_urls` に全て渡して横断検索するか、ユーザーに確認する |
| `notion-create-pages` が `data_source_id` エラーを返す | `database_id` 形式（`parent.type="database_id"`）に切り替えてリトライする |
| run 結果情報が不十分 | 「追記する run 結果の詳細を教えてください」とユーザーに確認する |
| Notion MCP が応答しない | エラー内容をそのまま報告し、手動対応を案内する |

---

## ツール呼び出し早見表

| 目的 | ツール | 主要パラメータ |
|---|---|---|
| RL学習リストDBを探す | `notion-search` | `query="RL学習リスト"` |
| DBの schema/data_source_url を取得 | `notion-fetch` | `id=<DB URL or ID>` |
| run ページを SQL で絞り込む | `notion-query-data-sources` | `data_source_urls`, `query` (LIKE 検索) |
| ページ本文を読む | `notion-fetch` | `id=<ページ URL or ID>` |
| 末尾に追記する | `notion-update-page` | `command="insert_content"`, `position={"type":"end"}` |
| 新規行（ページ）を作成する | `notion-create-pages` | `parent.data_source_id`, `properties`, `content` |

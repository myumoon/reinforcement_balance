---
name: setup-run
description: Use when the user says "{version}で{run-name}をセットアップ", "{run-name}をセットアップ", or wants to set up a new training run folder for Survivors. Examples: "v10で05_my_runをセットアップ", "05_fix_penaltyをセットアップ".
---

# Survivors run フォルダセットアップ

Survivors の訓練 run フォルダをセットアップする。

## 引数の解釈

ユーザーの発言から以下を読み取る:
- `VERSION`: `v` で始まる文字列（例: `v10`）
- `RUN_NAME`: それ以外の文字列（例: `05_reduce_penalty`）

どちらか一方または両方が省略された場合は Step 1 で決定する。

## Step 1: VERSION と RUN_NAME を決定する

### VERSION の決定

発言に VERSION が含まれる場合はそれを使う。含まれない場合:

```bash
ls "Tools/Training/runs/survivors/" | sort | tail -1
```

で最新バージョンを自動選択し「バージョンを {VERSION} と判定しました」と報告する。

### RUN_NAME の決定

発言に RUN_NAME が含まれる場合はそれをそのまま使う。含まれない場合:

1. 直近コミット履歴を確認する:
   ```bash
   git log --oneline -10
   ```
2. 現バージョンの既存 run 一覧を確認する:
   ```bash
   ls "Tools/Training/runs/survivors/{VERSION}/train/" | sort
   ```
3. 「このrunで何を変更・試みますか？」とユーザーに聞く
4. 回答をもとに `{連番}_{内容の英語スネークケース}` 形式で命名案を提示し確認を取る

命名例: `05_reduce_penalty_coeff`、`06_fix_early_death`
連番は既存 run の最大番号 + 1。

## Step 2: 前回 run フォルダを特定する

```bash
ls "Tools/Training/runs/survivors/{VERSION}/train/" | sort
```

名前順で最後のフォルダが前回 run。フォルダが存在しない（初回）場合は前バージョンを探す:

```bash
ls "Tools/Training/runs/survivors/" | sort
# 現バージョンのひとつ前のバージョンで同様に最後のフォルダを取得
ls "Tools/Training/runs/survivors/{PREV_VERSION}/train/" | sort | tail -1
```

## Step 3: フォルダを作成する

同名フォルダが既に存在する場合は「`{RUN_NAME}` は既に存在します。続行しますか？」と確認してから進む。

```bash
mkdir -p "Tools/Training/runs/survivors/{VERSION}/train/{RUN_NAME}/config"
```

## Step 4: reward_fn.py をコピーする

```bash
cp "Tools/Training/configs/{VERSION}/reward_fn.py" \
   "Tools/Training/runs/survivors/{VERSION}/train/{RUN_NAME}/config/reward_fn.py"
```

`Tools/Training/configs/{VERSION}/reward_fn.py` が存在しない場合は「reward_fn.py が見つかりません」と報告して中断する。

## Step 5: train_config.yaml をコピーする

前回 run の config フォルダから train_config.yaml をコピーする:

```bash
cp "Tools/Training/runs/survivors/{PREV}/config/train_config.yaml" \
   "Tools/Training/runs/survivors/{VERSION}/train/{RUN_NAME}/config/train_config.yaml"
```

前回 run の train_config.yaml が存在しない場合は「前回 run の train_config.yaml が見つかりません。コピー元のパスを教えてください」とユーザーに確認する。

## Step 6: train_config.yaml を更新する

Edit ツールで以下の4フィールドを更新する:

| フィールド | 更新値 |
|---|---|
| `version_name` | `{VERSION}` |
| `run_name` | `{RUN_NAME}` |
| `wandb_project` | `survivors-{VERSION}` |
| `wandb_run_name` | `{RUN_NAME}` のアンダースコアをハイフンに変換した値 |
| `reward_fn` | `runs/survivors/{VERSION}/train/{RUN_NAME}/config/reward_fn.py` |

`wandb_run_name` の変換例: `05_reduce_penalty_coeff` → `05-reduce-penalty-coeff`

## Step 7: 完了報告

```
セットアップ完了:
  📁 Tools/Training/runs/survivors/{VERSION}/train/{RUN_NAME}/
     └── config/
         ├── reward_fn.py        (configs/{VERSION}/reward_fn.py からコピー)
         └── train_config.yaml   (前回 run から引き継ぎ・更新済み)

train_config.yaml の更新箇所:
  version_name:   {VERSION}
  run_name:       {RUN_NAME}
  wandb_project:  survivors-{VERSION}
  wandb_run_name: {WANDB_RUN_NAME}
  reward_fn:      runs/survivors/{VERSION}/train/{RUN_NAME}/config/reward_fn.py
```

## エラーハンドリング

| ケース | 対応 |
|---|---|
| `Tools/Training/runs/survivors/` が存在しない | 「runs ディレクトリが見つかりません」と報告して中断 |
| `configs/{VERSION}/reward_fn.py` が存在しない | 「reward_fn.py が見つかりません」と報告して中断 |
| 前回 run の `train_config.yaml` が見つからない | コピー元パスをユーザーに確認 |
| 同名 run フォルダが既に存在する | 上書き確認をユーザーに取ってから続行 |

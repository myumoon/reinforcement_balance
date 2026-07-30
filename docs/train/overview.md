# 訓練スクリプト概要

`Tools/Training/` 配下の Python スクリプト群の役割と起点。
CLI オプションの詳細は `python <script>.py --help` またはソースコードを参照すること。

## エントリーポイント

### `train.py`
PPO 訓練のメインエントリーポイント。ゲーム選択・並列環境数・resume・各種コールバックを制御する。
`--dry-run` で UE5 なしのスタブ環境で動作確認できる。

### `eureka_loop.py`
LLM が報酬シェーピング関数を反復生成・改良する EUREKA ループ。
生成した関数は `eureka_results/<run_name>/` に保存され、`train.py --reward-fn` で適用できる。
reward_fn を実装・修正する際は [`reward_fn_policy.md`](reward_fn_policy.md) のチェックリストを参照すること。

### `export_onnx.py`
訓練済みモデルを ONNX 形式に変換し `ReinBalance/Content/Models/` に出力する。
UE5 の NNERuntimeORT で推論するために必要。

### `games/survivors/survivors_curriculum_test.py`
訓練済みモデルにカリキュラム昇格チェックを行う推論専用スクリプト。
特定フェーズでのパフォーマンスを訓練なしで単体検証できる。

## Python モジュール構成

`Tools/Training/` は 3 層で構成されている。

| ディレクトリ | 役割 | 変更頻度 |
|---|---|---|
| `base/` | 全ゲーム共通の抽象基底クラス | 低 |
| `common/` | ゲーム非依存のユーティリティ | 低 |
| `games/<game>/` | ゲーム固有の実装 | 高 |

新しいゲームを追加する場合は `games/<game>/` にモジュールを作成し、
`train.py` の `_GAME_DEFAULTS` と env 選択分岐に追加する。

## DeployObsV1 wrapper

`games/survivors/deploy_obs_wrapper.py` は privileged raw observation を直接 slice せず、target camera の screen projection と visibility/occlusion/clipping を経て、Deployment と共有する named-estimate adapter から `value + validity + age` tensor を生成する。

本番相当の学習・評価には `DeployObsWrapper.release()` を使う。`oracle_diagnostic()` は全 state と比較する診断専用で release artifact を生成できない。VecNormalize は deploy tensor の外側へ新規 fit し、既存 privileged observation の統計を流用しない。

DeployObs schema または release adapter の producer hash を変更した場合、既存 00-05 baseline は意図的に失効する。00-05 の verdict/gating 契約は変更せず、01-05 formal 収集前に integration fidelity verdict を再発行する。詳細は [`docs/deployment/deploy_obs_v1.md`](../../deployment/deploy_obs_v1.md) を参照。

## Perception error profile

`--perception-error-profile` に
`Tools/Training/configs/perception_error_bootstrap_v1.json` のような
`perception_error.v1` JSON を指定できる。03-02 では profile を共有契約で
fail-closed 検証し、その canonical SHA-256 を `log/run_meta.json` の
`perception_error_profile_hash` と resolved config に記録する。

この段階では profile の学習環境への自動適用は行わない。画面由来の
`DeployObsV1` tensor を個別に検証する場合は
`games.survivors.perception_error_wrapper.PerceptionErrorWrapper` を使う。
wrapper は latency、burst dropout、座標ノイズ、categorical confusion、
false entity count clipping の順序を固定し、worker ごとの corruption state を
`get_corruption_state()` / `set_corruption_state()` で保存・再開できる。

## Survivors Value Source descriptor

Survivors IS2 が `curriculum_complete` で正常終了すると、`train.py` は model /
VecNormalize、obs schema、resolved config、code、package freeze、action semantics、
coverage を束縛した `survivors.value_source_descriptor.v1` を
`result/value_source_descriptor.json` へ atomic publish する。SIGINT、接続断、例外時は
`log/value_source_descriptor.incomplete.json` だけを残し、descriptor を昇格しない。

source worktree が dirty の場合はデフォルトで開始を拒否する。意図的に許可する場合だけ
`--allow-dirty-value-source --value-source-artifact-store <store-root>` を指定する。
この場合は binary patch が content-addressed artifact store に保存され、その SHA-256 が
descriptor identity に含まれる。

保存済み run は次の CLI でも再監査できる。

```bash
python survivors_value_source_audit.py \
  --run-dir runs/survivors/<version>/train/<run> \
  --obs-schema-json runs/survivors/<version>/train/<run>/result/obs_schema.json \
  --created-at-utc 2026-07-29T00:00:00Z
```

終了コードは `0=probe ready`、`2=not ready`、`3=invalid input`。descriptor は immutable
source のみを表し、`ready_for_labels`、teacher validation、verdict は含めない。これらは
後続 phase で descriptor を参照する別 artifact として発行する。

## Survivors recurrent value choice scorer

`survivors_value_choice_probe.py` は immutable Value Source descriptor、level-up preview
JSON、policy-bound critic context NPZ を読み、`survivors.value_choice_ranking.v3` JSONL を
生成する。PPO/RecurrentPPO は保存 zip の policy class から自動判定し、model /
VecNormalize の raw byte hash、observation schema、`shared_lstm`、
`enable_critic_lstm`、hidden size、layer count、policy state schema hash をロード時に
再検証する。

```bash
/usr/bin/python3 survivors_value_choice_probe.py \
  --manifest runs/survivors/<run>/result/value_source_descriptor.json \
  --preview-json /path/to/level_up_preview.json \
  --context-npz /path/to/critic_context.npz \
  --output-jsonl /path/to/value_choice_ranking.jsonl
```

preview JSON は HTTP preview の field に `environment_step` を加えた固定形式とする。
pending/base observation は recurrent state を進めず、全 candidate は同一 `hidden_in`
から評価する。selected post-choice observation の state 更新は
`RecurrentPolicySession.commit_selected()` だけが exactly once 実行する。

`--zero-state-smoke` は診断専用で、出力の
`ready_for_training_label=false` を強制する。formal ranking 自体も label release verdict
ではないため ready を主張せず、tie は epsilon `1e-5` のまま後続 verdict へ渡す。

UE5 build / LLT: 未実行（Windows 専用）。real completed source model の label release
判定と HTTP 接続を使う burn-in end-to-end integration は後続 phase で実施する。

## Survivors choice trace dataset collection

`collect_survivors_value_choices.py` は current-hash `integration` fidelity verdict、immutable
Value Source、UE5 external choice API を親に持つ formal collector である。baseline、
stale、missing、blocking verdict では起動しない。behavior は既定
`epsilon_source_scorer` / `epsilon=0.20` で、teacher ranking と selected behavior は別
field に保存する。propensity は候補数を `K` として、teacher best は
`1-epsilon+epsilon/K`、その他は `epsilon/K` を記録する。

```bash
python collect_survivors_value_choices.py \
  --manifest /artifact/value-source/result/value_source_descriptor.json \
  --fidelity-verdict /artifact/fidelity/integration-verdict.json \
  --current-producer-hashes /artifact/fidelity/current-producer-hashes.json \
  --artifact-store /artifact/survivors-choice-datasets \
  --seed-start 1000 --seed-end 1099 --episode-count 100 \
  --shard-size 100 --ue5-ports 8767 8777 --epsilon 0.20
```

dataset ID の既定形式は
`survivors-value-choices-<source identity先頭12桁>-seeds-<開始>-<終了>` である。明示
`--dataset-id` も使用できるが、同じ ID の既存 manifest へ異なる source identity を追加
することはできない。manifest と各 row には source、model、VecNormalize、observation
schema、policy state schema、fidelity verdict の content hash を provenance として保存
する。reset、全 movement action、全 preview/choice request/ack、external decision ID も
ordered replay events に残す。

dataset は Git worktree 外の `--artifact-store` にのみ作成する。各 shard は canonical
JSONL metadata、pickle-free compressed NPZ、commit marker の組で、全 ndarray は
`float32` / `int32` に固定する。actor/critic LSTM h/c の元 shape は維持する。容量は概ね
`decision数 × (base observation + 全candidate observation + selected observation +
pi/vf h/c) × 4 byte` に JSONL/NPZ overhead を加えた値であり、圧縮率は observation と
state に依存する。formal 件数の storage/wall-clock 上限は pilot 後に固定するため、この
CLI は budget を推測しない。

停止時は commit 前 shard が `.staging` に残る。次回起動は中断 staging、partial NPZ、
JSONL/NPZ count mismatch を `quarantine/` へ隔離する。shard directory の確定後かつ
manifest 更新前に停止した場合だけ、commit marker と全 hash/count の read-back 成功を
条件に manifest へ exactly once recovery する。collector journal は選択と ack を
decision ID ごとに保存し、process resume でも別 choice/record を発行しない。record ID は
source identity + episode logical ID + external decision ID の canonical SHA-256 なので、
HTTP retry と episode 再実行は deduplicate される。manifest histogram は shard commit
時だけ更新される。

artifact store の backup/retention は dataset directory 全体（manifest、`shards/`、
`journals/`、`quarantine/`）を同じ世代として扱う。NPZ/JSONL は Git に追加しない。
formal collection 前に current producer hashes で integration fidelity verdict を再検証
し、source descriptor と fidelity verdict は dataset と同じ artifact store の immutable
親として保持する。

UE5 build / LLT: 未実行（Windows 専用）。PIE 100 decisions pilot、formal storage/
parallel-port/wall-clock budget の固定は後続 phase で実施する。

## 関連ドキュメント

- UE5 との通信仕様: [`ue5_env.md`](ue5_env.md)
- 実装上の注意事項・既知の問題: [`impl_notes.md`](impl_notes.md)
- Survivors reward_fn 設計ポリシー: [`reward_fn_policy.md`](reward_fn_policy.md)

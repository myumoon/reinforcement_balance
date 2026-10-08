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

## DeployObs v2 wrapper と deploy_raw

Training の既定 DeployObs schema は v2（`DeployObsSchema.default_v2()`）。`deployable_policy_trainer.py` と `perception_error_wrapper.py` の既定が v2 になり、v1 で作った dataset / checkpoint は schema hash 不一致で拒否される。`collect_survivors_combat_distillation.py` も 03-08 で v2 に切り替えた（synthetic 経路を含む）。

UE5 から v2 観測を作る経路は次のとおり。

```
SurvivorsEnv（/params deploy_raw=true）
  → DeployRawEnv（games/survivors/deploy_raw_env.py）: HTTP の deploy_raw → v2 raw dict
  → DeployObsWrapper.release(env, DeployObsSchema.default_v2())
  → Common の build_deploy_obs_v2（特徴量の計算はここだけ）
```

- `DeployRawEnv` は reset のたびに `/params {"deploy_raw": true}` を送り、`/reset` 応答のトップレベルと `/step` 応答の `info` から `deploy_raw` を読む（仕様は [`ue5_env.md`](ue5_env.md#deploy_rawopt-in)）。未知キー・欠損キー・非数・語彙外の class 名は例外にする（fail-closed）。武器・パッシブの `type_id` は Common 語彙の名前に、空き枠は `None` に変換する。移動方向は自機の世界座標の前フレームとの差から作る。
- v2 raw dict の entity は `entity_id`・`class_name`（Common `deploy_obs_v2_features.yaml` の entity 語彙）・`radius_world`・`world_x/y`・`occluded`・`timestamp_ns` に加えて、武器エフェクトだけが `slot`・`ttl_true_s`・`warning` を持つ。inventory は武器・パッシブ各 6 枠（`index`・`type_name`・`level`）と `duration_mult`。v2 raw に `privileged` mapping は無い。v1 raw は v1 schema のときだけ受け付ける。
- release は world 座標を target camera で px へ投影し（半径は `radius_world × viewport幅 / (2 × 半幅)`）、可視判定（中心が画面内・遮蔽なし）は Common の `is_track_visible` に任せる。entity の `slot`・`ttl_true_s`・`warning` は読まない。
- 武器エフェクトの残り時間は、wrapper が entity_id ごとに「初めて画面内に見えた時刻」を保持し、Common の持続時間表から推定する。Common の `track_max_age_frames`（class ごと）フレーム続けて見えなければ記録を捨て、reset ですべて捨てる（実機 tracker が track を作り直すのと同じ規則）。
- release の v2 は縦横とも viewport 幅/2 で正規化するので、viewport の縦横比が target camera の `half_width / half_height`（sim は 16:9）と違う raw は拒否する。
- `oracle_diagnostic()` は同じ tensor に加えて、「release ビルダーが実際に出した zone / orbit の残り時間 − sim の真の残り時間（秒、0..8 に clip）」を `info["deploy_ttl_error_s"]`（entity_id → 秒）に出す。推定式は Training に持たず、ビルダー出力を秒へ戻して比べる。ビルダーが無効にした値（emitter が一意に決まらない構成など）は含めない。fidelity の診断専用。
- `deploy_raw_env.py` と `deploy_obs_wrapper.py` は fidelity の `deploy_release_adapter` producer 閉包に入っているので、変更すると deploy 系 gating hash が変わり既存 verdict は失効する。
- Python テストの正は実 UE5 Editor PIE から取得した `Tools/Training/tests/survivors/fixtures/deploy_raw_pie_v1.json`。`deploy_raw_llt_v1.json` は C++ LLT の `[fixture]` テスト専用。再取得は PIE 起動中に `python Tools/Training/capture_survivors_deploy_raw_fixture.py --output Tools/Training/tests/survivors/fixtures/deploy_raw_pie_v1.json`（既定 port 8767、変更時は `--port`）を実行する。

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

最初にfixed subject (`ReinBalanceEditor / Win64 / Development`) のfresh UBT action graphを
current checkoutから発行する。UBTまたはbuild inputが不整合ならoutputは発行されない。

```powershell
Set-Location "<PROJECT_ROOT>\Tools\Training"
python export_survivors_ubt_action_graph.py `
  --engine-root "C:\UnrealEngine\UE_5.4" `
  --output D:\repo\ue5_reinforcement_balance\FidelityInputs\current\ubt-action-graph.json
```

`generated-inputs.json` は次のexact-key schemaをcanonical JSONで保存する。各`path`は
current checkout内に実在するrepo-relative POSIX pathでなければならず、絶対path、`..`、
checkout外のartifact、旧checkoutのcopyは拒否される。次の`FidelityInputs/current/*.json`は
operatorがcurrent HTTP/schema captureからcheckout内へ発行したsnapshotの配置例である。
DeployObsが未導入のbaseline/integrationでは、指定の2入力だけ`"absent"`にする。

```json
{
  "inputs": {
    "action_time_schema": {"format": "json", "path": "FidelityInputs/current/action_time_schema.json"},
    "content_schema": {"format": "json", "path": "FidelityInputs/current/content_schema.json"},
    "deploy_obs_schema": "absent",
    "deploy_release_adapter": "absent",
    "external_decision_schema": {"format": "json", "path": "FidelityInputs/current/external_decision_schema.json"},
    "preview_schema": {"format": "json", "path": "FidelityInputs/current/preview_schema.json"},
    "target_profile": {"format": "yaml", "path": "Tools/Deployment/configs/mad_forest_standard_v1.yaml"}
  },
  "schema_version": "survivors.generated_fidelity_inputs.v1"
}
```

```bash
python collect_survivors_value_choices.py \
  --manifest /artifact/value-source/result/value_source_descriptor.json \
  --fidelity-verdict /artifact/fidelity/integration-verdict.json \
  --generated-input-descriptor /path/to/ue5_reinforcement_balance/FidelityInputs/current/generated-inputs.json \
  --ubt-action-graph /path/to/ue5_reinforcement_balance/FidelityInputs/current/ubt-action-graph.json \
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
formal collection 前にcurrent checkout、generated input descriptor、fresh UBT action graphから
再解決したproducer hashesでintegration fidelity verdictを再検証
し、source descriptor と fidelity verdict は dataset と同じ artifact store の immutable
親として保持する。

UE5 build / LLT: 未実行（Windows 専用）。PIE 100 decisions pilot、formal storage/
parallel-port/wall-clock budget の固定は後続 phase で実施する。

## Paired rollout teacher validation

`validate_survivors_value_teacher.py` は `survivors.paired_rollout_corpus.v1`、immutable
Value Source descriptor、current integration fidelity verdict を入力とし、episode split、
teacher score scale、reliability calibration、short/full validation report、
`survivors.label_release_verdict.v1` を一方向に生成する。release threshold の CLI override
は提供しない。development train で outcome normalization、development train/validation
で score scale と calibration を確定した後、untouched final test を一度だけ評価する。

```powershell
Set-Location "<PROJECT_ROOT>\Tools\Training"
python validate_survivors_value_teacher.py `
  --corpus <ARTIFACTS_DIR>\paired-rollout-corpus.json `
  --source-descriptor <ARTIFACTS_DIR>\value_source_descriptor.json `
  --integration-fidelity-verdict <ARTIFACTS_DIR>\integration-fidelity-verdict.json `
  --generated-input-descriptor <PROJECT_ROOT>/FidelityInputs/current/generated-inputs.json `
  --ubt-action-graph <PROJECT_ROOT>/FidelityInputs/current/ubt-action-graph.json `
  --output-dir <ARTIFACTS_DIR>\teacher-validation
```

追加モジュールの責務は次のとおり。

| モジュール | 責務 |
|---|---|
| `games/survivors/choice_branch_rollout.py` | complete semantic trace、candidate/worker 非依存 replication key と post-decision RNG stream、quarantine/RNG seam |
| `games/survivors/teacher_validation_split.py` | episode-group development/final split の freeze と final lineage sealing |
| `games/survivors/teacher_score_scale.py` | development score-difference refs の q05/q95 scale fit・create-once commit |
| `games/survivors/teacher_reliability.py` | exact/fallback slice の episode/seed-cluster residual CI・error UCB・weight |
| `games/survivors/teacher_validation.py` | short/full の tie-aware top-1、pairwise、NDCG@3、regret と固定 release gate |

UE5 Editor の `/validation_branch_rng` は semantic replay 後の validation worker 専用であり、
通常 `/reset` が必ず解除する。`/step` info の `elapsed` / `level` / `gems` / `kills` /
`alive` / `stage_clear` は read-only outcome metrics で、base/shaped/HP penalty reward
semantics は変更しない。NovelD、shaped reward、HP penalty は provenance に残るが
ground-truth utility には入らない。

回帰テストは Windows conda env で skip なしに実行する。

```powershell
Set-Location "<PROJECT_ROOT>\Tools\Training"
python -m pytest tests\survivors\ -q --tb=short
```

formal corpus（300 decisions / 30 episodes 以上）と formal gate 達成は実環境収集後に行う。
UE5 build / LLT: 未実行（Windows 専用）。

## Survivors extrinsic utility value overlay

raw PPO critic が paired gate を満たさない場合は
`train_survivors_value_overlay.py` で source-bound critic latent dataset から utility head を
学習する。source policy 全体（actor、feature extractor、critic LSTM を含む）は freeze
され、optimizer には `Linear(latent, 128) → GELU → Linear(128, 1)` の head parameter
だけが登録される。loss は `Huber + 0.5 * pairwise logistic`、Adam は
`lr=1e-3` / `weight_decay=1e-4`、上限100 epoch / patience 10で、development validation
NDCG が最良の checkpoint だけを保存する。

```powershell
Set-Location "<PROJECT_ROOT>\Tools\Training"
& 'C:\Users\neko\anaconda3\envs\reinbalance\python.exe' `
  train_survivors_value_overlay.py `
  --source-manifest <ARTIFACTS_DIR>\value_source_descriptor.json `
  --dataset <ARTIFACTS_DIR>\extrinsic-value-training-set.json `
  --output-dir <ARTIFACTS_DIR>\extrinsic-value-overlay
```

dataset split は episode/decision seed group で固定し、target mean/std は
`development_train` だけから fit する。censored、early failure、quarantine branch は
loss から除外し、primary tie は pairwise loss に使用しない。`final_test` は学習・
checkpoint 選択では開封せず、後続 revalidation の method lock 後だけ利用する。

出力は `extrinsic_value_overlay.pt` と
`extrinsic_value_overlay_manifest.json` の組である。manifest は source model、
policy-state schema、VecNormalize、context shape、actor invariance、paired dataset、
state dict hash、target normalization、best validation NDCG を束縛する。片方欠損、
hash 不一致、別 source/context ではロードを拒否する。

choice probe/collector は任意の
`--value-overlay <overlay-manifest-or-directory>` を受け取る。未指定時は従来の raw
critic path をそのまま使用する。

対象回帰テストは Windows conda env で skip なしに実行する。

```powershell
Set-Location "<PROJECT_ROOT>\Tools\Training"
& 'C:\Users\neko\anaconda3\envs\reinbalance\python.exe' -m pytest `
  tests\survivors\test_extrinsic_value_overlay.py `
  tests\survivors\test_train_value_overlay.py -q --tb=short
& 'C:\Users\neko\anaconda3\envs\reinbalance\python.exe' -m pytest `
  tests\survivors\ -q --tb=short
```

UE5 build / LLT: 未実行（Windows 専用）。実データによる promotion gate、overlay
reliability calibration、raw critic 比較、label release verdict は後続 revalidation
フェーズで実施する。

## Survivors ItemSelector training

`games/survivors/item_selector_model.py` は context MLP と candidate MLP を分離し、全 card に
同じ scorer を適用する。候補 index は入力に含めず、padding slot は logits を `-inf` にする。
`item_selector_loss.py` は teacher soft target の cross entropy、非 tie pair ranking、soft Brier
を `CE + 0.5 * Pair + 0.05 * Brier` で合成する。train-only class weight と reliability は完成した
row loss へ一度だけ適用される。

`item_selector_trainer.py` は共有 `ItemDecisionFeatures.to_wire()` から model feature を encode し、
dataset の target/mask/reliability を raw teacher score から再計算しない。optimizer は AdamW
(`lr=3e-4`, `weight_decay=1e-4`)、batch 512、最大100 epoch、patience 12で、development
validation NDCG が最良の checkpoint を選ぶ。resume は target capability hash と Nmax の両方が
一致する場合だけ許可し、trainer の partition reader は train/validation だけを要求する。

artifact packaging と sealed evaluation の入口は次のとおり。

| モジュール | 責務 |
|---|---|
| `evaluate_survivors_item_selector.py` | ItemSelector test-set evaluation + verdict generation |
| `export_survivors_item_selector.py` | ItemSelector artifact packaging (TorchScript + ONNX) |
| `games/survivors/item_selector_artifact.py` | manifest/model/dataset/feature/UI policy bindingを検証するruntime loader |

export はvalidation splitだけでstudent output temperatureを固定してから、sealed test readerを
ONNX/TorchScript parity確認のために一度だけ開く。packageにはteacher/source binaryを含めず、
lineage identityだけを記録する。runtime load時には全file hash、埋め込みmodel binding、dynamic
ONNX signature、install済み`NonModelUiPolicyV1`のconfig/schema/implementation hashを照合する。
evaluatorはseed 42・2000 resampleのbootstrap CIとoverall/per-slice/calibration gateを
`item_selector_verdict.json`へ保存する。

## Survivors ItemSelector closed-loop evaluation

`games/survivors/item_selection_strategy.py` は release 済み `ItemDecisionFeatures` を model
tensor に変換し、valid card の最大 logit を live choice ID として返す。teacher target、
reliability、split は推論入力に含めない。同点は candidate index ではなく choice ID の辞書順で
解決するため、UI の表示順が変わっても同じ card を選ぶ。

`games/survivors/item_selector_eval.py` は movement step の pending item decision を検出すると、
次の movement より先に選択を一度だけ `/level_up_choice` 相当の endpoint へ適用する。response の
decision ID と choice ID が request と一致しない場合は episode を fail-closed で停止する。評価
adapter は step info に `item_decision_features`、`decision_id`、`choice_ids` をセットし、choice
endpoint は `(post_choice_observation, acknowledgement)` を返す必要がある。

`eval_survivors_item_selector.py` の `load_item_selector()` は checkpoint の target capability hash、
Nmax、context/candidate feature 次元を検証して strategy を復元する。`run_closed_loop_evaluation()`
へ UE5 adapter と既存 movement policy を渡すと、episode return、movement step、ack 済み choice
履歴を JSON 化できる。

## Survivors Deploy可能 Student Policy (03-05)

`games/survivors/combat_distillation_dataset.py` は teacher trajectory から固定長 sequence dataset を
構築する。各 sequence は `valid_mask`・`burn_in_mask`・`episode_reset_mask` を保持し、
`action_logits`・`teacher_values` を `deploy_schema_hash` へ束縛して NPZ + manifest として保存する。
`unobservable` source class、teacher actions、train 以外の split を `assert_release_training_ready()`
で拒否し、release training への漏洩を step 0 で止める。

`games/survivors/deployable_policy_trainer.py` は GRU ベースの `DeployableCombatPolicy` に対し
actor KL + value Huber distillation loss を `burn_in_mask` と `valid_mask` で制御しながら学習する。
4 段階の corruption curriculum (clean → light → measured → full) と DAgger shard 追加境界を
`CurriculumConfig` に固定し、model / VecNormalize / error wrapper RNG / curriculum / DAgger state を
まとめて checkpoint resume できる。`FormalDependencies.validate()` は fidelity verdict の
`gating_producer_hashes` と measured perception profile hash を step 0 で照合し、
missing/stale な場合は `ValueError` で fail closed にする。development run は常に
`development_only=true`・`formal_student_eligible=false` の checkpoint を生成する。

`games/survivors/deployable_policy_package.py` は eligible checkpoint から model-only runtime
package を原子的に構築する。`development_only` または `formal_student_eligible` が不正なら
package 前に `ValueError` を送出し、load 側でも同じ gate を繰り返す。

`collect_survivors_combat_distillation.py` は蒸留 dataset の収集 CLI。既定 schema は DeployObs v2。
`--source-descriptor` を省略すると UE5 に接続しない synthetic fixture を `development_only` として保存する
（正式 student 訓練には使えない）。`--source-descriptor` を指定すると正式収集になる。

### 正式蒸留データ収集 (03-08)

`games/survivors/distillation_collector.py` が、UE5 で教師を動かしながら同じ step の DeployObs v2 を記録する。

- 1 step ごとに、UE5 応答の flat obs（教師の入力）と同じ応答の `deploy_raw` を受け取る。
  `DeployRawEnv` が flat obs を `last_flat_obs` に残し、`deploy_raw` は `DeployObsWrapper.release()`（Common の共有ビルダー）だけで v2 tensor にする。
- 教師（`ValueSourceTeacher`: 01-01 の `load_value_source` で検証した model + VecNormalize）は決定的に推論し、
  行動分布の logits・value・行動を同じ添字に記録する。教師の行動は release dataset に入れない（`CollectedSequences.teacher_actions` で診断用に保持するだけ）。
- 教師の LSTM 状態は episode ごとに零へ戻し、最初の step だけ `episode_start=True` を渡す。
- episode は `--sequence-length` ごとの行に分ける。各行の先頭が reset 境界で、そこから `--burn-in` step が burn-in。
  末尾は padding（logits / value は 0、`valid_mask` は False）。
- split は episode 単位で割り当てる（`--validation-every` 個目ごとに validation、0 なら全部 train）。同じ episode の行が別 split に分かれることはない。
- 保存前に train は `assert_release_training_ready()`、validation は `assert_release_observations()` を通す。
  unobservable segment（enemy_hp・cooldown）に oracle 値が 1 step でも入っていれば拒否する。
- `deploy_raw` が無い応答、`/params deploy_raw=true` の失敗、非数・形違いの flat obs や教師出力は、黙ってスキップせず収集を失敗させる。

開始条件は fail closed で、次の順に検査する。どれかが欠ける・失敗すると stderr に 1 行出して **終了コード 3** を返し、UE5 へは接続しない（dataset も保存しない）。

1. `--fidelity-verdict` / `--artifact-store` / `--generated-input-descriptor` / `--ubt-action-graph` がすべて指定されている
2. 教師 source descriptor が存在し、`load_value_source` の検証（hash・model・VecNormalize）を通る
3. fidelity verdict が読めて、現在の producer hash（`resolve_current_gating_producer_hashes`）に対する `integration` stage として `verify_current_fidelity` を通る
4. verdict に `blocking_reasons` が無い

モジュール関数 `run_formal_collection()` も同じ順序で検査し、保存関数 `save_dataset_artifact()` も保存直前に verdict を再検証する。

保存先: `--output` に `train/`・`validation/`（`data.npz` + `manifest.json`）と `artifact_descriptor.json` を書き、
同じファイルを `--artifact-store` に登録する。descriptor は `node_kind=combat_distillation_dataset` で、
親は教師 source descriptor と fidelity verdict の 2 つの `source_descriptor` node。`identity_metadata` には
DeployObs schema hash / version、dataset schema version、`deploy_raw` schema version、教師 identity、verdict identity、
episode 数・sequence 長・burn-in・seed・収集 step 数を記録する（dataset 自体の schema version は v1 のまま）。
dataset の logical id は教師 identity・verdict identity と、収集設定・教師 descriptor・保存ファイル内容の digest から作るため、
同じ store・教師・verdict・seed で再収集しても別 dataset として登録される。書き出しは `--output` の隣の一時 directory で行い、
store 登録まで成功したときだけ `--output` へ rename する（失敗時は `--output` を作らず終了コード 3）。
`--artifact-store` を開けない場合は UE5 へ接続する前に終了コード 3 で止まる。

前提 artifact（merge 後の手動作業）: 03-07 merge 後に再発行した integration fidelity verdict と、Phase 5 教師の source descriptor（01-01 release）。
正式収集は 03-05 `D03-DEPLOY-STUDENT-RELEASE` の最初の手順として、学習とは別プロセスで実行する（`deploy_raw` 付き応答は大きく収集が遅い）。

```bash
python collect_survivors_combat_distillation.py \
  --output D:/reinbalance-data/combat-distillation/run-001 \
  --source-descriptor D:/reinbalance-artifacts/phase5-teacher/source_descriptor.json \
  --fidelity-verdict D:/reinbalance-artifacts/fidelity/integration-verdict.json \
  --artifact-store D:/reinbalance-artifacts/store \
  --generated-input-descriptor D:/reinbalance-artifacts/fidelity/generated-inputs.json \
  --ubt-action-graph D:/reinbalance-artifacts/fidelity/ubt-action-graph.json \
  --ue5-port 8767 --episodes 64 --sequence-length 64 --burn-in 16 --validation-every 5 --seed 0
```

終了コード: 0 = 保存成功、3 = 開始条件の不足・検証失敗、または収集中の契約違反（正式経路）、2 = synthetic 経路の保存失敗。

`train_survivors_deployable_policy.py` は distillation 訓練の CLI エントリポイント。
`--formal-deps` を省略すると development mode で動作し、生成 checkpoint は正式 package に昇格できない。

`--formal-deps` に store 形式の JSON（`perception_profile_store_root` +
`perception_calibration_commit_logical_id`）を渡す場合は、
`--required-calibration-descriptor-hash` が必須。これは producer の
`perception_calibration_profile` descriptor identity の期待値で、**JSON 側には書けない**。
store の所在を指す JSON と期待値を同じ入力から読むと、その入力を書ける主体が
store・commit・期待値を自己整合的に用意できてしまい照合が無意味になるため、
期待値は起動スクリプト / CI 設定など別チャネルで管理・レビューする。

`eval_survivors_deployable_policy.py` は packaged policy を dataset（synthetic、または正式収集の `validation/`）で再評価して
actor KL / value Huber を JSON report として保存する evaluation CLI。

```bash
# 開発用: synthetic dataset 生成
python collect_survivors_combat_distillation.py --output /tmp/dev_dataset

# 開発用: distillation 訓練 (development_only)
python train_survivors_deployable_policy.py \
  --dataset /tmp/dev_dataset \
  --output-dir /tmp/dev_checkpoints \
  --updates 10
```

正式 student 訓練 (`D03-DEPLOY-STUDENT-RELEASE`) は 04-07 measured calibration profile と
03-04 post-curriculum fidelity verdict が揃うまで開始しない。

## 関連ドキュメント

- UE5 との通信仕様: [`ue5_env.md`](ue5_env.md)
- 実装上の注意事項・既知の問題: [`impl_notes.md`](impl_notes.md)
- Survivors reward_fn 設計ポリシー: [`reward_fn_policy.md`](reward_fn_policy.md)

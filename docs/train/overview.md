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

## 関連ドキュメント

- UE5 との通信仕様: [`ue5_env.md`](ue5_env.md)
- 実装上の注意事項・既知の問題: [`impl_notes.md`](impl_notes.md)
- Survivors reward_fn 設計ポリシー: [`reward_fn_policy.md`](reward_fn_policy.md)

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

## 関連ドキュメント

- UE5 との通信仕様: [`ue5_env.md`](ue5_env.md)
- 実装上の注意事項・既知の問題: [`impl_notes.md`](impl_notes.md)
- Survivors reward_fn 設計ポリシー: [`reward_fn_policy.md`](reward_fn_policy.md)

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

## 関連ドキュメント

- UE5 との通信仕様: [`ue5_env.md`](ue5_env.md)
- 実装上の注意事項・既知の問題: [`impl_notes.md`](impl_notes.md)

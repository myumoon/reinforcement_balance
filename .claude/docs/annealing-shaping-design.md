# shaped_reward 線形アニーリング 設計メモ

作成日: 2026-04-29  
**実装状況: ✅ 実装済み**（`Tools/Training/train.py` の `_AnnealingShapingCallback` クラス）

---

## 目的

EUREKA ループが生成した `reward_fn` は訓練初期の「補助輪」として機能するが、  
エージェントが基本行動を習得した後も使い続けると報酬ハッキングのリスクがある。  
`|shaped_reward_mean| / base_reward_mean` の比率を監視し、比率が閾値を下回ったら  
自動的に shaped_reward を線形に 1.0 → 0.0 へ落とす仕組みを train.py に追加する。

---

## 比率の意味

```
ratio = |shaped_reward_mean| / base_reward_mean  (エピソード単位の平均)

ratio 高い → shaped が学習を駆動している（補助輪が効いている）
ratio 低い → shaped がほぼ 0 → エージェントが shaped の意図を内在化
           → アニーリング開始の合図
```

---

## 実装済みクラス

`Tools/Training/train.py` の `_AnnealingShapingCallback` クラスに実装済み。  
`--anneal-*` 系 CLI 引数で制御する。

### CLI 引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--anneal-threshold` | 0.1 | `\|shaped\|/base` がこれを下回ったらアニーリング開始 |
| `--anneal-steps` | 50000 | アニーリングにかけるステップ数 |
| `--anneal-check-freq` | 5000 | 比率のチェック間隔（ステップ） |
| `--anneal-min-steps` | 50000 | アニーリングチェックを開始する最小ステップ数 |
| `--anneal-min-weight` | 0.0 | アニーリング後の最小 shaping_weight（0=完全無効化） |

### 動作条件

- `--game coin` または `--game survivors` かつ `--reward-fn` 指定時のみコールバックが有効
- `--dry-run` 時は DummyCoinEnv / DummySurvivorsEnv に `shaping_weight` が存在しないため適用外

---

## 使用例

```bash
# アニーリングあり（デフォルト設定）
python train.py --game coin --reward-fn eureka_results/my_run/best/reward_fn.py \
    --entity-attention

# 即時アニーリング確認用（テスト）
python train.py --game coin --reward-fn ... \
    --anneal-threshold 999 --anneal-min-steps 0 --anneal-steps 10000

# アニーリングをより積極的に
python train.py --game coin --reward-fn ... \
    --anneal-threshold 0.3 --anneal-steps 30000
```

---

## 検証手順

1. 起動後 5000 step ごとに `[INFO] シェーピング比率: X.XXX` が表示される
2. ratio < threshold になったとき `[INFO] shaped_reward アニーリング開始` が表示される
3. anneal_steps 後に `[INFO] shaped_reward アニーリング完了 → reward_fn を無効化` が表示される
4. `--anneal-threshold 999 --anneal-min-steps 0` で即時トリガーを検証できる

# world_detector — 本家 Survivors world entity 検出・追跡

## 概要

本家 Vampire Survivors の 1920×1080 ゲーム画面から、プレイヤー・敵・宝石などの
world エンティティを検出し、フレーム間でトラック ID を維持して deploy obs へ渡す。

- **検出:** torchvision SSDLite320_MobileNet_V3_Large（head を 12 クラスに置換）
- **追跡:** normalized center distance + IoU + class penalty の deterministic greedy matching
- **出力:** `TrackedWorldStateV1`（04-09 が deploy obs に変換）

## クラス定義

`configs/world_class_map_v1.yaml` で固定（background 0 + foreground 11 = 12 クラス）。

| ID | 名前 | coarse_category |
|----|------|----------------|
| 0 | `__background__` | — |
| 1 | `player_anchor` | anchor |
| 2 | `enemy_normal` | enemy |
| 3 | `enemy_elite` | enemy |
| 4 | `enemy_boss` | enemy |
| 5 | `gem_blue` | gem |
| 6 | `gem_green` | gem |
| 7 | `gem_red` | gem |
| 8 | `pickup_heal` | pickup |
| 9 | `pickup_special` | pickup |
| 10 | `hazard_projectile` | hazard |
| 11 | `hazard_area` | hazard |

クラス ID / 名前の変更は `class_map_version` を上げてから行う。

## ファイル構成

```
Tools/Deployment/
├── survivors/vision/
│   ├── world_dataset.py        # COCO dataset loader + preflight + session split
│   ├── world_detector.py       # SSDLite320 adapter + DetectionResult + CheckpointManifest
│   └── entity_tracker.py       # greedy tracker + TrackedWorldStateV1
├── configs/
│   ├── world_class_map_v1.yaml # 12 クラス固定 class map
│   └── world_detector_v1.yaml  # 学習・推論設定（formal_detector_eligible=false）
├── train_survivors_world_detector.py   # 学習 CLI
├── eval_survivors_world_detector.py    # 評価 CLI（mAP50:95 等）
└── mine_survivors_active_learning.py   # active learning マイニング CLI
docs/deployment/
└── world_detector.md           # 本ドキュメント
Tools/Deployment/tests/vision/
├── test_world_dataset.py
├── test_world_detector.py
└── test_entity_tracker.py
```

## 使い方

### 学習（開発用 dry-run）

```bash
cd Tools/Deployment
python train_survivors_world_detector.py \
    --annotations data/world_annotations.json \
    --config configs/world_detector_v1.yaml \
    --class-map configs/world_class_map_v1.yaml \
    --output runs/world_detector_dev \
    --dry-run
```

`--dry-run` は preflight チェックのみ実行して終了する。
生成される checkpoint は `formal_detector_eligible=false` のため、
04-10 の formal loader には拒否される。

### 評価

```bash
python eval_survivors_world_detector.py \
    --annotations data/world_val.json \
    --predictions data/world_predictions.json \
    --config configs/world_detector_v1.yaml \
    --class-map configs/world_class_map_v1.yaml \
    --output runs/metrics.json
```

### active learning マイニング

```bash
python mine_survivors_active_learning.py \
    --annotations data/world_train.json \
    --candidate-frames data/unlabeled.json \
    --top-k 100 \
    --output data/mining_top100.json
```

## テスト実行

```bash
bash Tools/run-pytest.sh Tools/Deployment/tests/vision -q -rs
```

synthetic fixture のみ使用。GPU・実画像・実 weight 不要。

## 制約と非目標

- **formal_detector_eligible=false:** 本 PR の checkpoint / config はすべて開発用。
  正式パッケージへの昇格は 04-08 が確定する。
- **weight なし:** 学習 CLI はスタブ実装。実際の weight 更新は 04-07 が実装する。
- **新規 detection framework 依存なし:** torchvision だけを使用する。
- **実動画の recall / latency 合格は本 PR の完了条件ではない。**

## 04-07 / 04-08 との境界

| 責務 | 担当 PR |
|------|---------|
| class map / schema / tracker / dataset loader | **04-06 (本 PR)** |
| DataLoader / optimizer / augmentation 本格実装 | 04-07 |
| formal weight / threshold / formal_detector_eligible=true | 04-08 |
| deploy obs への変換 | 04-09 |

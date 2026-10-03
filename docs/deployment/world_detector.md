# world_detector — 本家 Survivors world entity 検出・追跡

## 概要

本家 Vampire Survivors の 1920×1080 ゲーム画面から、プレイヤー・敵・宝石などの
world エンティティを検出し、フレーム間でトラック ID を維持して deploy obs へ渡す。

- **検出:** torchvision SSDLite320_MobileNet_V3_Large（head を class map の 16 クラスに置換）
- **追跡:** normalized center distance + IoU + class penalty の deterministic greedy matching（大分類が違う組・weapon で細分類が違う組は対応付けない）
- **出力:** `TrackedWorldStateV2`（04-09 が deploy obs に変換）

## クラス定義

`configs/world_class_map_v2.yaml` で固定（background 0 + foreground 15 = 16 クラス）。
ID 1〜11 は旧 `world_class_map_v1.yaml`（12 クラス）と名前・順序・大分類が完全に同じで、ID 12〜15 に武器エフェクトを追加した。
ファイル形式は v1 と同じなので `schema_version` は `world_class_map.v1` のまま、内容の版を `class_map_version: 2` で表す。

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
| 12 | `weapon_projectile` | weapon |
| 13 | `weapon_zone` | weapon |
| 14 | `weapon_orbit` | weapon |
| 15 | `weapon_aura` | weapon |

クラス ID / 名前の変更は `class_map_version` を上げてから行う。
クラス数・大分類・クラス名は detector 構築・tracker・package・annotation・COCO 出力のすべてで同じ class map から導出する。
v1 の class map / detector config（`world_class_map_v1.yaml` / `world_detector_v1.yaml`）は残しているが、既定はすべて v2。
v1 の class map hash を持つ checkpoint・package は、既存の `class_map_hash` / `contract_hash` 照合で拒否される。
`world_detector.v1` config（12 クラス）と v2 class map、`world_detector.v2` config（16 クラス）と v1 class map の組み合わせも `num_classes` 不一致で拒否される。

## ファイル構成

```
Tools/Deployment/
├── survivors/vision/
│   ├── world_dataset.py          # COCO dataset loader + preflight + session split
│   ├── world_detector.py         # SSDLite320 adapter + DetectionResult + CheckpointManifest
│   ├── entity_tracker.py         # greedy tracker + TrackedWorldStateV2
│   └── world_detector_package.py # 開発 package writer / restore (04-07)
├── configs/
│   ├── world_class_map_v2.yaml   # 16 クラスの class map（既定）
│   ├── world_class_map_v1.yaml   # 旧 12 クラスの class map（旧成果物の照合用に残す）
│   ├── world_detector_v2.yaml    # 学習・推論設定（既定、formal_detector_eligible=false）
│   └── world_detector_v1.yaml    # 旧 12 クラス用の設定
├── train_survivors_world_detector.py   # 学習 CLI（DataLoader / optimizer / checkpoint selection）
├── eval_survivors_world_detector.py    # 評価 CLI（mAP50:95 / performance gate）
└── mine_survivors_active_learning.py   # active learning マイニング CLI
docs/deployment/
└── world_detector.md           # 本ドキュメント
Tools/Deployment/tests/vision/
├── test_world_dataset.py
├── test_world_detector.py
├── test_entity_tracker.py
└── test_world_detector_package.py  # 04-07 tooling テスト
```

## 使い方

### 学習（開発用 dry-run）

```bash
cd Tools/Deployment
python train_survivors_world_detector.py \
    --annotations data/world_annotations.json \
    --split data/split.json \
    --config configs/world_detector_v2.yaml \
    --class-map configs/world_class_map_v2.yaml \
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
    --config configs/world_detector_v2.yaml \
    --class-map configs/world_class_map_v2.yaml \
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

## config 共有バリデーション

すべての public 入口（`load_detector_config()`, `WorldDetector.from_config()`,
学習 CLI, 評価 CLI, `publish_development_package()`, `restore_package()`）は
`survivors.vision.world_detector.validate_detector_config()` を必ず通る。

バリデーション規則はこの共有 validator にのみ定義する。各 caller へ複製しない。

**対応範囲（world_detector.v1 / world_detector.v2）**

| フィールド | 許容値 |
|---|---|
| `schema_version` | 文字列 `world_detector.v1`（v1 class map 用）または `world_detector.v2`（v2 class map 用） |
| `formal_detector_eligible` | bool `false` のみ |
| `model.architecture` | `ssdlite320` のみ |
| `model.backbone` | `mobilenet_v3_large` のみ |
| `model.pretrained_backbone` | bool `false` のみ（int 1 / 文字列 `'false'` は不可） |
| `model.num_classes` | schema_version に対応する class map の background + foreground 数（v1=12, v2=16）の int のみ（bool 不可） |
| `model.input_size` | `[320, 320]` のみ（bool 不可） |
| `input.normalize.mean` | `[0.485, 0.456, 0.406]` のみ |
| `input.normalize.std` | `[0.229, 0.224, 0.225]` のみ |
| `training.optimizer` | `sgd` キーのみ（adam 等は未実装） |
| `training.batch_size` | int ≥ 3（bool 不可） |
| `tracker.max_age_by_class` / `dev_diagnostics.class_recall_min` のキー | schema_version に対応する class map 内の名前のみ |
| 数値フィールド全般 | 有限値のみ（NaN / Infinity / bool / 文字列は拒否） |

**未対応値はサイレント受理されない。** 未知 top-level key, 未知 nested key, 型違い, 非有限値,
未実装 optimizer (adam 等) / scheduler / augmentation はすべてバリデーション段階で拒否される。

**development score_threshold について**

04-07 の `score_threshold` は development package を同じ条件で restore・推論するための
再現用値であり、formal threshold・formal PASS・formal eligibility の根拠ではない。
formal threshold は 04-08 で正式 dataset / target / validation verdict に基づいて別途選定する。
04-07 の値をそのまま formal threshold へ昇格しない。

## tracker と TrackedWorldStateV2

`EntityTracker` は `coarse_by_class_id`（class_id → class map の大分類）を必須引数に取る。
構築する全経路（`run_survivors_controller.py`、`restore_package()`、replay fixture）は class map の
`coarse_by_class_id()` を渡す。マッチングの規則は次のとおり。

- 大分類が違う track と検出は対応付けない（cost を無限大にする）。プレイヤーと同心の大きい
  `weapon_aura` が `player_anchor` の track に割り当てられて anchor の矩形が膨らむのを防ぐ。
- 大分類 `weapon` では細分類（class）が一致するときだけ対応付ける。track の class は生成時から変わらないため、
  火球（`weapon_projectile`）→爆発（`weapon_zone`）のように種類が変わったら新しい track にする。
- 同じ大分類内の他のクラス（例: `enemy_normal` と `enemy_elite`）は従来どおり class penalty 付きで対応付けられる。

weapon の track は `coarse_class == "weapon"` で出力され、大分類 `enemy` / `hazard` だけを見る
`boss_flag` / `hazard_flag` には影響しない。

`TrackedWorldStateV2`（`frame_index`, `timestamp_ns`, `tracks`, `player_anchor`）の各 track は
`TrackedEntityV2` で、次の field を持つ。v1 の 15 field の末尾に 3 field を足したもので、
`world_detector_package` の `contract_hash` も変わったため旧 package は restore 時に拒否される。

| field | 内容 |
|---|---|
| `track_id` | track の ID |
| `class_id` / `class_name` | class map の ID と名前 |
| `coarse_class` | class map の大分類（anchor / enemy / gem / pickup / hazard / weapon） |
| `confidence` | 信頼度（未検出フレームでは decay） |
| `age` / `last_seen_frame_index` | 経過フレーム数と最後に検出されたフレーム |
| `normalized_cx` / `normalized_cy` | 矩形中心（画像サイズで正規化） |
| `player_relative_x` / `player_relative_y` | player_anchor からの相対位置 |
| `velocity_x` / `velocity_y` | 中心移動量の EMA（pixel/frame） |
| `on_screen` / `clipped` | 画面と重なるか / 画面外にはみ出すか |
| `normalized_width` / `normalized_height` | **v2 追加。** 最新の検出矩形の幅・高さ（画像サイズで正規化、平滑化なし） |
| `first_seen_timestamp_ns` | **v2 追加。** その track を生成したフレームの時刻。track が途切れて作り直されると新しい時刻になる |

## 制約と非目標

- **formal_detector_eligible=false:** 本 PR の checkpoint / config はすべて開発用。
  正式パッケージへの昇格は 04-08 が確定する。formal publish は常に `FormalPackageRejectedError` で拒否される。
- **development-only:** 学習 CLI は DataLoader / optimizer / SGD を実装する（開発用）。
  manifest に `development_only=true` と `training_mode: "smoke" | "development"` を記録する。
- **新規 detection framework 依存なし:** torchvision だけを使用する。
- **実動画の recall / latency 合格は本 PR の完了条件ではない。**
- **formal 機能は 04-08 へ委譲:** formal preflight, 正式 augmentation, 性能合格,
  threshold 選定, formal package, formal release はすべて 04-08 の責務。
  04-07 (本 PR) は development-only tooling を提供する。

## 04-07 / 04-08 との境界

| 責務 | 担当 PR |
|------|---------|
| class map / schema / tracker / dataset loader | 04-06 |
| DataLoader / optimizer / development checkpoint / development package | **04-07 (本 PR)** |
| 正式 augmentation / formal weight / threshold / formal_detector_eligible=true | 04-08 |
| deploy obs への変換 | 04-09 |

## 04-07 tooling — session 拒否 / checkpoint selection / package writer

### session 拒否（error_calibration / final_e2e_test）

`WorldDataset` に `rejected_sessions` を渡すと、
指定セッションを含むフレームが dataset loader へ入った時点で `DatasetPreflightError` を送出する。

```python
ds = WorldDataset(
    ann_path, cm_path,
    validate_bounds=True,
    rejected_sessions={"error_cal_01", "final_test_01"},
)
```

`run_split_preflight(split)` は `SessionSplit` の `error_calibration` / `final_e2e_test`
がデータ内に混入していないことを確認する。

### checkpoint selection

学習前に `CheckpointSelector` にルールを固定する:

```python
selector = CheckpointSelector(metric="val_map50_95", keep_top_k=3)
```

validation 後に `selector.record(CheckpointRecord(...))` を呼ぶと、
`keep_top_k` 内の best が `selector.best` で取得できる。
`selector.to_dict()` は `manifest.json` に追記される。

### development diagnostics gate

```python
from eval_survivors_world_detector import compute_dev_diagnostics

gate_result = compute_dev_diagnostics(
    metrics, gate_cfg, class_name_by_id,
    slice_annotations=slice_annotations,
)
# gate_result.passed が False なら開発 diagnostic FAIL
```

`world_detector_v2.yaml` の `dev_diagnostics` セクションに閾値を定義する。
formal 性能判定・threshold・session-cluster CI は 04-08 に委譲する。
development diagnostics の `passed=True` は formal PASSではない。

### package writer

```python
from survivors.vision.world_detector_package import publish_development_package, restore_package

pkg_path = publish_development_package(
    checkpoint_manifest,
    metrics_dict,
    checkpoint_selection,
    store_dir,
    cfg_path=cfg_path,
    cm_path=cm_path,
    weight_path=weight_path,
)
state = restore_package(pkg_path, frame_bgr)  # TrackedWorldStateV2 を返す
```

- `formal_detector_eligible=false` の package は `assert_formal_eligible()` で拒否される。
- `contract_hash` が TrackedWorldStateV2 フィールド定義と一致しない場合は `PackageSchemaError`。
- `publish_formal_package()` は引数にかかわらず `FormalPackageRejectedError` を送出する（04-08 で実装）。
- manifest には `development_only=true` と `training_mode` が記録される。

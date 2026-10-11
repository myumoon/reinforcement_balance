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
│   ├── world_detector_package.py # development / formal package writer・restore
│   ├── world_formal_sampling.py  # session sampler・weights・augmentation
│   ├── world_formal_release.py   # formal preflight・verdict・artifact release
│   └── session_cluster_ci.py     # session 単位の比率 bootstrap CI
├── configs/
│   ├── world_class_map_v2.yaml   # 16 クラスの class map（既定）
│   ├── world_class_map_v1.yaml   # 旧 12 クラスの class map（旧成果物の照合用に残す）
│   ├── world_detector_v2.yaml    # 学習・推論設定（既定、formal_detector_eligible=false）
│   └── world_detector_v1.yaml    # 旧 12 クラス用の設定
├── train_survivors_world_detector.py   # 学習 CLI（DataLoader / optimizer / checkpoint selection）
├── eval_survivors_world_detector.py    # development / formal 評価 CLI
├── publish_survivors_world_detector_formal.py # formal package / artifact publisher
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

## 制約と境界

- checkpoint は `formal_detector_eligible=false` のまま保存する。formal eligibility は PASS verdict と対応 hash を確認した publisher だけが package に設定する。
- development / smoke の checkpoint と package は従来どおり formal runtime に拒否される。
- formal package restore は verdict hash、threshold、weight、dataset、build、resolved config を再検証する。
- この実装の合成データ検証は正式 dataset 上の訓練や target GPU latency 実測を代替しない。正式 release の運用実行は 04-08 で行う。
- 新しい detection framework 依存は追加せず、torchvision と既存の OpenCV / NumPy を使用する。

| 責務 | 担当 |
|------|------|
| class map / schema / tracker / dataset loader | 04-06 |
| development sampler / checkpoint / package | 04-07 |
| formal preflight / training / validation verdict / package / artifact / restore | **04-15** |
| 正式 dataset での訓練、target GPU latency 実測、release config の確定 | 04-08 |
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
- formal package は `validation_verdict_hash` が無ければ読み込みを拒否する。
- manifest には `development_only=true` と `training_mode` が記録される。

## 04-15 formal release

formal 入口は COCO、04-17 dataset descriptor、split manifest、restore verdict、feasibility PASS、annotation agreement を照合する。split manifest の seal と session の `formal_dataset_eligible`、`lossless_png`、build / target identity を確認し、sealed session は学習と評価の COCO に入れない。preflight は code hash と dataset / resolved config / build identity を含む canonical report を出し、学習開始前に失敗した場合は checkpoint を作らない。

formal training は descriptor が示す `model_train` / `model_validation` だけを使う。session-balanced sampler、class weights、near-duplicate 抑制、水平反転・scale・HSV augmentation、weighted SSD loss、identity 検査付き resume を有効にする。development の config 既定は sampler=`interleaved`、augmentation と duplicate suppression は disabled のため従来挙動を維持する。

`input.ignore_regions` は normalized `[x1, y1, x2, y2]` で指定し、box center が領域内の検出を全 class で除外する。既定 item HUD 領域は `[0.050, 0.033, 0.203, 0.120]`（1920×1080 で `[96, 36, 390, 130]` px）。formal eval と package restore が同じ filter を tracker より前に適用し、formal GT も同じ中心規則で除外する。
sim の DeployObs builder にはこの除外範囲を適用していないため、sim と実機の差は fidelity の未解消項目として扱う。

formal eval は外部 `--predictions` を受け付けず、checkpoint の weight で `model_validation` を in-process 推論する。COCO 101 点 AP、session-cluster 比率 CI、class/slice recall、距離・密度指標、GPU latency gate を評価し、raw score に threshold rule を適用して選んだ threshold と各 gate を verdict に固定する。threshold 選定と gate は同じ validation split を使うため、verdict に selection bias の注記が残る。`error_calibration` と `final_e2e_test` は選定にも評価にも使用しない。

04-08 での運用コマンド例（パスは正式 dataset と target 機の配置に合わせる）:

```powershell
python train_survivors_world_detector.py `
  --annotations data/world_coco_model.json --formal `
  --feasibility-verdict data/feasibility.json --capture-store data/capture-store `
  --target-profile-hash <target-profile-sha256> --hardware-id <target-gpu-id> `
  --dataset-descriptor data/world-dataset-descriptor.json `
  --restore-verdict data/dataset-restore-verdict.json `
  --annotation-agreement data/annotation-agreement.json `
  --minimum-counts data/world-detector-minimum-counts.json `
  --preflight-report-out runs/formal/preflight.json `
  --config configs/world_detector_v2.yaml --class-map configs/world_class_map_v2.yaml `
  --output runs/formal/checkpoint

python eval_survivors_world_detector.py `
  --annotations data/world_coco_model.json --formal `
  --preflight-report runs/formal/preflight.json --checkpoint-dir runs/formal/checkpoint `
  --latency data/target-gpu-latency.json --gates configs/world_detector_formal_gates.yaml `
  --config configs/world_detector_v2.yaml --class-map configs/world_class_map_v2.yaml `
  --verdict-out runs/formal/validation-verdict.json

python publish_survivors_world_detector_formal.py `
  --verdict runs/formal/validation-verdict.json --preflight-report runs/formal/preflight.json `
  --checkpoint-dir runs/formal/checkpoint --out-root runs/formal/packages `
  --artifact-store data/artifact-store --backup-root data/artifact-backup `
  --restore-root data/artifact-restore --dataset-descriptor data/world-dataset-descriptor.json `
  --logical-prefix survivors/world-detector --resolved-yaml-out configs/world_detector_mad_forest_1080p_ja_v1.yaml
```

publisher は package file と 04-10 `world_detector_package.v1` envelope を ArtifactStore に保存し、descriptor を発行する。`run_restore_test` が backup 経由で空 store への full restore を確認し、その verdict が PASS の場合だけ resolved YAML を出力する。sha256 形式の `game_build_id` を使う場合は、同一 hash になる build attestation を `--build-attestation` で publisher に渡す。resolved YAML、weight、正式 capture pixel はこの code-only 変更に含めず、04-08 の実行時に作る。

package directory key は manifest 内容の SHA-256 であり、package 内 file 内容を単独で改ざん不能にする仕組みではない。restore が verdict / weight / config の hash を再照合し、ArtifactStore descriptor と restore verdict が release object 集合を固定する。

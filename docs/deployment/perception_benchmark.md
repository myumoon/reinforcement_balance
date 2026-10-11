# Perception Benchmark

`perception_benchmark` は calibration residuals から `PerceptionErrorProfile` を fit し、
formal またはsynthetic セッションで分類・回帰・レイテンシ・UI ROI メトリクスを集計する 04-10 の benchmark ツール群です。

## モジュール

| モジュール | 役割 |
|---|---|
| `survivors/perception_session_split.py` | calibration/final の overlap・mixed build・underpowered slice を fail-closed 検証 |
| `survivors/perception_benchmark.py` | screen F1・HP/XP MAE・latency p95/p99・UI ROI・cluster CI を集計 |
| `survivors/perception_error_fit.py` | calibration residuals から `PerceptionErrorProfile` を fit し、final lineage seal と stale-verdict 検証を管理 |
| `benchmark_survivors_perception.py` | CLI エントリポイント。formal 入力欠落時は BLOCKED で終了 |

## 境界

- `development_only=True` の synthetic fixture 結果は `D04-PERCEPTION-CALIBRATION` / `D04-PERCEPTION-FINAL` に昇格できません。
- calibration/final セッションの重複を `SplitOverlapError` で拒否します。
- `PerceptionFinalVerdict` は parser/detector/assembler/config/UI schema いずれかの hash が変化すると `StaleVerdictError` を送出します。
- `FinalLineageSeal.open_session()` は create-once で、2 回目の開封を `FinalSessionAlreadyOpenedError` で拒否します。
- MP4 decode は domain-shift 比較専用で、benchmark 入力（`--source-policy=raw` または `lossless`）には使えません。
- formal capture descriptor は各 calibration/final session の `annotations.jsonl` と
  `session_manifest.json` を `annotation_bindings` で immutable file ref に結びます。
  annotation JSONL の `ground_truth_semantic_hash` と raw UI evidence は runner が独立に読み、
  ground-truth loader は snapshot と検証 evidence を同時生成できません。
- calibration profile を publish/freeze して lineage seal を確定した後にのみ final PNG を
  create-once 予約・restore します。commit 済み batch の再実行は現在の全 subject file hash
  と fit code hash を再計算し、同一なら final を再予約せず canonical alias だけを回復します。
- formal rare-slice gate は `world_class_map_v2` の foreground 全 15 class（weapon 4 class を含む）を
  各200 entities（`weapon_orbit` だけ 50）、boss/hazard を各100、level-up 100、chest 30、death/result を各20要求し、
  slice ごとの session-cluster bootstrap 95% CI lower bound を検証します。
  `weapon_orbit`（King Bible）は Mad Forest 標準ビルドで出現が少ないため下限を 50 にしています。

## 出現数依存 slice の「該当なし」規則（04-13）

`foreground_class:hazard_projectile` / `hazard_area` / `weapon_projectile` / `weapon_zone` / `weapon_aura` / `weapon_orbit`
と `event:hazard` は、ステージによって出現自体が少ない「出現数依存 slice」です（04-11 の付け替えで、
Mad Forest の確認済みデータの hazard はほぼ全てプレイヤーの武器エフェクトだった）。

- 正式収録で実測した出現数が下限未満なら、その slice を「該当なし」とし、`BenchmarkReport.absent_slices`
  に `slice 名 → 実測出現数` を記録します。`absent_slices` は `metrics_wire()` 経由で `PerceptionFinalVerdict`
  の `metrics` に残るので、外した slice と根拠（出現数）が final verdict から読めます。
- 「該当なし」の slice は formal gate の下限判定と CI 判定（必須 slice）から外します。下限の値そのものは下げません。
- 出現数依存でない slice（敵・ジェム・pickup・boss・level-up など）は従来どおり下限未満で失敗します。
- verdict の再計算（`recompute_gate_from_metrics`）は `absent_slices` を `slice_counts` から計算し直した値と比べ、
  食い違えば blocking にします。出現数依存でない slice 名を書き足した metrics は読み込み時に拒否します。

`band_floor_mode: reached_only` では session floor 未達の time band を session floor 判定と cluster CI 判定の両方から外します。外した band は verdict metrics と formal CLI 成功時の stdout JSON の `formal_skipped_bands` に列挙されます。そのため `passed: true` だけで全 band が評価済みとは判断できません。

## DeployObs v2 の誤差指標と calibration 項目（04-13）

`obs_v2_residuals(ground, predicted)` が正解・予測の DeployObs v2 から新 segment ごとの残差を作り、
benchmark の `BenchmarkReport.obs_v2_errors`（field → 絶対値の平均 `mean_abs` と件数 `count`）と、
calibration 残差（`_derive_calibration_residuals`）の両方に使います。正解・予測の両方で有効な要素だけを比べます。

| 残差 field | 内容 |
|---|---|
| `obs_v2_enemy_dir16_l1` / `obs_v2_gem_dir16_l1` / `obs_v2_rare_gem_dir16_l1` / `obs_v2_projectile_dir16_l1` | 16 方向特徴（最寄り距離・近距離密度・中距離密度、projectile 密度）の segment ごとの L1 誤差 |
| `obs_v2_zone_position` / `obs_v2_zone_radius` | 両方で見えている zone 枠の位置の距離と半径の差（半幅正規化） |
| `obs_v2_slot_id_mismatch` | 武器・パッシブのスロット種類 id の不一致（0/1）。平均が誤り率、1−平均が正解率 |
| `obs_v2_ttl_first_seen_offset_s` | 両方で見えている orbit / zone の残り時間の「予測 − 正解」（秒、符号付き） |

- calibration profile の artifact は `perception_calibration_profile.v2` に上げ、`segment_error_stats`
  （field → `mean`・`std`・`count`）を追加しました。2 件未満の field は統計を出しません（誤差は不明のまま）。
- `obs_v2_ttl_first_seen_offset_s` の符号付き平均が「初観測の基準ずれ」の calibration 項目です。sim は Santa Water
  の 0.1 秒の警告時間を含めて初観測時刻を数えますが、本家で最初に見えるのは炎が出た時点の可能性があり
  （Peachone の照準も同様）、初観測が遅れると残り時間は過大評価（正の値）になります。
- `obs_v2_errors` には実データでの基準が無いので、現時点では gate に使わず記録だけします。

## Formal provider と ground truth loader

`survivors.formal_perception_provider.build_formal_predictor` は formal parser／detector package だけを復元し、frame ごとに HUD parse、world detection、tracking、`RealObsAssembler` を実行します。予測 provider は annotation、HUD truth、COCO label を読みません。runner は各評価窓の先頭で `reset()` を必須呼び出しし、窓をまたぐ temporal state を持ち越しません。

`survivors.formal_annotation_loader.build_ground_truth_loader` は descriptor で検証した `hud_truth.jsonl`、`frames.jsonl`、session 単位の COCO bytes を index 化し、要求された frame だけを読みます。`formal_ground_truth.ground_truth_snapshot` が HUD truth と COCO box から状態を持たない 1 frame の snapshot を作り、provider と同じ ID 正規化・assembler 規則に通します。`build_survivors_formal_annotations.py` もこの関数を使うため、`annotations.jsonl` の semantic hash と loader の snapshot hash、inventory／candidate hash、foreground／slice diagnostics は一致します。annotation は formal dataset publish の前に生成します。

GT `item_context` の level-up screen はその frame の label box から作ります。runtime provider は直前の gameplay world を level-up context に使うため、時間的な意味は異なります。gameplay の boss／hazard event は visible な label box の rising edge で数え、継続表示中は 1 occurrence、消失後の再出現は次の occurrence になります。foreground entity は frame-local box 単位なので、entity floor は box 出現数で満たされます。

Benchmark の入力は stride 3（約 10 Hz）で評価され、runtime の 30 fps 推論とは時間条件が異なります。遷移確率は calibration の標本平均を出してから一度だけ consumer tick 単位へ戻します（`p_consumer = 1 - (1 - p_sampled) ** (1 / (2 × stride))`）。係数2は capture 30 fps と consumer 60 Hz の換算です。gameplay の `event:boss`／`event:hazard` は GT と prediction の存在一致を使い、位置精度は foreground class slice が評価します。

## 事前登録した評価窓（04-18）

formal `error_calibration`／`final_e2e_test` の expected tick は、capture descriptor の `identity_metadata.evaluation_windows` に束縛された `evaluation_windows.v1` の窓内だけです。各窓では plan の `stride` で frame を間引き、窓外 frame は expected tick、invalid tick、遷移残差に含めません。窓内で HUD truth または COCO label が欠けた frame は従来どおり invalid tick になります。

窓は時間帯ごとの固定 offset、run tail、development parser の state rising edge、development world detector の boss rising edge から規則で作ります。sealed session の内容を見て手動で窓を選びません。評価結果が示すのは session 全体ではなく事前登録窓内の性能です。formal runner は verdict 作成時と batch 復元時に、metrics の `formal_gates.yaml_sha256` と `evaluation_windows_hash` を capture descriptor の YAML／plan identity と照合し、不一致なら拒否します。`recompute_gate_from_metrics` は metrics 内 gate 値の canonical hash を再計算しますが、`verify_formal_runtime_release` は YAML bytes hash や窓 plan hash を descriptor と再照合しません。gate の既定 floor は変更せず、世代固有値は descriptor で指定された YAML を使います。

annotation 前に `plan_survivors_evaluation_windows.py` で plan を生成して session log と artifact store に保管します。`check_survivors_evaluation_coverage.py` は同じ `ground_truth_snapshot`／`slice_counts_for_snapshot` helper で窓内 floor を集計し、不足 frame と slice を表示します。

## CLI

```bash
# formal 実行（D04-CAPTURE-DATASET + 04-05 + 04-08 が必要）
python benchmark_survivors_perception.py \
  --capture-dataset /path/to/d04_capture_manifest.json \
  --parser-package /path/to/hud_parser_package.json \
  --detector-package /path/to/detector_package.json

# development-only dry-run（formal 入力なしで synthetic fixture のみ）
python benchmark_survivors_perception.py --dry-run
```

## Formality

このモジュールは code-only PR（04-10）の成果物です。
`D04-CAPTURE-DATASET`（04-02）、formal HUD parser package（04-05）、formal world detector package（04-08）が揃うまで、calibration/final セッションは開封しません。
formal benchmark 実行後に `D04-PERCEPTION-CALIBRATION` と `D04-PERCEPTION-FINAL` を別 DAG node として atomic publish します。

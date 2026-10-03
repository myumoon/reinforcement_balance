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

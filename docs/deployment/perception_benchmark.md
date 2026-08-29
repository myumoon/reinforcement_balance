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

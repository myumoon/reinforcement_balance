# HUD calibration tooling

Survivors HUD parser の ROI fitting、11 gate validation、development package 発行を行う。
入力は `model_train` と `model_validation` だけで、`error_calibration` と
`final_e2e_test` は拒否する。`CalibrationConfig` は validation 前に固定し、結果から調整しない。

```bash
python Tools/Deployment/calibrate_survivors_hud_parser.py \
  --train-annotations /artifact/hud_train.json \
  --validation-annotations /artifact/hud_validation.json \
  --output-package /artifact/hud_parser_dev.json
python Tools/Deployment/eval_survivors_hud_parser.py \
  --annotations /artifact/hud_validation.json --output /artifact/hud_report.json
```

注釈 JSON は `sample_id`、`split`、正解/予測値、latency、正解/予測 ROI を持つ object の配列。
floor gate は timer_exact/level/item/choice/screen_state_f1/roi_inside_rate、ceiling gate は
hp_mae/xp_mae/latency/roi_center_error/roi_false_positive で、いずれも境界値を合格とする。

development package は `development_only=true`, `formal_parser_eligible=false` です。fit/report/用途属性を
`package_hash` に束縛し、同一 directory で fsync 後に atomic replace します。生成物と実データは
Git 外へ置き、formal load は development package を拒否します。

## Formal release tooling (04-14)

`--formal` は `hud_truth.jsonl` を dataset descriptor の session/hash binding から読み、frozen split、
formal eligibility、lossless PNG、frame id、restore verdict と validation slice の件数を preflight します。
`error_calibration` と `final_e2e_test` は開封しません。確認済みの train truth だけで fit と実 frame atlas を作り、
validation frame は `HudParser` で in-process 予測して pooled 11 gate と session cluster 95% CI を評価します。

formal 実行には `--capture-store`、`--dataset-descriptor`、`--restore-verdict`、`--target-profile-hash`、
`--vocabulary`、`--minimum-counts`、`--artifact-store`、`--backup-root`、`--restore-root`、
`--logical-prefix`、`--out-dir` が必要です。minimum counts は `time_bands`（秒の `開始-終了`）、
`screen_states`、`card_counts`、`item_classes`、`buttons`、`fallback`、`roi_negative` を指定します。
`buttons` は任意 slice で、空 mapping にできます。`hud_truth.v1` に button 正解 field がないため既定では対象外です。
下限を設定した場合は、対応する正解件数が不足すると preflight が拒否します。
gate が落ちたら exit 2、preflight／入力不正なら exit 1 です。

package schema は `hud_parser_package.formal.v1` で development schema と分離しています。publisher は
package、atlas、preflight、formal report の hash chain と profile/build を照合してから書き込みます。
ArtifactStore publish 後に空 restore root へ全 object を復元し、full restore verdict を保存します。
resolved layout yaml は `--out-dir` に出ます。正式 dataset での実行・release yaml の管理は 04-05 の対象です。
validation の parser hash は placeholder なので、04-16 provider は package JSON の SHA-256 を `HudParser` に渡してください。

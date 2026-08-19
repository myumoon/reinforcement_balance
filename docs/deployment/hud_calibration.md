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

package は常に `development_only=true`, `formal_parser_eligible=false`。fit/report/用途属性を
`package_hash` に束縛し、同一 directory で fsync 後に atomic replace する。生成物と実データは
Git 外へ置く。formal load は development package を拒否し、formal publish は04-05まで未実装。

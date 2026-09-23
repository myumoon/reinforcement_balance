# Survivors perception feasibility gate

この文書は00-04 spikeの判定契約であり、実測PASSの代替ではない。設定値は
`Tools/Deployment/configs/perception_feasibility_v1.yaml`、判定実装は
`spikes/survivors_vertical_feasibility.py`を正とする。

## Gate

3～5独立session（各10分以上）、必須slice各2 session、二重annotation 300
frame以上を必要とする。build/profile/sessionの混在、target audit未合格、slice
不足、独立した2名のannotation担当不在、比較architecture不足のいずれかで
FAILとなり、PASSは発行できない。

320入力のp10 short-sideが4px未満、またはlate/heavy oracle-assisted recallが
0.85未満ならSSDLite320を却下する。single-pass p95が25msを超えた場合もtileや
multi-scaleを自動採用せず、utility/latencyで全候補を比較する。bbox QA IoU
0.80未満、class agreement 0.95未満、dense annotation 300 entities/hour未満なら
segmentation/density/count supervisionを候補化する。

JSON/Markdown verdictはselected architecture、rejected alternatives、切替条件、
unresolved risk、split別session/frame/entity/UI-event、annotation/GPU/storage/
wall-clock/worker予算を含む。FAIL時は04-01以降およびlong-run student学習を開始
してはならない。

action displacementは00-03 golden telemetryをread-only parentとして再生する。
`proposal_vector`と`measured_screen_displacement`は別fieldであり、提案値を実測値
として扱わない。このspikeはlive inputを送信しない。capture/video/atlasと実測
verdictはGit外artifact storeへ保存し、後のuntouched final splitへ再利用しない。

## 実測pilot証拠からの判定発行（CLI）

`spikes/feasibility_gate_cli.py`は、実測pilotのセッションmetadata・probe結果・
annotation結果・TargetProfileから`GateEvidence`を組み立て、既存の`write_verdict()`
をそのまま呼び出すだけの薄いドライバである。判定ロジックそのものは変更しない。

```bash
python -m spikes.feasibility_gate_cli \
  --sessions pilot_sessions.json \
  --probe-results probe_run1.json probe_run2.json \
  --annotation-results annotation_summary.json \
  --annotators alice bob \
  --representative-frames 320 \
  --target-profile .env/target_profile.resolved.yaml \
  --output-json verdict.json \
  --output-markdown verdict.md
```

`--probe-results`は複数ファイルを指定でき、pilotが複数セッションに分割されて
実施された場合の`evaluate_probe()`出力を束ねる。統合方針は対応する数値指標の
単純平均であり、標本への再集計は行わない（`evaluate_probe()`は集計済み出力しか
持たないため）。`--target-profile`省略時は`target_profile.py`の既定
（`.env/target_profile.resolved.yaml`）を読む。

入力JSONの形は次の3種：

- `--sessions`: `{"session_id","build_id","profile_id","minutes","slice"}`を
  持つオブジェクトの配列。`slice`は本文中の必須slice名（early/mid/late/heavy/
  level_up/chest/death_result）のみを含む非空リストでなければならない。
- `--probe-results`: `perception_probe.evaluate_probe()`の戻り値そのもの
  （`pixel_size`/`recall_upper_bound`/`latency_ms`/`annotation_seconds`/
  `architectures`）。
- `--annotation-results`: `annotation_throughput.summarize_annotation()`の
  戻り値そのもの（`entities_per_hour`/`dense_entities_per_hour`/
  `qa_rework_rate`/`bbox_qa_iou`/`class_agreement`/`annotation_hours`）。

入力の欠損・型不正・schema不一致（未知フィールド）・`--annotators`不足
（2名未満）は、CLIが非ゼロ終了・原因を含むstderrメッセージでfail-closedに
拒否する。一方、`--target-profile`のprovenanceが`operator-attested`でない
場合や、`--annotators`に重複名がある場合は、CLIは正常終了（終了コード0）した
うえで判定を`FAIL`として発行する（本文中のtarget audit未合格・独立annotation
担当不在の既存fail-closed業務仕様に従うため）。

既知の制約: `merge_probe_evaluations()`のutility変換は
`utility = utility_per_latency * latency_p95_ms`（`evaluate_probe()`の
`utility_per_latency = mean(oracle_detectable) / latency_p95_ms`の逆算）を
採用している。分子（oracle検出率の平均）がarchitecture非依存の合成fixtureでは
4方式のutilityが同一値に潰れ、`issue_verdict()`のswitch条件
（`max(utility/latency)`）がlatencyの短い順に単純化されてしまう。判定ロジック
自体（`issue_verdict`/`GateEvidence`）はこのCLI追加の対象外で変更していないため、
実測pilotデータでもこの潰れが起こるかは別途確認が必要であり、起こる場合は
utility定義の見直しを別PRで検討する。

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

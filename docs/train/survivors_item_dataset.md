# Survivors ItemSelector 教師 dataset

Phase 02-01 は、Phase 01-06/01-07 で承認された teacher label を ItemSelector 用の
development dataset として release し、画面から取得できる feature だけで label を説明できる
上限を測るフェーズです。ItemSelector の実訓練と closed-loop 評価は含みません。

## Feature 契約

`context_only_v1` は level-up UI とプレイヤー状態だけを使います。`context_danger_v1` はそれに
直近の画面内 enemy/gem density、nearest enemy、boss/hazard と snapshot validity/age を加えます。
候補は `max_item_cards` まで padding し、`card_mask` が有効位置を決めます。

teacher score、raw value、critic latent、privileged world state、off-screen count は context/candidate
feature に含めません。teacher score は label 生成側にだけ存在し、既存の
`teacher_score_scale` と `teacher_reliability_calibration` artifact identity へ束縛されます。

## Split と test sealing

`SplitManifest` は episode、またはより粗い prefix group を stable hash 順で 70/15/15 の
train/validation/test に一度だけ割り当てます。decision 行単位の再分割は行いません。exact trace
fingerprint と near-duplicate key が異なる group をまたぐ入力は release 前に拒否します。

ablation、Bayes ceiling、temperature selection、learning curve は train/validation のみを読みます。
test は `SplitManifest.read_test_rows()` 以外から読めず、この専用 API でも ablation/ceiling/
learning-curve purpose は拒否されます。

## Soft target と reliability

score は 01-06 artifact の `sigma` で標準化し、降順隣接差が `tie_epsilon_z=0.02` 未満の候補を
tie group として同じ logit に pool します。無効候補を除外した masked softmax の温度は
`[0.25, 0.5, 1.0, 2.0, 4.0]` から development NLL、次に Brier で選びます。

reliability weight は 01-07 artifact が保持する
`support_factor * clip(1 - error_ucb_z / 1.0, 0, 1)` を再計算検証した値です。
underpowered、unknown/final-derived lineage、fabricated support、`error_ucb_z >= 1.0` は release
blocker です。02-01 で scale を再 fit したり別 teacher の scale/calibration を代用したりしません。

## Feature feasibility data card

`item_feature_feasibility.run_feature_feasibility` は development partition だけで両案の label entropy、
Bayes top-1/NDCG ceiling と episode bootstrap 95% CI を計算します。data card JSON/Markdown には
入力次元、実 availability、latency class、partition access audit と minimum viable feature の根拠を
保存します。実 data の出力先は収集 run ごとに明示し、このリポジトリには固定しません。

offline top-1 floor は固定 0.85 ではなく、選定案の Bayes ceiling lower bound、事前指定の ceiling
retention、business floor から test open 前に登録します。収集は初期 2,000 decisions、その後
2,000 件 block とし、coverage 達成後に validation NDCG gain `< 0.005` が 2 block 継続すると停止します。
hard cap は Phase 01-05 で得た throughput/time budget と storage budget の小さい方です。

## Release CLI

`Tools/Training/build_survivors_item_dataset.py` は source identity、approved label verdict、teacher
identity、acyclic ancestry、manifest split binding を全行で検証します。test 行は derived shard へ
書き込めません。manifest の `source_trace_index` から任意 decision を source trace ref へ逆参照
できます。

UE5 build / LLT: 未実行（Windows 専用）

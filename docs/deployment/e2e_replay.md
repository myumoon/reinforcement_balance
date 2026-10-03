# Survivors recorded replay end-to-end 回帰検証 (06-01)

`Tools/Deployment/replay_survivors_session.py`(recorded replay CLI)と
`Tools/Deployment/configs/e2e_replay_suite_v1.yaml`(synthetic 検証 suite)の使い方です。
記録済み capture session を実 `SurvivorsController` へ仮想時計で流し、perception・DeployObs・
行動決定・state 遷移・shadow effect を決定的に再生して、artifact 更新時の回帰を見つけます。
判定値・schema 名はすべて `survivors/replay/e2e_replay.py` の定数が唯一の正で、本ドキュメントは
それを転記したものです(食い違った場合は実装を優先してください)。

コマンドはいずれも `Tools/Deployment` をカレントディレクトリにして実行します。

## 仕組み

- **仮想時計**: `VirtualClock` を controller の `clock_ns`/`sleep` へ注入します。replay path は
  `time.sleep`/`time.perf_counter_ns` などの実時計を一切呼ばず、記録 event の timestamp で時計を進めます。
- **frame source**: `RecordedFrameSource` が capture manifest の event 列(frame / duplicate / timeout /
  focus_lost / inference_stall)を記録された completion 順に再生し、`CapturedFrame` を返します。
  inference_stall は直前の frame の推論が記録時刻まで返らなかったことを表し、runtime へ注入した時計
  (`inference_clock_ns`)が推論中に仮想時計を進めるので、runtime の inference timeout gate を実際に通ります。
- **shadow 固定**: controller は常に shadow mode で、`execute_effect` も live input backend も使いません。
  effect は telemetry の `effect` 行から semantic JSON(`effects.json`)として保存します。
- **決定性 manifest**: `torch.use_deterministic_algorithms`・device・NMS 実装・threads を記録時と照合し、
  食い違えば1 frame も流さずに拒否します。再生側の値は申告ではなく実測です。device は読み込んだ detector と
  combat policy の parameter から読み(`--device` を渡して実測と違えば `DeterminismMismatchError`)、
  NMS は detector が実際に使う `torchvision.ops.nms` だけを受け付けます。
- **2経路の比較**: state/action/effect などの discrete 出力は canonical hash の exact 一致、obs/logits/latency
  などの numeric 出力は量子化 hash → segment 別 tolerance の順で判定します。全比較対象に同じ2経路を当てます。

## replay の実行

```bash
python replay_survivors_session.py replay \
  --capture-manifest <session>/capture.json --output-dir <out> \
  --combat-package <pkg> --detector-config <cfg> --class-map configs/world_class_map_v2.yaml \
  --detector-weights <weights> --detector-manifest <manifest> [--runs 3]
```

- capture の target profile / game build が今の profile と合わなければ、artifact を読む前に
  `ReplayIntegrityError` で止まります。
- `--runs 1`(既定)は `<out>/` へ、`--runs N` は `<out>/run-1..N/` へ出力し、1回目と各回を比べた結果を
  `<out>/comparisons.json` に書きます。artifact は run ごとに読み直します(tracker などの内部状態を共有しない)。
- 出力: `telemetry.jsonl` / `discrete.jsonl` / `numeric.jsonl` / `numeric_obs.npz` / `effects.json` / `manifest.json`。
- 終了コード: 1回なら controller の終了コード(0/2/3/4)、複数回なら比較が1つでも失敗すると 1。

これが local full suite の実行方法です。実 recorded session ごとに `--runs 3` で回し、discrete hash 一致と
numeric segment の quantized hash / tolerance gate を確認します。

## golden 比較

```bash
python replay_survivors_session.py compare --old <run-a> --new <run-b> [--golden golden.json] [--report diff.json]
```

- `--old`: 旧 run と比べ、合否・metrics と最初の分岐点(first divergence)を tree で出します。tree は
  frame(correlation id)→ stage → 分岐の種類(parser field / obs の平面・index・segment / model action /
  state / effect)の順に並び、どの segment・stage が最初に崩れたかが分かります。
- `--golden`: golden に固定した参照 run 出力(`<golden 名>.reference/`)と `--old` と同じ規則で比べます。
  discrete は exact hash、numeric は stage 値・latency・obs 3平面の全 segment を量子化 hash → segment tolerance の
  順で判定するので、quantum 境界をまたぐだけの許容内の差は食い違いになりません。artifact hashes は exact に比べます。
  食い違った項目名を `golden_mismatches` に、分岐点を `golden_first_divergence` に出します。
  参照 run 出力が golden の `reference_sha256` と合わなければ `ReplayIntegrityError` で止まります。
- 合格なら 0、1つでも食い違えば 1 を返します。

## golden update

```bash
python replay_survivors_session.py update-golden --golden golden.json \
  --old-run <旧 golden 側の run> --new-run <新しい run> --artifact-hashes hashes.json [--approved-by <担当者>]
```

- 旧/新 run の集計 metrics と新旧の比較 metrics は run から計算し、golden に保存します。
- 新 run の `discrete.jsonl` / `numeric.jsonl` / `numeric_obs.npz` を `<golden 名>.reference/` へ複製し、
  その sha256 を golden の `reference_sha256` に固定します(golden 比較の tolerance 再判定に使う)。
- `--artifact-hashes` は新 run の artifact hashes(bundle/detector 等と `capture_manifest_sha256`、NPZ があれば
  `frames_sha256`)を明示した JSON で、新 run の manifest と1つでも食い違えば `GoldenUpdateError` です。
- 前の golden から artifact hashes が変わる更新(model/parser の版が変わる)は、独立検証担当者の
  `--approved-by` が無ければ拒否されます。

## formal verdict の publish

```bash
python replay_survivors_session.py publish-formal-verdict \
  --capture-manifest <session>/capture.json <artifact 引数> \
  --run <out>/run-1 --run <out>/run-2 --run <out>/run-3 --verdict verdict.json
```

次のどれかに当たると `FormalReplayRejectedError` で publish を拒否し、verdict ファイルを書きません(fail closed)。

- runtime bundle / detector / capture の正式 parent が揃わない(`development_only` を含む)
- run 数が3未満、または比較結果が足りない
- run の manifest が `development_only` / `formal_replay_eligible=false`、別 capture、別 artifact
- run の artifact hashes が今読んだ artifact と違う

## synthetic と formal の違い

| | synthetic(本 PR) | formal(`D06-RECORDED-REPLAY`) |
|---|---|---|
| 入力 | `e2e_replay_suite_v1.yaml` からコードで作る画素なし mini session | lossless で記録した実 session |
| parent | development bundle・fixture detector/HUD | 03-05 student release・04-10 perception・05-04 formal shadow |
| 結果 | `development_only=true`・`formal_replay_eligible=false` | 条件が揃えば `formal_replay_eligible=true` |
| 実行 | 標準 pytest | 手動ゲート(本 PR の範囲外) |

synthetic suite からは formal replay verdict を発行できません。`D06-RECORDED-REPLAY` の手動実行(正式 parent と
lossless session を固定して3回再生、安全 fixture の hard failure 0、30分 replay を wall-clock 10分以内、
first divergence 空、verdict の backup/restore と 06-04 への昇格)は merge 後に別途行います。
MP4 replay は正規入力ではなく codec domain-shift の別テストとして扱います。

## repo に置くもの

actual frames は local artifact として手元に保存し、repo へは commit しません。repo に置くのは replay manifest・
golden hashes と synthetic mini session(`e2e_replay_suite_v1.yaml` と、それを組み立てるテストコード)だけです。
capture manifest の `frames_path` は manifest からの相対パスの NPZ(`frame_<index>` に 1080x1920x4 uint8)で、
`frames_sha256` で中身を固定します。

## synthetic suite

`configs/e2e_replay_suite_v1.yaml` は次の fixture を列挙します。各 fixture は画面の台本(script)と
capture 側の異常(faults)と、1回目の再生で必ず起きること(expect)を持ちます。

- early normal gameplay + gem collection
- dense late-game / effects / occlusion
- target-derived 1..3 card(new / upgrade / evolution / union / fallback)と reroll・skip ボタン
- chest、pause/resume、death、result、run restart
- fault: capture gap(drop・duplicate を含む)、focus loss、unknown UI、parser low confidence、inference timeout

テストは各 fixture を同じ bundle で3回再生し、discrete hash 一致と numeric gate を確認します。
`safety: true` の fixture は、fault を観測した tick(unknown 中の tick、focus loss / capture timeout の後に最初に
処理された frame、推論が遅れた frame)の effect が release / no-op だけであることを hard assertion します
(aggregate tolerance では許容しない)。観測 tick が0件の区間は検証済みとみなさず、run が health_stop で終わった
場合だけ区間開始以降の全 effect を対象にして許します。inference timeout の fixture は policy 行の reason に
`inference timeout gate failed` が出ることも確かめ、timeout gate の無効化・health stop の無効化・区間内での移動を
入れた改変で各 safety fixture が `SafetyAssertionError` になる mutation test を持ちます。
low confidence の frame からは移動を出さないことも確かめます。
さらに仮想時計で30分進む schedule(200ms 間隔・約9000 frame)が wall-clock 10分以内に終わることを測ります
(手元で約90秒)。

## ローカルの検証コマンド

```bash
USER=$USERNAME bash Tools/run-pytest.sh Tools/Deployment/tests/replay -q -rs
```

リポジトリ直下で実行します。Git Bash では `USER` が未定義なので `USER=$USERNAME` を付けます。

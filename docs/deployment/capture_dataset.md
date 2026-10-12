# Capture Dataset

Survivors の frame は、明示した Git 外 store の一時 session へ lossless PNG と JSONL で逐次保存し、完成後に directory rename で publish する。収録時の manifest は `formal_dataset_eligible=false` であり、operator が resolved target profile と build を照合して昇格 CLI を実行した後だけ `true` になる。live capture の成功後は開始時の resolved profile bytes、統計 JSON、記録雛形から埋めた Markdown を `<RecordRoot>/session_logs/<SessionId>/` に保存する。これは `capture_sessions/` の外なので capture hash を変えない。synthetic／dry-run は session log を作らず、log 保存に失敗しても公開済み session は残る。MP4 は目視・ドメイン比較専用で、正式な pixel source にはできない。

## Synthetic smoke

実ゲーム画像を使わない disposable pixel の動作確認:

```bash
python Tools/Deployment/capture_survivors.py \
  --store-root /mnt/d/reinbalance-capture \
  --session-id synthetic-smoke-001 --duration-sec 1 --synthetic
```

## Dry-run

frame 契約まで検証し、PNG / JSONL / manifest を書かない:

```bash
python Tools/Deployment/capture_survivors.py \
  --store-root /mnt/d/reinbalance-capture \
  --session-id synthetic-dry-001 --duration-sec 1 --synthetic --dry-run
```

## Live pilot

Windows 上で operator-attested target profile、前面の 1920x1080 borderless target、固定済み `opencv-python==4.10.0.84` / `dxcam==0.3.0` を準備して短時間実行する:

```powershell
python Tools/Deployment/capture_survivors.py `
  --store-root D:\reinbalance-capture `
  --session-id live-pilot-001 --duration-sec 30
```

`--store-root` の省略、build/profile の変化、解像度・monitor の変化は fail-closed で終了する。実ゲーム PNG / MP4 / annotation は `capture_sessions/` 配下に置き、Git へ追加しない。

`--session-id` は `auto` を指定すると `capture_sessions/` 配下の既存連番の次(`session-0001` から)を自動採番する。毎回一意な ID を手入力する必要はない:

```powershell
python Tools/Deployment/capture_survivors.py `
  --store-root D:\reinbalance-capture `
  --session-id auto --duration-sec 30
```

起動直後に対象ウィンドウが前面へ来るまで自動でポーリング待機するため、従来 operator が別途行っていた `Start-Sleep` での起動待ちは不要になった。

## 中断・一時停止

収録中に alt-tab 等で対象ウィンドウが一時的にフォアグラウンドを失っても、収録は中断されない。自動で一時停止し、ウィンドウが前面へ戻ると自動で再開する(解像度変更やプロセス差し替えなど、alt-tab 以外の状態変化は従来通り即座に fail-closed で終了する)。一時停止・再開のたびに `capture paused: target window lost foreground` / `capture resumed: target window regained foreground` を stderr へ出力するため、operator はコンソールで一時停止が起きたことを確認できる。

Ctrl+C は 1 回目と 2 回目で挙動が異なる:

- **1 回目**: 収録ループを安全に打ち切り、それまでに撮れたフレームがあればそのまま `PUBLISHED` として公開する。1 枚も撮れていなければ何も公開せず `ABORTED_EMPTY` として終了する（従来の「未公開 temp を丸ごと破棄する」動作による全データ消失を避けるための変更）。
- **2 回目**: 通常の `KeyboardInterrupt` として即座に中断する。この場合は未公開の一時領域が破棄される。意図的に全て破棄したい場合のエスケープハッチとして残している。

終了時の JSON (`PUBLISHED` / `ABORTED_EMPTY`) には `ended_reason`(`duration_elapsed` / `source_exhausted` / `interrupted`)・`requested_duration_sec`・`elapsed_sec` が含まれる。

## Annotation workflow

画面 annotation は [D04 capture と annotation の手順](capture_annotation_manual.md) の 2-2〜2-7 に従ってください。`annotate_survivors_frames.py` は旧 stdin CLI で、class 定義が world class map v2 と一致しないため正式 annotation に使用しません。

二重 annotation を行う場合は2人目に別の work-root を用意し、1人目の label と、1人目の frame で学習した prelabel を見せません。agreement は2つの work-root を入力にして作成します。
二重 annotation と agreement は任意です。publish の `--agreement` も省略でき、agreement は 04-10／04-14／04-15 の判定には使いません。

## Split freeze workflow

収録時に `capture_survivors.py --planned-split <model_train|model_validation|error_calibration|final_e2e_test>` で用途を登録します。昇格 CLI は `operator_checkpoint` の planned split を使い、`freeze_survivors_capture_split.py` が session manifest から割り当てます。個々の session を手で別用途に割り当てず、freeze 後は用途や manifest を変更しません。詳細な CLI と実行順は「Formal promotion と publish」を参照してください。

同一 session の複数用途への割り当て、特に `final_e2e_test` から train/calibration への参照は `SplitConflictError` になる。

## Formal promotion と publish

session の annotation 作業中でも昇格 CLI は使えます。昇格が変更するのは `session_manifest.json` の eligibility と `operator_checkpoint` だけで、frame PNG、`frames.jsonl`、`annotations.jsonl`、build/profile identity は変更しません。**順序は必ず「昇格 → 割当 → freeze」**です。freeze 済み split に入った session は昇格できません。

`--planned-split` で保存する用途は次の4種類です。

| 値 | 用途 |
|---|---|
| `model_train` | 練習用の学習 |
| `model_validation` | 練習用の確認 |
| `error_calibration` | 本番テスト用の誤差統計 |
| `final_e2e_test` | 本番テスト用の最終判定 |

収録前に用途を決め、`capture_survivors.py --planned-split` で記録します。既存 session は `planned_split` と evaluation window plan が無く、model 調整に使ったものもあるため、`model_train`／`model_validation` または development に限ります。calibration／final へ移すための再 annotation はしません。

```powershell
python Tools/Deployment/capture_survivors.py `
  --store-root "<RecordRoot>" --session-id auto --duration-sec 2700 `
  --planned-split model_train
python Tools/Deployment/promote_survivors_capture_session.py `
  --store-root "<RecordRoot>" --session-id "<SessionId>" `
  --operator-id "<OperatorId>" --note "実機 build と収録内容を確認"
python Tools/Deployment/freeze_survivors_capture_split.py `
  --store-root "<RecordRoot>" `
  --manifest-path "<RecordRoot>/capture_split_manifest.<YYYYMMDD>.json"
```

`--target-profile` は省略できます。省略時は session ごとの `<RecordRoot>/session_logs/<SessionId>/target_profile.resolved.yaml` を読み、`--all-planned` でも各 session の copy を個別に照合します。ファイルが無い場合は昇格を拒否します。別の profile copy を使うときは `--target-profile "<ProfilePath>"` を指定できます。どちらの場合も `target_resolve` を再実行しません。`--all-planned` は checkpoint に用途が入った session をまとめて昇格します。freeze 前なら同じ昇格 CLI の `--planned-split` で用途を変更できます。既存の frozen manifest は上書きせず、sealed session を加える世代には別の `--manifest-path` を使います。

session 単位の COCO export は capture store の pixel hash と split を束縛します。model train/validation の world/UI 出力はそれぞれ2 split を merge し、二人の独立 annotation の agreement を作ってから publish します。

```powershell
python Tools/Deployment/export_survivors_annotations_coco.py `
  --work-root "<WorkRoot>" --store-root "<RecordRoot>" `
  --split-manifest "<SplitManifest>" --split model_train `
  --output-dir "<DatasetDir>/model_train"
python Tools/Deployment/export_survivors_annotations_coco.py `
  --work-root "<WorkRoot>" --store-root "<RecordRoot>" `
  --split-manifest "<SplitManifest>" --split model_validation `
  --output-dir "<DatasetDir>/model_validation"
python Tools/Deployment/merge_survivors_coco.py `
  --world-train "<DatasetDir>/model_train/world_coco.json" `
  --world-validation "<DatasetDir>/model_validation/world_coco.json" `
  --ui-train "<DatasetDir>/model_train/ui_coco.json" `
  --ui-validation "<DatasetDir>/model_validation/ui_coco.json" `
  --output-dir "<DatasetDir>/model"
python Tools/Deployment/build_survivors_annotation_agreement.py `
  --work-root-a "<WorkRootA>" --work-root-b "<WorkRootB>" `
  --session-id "<TrainSession>" --session-id "<ValidationSession>" `
  --output "<DatasetDir>/annotation_agreement.json"
```

04-18 merge 済みの現行 plan では、model session は従来の候補選択、sealed session は `plan_survivors_evaluation_windows.py` で固定した window plan に従って窓内の全 frame を annotation します。sealed の窓内 frame は画面全体をラベルして `labeled_region` を使いません。

04-16 merge 後、sealed session の formal `annotations.jsonl` は `build_survivors_formal_annotations.py` で作成します。formal schema は旧 `AnnotationWriter.read_annotations()` では読めません。session ごとの COCO は `<RecordRoot>/coco/<SessionId>/` へ出力します。正式 gate の既定構成 `Tools/Deployment/configs/formal_gates_minimal_v1.yaml` は 04-18 merge 後に使います。最短 session 長は300秒、`event:death` は20回、`event:result` は floor 未満の実測件数を記録して gate にせず、time band は到達 session が2本以上のときだけ判定します。annotation 前に YAML と SHA-256 を session log に記録し、publish descriptor の `formal_gates.sha256` と照合します。coverage と verdict notes でデータ不足と性能不足を区別します。

```powershell
python Tools/Deployment/publish_survivors_capture_dataset.py `
  --store-root "<RecordRoot>" --split-manifest "<SplitManifest>" `
  --artifact-store "<ArtifactStore>" --backup-root "<EmptyBackupRoot>" `
  --restore-root "<EmptyRestoreRoot>" --logical-prefix "survivors/capture/<Generation>" `
  --world-coco-model "<DatasetDir>/model/world_coco_model.json" `
  --ui-coco-model "<DatasetDir>/model/ui_coco_model.json" `
  --evaluation-windows "<EvaluationWindows.json>" `
  --formal-gates "<formal_gates_minimal_v1.yaml>"
```

`--agreement` は二重 annotation をした場合だけ指定する任意引数です。04-10／04-14／04-15 の判定には使いません。model-only 世代では `--formal-gates` と `--evaluation-windows` も省略できます。publish は descriptor と全 object を発行し、別の空 restore store への復元 verdict を同時に作ります。

`--evaluation-windows` は publish 時に evaluation windows artifact を指定します。現行の 04-18 merge 済み plan では window plan を annotation 前に保存します。正式 test 用の最短長 300 秒は pinned `formal_gates.v1` の値であり、現行 `perception_session_split.py` の1800秒条件を適用する運用では `--duration-sec 1800` ちょうどを使わず余裕を取ります。

## Verification

```bash
bash Tools/run-pytest.sh Tools/Deployment/tests -q -rs
git status --short
```

### PR smoke 実行結果 (M11)

```
bash Tools/run-pytest.sh Tools/Deployment/tests -q -rs
# exit code: 0
# 294 passed, 0 failed, 1 skipped
# skipped: requires a real Windows file handle (test_target_audit.py:193)
```

### git status 実データ非表示確認 (M12)

PR smoke 実行時の `git status --short` 出力（実ゲーム pixel / video / annotation なし）:

```
 M .gitignore
 M Tools/Deployment/pyproject.toml
?? Tools/Deployment/annotate_survivors_frames.py
?? Tools/Deployment/capture_survivors.py
?? Tools/Deployment/survivors/capture_dataset.py
?? Tools/Deployment/tests/test_capture_annotation.py
?? Tools/Deployment/tests/test_capture_dataset.py
?? docs/deployment/capture_dataset.md
```

`.png`, `.mp4`, `capture_sessions/`, `capture_store/` はいずれも現れない。`formal_dataset_eligible` は常に `false`。

# Capture Dataset

Survivors の frame は、明示した Git 外 store の一時 session へ lossless PNG と JSONL で逐次保存し、完成後に directory rename で publish する。writer が作る manifest は常に `formal_dataset_eligible=false` であり、formal dataset への昇格は merge 後に operator が artifact store で行う手動 gate のみとする。MP4 は目視・ドメイン比較専用で、正式な pixel source にはできない。

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

GUI を使う手順は [Survivors キャプチャの GUI アノテーション](annotation_gui.md) を参照してください。

```bash
python Tools/Deployment/annotate_survivors_frames.py \
  --store-root /mnt/d/reinbalance-capture --session-id live-pilot-001 \
  --annotator-id operator-01 --resume
```

stdin へ `FRAME_ID CLASS LEFT TOP RIGHT BOTTOM` を入力する。`undo` は最後の annotation を削除、`skip FRAME_ID` は指定フレームをスキップとして記録して進む（再アノテーションするには `write_unskip()` が必要）、`done` は終了する。second review は `--second-review` を付ける（初回レコードは JSONL に残り監査証跡として保持される）。各 annotation は即時 `annotations.jsonl` へ autosave される。

## Split freeze workflow

4 用途は session/build 単位で排他的に割り当て、一度 freeze したら追記も用途変更もしない:

```python
from survivors.capture_dataset import SplitFreezer

freezer = SplitFreezer("/mnt/d/reinbalance-capture")
freezer.assign("model_train", ["train-session-001"])
freezer.assign("model_validation", ["validation-session-001"])
freezer.assign("error_calibration", ["calibration-session-001"])
freezer.assign("final_e2e_test", ["final-session-001"])
manifest = freezer.freeze()
print(manifest.manifest_sha256)
```

同一 session の複数用途への割り当て、特に `final_e2e_test` から train/calibration への参照は `SplitConflictError` になる。

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

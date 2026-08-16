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

`--store-root` の省略、focus loss、build/profile の変化は fail-closed で終了する。実ゲーム PNG / MP4 / annotation は `capture_sessions/` 配下に置き、Git へ追加しない。

## Annotation workflow

```bash
python Tools/Deployment/annotate_survivors_frames.py \
  --store-root /mnt/d/reinbalance-capture --session-id live-pilot-001 \
  --annotator-id operator-01 --resume
```

stdin へ `FRAME_ID CLASS LEFT TOP RIGHT BOTTOM` を入力する。`undo` は最後の annotation を削除、`skip` は現在の入力を保存せず進み、`done` は終了する。second review は `--second-review` を付ける。各 annotation は即時 `annotations.jsonl` へ autosave される。

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

最終検証の exit code と passed / failed / skipped 件数、および `git status` の実データ非表示確認は implementation evidence JSON に記録する。

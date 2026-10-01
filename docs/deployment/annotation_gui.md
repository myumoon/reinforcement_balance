# Survivors キャプチャの GUI アノテーション

候補フレームを抽出し、自動下書きを作ってから X-AnyLabeling で確認・修正し、COCO JSON を出力します。ゲーム本体や Python の依存関係を追加する必要はありません。

## 1. X-AnyLabeling を用意する

[X-AnyLabeling v4.0.6 の GitHub Releases](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v4.0.6) から Windows 用 exe を取得します。

- 通常は `X-AnyLabeling-v4.0.6-CPU.exe` を使います。
- 対応する CUDA 環境がある場合は `X-AnyLabeling-v4.0.6-Windows-CUDA12.exe` を選べます。

この手順はラベル JSON の `checked` と矩形形式を v4.0.6 に合わせています。

## 2. 候補フレームを抽出する

公開済みキャプチャから、foreground のフレームを選んで work-root に配置します。work-root を省略すると `<store-root>/annotation_work` が使われ、既定で最大 200 枚を stride 15 で抽出します。

```powershell
python Tools/Deployment/select_survivors_annotation_candidates.py `
  --store-root "D:\captures" `
  --session-id "session-0001"
```

必要なら `--work-root "D:\annotation_work"`、`--count 200`、`--stride 15` を指定します。`classes.txt` はこの作業領域に作成されます。

## 3. 自動下書きを作る

固定矩形（`player_anchor` / `hud_hp` / `hud_xp`）と、確認済みラベルで学習した下書き用検出器（Faster R-CNN）の推論結果を使い、指定セッション内の JSON がまだ無い PNG に下書きを作ります。

### 3-1. 学習用の確認済みフレームを用意する（最初の周回）

時期の違うフレームを数枚選び、X-AnyLabeling で確認済みにします（手順は「4.」）。画面全体をラベルする必要はありません。

1. `labeled_region` の矩形を1つ描きます。画面の 1/4 程度で、敵が 5〜10 体入る場所を選びます。
2. **その矩形の内側にある** 敵・ジェム・ピックアップを漏れなくラベルします。矩形の外側はラベル不要です。
3. `Ctrl+Alt+K` で確認済みにします。

学習では `labeled_region` の内側だけを切り出して使うため、範囲外の未ラベルの物体が「背景」として誤って学習されることはありません。画面全体をラベル済みのフレームは、`labeled_region` なしでそのまま学習に使えます。

### 3-2. 下書き用検出器を学習する

```powershell
python Tools/Deployment/train_survivors_prelabel_detector.py `
  --work-root "D:\captures\annotation_work"
```

work-root 内の `checked: true` の JSON だけを使って学習し、重みを `<work-root>/prelabel_detector.pt` に保存します（`--output` で変更可）。確認済みフレームが1枚も無い場合はエラーで終了します。初回は pytorch.org から事前学習重み（約 74MB）を自動でダウンロードします。GPU があれば自動で使い、数分で終わります。

### 3-3. 下書きを作る

```powershell
python Tools/Deployment/prelabel_survivors_frames.py `
  --work-root "D:\captures\annotation_work" `
  --session-id "session-0001"
```

既定では `<work-root>/prelabel_detector.pt` を読みます（`--detector` で変更可）。重みファイルが無い場合は警告を表示し、固定矩形だけの下書きを作ります。

設定は [annotation_prelabel_v2.yaml](../../Tools/Deployment/configs/annotation_prelabel_v2.yaml) です。固定矩形・検出するラベル・スコアのしきい値・学習回数などを調整できます。設定は学習 CLI と下書き CLI で同じ検証を通り、ラベル名は共通クラス一覧で確認され、矩形は 1920x1080 の範囲内に置く必要があります。学習後に `input_scale` や `labels` を変えた場合は、重みと設定が一致しないためエラーになります。再学習してください。

下書きは `checked: false` で保存されます。既存 JSON はスキップされます。下書きに `labeled_region` は書かれません。

### 既存 work-root の classes.txt

`labeled_region` を追加する前に作った work-root では、`classes.txt` に `labeled_region` がありません。末尾に1行 `labeled_region` を追記してから X-AnyLabeling で読み込み直してください。

## 4. X-AnyLabeling で確認・修正する

1. X-AnyLabeling で `work-root/<session-id>` フォルダを開きます。
2. `Upload` → `Upload Label Classes File` を選び、`work-root/classes.txt` を読み込みます。
3. `R` で矩形を作成し、右側のラベル一覧からクラスを選びます。下書きの位置やラベルが違う場合は修正します。
4. `D` / `A` で次 / 前の画像へ移動します。
5. 確認が終わった画像で `Ctrl+Alt+K` を押して確認済みにします。`Ctrl+Shift+D` は次の未確認画像へ移動します。

自動保存は既定で ON です。COCO 出力に含める画像は `checked: true` にしてください。確認済みで矩形が 0 個の画像も負例として出力されます。

## 5. COCO JSON を出力する

確認済みラベルだけを出力します。`--session-id` を省略すると work-root 直下の全セッションを対象にします。

```powershell
python Tools/Deployment/export_survivors_annotations_coco.py `
  --work-root "D:\captures\annotation_work" `
  --output-dir "D:\datasets\survivors"
```

`world_coco.json` と `ui_coco.json` が作られます。

## クラス早見表

| クラス | 対象 |
|---|---|
| `player_anchor` | プレイヤーの位置 |
| `enemy_normal` / `enemy_elite` / `enemy_boss` | 通常 / エリート / ボス敵 |
| `gem_blue` / `gem_green` / `gem_red` | 青 / 緑 / 赤の経験値ジェム |
| `pickup_heal` / `pickup_special` | 回復 / 特殊ピックアップ |
| `hazard_projectile` / `hazard_area` | 敵が放つ弾 / 敵の範囲攻撃（プレイヤーの武器は含めない） |
| `hud_hp` / `hud_xp` | HP / XP の HUD 領域 |
| `card` / `button` | レベルアップカード / 選択ボタン |
| `death_result` | 死亡・結果画面の領域 |
| `labeled_region` | ラベルを付け終えた範囲（学習用。物体ではない） |

宝箱は `chest` ではなく `pickup_special` として付けます。`card`、`button`、`death_result`、`hazard_projectile`、`hazard_area` は自動下書きされないため、GUI で必要な矩形を追加してください。

`enemy_elite` は下書き用検出器が区別して学習するため、下書きでも `enemy_elite` として出ます。ただし確認済みのサンプルが少ないうちは `enemy_normal` と取り違えることがあるので、GUI で確認してください。`enemy_boss` は検出器の学習時に `enemy_normal` として扱われるため、下書きでは `enemy_normal` として出ます。GUI で正しいクラスへ直してください。

`hazard_projectile` / `hazard_area` は、プレイヤーに害を与える**敵側**の弾や範囲攻撃にだけ付けます。Garlic のオーラ、斧、Santa Water の炎、Peachone / Ebony Wings の照準など、**プレイヤー自身の武器エフェクトには付けません**（燭台などの壊せる置物も対象外です）。実機の観測では `hazard_*` が 1 つでも見えると `hazard_flag` が真になり、アイテム選択の判断材料として方策へ渡されます（`Tools/Deployment/survivors/real_obs_assembler.py`）。プレイヤーの周りに常にある Garlic を `hazard_area` にすると、この値がほぼ常に真になり意味を失います。

`labeled_region` を含む確認済みフレームは範囲外が未ラベルのため、COCO 出力から除外されます（出力時に「範囲限定でスキップした数」として表示されます）。COCO に含めたいフレームは画面全体をラベルし、`labeled_region` を消してください。

`hazard_projectile` / `hazard_area` は円（circle）ツールで描いても構いません。中心点と円周上の一点から外接する矩形へ自動変換されます。

タイプミスで未知のラベル名が付いた shape や、rectangle/circle 以外の未対応図形は、そのファイルを読む際に警告を表示して読み飛ばされます（ファイル全体は失敗しません）。警告は `prelabel_survivors_frames.py` や `export_survivors_annotations_coco.py` の実行時に標準エラー出力へ表示されるので、意図しない読み飛ばしがないか確認してください。

### ボスと元ボスの雑魚敵

`enemy_boss` は、ボスとして出現した個体（倒すと宝箱を落とす個体）にだけ付けます。ステージ序盤〜中盤にボスとして出た敵は、後半になると同じ見た目のまま雑魚敵として群れで出てきます。この元ボスの雑魚敵は `enemy_normal` として付けてください。

- ボスは通常の敵より大きく描かれることが多く、1 画面に普通 1 体です。同じ見た目の敵が何体もいる場合は雑魚敵と判断できます。
- 迷った場合は `enemy_normal` にします。

`enemy_boss` が 1 体でも見えると、実機の観測では `boss_flag` が真になり、アイテム選択の判断材料として方策へ渡されます（`Tools/Deployment/survivors/real_obs_assembler.py`）。元ボスの雑魚敵を `enemy_boss` で付けると、ステージ後半がずっとボス戦として扱われてしまいます。

下書き用検出器は `enemy_boss` を `enemy_normal` にまとめて学習するため、この区別は下書きの精度には影響しません。効くのは COCO 出力で学習する本番の検出器（[world_detector.md](world_detector.md)）です。

## 2周目以降の運用

下書きを直して確認済みにするほど、学習データが増えて検出器の精度が上がります。次の流れを繰り返します。

1. 下書きを GUI で修正し、確認済みにする
2. `train_survivors_prelabel_detector.py` で再学習する
3. `prelabel_survivors_frames.py --refresh-unchecked` で未確認の下書きを作り直す

既存 JSON は既定では上書きされないため、`--refresh-unchecked` を付けない場合は下書きが無い PNG にだけ新しい JSON が作られます。`--refresh-unchecked` を付けると `checked: false` の既存下書きだけが最新の検出器で再生成され、`checked: true` にしたファイルは変更されません。

```powershell
python Tools/Deployment/prelabel_survivors_frames.py `
  --work-root "D:\captures\annotation_work" `
  --session-id "session-0001" `
  --refresh-unchecked
```

未確認のまま GUI で手動編集を始めているフレームがある場合、`--refresh-unchecked` を実行すると下書きが編集前の状態に戻ってしまいます。手を付けたフレームは先に確認済みにしてから実行してください。

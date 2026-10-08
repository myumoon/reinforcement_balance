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

武器エフェクト4クラス（`weapon_projectile` / `weapon_zone` / `weapon_orbit` / `weapon_aura`）を検出するラベルに追加したため、追加前に学習した `prelabel_detector.pt` はラベル不一致のエラーで読み込めません。この CLI で**再学習**してください。Garlic の輪（`weapon_aura`）の下書き精度が出ない場合は、設定の `detector.labels` から `weapon_aura` だけ外して再学習します。

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

武器エフェクトクラスを追加する前に作った work-root では、`classes.txt` の `labeled_region` の後ろに次の4行を**この順で**追記し、X-AnyLabeling で読み込み直してください（新しく候補を抽出した work-root には最初から入っています）。

world class map v2 で武器エフェクトが world クラス（ID 12〜15）になったため、新しく作る `classes.txt` では4行が `hazard_area` の後ろ（UI クラスより前）に並びます。**既に末尾へ4行を追記した `classes.txt` はそのまま使えます。** JSON にはラベル名で保存されるので、並びの違いは X-AnyLabeling のラベル一覧の表示順が変わるだけで、移行作業は不要です。

`weapon_target` を追加する前に作った `classes.txt` には `weapon_target` がありません。末尾に1行 `weapon_target` を追記してから読み込み直してください（新しく候補を抽出すると最初から末尾に入ります）。

```text
weapon_projectile
weapon_zone
weapon_orbit
weapon_aura
```

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

武器エフェクト（`weapon_*`）は world class map v2 の world クラスなので、`world_coco.json` に category 12〜15（`weapon_projectile` = 12、`weapon_zone` = 13、`weapon_orbit` = 14、`weapon_aura` = 15）として出力されます。以前の「武器エフェクトで除外した矩形数」の表示と除外は無くなりました。

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
| `weapon_projectile` | プレイヤーの武器の弾・斬撃（1つごとに1矩形） |
| `weapon_zone` | プレイヤーの武器が地面に出す範囲（炎・爆発・落雷） |
| `weapon_orbit` | プレイヤーの周りを回る武器（King Bible の本。1冊ごとに1矩形） |
| `weapon_aura` | プレイヤーを中心とするオーラ（Garlic の輪。見えている輪を囲む1矩形） |

宝箱は `chest` ではなく `pickup_special` として付けます。`card`、`button`、`death_result`、`hazard_projectile`、`hazard_area` は自動下書きされないため、GUI で必要な矩形を追加してください。

### 武器エフェクトのクラス対応表

プレイヤー自身の武器エフェクトには `weapon_*` を付けます。クラスはシミュレーターが観測する4種類（Projectile / GroundZone / Orbit / Aura）に合わせています。`Tools/Deployment/survivors/annotation_labels.py` の `WEAPON_EFFECT_KINDS` は、Common の `deploy_obs_v2_features.yaml` の `weapon_effect_kinds` から作られます（表を二重に持ちません）。

| 武器（進化後） | ラベル |
|---|---|
| Whip（Bloody Tear）、Magic Wand（Holy Wand）、Knife（Thousand Edge）、Axe（Death Spiral）、Cross（Heaven Sword）、Runetracer（NO FUTURE） | `weapon_projectile`（弾・斬撃1つごとに1矩形） |
| Fire Wand（Hellfire） | 火球は `weapon_projectile`、着弾後の爆発範囲は `weapon_zone` |
| Santa Water（La Borra） | `weapon_zone`（炎の範囲） |
| Lightning Ring（Thunder Loop） | `weapon_zone`（落雷の範囲） |
| King Bible（Unholy Vespers） | `weapon_orbit`（1冊ごとに1矩形） |
| Garlic（Soul Eater） | `weapon_aura`（見えている輪を囲む1矩形） |
| Peachone、Ebony Wings、Vandalier | 着弾の爆発（小さな光）は `weapon_projectile`、照準の大きな円は `weapon_target`。**鳥本体には付けない** |

次のものには何も付けません。

- Pentagram（Gorgeous Moon）の画面フラッシュ、Laurel の盾（物体として観測されないため）
- 燭台などの壊せる置物（武器ではなく、`hazard_*` でもない）
- Peachone / Ebony Wings / Vandalier の鳥本体

照準の円は `weapon_projectile` ではなく、**アノテーション専用クラス `weapon_target`** を付けます。シミュレーターは照準の円の中に小さな着弾を 0.1 秒ずつ出すだけで、円そのものを観測しません。円を `weapon_projectile` にすると、実機の projectile 密度だけが円の表示中ずっと高くなり、教師と食い違います。一方で照準の位置は移動の判断材料になりうるため、観測に入れるかを後で決められるよう、データだけ `weapon_target` として集めます。`weapon_target` は world class map・下書き検出器・COCO 出力のどれにも入りません（COCO 出力では「アノテーション専用で除外した矩形数」として数えられます）。下書きには出ないので、手で付けてください。

矩形は `R`（または円ツール）で、見えているエフェクトの外形をぴったり囲みます。複数の弾が重なっていても、見分けられる限り1つずつ付けます。

`enemy_elite` は下書き用検出器が区別して学習するため、下書きでも `enemy_elite` として出ます。ただし確認済みのサンプルが少ないうちは `enemy_normal` と取り違えることがあるので、GUI で確認してください。`enemy_boss` は検出器の学習時に `enemy_normal` として扱われるため、下書きでは `enemy_normal` として出ます。GUI で正しいクラスへ直してください。

`hazard_projectile` / `hazard_area` は、プレイヤーに害を与える**敵側**の弾や範囲攻撃にだけ付けます。Garlic のオーラ、斧、Santa Water の炎、Peachone / Ebony Wings の照準など、**プレイヤー自身の武器エフェクトには付けず、上の対応表に従って `weapon_*` を付けるか、何も付けません**（燭台などの壊せる置物はどちらの対象でもありません）。実機の観測では `hazard_*` が 1 つでも見えると `hazard_flag` が真になり、アイテム選択の判断材料として方策へ渡されます（`Tools/Deployment/survivors/real_obs_assembler.py`）。プレイヤーの周りに常にある Garlic を `hazard_area` にすると、この値がほぼ常に真になり意味を失います。

`labeled_region` を含む確認済みフレームは範囲外が未ラベルのため、COCO 出力から除外されます（出力時に「範囲限定でスキップした数」として表示されます）。COCO に含めたいフレームは画面全体をラベルし、`labeled_region` を消してください。

`hazard_projectile` / `hazard_area` / `weapon_*` は円（circle）ツールで描いても構いません。中心点と円周上の一点から外接する矩形へ自動変換されます。

タイプミスで未知のラベル名が付いた shape や、rectangle/circle 以外の未対応図形は、そのファイルを読む際に警告を表示して読み飛ばされます（ファイル全体は失敗しません）。警告は `prelabel_survivors_frames.py` や `export_survivors_annotations_coco.py` の実行時に標準エラー出力へ表示されるので、意図しない読み飛ばしがないか確認してください。

### ボスと元ボスの雑魚敵

`enemy_boss` は、ボスとして出現した個体（倒すと宝箱を落とす個体）にだけ付けます。ステージ序盤〜中盤にボスとして出た敵は、後半になると同じ見た目のまま雑魚敵として群れで出てきます。この元ボスの雑魚敵は `enemy_normal` として付けてください。

- ボスは通常の敵より大きく描かれることが多く、1 画面に普通 1 体です。同じ見た目の敵が何体もいる場合は雑魚敵と判断できます。
- 迷った場合は `enemy_normal` にします。

`enemy_boss` が 1 体でも見えると、実機の観測では `boss_flag` が真になり、アイテム選択の判断材料として方策へ渡されます（`Tools/Deployment/survivors/real_obs_assembler.py`）。元ボスの雑魚敵を `enemy_boss` で付けると、ステージ後半がずっとボス戦として扱われてしまいます。

下書き用検出器は `enemy_boss` を `enemy_normal` にまとめて学習するため、この区別は下書きの精度には影響しません。効くのは COCO 出力で学習する本番の検出器（[world_detector.md](world_detector.md)）です。

## 付け替え CLI（propose → 確認 → apply）

武器エフェクト用クラスを追加する前は、武器エフェクトに `hazard_projectile` / `hazard_area` を付けていた確認済みフレームがあります。`relabel_survivors_annotations.py` で `weapon_*` へ付け替えます。確認済みデータを書き換えるので、**対応表をユーザーが確認してから** apply します。

1. **propose（データは書き換えない）**

   ```powershell
   python Tools/Deployment/relabel_survivors_annotations.py propose `
     --work-root "D:\captures\annotation_work" `
     --output "D:\captures\relabel_mapping.json" `
     --sheet "D:\captures\relabel_sheet.png"
   ```

   旧ラベル（既定 `hazard_projectile` と `hazard_area`。`--old-labels` で変更可）の shape を全 session から集め、対応表 JSON と番号付きの一覧画像を作ります。一覧画像の番号は対応表の `id` です。既存の対応表は上書きしません。提案の規則は次のとおりです。

   - `hazard_projectile` → `weapon_projectile`
   - `hazard_area` で、矩形の中心が `player_anchor` の中心（JSON に無ければ設定の固定矩形の中心）から `--aura-radius-px`（既定 40）以内 → `weapon_aura`
   - それ以外 → `review`（人が決める）

   距離だけで判定するため、プレイヤー付近の Santa Water の炎を `weapon_aura` と誤提案することがあります。

2. **対応表を確認・修正する**

   一覧画像を見ながら、対応表の各行の `proposed_label` を決めます。使える値は共通クラス名（`weapon_zone` など）、`delete`（shape を削除）、旧ラベルと同じ名前（変更なし）です。`review` を1行でも残すと apply は失敗します。`file`・`shape_index`・`old_label`・`bbox` は変えないでください。

3. **apply**

   ```powershell
   python Tools/Deployment/relabel_survivors_annotations.py apply `
     --work-root "D:\captures\annotation_work" `
     --mapping "D:\captures\relabel_mapping.json"
   ```

   書き込む前に全行を検証し、`review` が残っている行、未知のラベル、propose の後で旧ラベルや矩形が変わった shape が1つでもあれば、**何も書かずに終了コード 1** で止まります。検証が通ると、書き換える JSON を丸ごと `<work-root>/_relabel_backup/<YYYYmmdd-HHMMSS>/`（`--backup-dir` で変更可）へコピーしてから、該当 shape のラベル変更・削除だけを行います。`checked`、円（circle）の形、`flags` などその他の内容はそのまま残ります。元に戻すときはバックアップのファイルを同じ相対パスへコピーし直します。

## 確認済みフレームの見直し手順

確認済み（`checked: true`）フレームは「対象クラスがすべてラベル済み」という意味です。武器エフェクト用クラスを追加する前に確認したフレームには、`hazard_*` で付けていたもの以外の武器エフェクト（鞭・ナイフ・Magic Wand の弾など）が未ラベルのまま残っています。このまま学習すると、未ラベルのエフェクトが背景として学習されてしまいます。付け替え（apply）と同時に、次の手順で見直してください。

1. `classes.txt` に `weapon_*` の4行があることを確認する（[既存 work-root の classes.txt](#既存-work-root-の-classestxt)）。
2. 上の付け替え CLI で `hazard_*` を `weapon_*` へ付け替える。
3. 既存の確認済みフレームをすべて X-AnyLabeling で開き直し、上の対応表に従って未ラベルの武器エフェクトを漏れなく追加する。`labeled_region` 付きのフレームは範囲の内側だけで構いません。
4. 追加し終えたら確認済み（`Ctrl+Alt+K`）に戻す。
5. 下書き用検出器を再学習する（[3-2](#3-2-下書き用検出器を学習する)）。

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

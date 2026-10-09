# Survivors 実機キャプチャの録画とアノテーション手順

本家 Vampire Survivors の画面を録画し、X-AnyLabeling でアノテーションして COCO データセットを作るまでの一連の手順をまとめたオペレーター向けマニュアルです。各ステップの実装詳細は [`capture_dataset.md`](capture_dataset.md)（録画）と [`annotation_gui.md`](annotation_gui.md)（アノテーション）を参照してください。

```
録画 → 候補フレーム抽出 → 自動下書き → GUIで確認・修正 → COCO JSON 出力
```

このドキュメントで `<ProjectDir>` はリポジトリのルート、`<RecordRoot>` は録画データの保存先、`<SessionId>` は録画のセッション ID、`<WorkRoot>` はアノテーション作業フォルダ、`<DatasetDir>` は COCO 出力先を表します。実際のパスに置き換えて使ってください。

## 0. 前提

- 録画対象の実機 target profile（対象ウィンドウ、解像度、build など）が準備済みであること。詳細は [`target_profile.md`](target_profile.md) を参照してください。
- `<ProjectDir>/Tools/Deployment` を Python の実行ディレクトリの基準にし、`reinbalance` conda 環境が有効になっていること。

## 1. 録画する

対象ウィンドウ（1920×1080 borderless）を前面に表示した状態で実行します。

```powershell
python Tools/Deployment/capture_survivors.py `
  --store-root "<RecordRoot>" `
  --session-id auto --duration-sec 1800
```

- `--session-id auto` を指定すると `<RecordRoot>/capture_sessions/` 配下の既存セッション番号の次（`session-0001` から）が自動採番されます。毎回 ID を手入力する必要はありません。
- 起動直後は対象ウィンドウが前面に来るまで自動でポーリング待機するため、事前の待ち時間は不要です。
- 収録中に alt-tab などで対象ウィンドウが一時的に前面を失っても収録は中断されません。自動で一時停止し、ウィンドウが前面へ戻ると自動で再開します。一時停止・再開のたびに `capture paused: ...` / `capture resumed: ...` がコンソールに表示されます（解像度変更やプロセス差し替えなど、alt-tab 以外の状態変化は即座に終了します）。
- **Ctrl+C は 1 回目と 2 回目で挙動が違います。**
  - 1 回目: 収録を安全に打ち切り、撮れたフレームがあればそのまま公開します。1 枚も撮れていなければ何も公開しません。
  - 2 回目: 直ちに中断し、未公開の一時データを破棄します（全て破棄したい場合のみ使用）。
- 録画データは `<RecordRoot>/capture_sessions/<SessionId>/` に保存されます。このフォルダは Git 管理対象外です。

複数セッション分の録画が必要な場合は、`--session-id auto` で同じコマンドを繰り返し実行してください。

## 2. アノテーションする

### 2-1. X-AnyLabeling を用意する

[X-AnyLabeling v4.0.6 の GitHub Releases](https://github.com/CVHub520/X-AnyLabeling/releases/tag/v4.0.6) から Windows 用 exe を取得します。

- 通常は `X-AnyLabeling-v4.0.6-CPU.exe` を使います。
- 対応する CUDA 環境がある場合は `X-AnyLabeling-v4.0.6-Windows-CUDA12.exe` でも構いません。

### 2-2. 候補フレームを抽出する（セッションごとに実行）

録画済みの各セッションから、アノテーション対象にするフレームを選んで作業フォルダへ配置します。

```powershell
python Tools/Deployment/select_survivors_annotation_candidates.py `
  --store-root "<RecordRoot>" `
  --session-id "<SessionId>"
```

- `--work-root` を省略すると `<RecordRoot>/annotation_work` が使われます。
- 既定で foreground フレームの重複除去 + stride 15 間引き + 多様性確保の k-means により、最大 200 枚を自動選定してハードリンク配置し、`classes.txt` を出力します。
- 必要なら `--work-root "<WorkRoot>"`、`--count 200`、`--stride 15` を指定して調整できます。
- 録画した全セッションに対して同様に実行してください。

### 2-3. 自動下書きを作る

固定矩形（player/HUD）と、確認済みラベルで学習した下書き用検出器（Faster R-CNN）の推論結果を使って下書き JSON を生成します。

**(1) 最初の周回: 学習用の確認済みフレームを用意する**

時期の違うフレームを数枚選び、X-AnyLabeling（2-4.）で次のようにラベルして確認済みにします。全部をラベルする必要はありません。

- `labeled_region` の矩形を1つ描きます（画面の 1/4 程度、敵が 5〜10 体入る場所）。
- **その矩形の内側にある** 敵・ジェム・ピックアップを漏れなくラベルします。矩形の外側はラベル不要です。
- 画面全体をラベル済みのフレームは、`labeled_region` なしでそのまま学習に使えます。

**(2) 検出器を学習する**

```powershell
python Tools/Deployment/train_survivors_prelabel_detector.py `
  --work-root "<WorkRoot>"
```

- work-root 内の `checked: true` の JSON だけを使い、重みを `<WorkRoot>/prelabel_detector.pt` に保存します（`--output` で変更可）。確認済みフレームが無い場合はエラーで終了します。
- 初回は pytorch.org から事前学習重み（約 74MB）を自動でダウンロードします。GPU があれば自動で使い、数分で終わります。

**(3) 下書きを作る**

```powershell
python Tools/Deployment/prelabel_survivors_frames.py `
  --work-root "<WorkRoot>" `
  --session-id "<SessionId>"
```

- 既定で `<WorkRoot>/prelabel_detector.pt` を読みます（`--detector` で変更可）。重みファイルが無い場合は警告を表示し、固定矩形だけの下書きを作ります。
- 下書きは `checked: false` で保存され、JSON がまだ無い画像にのみ作成されます（既存 JSON は上書きされません）。下書きに `labeled_region` は書かれません。
- 設定は [annotation_prelabel_v2.yaml](../../Tools/Deployment/configs/annotation_prelabel_v2.yaml) です。学習後に `input_scale` や `labels` を変えた場合は重みと一致しないためエラーになるので、再学習してください。
- 既存 work-root の `classes.txt` に `labeled_region` が無い場合は、末尾に1行 `labeled_region` を追記してから X-AnyLabeling で読み込み直してください。
- 既存 work-root の `classes.txt` に武器エフェクト4クラスが無い場合は、`labeled_region` の後ろに `weapon_projectile`、`weapon_zone`、`weapon_orbit`、`weapon_aura` の4行をこの順で追記してから読み込み直してください。既に末尾へ4行を追記した `classes.txt` はそのまま使えます（新しく作る `classes.txt` では world class map v2 に合わせて `hazard_area` の後ろに並びますが、JSON はラベル名で保存されるため、違いは X-AnyLabeling の表示順だけです）。
- 既存 work-root の `classes.txt` に `weapon_target`（照準の円、アノテーション専用）が無い場合は、末尾に1行 `weapon_target` を追記してから読み込み直してください。
- 検出するラベルに `weapon_*` の4クラスを追加したため、追加前に学習した `prelabel_detector.pt` はラベル不一致のエラーになります。(2) で再学習してください。
- 武器エフェクト用クラスを追加する前に `hazard_*` で付けた武器エフェクトは、`relabel_survivors_annotations.py`（propose → 対応表の確認 → apply）で `weapon_*` へ付け替え、既存の確認済みフレームを開き直して未ラベルの武器エフェクトを追加します。手順は [`annotation_gui.md`](annotation_gui.md) の「付け替え CLI」「確認済みフレームの見直し手順」を参照してください。

**(4) 反復する**

下書きを修正して確認済みにする → (2) で再学習する → `--refresh-unchecked` を付けて下書きを作り直す、を繰り返すと検出器の精度が上がります。`--refresh-unchecked` では `checked: false` の既存下書きだけが最新の検出器で再生成され、`checked: true` にしたファイルは変更されません。

```powershell
python Tools/Deployment/prelabel_survivors_frames.py `
  --work-root "<WorkRoot>" `
  --session-id "<SessionId>" `
  --refresh-unchecked
```

未確認のまま GUI で手動編集を始めているフレームがある場合、`--refresh-unchecked` を実行すると下書きが編集前の状態に戻ってしまいます。手を付けたフレームは先に確認済みにしてから実行してください。

### 2-4. X-AnyLabeling で確認・修正する

1. `<WorkRoot>/<SessionId>` フォルダを X-AnyLabeling で開きます。
2. `Upload` → `Upload Label Classes File` を選び、`<WorkRoot>/classes.txt` を読み込みます。
3. `R` で矩形を作成し、右側のラベル一覧からクラスを選びます。下書きの位置やラベルがずれていれば修正します。
4. `D` / `A` で次 / 前の画像へ移動します。
5. 確認が終わった画像で `Ctrl+Alt+K` を押して確認済みにします（`Ctrl+Shift+D` で次の未確認画像へ移動）。

自動保存は既定で ON です。COCO 出力に含める画像は `checked: true` にしてください（矩形が 0 個の確認済み画像も負例として出力されます）。

### 2-5. クラス早見表

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

宝箱は `chest` ではなく `pickup_special` として付けます。

- `card` / `button` / `death_result` / `hazard_projectile` / `hazard_area` は自動下書きされないため、GUI で手動追加が必要です。
- `enemy_elite` は下書きでも `enemy_elite` として出ますが、サンプルが少ないうちは `enemy_normal` と取り違えることがあるので確認してください。`enemy_boss` は下書きでは `enemy_normal` として出るので、GUI で正しいクラスへ直してください。
- `hazard_projectile` / `hazard_area` は**敵側**の弾や範囲攻撃にだけ付けます。Garlic・斧・Santa Water・Peachone の照準など、プレイヤー自身の武器エフェクトには付けず、`weapon_*` を付けるか何も付けません（理由は [`annotation_gui.md`](annotation_gui.md) を参照）。
- 武器エフェクトは次の対応表どおりに付けます（詳細は [`annotation_gui.md`](annotation_gui.md) の「武器エフェクトのクラス対応表」）。
  - Whip・Magic Wand・Knife・Axe・Cross・Runetracer（進化後も同じ）の弾・斬撃 → `weapon_projectile`（1つごとに1矩形）
  - Fire Wand（Hellfire）の火球 → `weapon_projectile`、着弾後の爆発範囲 → `weapon_zone`
  - Santa Water（La Borra）の炎、Lightning Ring（Thunder Loop）の落雷 → `weapon_zone`
  - King Bible（Unholy Vespers）の本 → `weapon_orbit`（1冊ごとに1矩形）
  - Garlic（Soul Eater）の輪 → `weapon_aura`（見えている輪を囲む1矩形）
  - Peachone・Ebony Wings・Vandalier の着弾の爆発 → `weapon_projectile`、照準の大きな円 → `weapon_target`（アノテーション専用で COCO には出ないが、下書きには出る。鳥本体には付けない）
  - Pentagram（Gorgeous Moon）の画面フラッシュ、Laurel の盾、燭台などの壊せる置物 → 何も付けない
- `weapon_*` は world class map v1 に無いため、当面は COCO 出力から矩形だけ除外されます（フレームと他の矩形は出力されます）。
- `enemy_boss` はボスとして出現した個体（倒すと宝箱を落とす個体）にだけ付けます。ステージ後半に、序盤〜中盤のボスと同じ見た目の敵が雑魚敵として群れで出てきた場合は `enemy_normal` です。迷ったら `enemy_normal` にしてください（理由は [`annotation_gui.md`](annotation_gui.md) の「ボスと元ボスの雑魚敵」を参照）。
- `labeled_region` を含む確認済みフレームは範囲外が未ラベルのため、COCO 出力から除外されます。COCO に含めたいフレームは画面全体をラベルし、`labeled_region` を消してください。

### 2-6. COCO JSON を出力する

全セッションの確認が終わったら（またはセッション単位で都度）実行します。

```powershell
python Tools/Deployment/export_survivors_annotations_coco.py `
  --work-root "<WorkRoot>" `
  --output-dir "<DatasetDir>"
```

`checked: true` の画像だけが出力対象になり、`<DatasetDir>/world_coco.json` と `<DatasetDir>/ui_coco.json` が生成されます。`--session-id` を省略すると work-root 直下の全セッションが対象です。

武器エフェクト（`weapon_*`）は world class map v2 の world クラスとして、`world_coco.json` の category 12〜15（`weapon_projectile` / `weapon_zone` / `weapon_orbit` / `weapon_aura`）に出力されます（以前のような除外はありません）。

## 参考

- 録画の内部実装・アーキテクチャ・テスト: [`capture_dataset.md`](capture_dataset.md)、[`capture_core.md`](capture_core.md)
- アノテーションの下書き設定・2 周目以降の運用: [`annotation_gui.md`](annotation_gui.md)
- target profile の準備: [`target_profile.md`](target_profile.md)

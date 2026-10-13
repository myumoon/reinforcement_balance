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
  --session-id auto --duration-sec 2700 --planned-split model_train
```

- 標準の30分 run はステージ選択から死亡、または30:00到達後の result 画面までを1 session に収録します。収録時間は2700秒以上にし、用途は開始前に `--planned-split` で登録します（詳細は「3-1. session を収録し split を事前登録する」を参照）。
- `--session-id auto` を指定すると `<RecordRoot>/capture_sessions/` 配下の既存セッション番号の次（`session-0001` から）が自動採番されます。毎回 ID を手入力する必要はありません。
- 起動直後は対象ウィンドウが前面に来るまで自動でポーリング待機するため、事前の待ち時間は不要です。
- 収録中に alt-tab などで対象ウィンドウが一時的に前面を失っても収録は中断されません。自動で一時停止し、ウィンドウが前面へ戻ると自動で再開します。一時停止・再開のたびに `capture paused: ...` / `capture resumed: ...` がコンソールに表示されます（解像度変更やプロセス差し替えなど、alt-tab 以外の状態変化は即座に終了します）。
- **Ctrl+C は 1 回目と 2 回目で挙動が違います。**
  - 1 回目: 収録を安全に打ち切り、撮れたフレームがあればそのまま公開します。1 枚も撮れていなければ何も公開しません。
  - 2 回目: 直ちに中断し、未公開の一時データを破棄します（全て破棄したい場合のみ使用）。
- 録画データは `<RecordRoot>/capture_sessions/<SessionId>/` に保存されます。このフォルダは Git 管理対象外です。
- live capture が成功すると、開始時の `.env/target_profile.resolved.yaml` をそのまま `<RecordRoot>/session_logs/<SessionId>/target_profile.resolved.yaml` に退避し、operator、実測 session 長、run 終了状態、profile 情報を含む `session_stats.json` と `session_log.md` を自動生成します。収録データの hash 検証に影響しないよう、これらは `capture_sessions/` の外に保存されます。session log に手で書く欄はありません。
- session log の保存に失敗した場合も公開済み capture は残ります。コマンドは終了コード 1 と `session_log_error` を返すので、その session の log 保存先を確認してください。synthetic と dry-run は session log を作りません。

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

候補抽出をまとめて行う場合は `Tools\Utility\SurvivorsAnnotation\SelectCandidates.bat` をダブルクリックできます。既定では capture_sessions にあるうち annotation_work にまだ配置されていない session だけを処理します。設定ファイル、WorkRoot、session ID、枚数、stride は引数で指定できます。

```cmd
Tools\Utility\SurvivorsAnnotation\SelectCandidates.bat -EnvFile "<EnvFile>" -WorkRoot "<WorkRoot>" -SessionId "session-0001,session-0002" -Count 200 -Stride 15
```

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

bat からは `Tools\Utility\SurvivorsAnnotation\TrainPrelabelDetector.bat` を実行できます。既定の WorkRoot は `.env/survivors_deployment.env` の `SURVIVORS_ARTIFACT_PRIMARY_ROOT\annotation_work` です。別の設定ファイルや WorkRoot を使う場合は引数で指定します。

```cmd
Tools\Utility\SurvivorsAnnotation\TrainPrelabelDetector.bat -EnvFile "<EnvFile>" -WorkRoot "<WorkRoot>"
```

- work-root 内の `checked: true` の JSON だけを使い、重みを `<WorkRoot>/prelabel_detector.pt` に保存します（`--output` で変更可）。確認済みフレームが無い場合はエラーで終了します。
- 初回は pytorch.org から事前学習重み（約 74MB）を自動でダウンロードします。GPU があれば自動で使い、数分で終わります。

**(3) 下書きを作る**

```powershell
python Tools/Deployment/prelabel_survivors_frames.py `
  --work-root "<WorkRoot>" `
  --session-id "<SessionId>"
```

WorkRoot 直下の `session-*` をすべて処理する場合は `Tools\Utility\SurvivorsAnnotation\PrelabelAllSessions.bat` を使えます。`_relabel` などのディレクトリは対象外です。既定では JSON がまだ無い画像にだけ下書きを作る安全版です。WorkRoot と env ファイルも必要に応じて指定できます。

```cmd
Tools\Utility\SurvivorsAnnotation\PrelabelAllSessions.bat -EnvFile "<EnvFile>" -WorkRoot "<WorkRoot>"
```

未確認の下書きを更新するときは、通常版とは別の `Tools\Utility\SurvivorsAnnotation\PrelabelAllSessions_Refresh.bat` を使います。引数なしでダブルクリックでき、別の設定ファイルや WorkRoot は引数で指定できます。

```cmd
Tools\Utility\SurvivorsAnnotation\PrelabelAllSessions_Refresh.bat -EnvFile "<EnvFile>" -WorkRoot "<WorkRoot>"
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

GUI で手を付けた未確認フレームは、Refresh 版を実行すると編集前の下書きに戻るため、手動編集したフレームを先に確認済みにしてください。Refresh 版を通常版と分けているのは、この上書きを意図した場合だけ選べるようにするためです。

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
  - 画面左上のアイテム欄の中（アイコンが並ぶ範囲）には何も付けない。アイコンにも、アイコンの後ろを通る本物の敵・ジェムにも付けない（アイコンと見分けにくいため。下書きにも出ない）
  - Pentagram（Gorgeous Moon）の画面フラッシュ、Laurel の盾、燭台などの壊せる置物 → 何も付けない
- `enemy_boss` はボスとして出現した個体（倒すと宝箱を落とす個体）にだけ付けます。ステージ後半に、序盤〜中盤のボスと同じ見た目の敵が雑魚敵として群れで出てきた場合は `enemy_normal` です。迷ったら `enemy_normal` にしてください（理由は [`annotation_gui.md`](annotation_gui.md) の「ボスと元ボスの雑魚敵」を参照）。
- `labeled_region` を含む確認済みフレームは範囲外が未ラベルのため、COCO 出力から除外されます。COCO に含めたいフレームは画面全体をラベルし、`labeled_region` を消してください。

### 2-6. HUD の正解値を付ける

bbox とは別に、HUD の正解値（画面状態、タイマー、レベル、HP・XP の割合、所持アイテム、アイテムごとのレベル、カード）を frame ごとに確定し、セッションの `hud_truth.jsonl` に保存します。HUD を読む parser の精度を測るときの正解に使います。

- 書き込むのは `<RecordRoot>/capture_sessions/<session-id>/hud_truth.jsonl` だけです。`annotations.jsonl`、X-AnyLabeling の label JSON、COCO、frame PNG、session manifest は読むだけで変更しないので、既存の bbox アノテーションには影響しません。
- 画像は開きません。表示される `png:` の path を X-AnyLabeling や画像ビューアの別窓で見ながら作業します。

```powershell
python Tools/Deployment/annotate_survivors_hud_truth.py `
  --store-root "<RecordRoot>" `
  --session-id session-0004 `
  --annotator-id <自分の ID> `
  --work-root "<WorkRoot>" `
  --from-labels
```

- 対象 frame は、`--work-root` にある `checked: true` の label の frame（COCO 出力と同じ集合）です。`--frame-ids 7193 7194` のように直接指定もできます。どちらも無いとエラーになります。
- `--from-labels`: label に `card` の矩形がある frame は、画面状態の下書きを `level_up_items` にします。`death_result` の矩形がある frame は parser の `death`／`result` 判定を残し（それ以外の判定なら `death`）、どちらなのかは人が `set state death|result` で確定してください。`--work-root` が必要です。
- `--atlas <development atlas>`: アイコン照合に使う atlas。省略するとアイテムの下書きは入りません（`items: null`）。
- `--resume`: 前回の `hud_truth.jsonl` を読み、確定済みの frame を飛ばして続きから始めます。`hud_truth.jsonl` があるのに `--resume` を付けないとエラーになります（上書き防止）。
- 保存は `quit` のとき、20 frame 進むごと、入力の終わり（Ctrl+Z / Ctrl+D）やエラーで止まったときに行われます。

#### 画面状態は必ず目視で確認する

各 frame の先頭に「state は目視で確認」の注意が出ます。04-20／04-22 反映後は、実測した枠と色で level-up・chest・death・result・paused を判定し、card surface の atlas があれば `expected_choice` の下書きも入ります。それでも `expected_screen_state` とカード名は必ず画像で確かめてください。過渡などで状態が違う場合は、`--from-labels` または `set state <state>` で直してください。

カードは上から順に記録します。一枚でも item_id が読めなければ choice 全体が null になるので、`set choice id1|id2|id3`（枚数に合わせる）で全カードを入れてください。所持欄の inventory template とカードの card template は別 entry です。atlas なしではカード枚数と位置だけが読め、名前の下書きは入りません。

次の録画では宝箱で一度、何も押さずに待って自動で開くか確認してください。parser は契約上「開く」を ack_chest にせず、終了ボタンだけを返します。手動で開く必要がある場合、現 controller は60秒で停止します。

#### コマンド早見表

slot 番号は 0〜11 です（0〜5 が武器、6〜11 がパッシブ。画面左上のパネルの左から順）。値の `-` と `null` は「読めない」を表します。

| コマンド | 意味 |
|---|---|
| `ok` | 表示中の値で確定して次の frame へ |
| `ok-range A B` | 表示中の frame から B までを、直前に確定した値を引き継いで一括確定する。各 frame の値（表示中の frame は直した後の値、後の frame は `--resume` で読んだ未確定行か下書き）で入っている項目が引き継ぎ値と違えば、違う項目を一覧で出してそこで止まり、その frame の値は上書きしない（タイマーが 1 秒以上進む範囲では止まるので、その frame は 1 つずつ確定する） |
| `skip` | 確定せずに次の frame へ |
| `undo` | 直前の操作を取り消す（前の frame の確定も戻せる） |
| `quit` | 保存して終了 |
| `set state <state>` | 画面状態（`gameplay`、`level_up_items`、`level_up_fallback`、`chest`、`paused`、`target_reached_transition`、`death`、`result`、`unknown`）。HUD の ROI は自動で決まり、段階マークが出ない画面にするとアイテムのレベルは null に戻る |
| `set timer 123.0` | タイマー（秒）。`set timer null` で不明 |
| `set level 7` | プレイヤーのレベル |
| `set hp 0.85`／`set xp 0.2` | HP・XP バーの割合（0〜1） |
| `set item SLOT <item_id>`／`set item SLOT empty_slot` | 1 slot のアイテム。空き枠は `empty_slot` |
| `set items null` | アイテムを読めない frame（1 slot でも読めないとき）。アイテムのレベルも null になる |
| `set choice a\|b\|c`／`set choice null` | 画面のカードの item_id（上から `\|` 区切り） |
| `set levels v1 ... v12` | 12 slot 分のアイテムのレベル。空き枠は `-` |
| `set level-of SLOT N` | 1 slot だけレベルを直す（先に `set levels` で全体を入れておく） |
| `set levels null` | アイテムのレベルを読めない frame |

確定するときは、アイテムを 12 slot すべて入れるか `set items null` にします。1 slot でも読めないときに一部だけ入れて確定することはできません。規則に合わないコマンドは値を変えずに理由を表示します。

#### 段階マーク（アイテムのレベル）の読み方

level-up 画面の左上パネルでは、武器（上段 6 枠）とパッシブ（下段 6 枠）のアイコンの下に、小さな四角が 1 行 3 個まで並びます。

```
 [アイコン]      [アイコン]     [アイコン]
  ■■■            ■■□            ■□
  ■□□            □□
  □□
 武器 Lv4（最大 8） パッシブ Lv2（最大 5）  指輪 Lv1（最大 2）
 ■ = 点灯（金色）  □ = 消灯（暗色）
```

- 点灯している四角の数がレベルです。点灯と消灯を合わせた数は最大レベルで、入力には使いません。
- 段階マークが出るのは level-up 画面（`level_up_items`／`level_up_fallback`）だけです。gameplay や宝箱の画面ではアイコンしか出ないので、`set levels` は受け付けません。
- 空き枠は `-` にします。1 枠でも読めない frame は `set levels` を入れずに null のままにします。
- パネルが滑り込んでくる途中の frame でも、人が読めるなら付けてかまいません。parser がそうした frame で読めない（None）と返しても、それは「未読」であって誤りとは扱いません。

例: 武器 2 個（Lv3、Lv1）とパッシブ 1 個（Lv2）を持っているとき

```
set item 0 whip
set item 1 magic_wand
...（残りの slot も empty_slot まで入れる）
set levels 3 1 - - - - 2 - - - - -
```

#### 注意点

- 対象 frame から、段階マーク格子の証拠が無い frame まで遡り、さらに十枚前から連続して parser に通します。前の target と窓が連続しない場合は `reset_temporal_state()` で状態を消してから始めます。長い level-up の後半では数百枚を読み、下書き一枚に数十秒かかることがあります。
- level-up の下書きには、段階値の信頼度が .5 以上ならアイテムのレベルが入ります。`expected_items` が null の場合はレベルも null のままです。点灯数は選択前の値なので、目視で確認してください。
- level-up の `expected_items` は、訪問直前の gameplay 在庫を slot ごとに三枚一致で保存し、位置で結び付けた値です。訪問前に gameplay が三枚無い session 先頭や、identity が読めない枠がある場合は全体が null になりえます。パネルの空枠と種別が食い違う枠は結合しません。
- 正式 atlas が無い場合や `--atlas` を指定しない場合、所持 identity は読めません。段階マーク自体が読めても、items と levels の下書きは null になる場合があります。

### 2-7. COCO JSON を出力する

COCO JSON は、画像ごとの物体の位置や種類をまとめる、物体検出で広く使われる形式です。確認済み annotation をモデルの学習・評価へ渡すために使います。1 回の export で world 用の `world_coco.json` と HUD 用の `ui_coco.json` が出力されます。HUD truth を確定した後に実行し、`expected_screen_state` は `hud_parser.SCREEN_STATES` の語彙で `death` と `result` を分けて記録してください。

```powershell
python Tools/Deployment/export_survivors_annotations_coco.py `
  --work-root "<WorkRoot>" --store-root "<RecordRoot>" `
  --session-id "<SessionId>" --output-dir "<RecordRoot>/coco/<SessionId>"
```

`checked: true` の画像だけが対象です。model 用 dataset は `model_train` と `model_validation` を別々に export して、混ざらないよう別フォルダーに保存します。

WorkRoot 直下の `session-*` をまとめて出力する場合は `Tools\Utility\SurvivorsAnnotation\ExportCoco.bat` をダブルクリックできます。設定ファイル、WorkRoot、session ID は引数で指定できます。各 session の出力先は `<RecordRoot>/coco/<SessionId>` です。

```cmd
Tools\Utility\SurvivorsAnnotation\ExportCoco.bat -EnvFile "<EnvFile>" -WorkRoot "<WorkRoot>" -SessionId "session-0001,session-0002"
```

```powershell
python Tools/Deployment/export_survivors_annotations_coco.py `
  --work-root "<WorkRoot>" --store-root "<RecordRoot>" `
  --split-manifest "<RecordRoot>/capture_split_manifest.<YYYYMMDD>.json" `
  --split model_train --output-dir "<DatasetDir>/model_train"
python Tools/Deployment/export_survivors_annotations_coco.py `
  --work-root "<WorkRoot>" --store-root "<RecordRoot>" `
  --split-manifest "<RecordRoot>/capture_split_manifest.<YYYYMMDD>.json" `
  --split model_validation --output-dir "<DatasetDir>/model_validation"
python Tools/Deployment/merge_survivors_coco.py `
  --world-train "<DatasetDir>/model_train/world_coco.json" `
  --world-validation "<DatasetDir>/model_validation/world_coco.json" `
  --ui-train "<DatasetDir>/model_train/ui_coco.json" `
  --ui-validation "<DatasetDir>/model_validation/ui_coco.json" `
  --output-dir "<DatasetDir>/model"
```

`merge_survivors_coco.py` は4つの入力を用途ごとにまとめ、`world_coco_model.json` と `ui_coco_model.json` を作ります。`weapon_*` は world class map v2 の world class として `world_coco.json` の category 12〜15 に出力されます。

正式利用では、元の収録 session が無い `annotation_work/session-0001` を dataset に含めないでください。`labeled_region` 付き frame は COCO から除外されるため、正式利用する frame は画面全体をラベルして `labeled_region` を付けないでください。

04-08（04-15 の formal world detector 訓練）用 dataset では agreement が必須です。2人なら independent_pair、1人なら v2 self_repeat を作成します。04-10／04-14 の判定では不要です。手順は 3-3 を参照してください。

## 3. D04 の収録から正式 dataset への順序

既存 session と annotation は作り直しません。planned split の記録と窓 plan が無い既存 session は `model_train`／`model_validation` または development 用に使い、`error_calibration`／`final_e2e_test` へ移しません。正式 test 用 session は収録前に用途を選び、別の dataset 世代として扱います。

### 3-1. session を収録し split を事前登録する

1 session は1 run 全体です。ステージ選択から死亡または30:00到達後の result 画面までを続けて収録し、途中で session を分けたり途中から始めたりしません。`--duration-sec 2700` 以上で起動し、result 画面を数秒記録してから止めます。収録中は alt-tab しません。

```powershell
python Tools/Deployment/capture_survivors.py `
  --store-root "<RecordRoot>" --session-id auto --duration-sec 2700 `
  --planned-split model_train
```

`--planned-split` は `model_train`、`model_validation`、`error_calibration`、`final_e2e_test` のいずれかです。用途と `planned_at` は session manifest の `operator_checkpoint` に記録されます。迷ったら `model_train` を明示します。用途を省略した既存 session は model 用として扱い、必要に応じて警告を確認します。

live capture が公開されると、stdout には PR #373 と同じ収録結果 JSON（resolved YAML SHA-256、`target_profile_hash`、`game_build_id`、`planned_split`、`planned_at`、収録開始時刻など）が出ます。operator、実測 session 長、run 終了状態、Steam の自動更新設定は stdout には含まず、`session_stats.json` と `session_log.md` にだけ自動記録されます。Steam の自動更新設定は Steam appmanifest から検出し、取得できない場合は `unknown` として記録します。`session_log.md` への手入力は不要です。

現行の `perception_session_split.py` は session 長 1800 秒以上を要求します。実行時間は終了 deadline より少し短くなるため、1800 秒要件が適用される収録では `--duration-sec 1800` ちょうどを使わず余裕を取ります。calibration／final 用の新しい session は現行の `Tools/Deployment/configs/formal_gates_minimal_v1.yaml` に定めた最短長に従います。

### 3-2. 昇格して split を freeze する

昇格は `session_manifest.json` の eligibility と `operator_checkpoint` を更新します。frame PNG、`frames.jsonl`、`annotations.jsonl` は変更しません。順序は必ず **「昇格 → 割当 → freeze」** です。freeze 済み session の manifest は書き換えません。

```powershell
python Tools/Deployment/promote_survivors_capture_session.py `
  --store-root "<RecordRoot>" --session-id "<SessionId>" `
  --operator-id "<OperatorId>" --note "実機 build と収録内容を確認"
python Tools/Deployment/freeze_survivors_capture_split.py `
  --store-root "<RecordRoot>" `
  --manifest-path "<RecordRoot>/capture_split_manifest.<YYYYMMDD>.json"
```

`--target-profile` は省略できます。省略時は個別昇格なら `<RecordRoot>/session_logs/<SessionId>/target_profile.resolved.yaml`、`--all-planned` なら各 session の同じ保存先から profile を読みます。ファイルが無い場合は昇格を拒否します。別のコピーを使う場合は `--target-profile "<ProfilePath>"` を明示できます。どの場合も収録時 hash と照合し、`target_resolve` で hash を取り直しません。用途を後から直す場合は freeze 前に昇格 CLI の `--planned-split <用途>` を使えます。sealed session を追加する世代には新しい `--manifest-path` を指定します。

用途は次の4種類です。

| `--planned-split` | 用途 |
|---|---|
| `model_train` | 練習用。detector／HUD parser の調整に使う |
| `model_validation` | 練習用。調整結果の確認に使う。`model_train` と同じ run を入れない |
| `error_calibration` | 封印。本番 test 用の認識誤差統計を作る |
| `final_e2e_test` | 封印。本番 test 用の最終判定に1回使う |

### 3-3. annotation、HUD truth、COCO export

model split は従来の候補選択で飛び飛びの frame を annotation します。04-08（04-15 の formal world detector 訓練）用 dataset では agreement が必要です。1人で作業する場合は、A の最終 label 更新から設定 gap（既定24時間）以上待ってから画像だけの B root を準備し、A の label と prelabel を見ずに B を annotation します。

```powershell
python Tools/Deployment/prepare_survivors_self_agreement.py `
  --work-root-a "<WorkRootA>" --work-root-b "<WorkRootB>" `
  --count 300 --seed 0 --minimum-gap-hours 24
```

A の最終 label 更新から設定時間が経っていなければ準備 CLI は拒否します。B root の label は空の状態で始まり、作成済み B root にファイルがあると準備 CLI は拒否します。準備から agreement 作成まで A root の label は一切保存せず、選ばれた frame だけを B で annotation します。A の label / 下書きおよび A の frame で作った prelabel は開きません。04-08 の必要件数は `minimum_counts.agreement`（`pair_count`／`iou_mean`／`class_agreement`）に従い、閾値は変えません。04-10／04-14 では agreement は不要です。annotation の開始・終了時刻と frame 数、下書きから修正した frame 数は COCO export 時に session log へ自動記録されます。

現行 plan では、model split は従来の候補選択、sealed session は `plan_survivors_evaluation_windows.py` で固定した窓の全 frame を画面全体で annotation します（`labeled_region` は使いません）。sealed session は prelabel の学習、active learning、候補選びの反復に使いません。窓内 frame の annotation 後、この manual の 2-6 で HUD truth、2-7 で COCO export を行います。

HUD truth を先に確定してから、session ごとに capture store の hash と split を束縛した COCO を出力します。04-08 用 agreement は `build_survivors_annotation_agreement.py` で作ります。1人の場合は `--annotator-a` と `--annotator-b` に同じ ID を渡し、準備 CLI と builder の `--minimum-gap-hours` に同じ値を指定します。gap は `min(manifest.created_at, B label の最小 mtime) - A label の最大 mtime` で測り、manifest より前の B label、blind 条件、設定 gap のいずれかを満たさない self_repeat は拒否します。publish 時は `publish_survivors_capture_dataset.py --agreement <AgreementJson>` を指定し、04-15 の formal 訓練では `train_survivors_world_detector.py --formal --annotation-agreement <AgreementJson>` に同じ JSON を渡します。04-10／04-14 用の publish では `--agreement` を省略できます。

sealed session の formal `annotations.jsonl` は `build_survivors_formal_annotations.py` で作ります。`AnnotationWriter.read_annotations()` は旧 schema 専用なので使いません。session ごとの COCO は `<RecordRoot>/coco/<SessionId>/` に置き、現行の evaluation windows と `formal_gates.v1` を使って `publish_survivors_capture_dataset.py` を実行します。publish は descriptor と restore verdict を出力します。backup と restore には別々の空 directory を指定してください。

現行の最小 gate は `Tools/Deployment/configs/formal_gates_minimal_v1.yaml` です。最短 session 長 300 秒、`event:death` 20 回、`event:result` は floor 未満の実測件数を記録して gate にせず、time band は到達 session が2本以上ある場合だけ判定します。sealed session の annotation 前に YAML を確定し、annotation 後は変更しません。publish は YAML の `schema_version` を確認し、入力 YAML の SHA-256 を descriptor に記録しますが、事前に確定した値との一致は検証しません。session log への転記・照合は不要です。coverage と verdict notes でデータ不足と性能不足を区別します。

正式 annotation 用 CLI が利用できる時期と画面・HUD truth の運用:

- `annotate_survivors_hud_truth.py` は model split の HUD truth に使います。`expected_items`／`expected_slot_levels` の下書きを画像と照合し、必要なら修正して確定します。
- level-up／宝箱の選択は通常どおり行い、固定ルールや選択ログは設けません。level-up 画面の段階マークから item level を読み、HUD truth の `expected_slot_levels` に記録します。
- `expected_screen_state` は parser が使う状態語彙に合わせ、`death` と `result` を分けて記録します。両方の画面を収録・確認します。

live capture が `<RecordRoot>/session_logs/<SessionId>/session_log.md` を自動生成し、COCO export が annotation 工数を追記します。雛形 [`templates/d04_session_log.md`](templates/d04_session_log.md) の複製や session log への手入力は不要です。

### 3-4. sealed session の評価窓を annotation する（04-18）

`error_calibration`／`final_e2e_test` に使う sealed session 全件を、annotation を始める前に一つの窓 plan にまとめ、artifact store に保存します。plan は時間帯の固定 offset、run tail、development parser の event、development detector の boss 検出から決まり、frame 内容を見て人が選びません。parser/checkpoint hash と閾値も plan の rule に固定します。

```powershell
python Tools/Deployment/plan_survivors_evaluation_windows.py `
  --store-root "<RecordRoot>" `
  --session-id "<SessionIdA>" `
  --session-id "<SessionIdB>" `
  --rule "<RuleYaml>" `
  --output "<PlanJson>" `
  --detector-manifest "<DetectorManifest>" `
  --detector-weights "<DetectorWeights>" `
  --detector-config "<DetectorConfig>" `
  --class-map "<WorldClassMapV2>"
```

sealed session の数だけ `--session-id` を繰り返し、全 session を含む JSON を一度に生成します。複数の plan JSON を手で結合しないでください。plan を capture descriptor の `identity_metadata.evaluation_windows` と結び、formal gate YAML の `formal_gates.v1` binding と一緒に世代を publish します。annotation 用 frame は同じ plan の stride で抽出します。

```powershell
python Tools/Deployment/select_survivors_annotation_candidates.py `
  --store-root "<RecordRoot>" `
  --session-id "<SessionId>" `
  --window-plan "<PlanJson>"
```

配置された frame だけを画面全体で label し、`labeled_region` は付けません。これは COCO export 時に frame が除外されるためです。通常の stride／k-means 選択は model_train／model_validation 用 session に引き続き使います。

全 window frame の画面状態・HUD truth を 04-14 の手順で確定した後、session 単位で COCO を出力し、04-16 の `build_survivors_formal_annotations.py` で formal annotation を作ります。

```powershell
python Tools/Deployment/annotate_survivors_hud_truth.py `
  --store-root "<RecordRoot>" --session-id "<SessionId>" `
  --annotator-id "<AnnotatorId>" --work-root "<WorkRoot>" --from-labels
python Tools/Deployment/export_survivors_annotations_coco.py `
  --work-root "<WorkRoot>" --store-root "<RecordRoot>" `
  --session-id "<SessionId>" --output-dir "<RecordRoot>/coco/<SessionId>"
```

収録後は coverage CLI で window 内の expected frame、HUD truth、COCO export、slice floor を確認します。GT が欠けた report は `provisional: true` となり formal coverage の確認には使えません。

```powershell
python Tools/Deployment/check_survivors_evaluation_coverage.py `
  --store-root "<RecordRoot>" --plan "<PlanJson>" `
  --coco-dir "<RecordRoot>/coco" `
  --hud-truth "<RecordRoot>/capture_sessions" `
  --class-map "<WorldClassMapV2>"
```

## 参考

- 録画の内部実装・アーキテクチャ・テスト: [`capture_dataset.md`](capture_dataset.md)、[`capture_core.md`](capture_core.md)
- アノテーションの下書き設定・2 周目以降の運用: [`annotation_gui.md`](annotation_gui.md)
- target profile の準備: [`target_profile.md`](target_profile.md)

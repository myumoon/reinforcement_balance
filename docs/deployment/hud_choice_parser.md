# HUD・Inventory・Choice Parser Core

**Roadmap-Plan: 04-03** — 実ゲーム画面からHUD値・レベルアップ選択肢・UI状態を読み取るparserコアの実装。

## 概要

`Tools/Deployment/survivors/vision/` パッケージが提供する parser 群は、
CapturedFrame (04-01 契約) から `HudStateV1` を生成します。
OCR には依存せず、template matching と bar segmentation で値を抽出します。

```
CapturedFrame
    │
    ▼ HudParser.parse()
    HudStateV1   ← 04-09 Assembler が PerceptionSnapshot に変換
```

`HudStateV1` は 04-09 の中間出力であり、05-03 (UI state machine) は直接参照しません。

## ファイル構成

| ファイル | 責務 |
|---|---|
| `survivors/vision/roi_layout.py` | HUD の正規化 ROI と実測 UI の固定 PixelROI |
| `survivors/vision/screen_layout.py` | 共有の画素クラス・HUD 枠・パネル・縦積みカード行判定 |
| `survivors/vision/hud_types.py` | ParsedCard・ParsedButton・候補集合 hash（hud_parser からも再公開） |
| `survivors/vision/digit_parser.py` | binarize + connected components + template distance でタイマー/レベルを解析 |
| `survivors/vision/bar_parser.py` | HSV color segmentation で HP/XP バー充填率を解析 |
| `survivors/vision/slot_level_parser.py` | 段階マークの色割合・prefix・パネル下端から slot level と空枠を読む |
| `survivors/vision/icon_matcher.py` | AtlasManifest、IconMatcher、色+エッジ特徴距離マッチング |
| `survivors/vision/hud_parser.py` | HudStateV1 データ契約、画面判定、時間的保持、choice への配線 |
| `survivors/vision/choice_parser.py` | レベルアップカード・ボタン・fallback の choice parser |
| `build_survivors_icon_atlas.py` | 開発用合成 atlas ビルダー |

## HudStateV1 スキーマ

```python
HudStateV1(
    schema_version = "hud_state.v1"
    session_id, frame_index, captured_monotonic_ns  # フレーム識別
    parser_artifact_hash                             # 呼び出し側が渡す識別子（開発 CLI は hud_parser.py の sha256）
    screen_state         # "gameplay"|"level_up_items"|"level_up_fallback"|"chest"
                         # |"paused"|"target_reached_transition"|"death"|"result"|"unknown"
    screen_state_confidence, screen_state_reason
    timer_seconds,  timer_confidence,  timer_reason
    post_30_evidence                                 # 30:00超遷移の観測
    hp_ratio,       hp_confidence,     hp_reason     # 0..1 or None
    xp_ratio,       xp_confidence,     xp_reason
    level,          level_confidence,  level_reason  # 1..99 or None
    inventory,      inventory_confidence, inventory_hash  # 12スロット
    cards,          candidate_set_hash
    buttons
    reroll_available, skip_available, banish_available
    capability_confidence, capability_reason
    inventory_levels = (None,) * 12                 # 12個の int 1..9 または None
    inventory_levels_confidence = 0.0              # 採用した段階値・空枠の割合
)
```

## Low-confidence ポリシー

- **推測しない**: 読めない観測は `None`。段階値は二枚一致した枠を訪問中だけ保持し、短い画素の揺れでは取り消しません
- **temporal constraints**: タイマーの逆行 (>30s) とレベルの逆行を reject する
- **unknown/None は伝播する**: assembler (04-09) が invalid field として処理する

## 段階マークと在庫の位置結合

`INV_SLOT_ROIS` は gameplay 左上の武器六枠・パッシブ六枠へ修正しました。1920×1080 では x=`100+46i`、武器 y=40、パッシブ y=86、各枠42×42画素です。段階マークは `SLOT_LEVEL_PIP_GRID` の正規化格子で読み、内部画素の六割が金色なら点灯、暗色なら消灯、灰色ならセル無しとします。全幅の九割が金色かつ直下二行の九割が茶色の最初の行がパネル下端で、その外側の行は除外します。金色だけでは宝箱の光を下端と誤認するため、実測した茶色の縁も確認します。点灯→消灯→セル無しの prefix が崩れる枠や曖昧な枠は不明です。武器八セル・パッシブ五セルの上限超過は `over_max` になり、宝箱の全点灯を証拠にしません。

下端を確認でき、二セル以上の valid slot が一つでもあれば、HUD のある画面を `level_up_items`／reason `slot_panel` と判定します。一セルの進化武器も level 1 ですが、画面判定の証拠には数えません。下端未検出時は全行を解析して段階値を返しますが、背景の偶然の prefix を画面証拠にしないよう `evidence_slots=0` にします。証拠が消えてから三枚は `slot_panel_hold` で同じ状態を保ちます。hold は画面状態だけに効き、新しい段階値を採用しません。各 slot は二枚連続一致で採用し、訪問終了時と `reset_temporal_state()` で全消去します。

半透明パネル上のアイコン照合は行わず、直前 gameplay の在庫を同じ位置で結合します。gameplay は三枚連続の非 None 一致で枠ごとに保存し、None が三十枚続いた枠だけ消去します。chest／death／result／unknown と reset では在庫と一致履歴を全消去します。カードの滑り込み `card_transient`（confidence .45）と `slot_panel_hold` は保存在庫を消しません。

パネル空枠と gameplay の所持 identity が食い違えば `slotN:empty_mismatch`、段階ありと保存空枠が食い違えば `slotN:identity_mismatch`、表示セル総数と atlas entry の kind が食い違う・種別不明なら `slotN:kind_mismatch` を screen_state_reason に追記して両方を None にします。パネル空枠と保存 None は `empty_slot`、段階ありと保存 None は level のみを出します。`inventory_confidence` は identity（空枠含む）の確定割合、`inventory_levels_confidence` は段階値か空枠を採用した割合です。

wire の二キーは必須で、未知キー・欠落キーは従来どおり拒否します。`inventory_hash` は identity だけのままで、`parser_artifact_hash` の定義も変えていません。座標と色は 1920×1080・既定 UI scale・日本語 UI・左右約96 px の黒帯がある収録で測定しました。全武器進化後の行位置・九段階パッシブは未確認です。

## 実測カード配置と画面状態（04-22）

カードは中央ウィンドウ内へ上から縦に一〜四枚積まれます。新しい UI 定数は正規化せず `PixelROI` で持ち、他解像度は `unknown`／`unsupported_resolution`、空 cards・buttons にします。以下の座標は右端・下端を含みません。

| 領域 | px 座標 |
|---|---|
| 中央ウィンドウ | (642,111)〜(1278,965) |
| カード | x=656〜1265、`CARD_TOP_Y=(267,424,581,738)`、高さ154 |
| icon | x=669〜720、各カード上端+13〜+68（51×55） |
| リロール | (1404,247)〜(1702,329)、青い内側 (1413,253)〜(1693,317) |
| 宝箱終了 | (810,827)〜(1107,913)、青い内側 (822,835)〜(1098,902) |
| 死亡終了 | (810,702)〜(1110,789)、赤い内側 (819,708)〜(1101,780) |
| 結果終了 | (812,968)〜(1108,1049)、青い内側 (821,974)〜(1099,1040) |
| 一時停止の再開 | 青い内側 (1448,952)〜(1732,1033) |

画面判定は上から評価します。gold は R>190/G>140/B<130、border は gold または全成分>230 の白です。warm は R>190/G>90/B<130、dim は 100<R<150/80<G<120/30<B<70 です。

| 優先 | 条件 | 状態・confidence・reason |
|---|---|---|
| 1 | 黄色 R>190/G>180/B<180 が全体の .30 以上 | chest / .60 / chest_flash |
| 2 | y=2,32、x=300〜1600 の dim が両方 .80 以上、再開内側の青が .50 以上 | paused / .70 / pause_menu |
| 3 | 中央上下枠あり。右枠あり、先頭から連続したカード金枠行が1〜4、黄色<.05 | level_up_items / .70 / card_rows:N |
| 3a | 中央上下枠あり、上の条件なし、カード領域の灰色面が .20 以上 | level_up_items / .45 / card_transient |
| 3b | 中央上下枠あり、上の二条件なし | chest / .65 / chest_panel |
| 4 | 大きな結果パネルの上下金枠 .80 以上・背景 RGB(75,79,116) が .50 以上 | result / .70 / result_panel（HUD に依存しない） |
| 5 | warm の HUD 枠二本 .80 以上、段階格子の証拠あり | level_up_items / .60 / slot_panel |
| 6 | HUD と GAME OVER 金文字 .06 以上・赤い終了面 .50 以上 | death / .70 / game_over |
| 7 | HUD 枠二本あり | gameplay / .60 / hud_present |
| 8 | 上記なし | unknown / .35 dark_no_hud または .30 no_hud |

空 frame は `empty_frame`、白一色は枠の構造がないため `no_hud` にします。カード上枠は各 top+1〜+4 の三行、x=700〜1200 の gold が .60 以上で連続数を数えます。中央枠は y=108〜124／905〜970 の各帯に x=660〜1260 の border が .80 以上の一行、右枠は x=1268〜1282 の帯に y=300〜900 の border が .80 以上の一列を要求します。

`HudParser.parse` は常に choice 解析を配線し、fallback が過半なら `level_up_fallback` に精緻化します。atlas がなくても枚数と矩形は残します。skip／banish は未計測のため常に False、`capability_confidence=0.0`／`skip_banish_roi_undefined` です。リロールだけは内側の青で観測します。所持カード選択直後の +1 は skip の確定に信頼度 .5 が必要なので、次のパネルまで段階値は None になります。

宝箱の `ack_chest` は終了ボタンだけです。y=905〜925 にパネル下枠が縮んだことと青い内側 .50 以上を要求します。「開く」の下枠は y=954〜962 にあり、ボタン自身の短い下端だけでは .80 の閾値に届きません。Common の ACK_CHEST 契約では開く→終了を別 intent にできないため、開くボタンは出しません。自動で開く前提は未確認で、手動が必要なら60秒で停止します。次の録画では宝箱で一度、何も押さずに待ってください。

宝箱証拠の後は15 frame の保持を持ち、続く unknown 十四枚を `chest`／.50／`chest_hold` にします。他の既知状態や未対応解像度を上書きせず、reset で消えます。controller の宝箱 timeout は実測約21秒の演出に合わせて60秒です。retry は既存の同等性判定が初回・再観測とも confidence .99 以上を要求します。フェード中や文字を含む青い面の割合が .99 未満なら初回クリック後の再送は `ui_retry_precondition_failed` で停止し、回収を保証しません。

death／result の `confirm` は観測のみです。`perception_snapshot._UI_SCREEN_STATES` に両状態がないため button target は invalid になり、controller の DEATH_RESULT はクリックしません。paused の cards・buttons は空で、controller は入力なし・timeout なしの PAUSED に入ります。

## Development atlas

`build_survivors_icon_atlas.py` が生成する合成 atlas は常に:
- `development_only=True`
- `formal_parser_eligible=False`

`IconMatcher.load_formal()` はこの atlas を `FormalLoaderRejectedError` で拒否します。
正式 atlas は 04-05 で実ゲーム画像から生成します。

`TemplateEntry.surface` は `inventory`（省略時の既定）か `card` です。所持 icon とカード icon は描画が違うため、同じ manifest に両方の crop を別 entry として入れます。`match(..., surface="card")` は card entry のみを照合し、なければ `no_surface_entries` です。開発 builder は各 item/level に同じ合成 feature の二 entry を作り、item_id・surface・level 順に並べます。

カードは実測 crop と feature が完全一致するときだけ、従来の margin 判定へ進みます。一致しないカードは `card_template_mismatch`／confidence 0 です。候補間の差だけでは、未登録の「引き寄せのオーブ」が runetracer に割り当てられる実フレームがあったためです。inventory の距離・margin 判定は変えていません。

feature は histogram と edge を要約するため、完全一致しても画素の同一や item identity を保証しません。逆に描画が少し変わった既知カードも拒否します。card entry は実際の `card_icon_roi(k)` の crop から作り、描画差を許す距離の較正は後続の formal atlas 作業で行います。合成 development atlas は本家のカードを識別できる保証を持ちません。

## HUD truth（`hud_truth.v1`）

parser の精度を測るための正解値です。capture session 配下の `hud_truth.jsonl` に 1 行 1 frame で保存します。`annotations.jsonl`（bbox）や X-AnyLabeling の label JSON とは別の file で、既存の annotation は読み書きしません。付け方は [`capture_annotation_manual.md`](capture_annotation_manual.md) の「2-7. HUD の正解値を付ける」を参照してください。

- 実装: `survivors/vision/hud_truth.py`（record・検査・読み書き・下書き・行コマンド）、CLI は `Tools/Deployment/annotate_survivors_hud_truth.py`
- 読み書き: `read_hud_truth(session_path)`／`write_hud_truth(session_path, records)`。`frame_id` の昇順で書き、重複した `frame_id` は `ValueError` になります。書き込みは一時 file → fsync → replace の順です。
- 未知の key は読むときに `HudTruthRecord.extra` に入り、書き戻すときにそのまま残ります。

| field | 型 | 内容 |
|---|---|---|
| `schema_version` | str | `"hud_truth.v1"` |
| `session_id`／`frame_id` | str／int | 対象 frame |
| `annotator_id` | str | 確認者 |
| `confirmed`／`confirmed_at` | bool／ISO8601 または null | 人が確定したか、確定した時刻 |
| `expected_screen_state` | str | `SCREEN_STATES` のいずれか |
| `expected_timer_seconds`／`expected_level`／`expected_hp_ratio`／`expected_xp_ratio` | 数値または null | 読めない値は null |
| `expected_items` | 12 個の str（item_id か `"empty_slot"`）または null | slot 0〜5 が武器、6〜11 がパッシブ。確定行では slot 単位の null を禁止し、1 slot でも読めなければ配列全体を null にします |
| `expected_slot_levels` | 12 個の int または null、または全体が null | 段階マークの点灯数。`roi_layout.SLOT_LEVEL_VISIBLE_STATES`（`level_up_items`／`level_up_fallback`）の画面でだけ非 null にできます。item の slot は 1〜9、`empty_slot` の slot は null。`expected_items` が null なら必ず null |
| `expected_choice` | str の配列または null | 画面の card の item_id（上から）。card は level-up 画面にしか出ないので、確定行では `level_up_items`／`level_up_fallback` 以外の state なら null にします |
| `roi_name`／`expected_roi` | `"hud"`／`[l, t, r, b]` または null | HUD あり状態（gameplay／level_up_items／level_up_fallback／chest）では HP バー〜XP バーの矩形 `[0, 32, 1920, 75]`、HUD なし状態では null（負例）。state から自動で決まり、人は編集しません |
| `draft_source` | object | 下書きを作った `parser_artifact_hash` と `atlas_content_hash`（atlas 無しは `"none"`） |

規則は `validate_record(record)` だけにあり、CLI の行コマンドの受理判定、`write_hud_truth` の確定行チェック、test のすべてがこの関数を使います。確定していない下書きでは slot 単位の null を許します。

`hud_calibration._validate_annotation` は `expected_slot_levels` を知らない field として無視します。ただし `expected_choice` は `str|None` として検査するので、`validate` へ渡すときは list を `|` 連結の str に変換します（04-14 の責務）。

## テスト実行

```bash
cd <worktree>
bash <project_root>/Tools/run-pytest.sh Tools/Deployment/tests/vision -q -rs
```

## 制約と将来の作業

- カード・ウィンドウ・ボタンは1920×1080日本語 UI の実測値です。他解像度・言語・UI scale は未対応で、定数の再計測が必要です。
- digit templates は開発用の合成パターン。実データには 04-04 の calibration を要する。
- icon matching は color/edge 特徴のみ。04-05 で実画像 atlas に差し替える。
- 画面状態は実測の枠と色で判定します。一時停止のオプション画面、target_reached_transition、level_up_fallback の実画面は未確認です。
- skip／banish の ROI は未計測で、所持カード選択の +1 は発火しません。カードの名前・レベルの文字も読みません。
- 開発 CLI の `parser_artifact_hash` は hud_parser.py 一ファイルの sha256 です。screen_layout.py・choice_parser.py・roi_layout.py の変更はこの hash だけでは識別できません。formal 側の全 vision `code_hashes` は04-14の対象です。
- `atlas_content_hash` は順序付き feature 列の連結だけで、item_id・surface を含みません。同じ feature 列の surface を入れ替えても hash は同じです。二 entry 化で列自体が増えるため、以前の一 surface atlas と同じ値になるとは限りません。04-14 の provenance key は item_id@surface とします。
- color/edge の距離差だけで照合するため、未登録の別 icon を少数 template へ誤って割り当てる場合があります。scratch atlas は formal 判定の代用にはできません。

## 受け入れ条件チェック

| 条件 | 状態 |
|---|---|
| low-confidence field は validity 0 + reason を返す | PASS |
| card count / fallback / button を全構造化 | PASS (合成フィクスチャで検証) |
| development atlas が formal loader で拒否される | PASS |
| HudStateV1 フィールドセットが exact-set テスト済み | PASS |
| controller/runtime が HudStateV1 を import しない | N/A (04-09 実装時に検証) |
| 実動画の accuracy/latency 達成を完了と主張しない | N/A (formal dependency) |

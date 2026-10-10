# Real observation assembler
`RealObsAssembler` は `HudStateV1` と `TrackedWorldStateV2` を同じ policy tick に結合し、`PerceptionSnapshot` を生成する 04-09 の本番観測境界です。

## 境界
- `DeployObservation` は visible (`on_screen=true`, `clipped=false`) な track と HUD 値だけから生成する。
- `UiPresentationSnapshotV1` の ROI はクリック実行専用で、deploy/item tensor と diagnostics に含めない。
- presentation は `PerceptionSnapshot` に atomic に格納し、別の `HudStateV1` side channel を公開しない。
- `source_content_hash` は ROI/confidence を 1e-4 量子化し、`ui_state_key` は連続値と時刻を除外する。
- 15 Hz tick で join し、50 ms 超の skew と 200 ms 超の stale source は fail-closed にする。
- `gameplay` は combat、level-up/fallback/chest は item context のみ有効にする。
- Deployment の既定 schema は DeployObs v2（`DeployObsSchema.default_v2()`）。v1 schema を明示的に渡したときだけ旧経路（上の on_screen/clipped 規則）で v1 tensor を作る。
- item context の `boss_flag` / `hazard_flag` は coarse class `enemy`（boss）/ `hazard` の track だけで決め、weapon の track は含めない（v1 と同じ意味）。

## DeployObs v2 の作り方（04-13）
特徴量は `Tools/Common` の `build_deploy_obs_v2` だけが計算し、Deployment（`survivors/deploy_obs_v2_input.py`）は入力の変換だけを行う。

- track の中心 px は `normalized_cx × 画面幅`・`normalized_cy × 画面高さ`、半径は検出矩形の `max(normalized_width × 幅, normalized_height × 高さ) / 2`、初観測時刻は `first_seen_timestamp_ns / 1e9` 秒。
- 信頼度 0.35 未満の track は渡さない。実機 tracker は遮蔽を判定しないので `occluded=False`。
- 可視判定は Common の規則（中心が画面内・遮蔽なし）に任せ、tracker の `clipped` / `on_screen` は使わない。矩形が画面からはみ出していても中心が画面内なら数える。
- プレイヤー位置は `player_anchor` の中心 px。v1 の `player_relative_x/y` は使わない。anchor が無い・fallback のときは world 特徴を不明（validity 0）にする。
- combat が無効なフレーム（level-up 画面など）は world・HUD 由来の値を全て不明にする（v1 で validity に combat を掛けていたのと同じ扱い）。HP・レベル・在庫は HUD の信頼度が 0.35 以上のときだけ渡す。
- `movement_direction` は Deployment に情報源が無いので常に不明（validity 0）。
- `screen_space_features.directional_bin` は Common の `directional_bin` の再 export（式は Common の1か所だけ）。
- tracker 設定 `world_detector_v2.yaml` の weapon 4クラスの `max_age_by_class` は Common の `track_max_age_frames` と一致させる（テストで固定）。

### HUD スロット
- 在庫 12 枠（武器 6・パッシブ 6）の identity は `configs/hud_identity_vocabulary_v1.yaml` で Common の語彙名（C++ `EWeaponType` / `EPassiveItemType` の名前）へ写す。ローダー `survivors/hud_identity_vocabulary.py` は未知キー・語彙外の対応先・重複を拒否する。
- 空と未認識の区別: `HudStateV1.inventory` の `None` は icon_matcher が読めなかった枠（`low_margin` / `low_confidence` / エフェクトの遮蔽）と空枠を区別できないので、常に「不明」として扱い、その枠の `HudSlot` を渡さない（validity 0）。「空スロット確定」は atlas の空スロット用テンプレートが返す identity（対応表の `empty_slot`、既定 `empty_slot`）の枠だけで、`HudSlot(type_name=None)`（validity 1）にする。atlas に空スロット用テンプレートが無いあいだは、空枠も全て「不明」になる（武器枠が 6 つ埋まるまで aura / orbit / zone の「出し手無し確定」は出ず、パッシブ枠が 6 つ埋まるまで持続時間倍率は不明）。
- 表に無い identity（種別違いを含む）の枠も渡さず「不明」にする。
- この区別は兄弟経路にも同じに適用する: Common の `duration_mult_from_hud_slots`（パッシブ全枠が揃わなければ `None`）と `_emitter` の complete 判定（武器全枠が揃わなければ「無し確定」にしない）は渡されない枠を不明として扱う。`TemporalAssembler.observe_hud` は読めなかった枠で前の identity を保持するが、`empty_slot` は新アイテムで埋まりうるので保持せず `None` に戻す。v1 の occupancy（`inventory_levels`）は `empty_slot` も空（0）と数える。
- 持続時間倍率は Common の `duration_mult_from_hud_slots`（C++ `ComputePassiveEffects` と同じく `1 + Spellbinder 0.10×Lv + TorronasBox の加算`）で求める。パッシブ枠に不明な枠やレベル不明の Spellbinder / TorronasBox があれば倍率は不明（`None`）で、残り時間も不明になる。

### 武器・パッシブのレベルの取得

level-up 画面の左上パネルには、アイコン下に段階マークが出ます。点灯数が現在レベル、表示セル総数が最大レベルで、選択前の値です。`HudParser` が `inventory_levels` に読み、`SlotLevelTracker` が gameplay 復帰時に選択結果を反映します。

追跡器の規則（class docstring と共通）:

- (a) gameplay で新しい identity が現れたら Lv1（進化武器も Lv1）。直前の同じ枠が None なら None（session 最初の gameplay は除く）。
- (p) 画面信頼度と段階値信頼度が .5 以上のパネルを、訪問中の選択前の基準値として保持する。後の確定フレームで上書きし、読めたカード（信頼度 .35 以上）と skip（信頼度 .5 以上）も保持する。
- (r) gameplay 復帰時、基準が無ければ全所持 None。基準があれば identity と level が両方読めた値で辞書を作り直す。基準の空枠に identity が一つだけ現れたら fresh を Lv1、他は基準のまま。二つ以上なら fresh と所持カードを None。fresh が無く、基準・復帰在庫の両方に None が無く、所持カードが一枚で確定した skip が False なら基準値 +1。それ以外は所持カードを None。カードを一枚も読めていなければ全所持 None。
- (d) 宝箱画面の後は所持全スロットと、その直後に現れた identity を None。
- (e) None は次に (a) または (p→r) が起きるまで不明のまま。(f) session 変更・reset で全消去。

パネルの identity は、直前の gameplay 在庫を slot 位置で結合します。gameplay は同じ非 None の identity が三枚続いたときだけ保存し、不読では上書きせず三十枚続いた枠だけ消します。パネルは枠ごとに二枚一致で段階値・空枠を採用し、訪問中の一枚の揺れでは取り消しません。保存 identity と表示セル総数が食い違う枠、種別不明の枠は結合しません（一セルは進化武器だけ）。

既知の制約:

- ランの途中から観測を始めると、最初の level-up までは実レベルを確定できません。規則 (a) は session 先頭を Lv1 として扱うため、session とランの開始を揃えてください。
- 所持カードを選んだ直後の +1 は、skip／banish の ROI を観測して能力信頼度が .5 以上になるまで発火しません。現在は `capability_confidence=0.0` なので、選択後の所持 slot level は次の level-up の段階マークを読むまで None になります。
- 九段階のパッシブは二行の格子に収まらないため None になります。全武器が進化した場合のパネル形状は未確認です。一時停止のパネルは半暗なので段階値の証拠にしません。
- 04-22 の画面判定は level-up／chest／death／result／paused を実測の枠と色で区別します。chest／death／result／unknown で保存在庫を全消去し、card_transient と slot_panel_hold では消しません。宝箱後の slot level は規則 (d) に従い全て None です。
- 色と座標は 1920×1080、日本語 UI、既定 UI scale、左右約96画素の黒帯がある収録で測定した値です。設定を固定し、異なる解像度には格子定数を差し替えてください。二枚連続の段階誤読・三枚連続の identity 誤読や、決定後の遅い slot 採用による apply ack の変化は防げません。

## Formality
fixture は development-only です。04-10 の正式 parser artifact hash、calibration replay、fidelity verdict を発行する能力はありません。

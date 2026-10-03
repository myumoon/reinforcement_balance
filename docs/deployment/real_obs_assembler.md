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
- `None` の枠は「空スロット確定」、表に無い identity（種別違いを含む）の枠は渡さず「不明」にする。
- 持続時間倍率は Common の `duration_mult_from_hud_slots`（C++ `ComputePassiveEffects` と同じく `1 + Spellbinder 0.10×Lv + TorronasBox の加算`）で求める。パッシブ枠に不明な枠やレベル不明の Spellbinder / TorronasBox があれば倍率は不明（`None`）で、残り時間も不明になる。

### 武器レベルの取得（調査結果と採用方法）
- 本家の HUD は左上のスロットにアイコンだけを表示し、スロットごとのレベルは表示しない（右上の `LV` はプレイヤーレベル）。HUD からの読み取りは採用しない。
- 代わりに `survivors/slot_level_tracker.py` の最小のレベル追跡器を assembler の時系列状態に持ち、tick ごとに HUD を渡す。規則:
  - (a) 在庫に新しい identity が現れたら Lv1（進化武器も Lv1）。
  - (b) レベルアップ画面から gameplay に戻ったとき、プレイヤーレベルがちょうど 1 上がり、新しい identity が無く、所持 identity のカードがちょうど 1 枚で skip できない画面だったなら、そのスロットをカードの新レベルにする。所持カードが複数・skip 可能なら該当スロットを不明、レベル差が 1 でない（連続レベルアップ・レベル不明）なら所持全スロットを不明にする。
  - (c) カードを読めなかった（`item_id` 無し・低信頼・カード無し）ときは所持全スロットを不明。
  - (d) 宝箱画面の後は所持全スロットと、その直後に現れた identity を不明（結果を読めないため）。
  - (e) 不明になったスロットは推測で埋めず、そのランの間は不明のまま。(f) session 変更で全消去。
- 既知の制約: ランの途中から観測を始めると、最初に見えた在庫は Lv1 として数えられる（session の開始とランの開始を揃えて使う）。controller の選択結果（item_session の決定）を assembler へ渡す経路ができれば (b) の曖昧さは無くなる（別 plan として提案）。

## Formality
fixture は development-only です。04-10 の正式 parser artifact hash、calibration replay、fidelity verdict を発行する能力はありません。

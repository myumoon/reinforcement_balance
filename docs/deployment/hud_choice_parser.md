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
| `survivors/vision/roi_layout.py` | viewport-relative ROI anchors、PixelROI、正規化→ピクセル変換 |
| `survivors/vision/digit_parser.py` | binarize + connected components + template distance でタイマー/レベルを解析 |
| `survivors/vision/bar_parser.py` | HSV color segmentation で HP/XP バー充填率を解析 |
| `survivors/vision/icon_matcher.py` | AtlasManifest、IconMatcher、色+エッジ特徴距離マッチング |
| `survivors/vision/hud_parser.py` | HudStateV1 データ契約、HudParser、ParsedCard、ParsedButton |
| `survivors/vision/choice_parser.py` | レベルアップカード・ボタン・fallback の choice parser |
| `build_survivors_icon_atlas.py` | 開発用合成 atlas ビルダー |

## HudStateV1 スキーマ

```python
HudStateV1(
    schema_version = "hud_state.v1"
    session_id, frame_index, captured_monotonic_ns  # フレーム識別
    parser_artifact_hash                             # parser設定+atlasのhash
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
)
```

## Low-confidence ポリシー

- **推測しない**: confidence < 閾値の場合、前回値を再利用せず `value=None` + `reason` を返す
- **temporal constraints**: タイマーの逆行 (>30s) とレベルの逆行を reject する
- **unknown/None は伝播する**: assembler (04-09) が invalid field として処理する

## Development atlas

`build_survivors_icon_atlas.py` が生成する合成 atlas は常に:
- `development_only=True`
- `formal_parser_eligible=False`

`IconMatcher.load_formal()` はこの atlas を `FormalLoaderRejectedError` で拒否します。
正式 atlas は 04-05 で実ゲーム画像から生成します。

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

- ROI anchors は 1920x1080 基準の推定値。04-04 のキャリブレーションツールで調整する。
- digit templates は開発用の合成パターン。実データには 04-04 の calibration を要する。
- icon matching は color/edge 特徴のみ。04-05 で実画像 atlas に差し替える。
- 画面状態検出は輝度/前景量ベースの簡易実装。実データで改善予定。

## 受け入れ条件チェック

| 条件 | 状態 |
|---|---|
| low-confidence field は validity 0 + reason を返す | PASS |
| card count / fallback / button を全構造化 | PASS (合成フィクスチャで検証) |
| development atlas が formal loader で拒否される | PASS |
| HudStateV1 フィールドセットが exact-set テスト済み | PASS |
| controller/runtime が HudStateV1 を import しない | N/A (04-09 実装時に検証) |
| 実動画の accuracy/latency 達成を完了と主張しない | N/A (formal dependency) |

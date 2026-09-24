# Survivors gameplay/UI state machine (05-03)

`Tools/Deployment/survivors/controller/state_machine.py` と
`Tools/Deployment/survivors/controller/ui_navigation.py` が実装する、実機 Survivors
操作の唯一の入力 driver です。`PerceptionSnapshot` と `AgentDecision`/`UiIntentV1`
だけを入力にし、`InputLeaseController` の公開 API(`send_action`/`send_ui_click`/
`send_ui_key`/`emergency_release`)だけを呼ぶ決定的 hierarchical state machine を
実装します。座標は毎回 `PerceptionSnapshot.ui_presentation` の typed target から
取得し、固定座標や side channel は一切使いません。

## 状態一覧

`ControllerState` (`state_machine.py`):

| 状態 | 意味 |
|---|---|
| `DISARMED` | 起動直後の休止状態。`StateMachine.arm()` を明示的に呼ぶまで画面入力を一切見ない。 |
| `ACQUIRE_TARGET` | arm 直後、対象 window の画面分類が3連続フレーム確定するのを待つ。 |
| `RUN_SETUP` | 最初の `GAMEPLAY` 遷移を待つだけの状態。formal campaign ではここは最初の gameplay 開始前だけに使われ、gameplay artifact/evidence を生成しない。 |
| `GAMEPLAY` | 通常プレイ中。movement decision (`kind="move"`) をそのまま `Effect(kind="move")` へ転送する。 |
| `LEVEL_UP` | レベルアップの item card / fallback(gold・chicken)選択画面。 |
| `CHEST` | 宝箱の ack 画面。 |
| `TARGET_REACHED_PENDING_TRANSITION` | 30:00 到達 evidence を観測した直後。ここに入った時点で `success_latched=True` になり、以後は何が起きても最終的に `COMPLETE` に収束する。 |
| `PAUSED` | 一時停止画面。timeout なし(indefinite)で input を一切送らない。 |
| `UNKNOWN` | 認識できない画面(未知の screen_state・focus loss)。1 秒(profile 由来)で fail-closed する。input は一切送らない。 |
| `RECOVER` | `PAUSED`/`UNKNOWN` から `GAMEPLAY` へ戻れたかを再確認するだけのバッファ。ここでも input は一切送らない。 |
| `DEATH_RESULT` | 死亡/結果画面。侵入ごとに combat LSTM reset 相当の `Effect(kind="combat_reset")` をちょうど1回発行してから、terminal routing を決める。 |
| `COMPLETE` | formal terminal(成功)。 |
| `FORMAL_RUN_TERMINAL_FAILURE` | formal terminal(失敗)。 |
| `EMERGENCY_STOP` | fail-closed の内部概念(state として複数 tick 観測されることはない。安全確定を1 tick 以上遅らせないため、検知した同じ tick 内で `FORMAL_RUN_TERMINAL_FAILURE` か `DISARMED` へ即時確定する)。 |

## 画面分類とデバウンス

`classify_screen_state(raw_screen_state, *, window_focused=True)` が
`PerceptionSnapshot.screen_state` の生文字列(04-09 parser の語彙:
`gameplay`/`level_up_items`/`level_up_fallback`/`chest`/
`target_reached_transition`/`paused`/`death`/`result`/`unknown`)を
`ControllerState` のカテゴリへ写します。未知の文字列や `window_focused=False`
(フォーカス喪失)は安全側の `UNKNOWN` に倒します。

同じカテゴリが `NavigationProfile.debounce_frames`(既定 3)連続で観測されて
はじめて状態遷移が「確定」します。1 フレームだけの誤認識(flicker)は streak を
リセットするだけで遷移を確定させません。

## movement/UI の排他

- `GAMEPLAY` 以外では movement effect を一切発行しません。
- UI click/key effect は、常に `Effect(kind="release_all")` とペアで
  同じ tick に発行されます(`release_all` → `ui_click`/`ui_key` の順)。これは
  gameplay → level-up の「key release が choice click より先」という順序を
  05-03 側でも明示的に保証するためです(`InputLeaseController` 側の
  `HelperRuntime.handle_lease` も UI 送信前に movement chord を強制解放しますが、
  05-03 の effect 順序としても独立に固定します)。
- `GAMEPLAY` から他の状態(`LEVEL_UP`/`CHEST`/`PAUSED`/`UNKNOWN`/
  `TARGET_REACHED_PENDING_TRANSITION`/`DEATH_RESULT`)へ確定的に抜けるtick
  自体でも、`Effect(kind="release_all")` を1回発行します。UI click とペアの
  release だけに頼らず(click が発生するまで時間がかかる/一度も click が
  発生しない PAUSED・UNKNOWN・DEATH_RESULT のような遷移でも)movement lease を
  即座に手放すためです。
- 同一 tick で `move` と `ui_click`/`ui_key` の両方を送ることはありません
  (1 tick 1 input effect)。

## UI click の解決と retry

`ui_navigation.resolve_ui_target(intent, snapshot, mode, *, original_snapshot=None)`
が `UiIntentV1` を `PerceptionSnapshot.ui_presentation` 上の typed target
(`UiCandidateTargetV1`/`UiButtonTargetV1`)へ解決します。

- `mode="initial"`: `AgentDecision` が既に保証する
  `source_snapshot_hash`/`source_frame_hash`/`source_content_hash` の完全一致を
  前提に、`choose_card`(`candidate_set_hash` + `target_index` + `semantic_kind
  == "item_card"`)/`choose_fallback`(`target_id` + `target_index` +
  `semantic_kind == "fallback_reward"`)/button 系(`semantic_action` 一致)の
  いずれかで、0 件・複数件・`validity=False`・信頼度未達・(button なら)
  `capability=False` を除外して1件だけを解決します。
- `mode="retry"`: 新しい snapshot が `original_snapshot` より時系列で後で
  あること、同じ `ui_state_key` であること、そして
  `perception_snapshot.is_equivalent_ui_target()` が定める全 gate(semantic
  一致・validity・信頼度・IoU ≥ 0.90・中心移動量 ≤ 0.01)を満たすことを要求
  します。1つでも欠けたら `None` を返し、二重送信を防ぎます。信頼度の閾値は
  `NavigationProfile.min_target_confidence` を呼び出し側(state machine)から
  渡します(以前はモジュール既定値 0.5 が hardcode されており、profile の
  値を無視していました)。
- retry は初回 click 送信直後の ack 待ち window
  (`NavigationProfile.retry_after_ns`、既定 200ms)を経過するまで発生しません。
  capture/perception の遅延が1フレーム(16ms)を超える実機では、初回 click を
  ゲームがまだ処理し切っていない状態で newer snapshot が届くことがあるため、
  この window を空けるまでは precondition を満たしていても retry しません。
- retry は `NavigationProfile.retry_budget`(既定 1)回まで。
- 送信済み click に対して、`ui_state_key`/`candidate_set_hash`/
  `inventory_hash` のいずれかが元の snapshot から変わっても、今 tick の
  intent identity(`candidate_set_hash`/`target_index` 等を含む)が元の
  attempt と同じなら「選んでいる対象は変わっていないノイズ」と判断し、
  apply ack の候補にしません(UI フェード中の1フレームだけの ui_state_key
  の揺れで再クリックが繰り返されるのを防ぎます)。intent identity 自体が
  実際に変わって初めて「game 側の apply ack」の**候補**とみなします(OS への
  送信受理である `ExecutionOutcome.ack` とは別物です)。さらに、同じ新しい
  組(`ui_state_key`/`candidate_set_hash`/`inventory_hash`)が
  `NavigationProfile.debounce_frames` 回**連続**で観測され(signature が
  元へ戻ったら連続カウントは破棄されます)、かつ ack 待ち window
  (`retry_after_ns`)も経過して初めて apply ack と確定します。確定するまでは
  再送も新規 click もせず待ちます。確定した場合は intent identity の変化と
  はみなさず、`ui_attempt` をクリアして今 tick の intent を新規 initial
  resolve として扱います(再送はしません)。これにより reroll/banish 成功後
  の次の選択や連続 level-up を、誤って emergency stop しません。apply ack が
  確定しないまま intent identity が変わった場合や、retry の precondition を
  満たさない場合は、再送せず emergency stop へ倒します。
- 上記のどの送信経路でも、同一 UI 訪問(LEVEL_UP/CHEST/TARGET_REACHED へ
  入ってから抜けるまで)での総送信数は 2 件(initial+retry 相当)を超えたら
  それ以上送りません。apply ack の再確定が繰り返されるような未知の経路が
  残っていた場合の最後の安全弁です。
- `CONFIRM` intent だけ ROI クリックの代わりに `ENTER` キーを使います。それ
  以外(`CHOOSE_CARD`/`CHOOSE_FALLBACK`/`REROLL`/`SKIP`/`BANISH`/`ACK_CHEST`)は
  resolve 済み ROI の中心をクリックします。

## ack と telemetry の区別

`send_ui_click`/`send_ui_key` の戻り値(`ExecutionOutcome.ack`)は OS への
送信受理を意味するだけで、ゲーム側で実際に適用されたかどうかは保証しません。
適用確認(game apply ack)は次に観測する `PerceptionSnapshot` の
`ui_state_key`/`candidate_set_hash`/`inventory_hash` の変化で行います
(前段落を参照)。`build_ui_action_telemetry()` は「何を」「どの候補/ボタンへ」
「初回か retry か」を記録する監査用ペイロードを別途組み立てますが、これも
ack とは独立な記録であり、適用成功の証明ではありません。

`combat_reset`/`controller_stop`/`process_terminate` の3種類は OS への入力
そのものではないため、`execute_effect` はこれらに対して何も実行せず
`ack=True` を返すだけです。この `ack=True` は「OS が受理した」という意味では
なく、「実行責任を呼び出し側へ委譲する通知を発行した」という意味だけです。
実際の controller 停止処理・ゲームプロセス終了処理を行う責任者は、この
PR の範囲外である 05-04 launcher です。

## 30:00 成功優先ルール

`TARGET_REACHED_PENDING_TRANSITION` に入った時点(30:00 evidence 観測)で
`success_latched=True` になります。この共通の入口(`_enter_target_reached`)は
`GAMEPLAY` から直接見えた場合だけでなく、一瞬 `UNKNOWN`(画面切り替え中の
ノイズ等)を経由してから見えた場合でも必ず通るため、経路によって
`success_latched` が立ったり立たなかったりすることはありません。

ただし、evidence を観測しただけではまだ成功を確定させません。成功が確定する
のは、その後 profile の timeout(既定 5s)以内に、次の post-30 の**確定的な**
画面変化を確認できたときだけです(それ以外に `target_reached` 画面から遷移
する現実的な行き先は無いという前提です)。

- `GAMEPLAY` へ戻ったことが確定した(post-30 confirm を押せてゲームが再開
  した) → 即座に `COMPLETE`。
- `death`/`result` へ直接移った(confirm 前に Reaper に倒された等) →
  `success_latched` を優先して `COMPLETE`。

- `LEVEL_UP`/`CHEST` が確定した場合は、`GAMEPLAY` 中の modal screen 遷移
  (`unexpected_ui_transition`)と同じ「想定外の遷移」として即座に
  **fail-closed** します(状態間の一貫性を優先)。
- `UNKNOWN`(`window_focused=False` による強制 UNKNOWN を含む)/`PAUSED` の
  間は、confirm 済みかどうかに関わらず `_dispatch_ui_click` を一切呼ばず、
  UI 入力 0 のまま timeout を待ちます(M15 fix: 以前は confirmed
  UNKNOWN/PAUSED でも timeout の 5 秒間 Enter/click を送り続けており、
  「unknown/paused/focus loss 中は入力 0」という不変条件に反していました)。

それ以外の確定的な画面変化や、post-30 event を確認できないままの timeout は、
成功を確定させず **fail-closed**
(`campaign_run_mode=formal_single_attempt` では `FORMAL_RUN_TERMINAL_FAILURE`、
`operator_debug_restart` では `DISARMED`)に倒します。

30:00 到達前の `death`/`result` は failure として扱われ、
`campaign_run_mode=formal_single_attempt` では `FORMAL_RUN_TERMINAL_FAILURE`
に、`operator_debug_restart` では `RUN_SETUP` へ戻ります(次の gameplay
attempt を許可するのは debug restart モードだけ)。

## formal terminal の確定

`COMPLETE`/`FORMAL_RUN_TERMINAL_FAILURE` は compare-and-set で一度だけ確定
します(`StateContext.terminal_locked`)。確定した瞬間だけ
`Effect(kind="release_all")` → `Effect(kind="controller_stop")` →
`Effect(kind="process_terminate")` をこの順で必ず1回発行し、以後同じ process
内で受け取る frame/decision は無視します(2回目の確定要求や `RUN_SETUP` への
巻き戻しは拒否されます)。`campaign_run_mode=formal_single_attempt` では、
一度 `GAMEPLAY` に entry した後の2回目の entry も同じ guard で拒否されます。

## グローバルな safety signal

以下は現在の状態に関係なく最優先で emergency stop します:

- `manual_abort_requested=True`(手動中断)。
- `AgentDecision.kind == "stop"`(`AgentRuntime` が観測の陳腐化・global
  validity 未達・非モデル UI policy の contract 違反など、screen state 分類
  とは無関係な致命的 safety 違反を検出したときだけ返す kind。死亡/結果/
  unknown/paused の通常フレームでは `AgentRuntime` は代わりに `"no_op"` を
  返すため、通常のシナリオでは `"stop"` を誤って過剰反応しない)。

`campaign_run_mode=formal_single_attempt` では `FORMAL_RUN_TERMINAL_FAILURE`
へ terminal 化し、`operator_debug_restart` では入力を全解放して `DISARMED`
へ戻ります。

## arm() は毎回まっさらな実行文脈を作る

`StateMachine.arm()` は `DISARMED` から `ACQUIRE_TARGET` へ遷移させる際、
`profile` 以外の実行文脈(`StateContext`)を毎回 **完全に新しく作り直し**ます。
`success_latched`/`gameplay_entries`/`death_result_reset_emitted`/
`terminal_reason`/`ui_attempt` を含め、前 run から一切引き継ぎません。これは
非formalモードで emergency stop(`DISARMED` へ戻る)した後に再 `arm()` した
とき、前 run で 30:00 に到達していた事実(`success_latched=True`)や
`GAMEPLAY` に entry 済みだった事実(`gameplay_entries>=1`)が新しい run に
紛れ込み、新しい run の pre-30 death を誤って成功にしたり、formal モードの
最初の `GAMEPLAY` entry を誤って拒否したりすることを防ぐためです。

## 非保証範囲

- 実機での座標解決精度・実際の click 命中率は本ドキュメントの範囲外です
  (D04 capture dataset 以降のキャリブレーション対象)。`Tools/Deployment/configs/ui_navigation_1080p_ja_v1.yaml`
  は timeout/retry/confidence のプロファイルのみを持ち、座標は一切含みません。
  実機検証(04-10 final verdict、05-01 live-capable runtime bundle、05-02
  safety gate)が揃うまでは、recorded fixture ベースの検証に限られます。
  05-04 の shadow gate 合格前は live input を解禁しません。
- `RUN_SETUP` からの UI 操作(メニュー/リスタート)は本 PR の範囲では
  「`GAMEPLAY` 画面分類が確定するのを待つだけ」であり、typed target を
  持たない画面要素(タイトル/メニュー由来のボタン等)への click は行いません。
- `HudState`/`hud_parser` 等の画面解析内部表現、および非モデル UI 決定ロジック
  (`NonModelUiPolicy`/`ItemSelector` 系)は import せず、`UiIntentV1` を生成・
  再分類することもありません(所有権は 05-01/05-02 側)。
- C++ 側(UE5)の変更はこの PR の範囲外です。

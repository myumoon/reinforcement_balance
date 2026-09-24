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
  します。1つでも欠けたら `None` を返し、二重送信を防ぎます。
- retry は `NavigationProfile.retry_budget`(既定 1)回まで。1 回のクリック
  試行の間に候補/inventory/state key が変わった(intent identity 変化)場合や、
  retry の precondition を満たさない場合は再送せず emergency stop へ倒します。
- `CONFIRM` intent だけ ROI クリックの代わりに `ENTER` キーを使います。それ
  以外(`CHOOSE_CARD`/`CHOOSE_FALLBACK`/`REROLL`/`SKIP`/`BANISH`/`ACK_CHEST`)は
  resolve 済み ROI の中心をクリックします。

## ack と telemetry の区別

`send_ui_click`/`send_ui_key` の戻り値(`ExecutionOutcome.ack`)は OS への
送信受理を意味するだけで、ゲーム側で実際に適用されたかどうかは保証しません。
適用確認は次に観測する `PerceptionSnapshot` の `ui_state_key`/`screen_state`
の変化で行います。`build_ui_action_telemetry()` は「何を」「どの候補/ボタンへ」
「初回か retry か」を記録する監査用ペイロードを別途組み立てますが、これも
ack とは独立な記録であり、適用成功の証明ではありません。

## 30:00 成功優先ルール

`TARGET_REACHED_PENDING_TRANSITION` に入った時点(30:00 evidence 観測)で
`success_latched=True` になります。以後は:

- 画面が(post-30 の confirm を確認できて)他カテゴリへ確定 → `COMPLETE`。
- `death`/`result` へ直接移っても(Reaper が confirm 前に倒す等) → `COMPLETE`。
- timeout してしまっても → `COMPLETE`(既定で成功扱い)。

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

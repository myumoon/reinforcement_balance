# Survivors input lease safety

この helper は controller と OS 入力注入を process 分離し、wire 上では `ActionContract` の 0〜8 semantic action、UI lease(ROIクリック/Enter/Escape)、emergency key-up だけを受け付けます。helper は action hash / ui_action_hash、target hash、session nonce、増加 sequence、単調時計期限を検証し、さらに arm・foreground PID/HWND が一致した時だけ chord/UI action を適用します。注入可能な入力は WASD、Enter、Escape、left click に閉じています。

起動時は disarmed です。対象 window を foreground にしたうえで `Ctrl+Shift+F12` を一度押すと edge-triggered に arm が反転します。押し続けても再反転しません。通常 lease は 75ms で、helper の poll loop は expiry、disarm、focus loss、PID/HWND change、IPC EOF に対して held input の解放を試みます。audit JSONL には lease event/ack と release の timestamp・sequence・理由が残ります。

## UI lease(ROIクリック・Enter・Escape)

`UiLease`(schema `survivors_input_lease_ui.v1`)は movement `Lease` と同じ session_nonce・単調増加 sequence・PID/HWND binding を共有する、最大150msの一回限り command です。許可される `ui_action` は `CLICK`/`ENTER`/`ESCAPE` の3つに閉じており、`CLICK` のときだけ 0.0〜1.0 の正規化座標 `normalized_x`/`normalized_y`(NaN/Inf・範囲外は拒否)を必須とし、`ENTER`/`ESCAPE` ではこれらの座標 field 自体の存在を拒否します。`InputLeaseController.send_ui_click(normalized_x, normalized_y)` / `send_ui_key("ENTER"|"ESCAPE")` だけを公開し、任意 VK・text・任意座標の method や wire payload は提供しません。

`CLICK` は毎回 `GetClientRect`+`ClientToScreen` で対象 window の client rect を取得して正規化座標を screen 座標へ写像し、`WindowFromPoint` で click 先の root owner window が `target_hwnd` と一致することを確認してから、mouse move + left-down + left-up を1組の atomic `SendInput` として送ります。client rect 取得失敗・縮退した矩形・`WindowFromPoint` 不一致は fail-closed に no-op します。`ENTER`/`ESCAPE` は VK down+up を1組の atomic `SendInput` として送ります。UI lease 適用直前には保持中の movement chord を必ず強制解放し、movement と UI action が同時に held 状態になることはありません。UI action 自体は held 状態を残さないため、helper の expiry/tick は UI action に対して偽の release を記録しません。

## 非保証範囲

75ms expiry と p99 100ms以下・最大150ms以下は、user-space helper process が scheduling され、テスト用 dry-run backend が動作できる fault-injection 条件で確認する観測値です。すべての fault を100ms以内に解放する保証ではありません。OS-wide freeze、kernel/driver hang、machine suspend、電源断、user32/SendInput stall、helper 自身が scheduling されない状態、Windows secure desktop/UAC、別 process による入力、物理 keyboard や hardware failure には timing 保証を提供できません。

UI click については、`WindowFromPoint` による root owner 検査と実際の `SendInput` 発行の間に TOCTOU(Time-Of-Check-Time-Of-Use)レースが存在します。検査直後に対象 window が最小化・移動・別 window に覆われても、その変化を検査だけで完全に排除することはできません(既知の残存リスクとして受容し、追加の排他制御は本フェーズの対象外です)。また、click 実行後の cursor 位置は意図的に復帰しません。復帰用の追加 `SendInput` は新たな失敗点を増やすだけで安全性を高めないため、実装しません。

`dry_run_backend.py` はテスト専用であり、production/formal/release 実行の代替・安全性証明ではありません。本家 Vampire Survivors での live validation、正式 target profile artifact hash binding、HUD/model 統合もこのフェーズの対象外です。異常時は対象を foreground から外し、`Ctrl+Shift+F12` で disarm し、controller の emergency release を実行してください。OS 自体が応答しない場合は operator/runbook による復旧が必要です。

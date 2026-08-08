# Survivors input lease safety

この helper は controller と OS 入力注入を process 分離し、wire 上では `ActionContract` の 0〜8 semantic action と emergency key-up だけを受け付けます。helper は action hash、target hash、session nonce、増加 sequence、単調時計期限を検証し、さらに arm・foreground PID/HWND が一致した時だけ chord を適用します。注入可能な入力は WASD、Enter、Escape、left click に閉じています。

起動時は disarmed です。対象 window を foreground にしたうえで `Ctrl+Shift+F12` を一度押すと edge-triggered に arm が反転します。押し続けても再反転しません。通常 lease は 75ms で、helper の poll loop は expiry、disarm、focus loss、PID/HWND change、IPC EOF に対して held input の解放を試みます。audit JSONL には lease event/ack と release の timestamp・sequence・理由が残ります。

## 非保証範囲

75ms expiry と p99 100ms以下・最大150ms以下は、user-space helper process が scheduling され、テスト用 dry-run backend が動作できる fault-injection 条件で確認する観測値です。すべての fault を100ms以内に解放する保証ではありません。OS-wide freeze、kernel/driver hang、machine suspend、電源断、user32/SendInput stall、helper 自身が scheduling されない状態、Windows secure desktop/UAC、別 process による入力、物理 keyboard や hardware failure には timing 保証を提供できません。

`dry_run_backend.py` はテスト専用であり、production/formal/release 実行の代替・安全性証明ではありません。本家 Vampire Survivors での live validation、正式 target profile artifact hash binding、HUD/model 統合もこのフェーズの対象外です。異常時は対象を foreground から外し、`Ctrl+Shift+F12` で disarm し、controller の emergency release を実行してください。OS 自体が応答しない場合は operator/runbook による復旧が必要です。

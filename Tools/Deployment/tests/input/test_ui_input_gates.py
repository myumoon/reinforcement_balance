"""UI lease(ROIクリック・Enter・Escape)のarm・focus・target gate と相互排他の受入テスト。
movementと同じgateがUI操作にも対称に効き、movement/UIの同時保持が起きないこと(I5)を確認します。
"""
from __future__ import annotations
import pytest
from survivors.input.audit_log import AuditLog
from survivors.input.dry_run_backend import DryRunBackend
from survivors.input.helper import HelperRuntime
from survivors.input.lease_protocol import Lease, LeaseValidator, UiLease, ui_action_contract_hash
def _runtime(tmp_path, *, armed: bool, focused: bool, pid: int = 42, hwnd: int = 84):
    """指定 gate 状態の helper runtime を作る。
    OS API を使わず、呼び出し回数とUI観測を確認できる backend を組み合わせます。
    """
    backend = DryRunBackend(foreground_pid=pid, foreground_hwnd=hwnd,
                            focused=focused, armed=armed)
    runtime = HelperRuntime(
        validator=LeaseValidator("1" * 32, "a" * 64, "b" * 64, 42, 84),
        backend=backend,
        audit=AuditLog(tmp_path / "audit.jsonl"),
    )
    return runtime, backend
def _movement_lease(sequence: int = 1, *, pid: int = 42, hwnd: int = 84, action: int = 0) -> Lease:
    """gate テスト用の有効な movement lease を返す。
    target binding 以外の検証条件を固定し、注入 gate だけを観察します。
    """
    return Lease(
        session_nonce="1" * 32, sequence=sequence, issued_monotonic_ns=1_000_000_000,
        expires_monotonic_ns=1_075_000_000, target_hash="a" * 64, action_hash="b" * 64,
        action_index=action, target_pid=pid, target_hwnd=hwnd,
    )
def _ui_lease(
    sequence: int = 1, *, pid: int = 42, hwnd: int = 84, ui_action: str = "CLICK",
) -> UiLease:
    """gate テスト用の有効な UiLease を返す。
    CLICK以外は座標fieldを含めず、UiLease自体の閉じた行動空間を保ちます。
    """
    coords = {"normalized_x": 0.5, "normalized_y": 0.5} if ui_action == "CLICK" else {}
    return UiLease(
        session_nonce="1" * 32, sequence=sequence, issued_monotonic_ns=1_000_000_000,
        expires_monotonic_ns=1_075_000_000, target_hash="a" * 64,
        ui_action_hash=ui_action_contract_hash(), ui_action=ui_action,
        target_pid=pid, target_hwnd=hwnd, **coords,
    )
@pytest.mark.parametrize(
    "armed, focused, lease_pid, lease_hwnd, foreground_pid, foreground_hwnd",
    [
        (False, True, 42, 84, 42, 84),
        (True, False, 42, 84, 42, 84),
        (True, True, 43, 84, 42, 84),
        (True, True, 42, 85, 42, 84),
        (True, True, 42, 84, 43, 84),
        (True, True, 42, 84, 42, 85),
    ],
)
def test_unsafe_gate_never_calls_send_input_for_ui_lease(
    tmp_path,
    armed: bool,
    focused: bool,
    lease_pid: int,
    lease_hwnd: int,
    foreground_pid: int,
    foreground_hwnd: int,
) -> None:
    """disarm・非focus・PID/HWND不一致ではUI actionでもSendInput相当を0件にする。
    movementと同じ6条件をUiLeaseへ適用し、gateが対称に効くことを確認します(I6の前提)。
    """
    runtime, backend = _runtime(
        tmp_path, armed=armed, focused=focused, pid=foreground_pid, hwnd=foreground_hwnd,
    )
    runtime.handle_lease(_ui_lease(pid=lease_pid, hwnd=lease_hwnd), now_ns=1_001_000_000)
    assert backend.send_input_calls == 0
def test_ui_action_never_leaves_active_movement_state_and_forces_release(tmp_path) -> None:
    """UiLease適用は`_active_*`を汚さず(I3)、事前に保持中のmovement chordを解放する(I5)。
    action_index=0(W押下を伴うchord)を先に適用してからCLICKを送り、pressedが空になることを見ます。
    """
    runtime, backend = _runtime(tmp_path, armed=True, focused=True)
    movement_ack = runtime.handle_lease(_movement_lease(sequence=1, action=0), now_ns=1_001_000_000)
    assert movement_ack["kind"] == "ack" and movement_ack["applied"] is True
    assert backend.pressed, "movement chord must be held before the UI lease arrives"
    ui_ack = runtime.handle_lease(_ui_lease(sequence=2), now_ns=1_002_000_000)
    assert ui_ack["kind"] == "ack" and ui_ack["applied"] is True
    assert backend.pressed == set(), "UI lease must force-release the held movement chord"
    # UI actionは保持状態を残さないため、helperのactive stateはNoneのまま(I3)
    assert runtime._active_sequence is None
    assert runtime._active_expiry_ns is None
    assert runtime._active_target is None
def test_ui_action_applies_before_movement_lease_does_not_double_hold(tmp_path) -> None:
    """UI適用直後にmovement leaseが来ても、同時保持は発生しない(I5・逆方向)。
    CLICK適用(保持なし)の直後にW押下movementを送るとpressedはW単独になることを確認します。
    """
    runtime, backend = _runtime(tmp_path, armed=True, focused=True)
    runtime.handle_lease(_ui_lease(sequence=1), now_ns=1_001_000_000)
    assert backend.pressed == set()
    runtime.handle_lease(_movement_lease(sequence=2, action=0), now_ns=1_002_000_000)
    assert backend.pressed, "movement lease after UI action must still apply normally"
def test_click_no_ops_when_client_rect_is_unavailable(tmp_path) -> None:
    """client rect取得に失敗している状況ではCLICKをno-opにする(M7 fail-closed)。
    gateは開いていてもSendInput相当を1件も呼ばないことを確認します。
    """
    runtime, backend = _runtime(tmp_path, armed=True, focused=True)
    backend.client_rect = None
    result = runtime.handle_lease(_ui_lease(), now_ns=1_001_000_000)
    assert result["applied"] is True  # gateは開いているが、backend内部でno-opする
    assert backend.send_input_calls == 0
    assert backend.ui_calls == []
def test_click_no_ops_when_window_from_point_owner_mismatches_target(tmp_path) -> None:
    """click直前のWindowFromPoint検査がtarget_hwndと不一致ならno-opにする(I6)。
    gate通過後でも、別windowへ意図せずclickが飛ぶ状況を最終防御として遮断します。
    """
    runtime, backend = _runtime(tmp_path, armed=True, focused=True)
    backend.window_from_point_hwnd = 999
    result = runtime.handle_lease(_ui_lease(), now_ns=1_001_000_000)
    assert result["applied"] is True
    assert backend.send_input_calls == 0
    assert backend.ui_calls == []
def test_enter_and_escape_do_not_require_client_rect(tmp_path) -> None:
    """ENTER/ESCAPEは座標を使わないため、client rect失敗の影響を受けない。
    座標系検査はCLICKだけに閉じていることを確認します。
    """
    runtime, backend = _runtime(tmp_path, armed=True, focused=True)
    backend.client_rect = None
    result = runtime.handle_lease(_ui_lease(ui_action="ENTER"), now_ns=1_001_000_000)
    assert result["applied"] is True
    assert backend.send_input_calls == 1
@pytest.mark.parametrize(
    "rect, nx, ny, expected",
    [
        ((0, 0, 800, 600), 0.0, 0.0, (0, 0)),
        ((0, 0, 800, 600), 0.5, 0.5, (400, 300)),
        ((0, 0, 800, 600), 1.0, 1.0, (799, 599)),
        ((-1920, -200, -1120, 400), 0.0, 0.0, (-1920, -200)),
        ((-1920, -200, -1120, 400), 1.0, 1.0, (-1121, 399)),
        ((100, 50, 101, 51), 1.0, 1.0, (100, 50)),
    ],
)
def test_map_normalized_to_screen_stays_inside_client_rect(rect, nx, ny, expected) -> None:
    """ROI座標→screen座標の写像が、負原点のモニタや1px矩形でも常にclient rect内へ収まる(M14)。
    実機クリック位置を決める唯一の式なので、端点と多重モニタ配置を固定値で検査します。
    """
    from survivors.input.win32_backend import map_normalized_to_screen
    x, y = map_normalized_to_screen(rect, nx, ny)
    assert (x, y) == expected
    left, top, right, bottom = rect
    assert left <= x < right and top <= y < bottom
def test_dry_run_click_records_the_shared_mapping_result(tmp_path) -> None:
    """dry-run backendのCLICK記録座標がproduction写像関数の結果と一致する(式の複製防止)。
    両backendが同じ関数を使っていることを、非原点のclient rectで観測します。
    """
    from survivors.input.win32_backend import map_normalized_to_screen
    runtime, backend = _runtime(tmp_path, armed=True, focused=True)
    backend.client_rect = (1000, 200, 1640, 680)
    runtime.handle_lease(_ui_lease(), now_ns=1_001_000_000)
    call = backend.ui_calls[-1]
    assert (call["screen_x"], call["screen_y"]) == map_normalized_to_screen(backend.client_rect, 0.5, 0.5)

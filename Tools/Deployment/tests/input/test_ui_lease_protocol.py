"""UiLease(ROIクリック・Enter・Escape)の wire protocol と再送防止 validator の受入テスト。
movement用Leaseと同じ検証パターン(nonce・順序・期限・hash)がUI操作にも対称に効くことを確認します。
"""
from __future__ import annotations
import math
import pytest
from survivors.input.lease_protocol import (
    Lease,
    LeaseValidator,
    UiLease,
    ui_action_contract_hash,
)
def _ui_lease(**changes: object) -> UiLease:
    """正しいCLICK UiLeaseを作り、異常系で必要な項目だけ差し替える。
    各拒否条件をほかの条件から独立して試せるようにするテスト用の組み立て関数です。
    """
    values: dict[str, object] = {
        "session_nonce": "1" * 32,
        "sequence": 1,
        "issued_monotonic_ns": 1_000_000_000,
        "expires_monotonic_ns": 1_075_000_000,
        "target_hash": "a" * 64,
        "ui_action_hash": ui_action_contract_hash(),
        "ui_action": "CLICK",
        "target_pid": 42,
        "target_hwnd": 84,
        "normalized_x": 0.5,
        "normalized_y": 0.5,
    }
    values.update(changes)
    return UiLease(**values)
@pytest.mark.parametrize(
    "first, candidate, now_ns",
    [
        (None, {"session_nonce": "2" * 32}, 1_001_000_000),
        (_ui_lease(sequence=3), {"sequence": 2}, 1_001_000_000),
        (None, {"expires_monotonic_ns": 1_000_000_000}, 1_000_000_000),
        (None, {"target_hash": "c" * 64}, 1_001_000_000),
        (None, {"ui_action_hash": "d" * 64}, 1_001_000_000),
        (None, {"target_pid": 43}, 1_001_000_000),
        (None, {"target_hwnd": 85}, 1_001_000_000),
    ],
)
def test_rejects_every_ui_lease_binding_failure(
    first: UiLease | None, candidate: dict[str, object], now_ns: int
) -> None:
    """nonce・順序・期限・target hash・ui_action_hashの不一致をfail-closedにする。
    movement Leaseと同じ7つの反例をUiLeaseへ与え、どれも検証を通過しないことを確認します。
    """
    validator = LeaseValidator("1" * 32, "a" * 64, "b" * 64, 42, 84)
    if first is not None:
        validator.accept(first, 1_001_000_000)
    with pytest.raises(ValueError):
        validator.accept(_ui_lease(**candidate), now_ns)
def test_click_wire_schema_is_closed_and_rejects_extra_fields() -> None:
    """CLICKのwire payloadに未知fieldや余分なキーを許可しない。
    座標を持つCLICKでも閉じた集合であることを、任意キー混入で確認します。
    """
    data = _ui_lease().to_wire()
    assert set(data) == {
        "kind", "schema_version", "session_nonce", "sequence",
        "issued_monotonic_ns", "expires_monotonic_ns", "target_hash",
        "ui_action_hash", "ui_action", "target_pid", "target_hwnd",
        "normalized_x", "normalized_y",
    }
    for key in ("vk", "text", "shortcut", "absolute_click"):
        mutated = dict(data)
        mutated[key] = "unsafe"
        with pytest.raises(ValueError):
            UiLease.from_wire(mutated)
def test_enter_escape_reject_coordinate_fields() -> None:
    """ENTER/ESCAPEには座標fieldの存在自体を許可しない。
    CLICK専用の`normalized_x`/`normalized_y`がENTER/ESCAPEへ混入する抜け道を塞ぎます。
    """
    for action in ("ENTER", "ESCAPE"):
        with pytest.raises(ValueError):
            UiLease(
                session_nonce="1" * 32, sequence=1, issued_monotonic_ns=1_000_000_000,
                expires_monotonic_ns=1_075_000_000, target_hash="a" * 64,
                ui_action_hash=ui_action_contract_hash(), ui_action=action,
                target_pid=42, target_hwnd=84, normalized_x=0.5, normalized_y=0.5,
            )
        wire = {
            "kind": "ui_lease", "schema_version": "survivors_input_lease_ui.v1",
            "session_nonce": "1" * 32, "sequence": 1, "issued_monotonic_ns": 1_000_000_000,
            "expires_monotonic_ns": 1_075_000_000, "target_hash": "a" * 64,
            "ui_action_hash": ui_action_contract_hash(), "ui_action": action,
            "target_pid": 42, "target_hwnd": 84,
            "normalized_x": 0.5, "normalized_y": 0.5,
        }
        with pytest.raises(ValueError):
            UiLease.from_wire(wire)
def test_click_boundary_normalized_coordinates_are_accepted() -> None:
    """0.0/1.0という境界値のROI座標を有効として受理する。
    範囲チェックが `<` ではなく `<=` で書かれていることを両端で確認します。
    """
    for x, y in ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)):
        lease = _ui_lease(normalized_x=x, normalized_y=y)
        assert lease.normalized_x == x and lease.normalized_y == y
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.001, 1.001, 0, "0.5"])
def test_click_rejects_non_finite_or_out_of_range_or_wrong_type_coordinates(bad: object) -> None:
    """NaN/Inf・範囲外・int/strの型混入をすべてCLICK座標として拒否する。
    画面外座標や型混同によるOS API誤呼び出しを事前に防ぎます。
    """
    with pytest.raises(ValueError):
        _ui_lease(normalized_x=bad)
    with pytest.raises(ValueError):
        _ui_lease(normalized_y=bad)
def test_ui_lease_and_movement_lease_share_one_monotonic_sequence() -> None:
    """UiLeaseとLeaseが同一validatorの単調増加sequenceを共有する。
    種別ごとに別カウンタを持たないため、混在した送信順でもreplayを検出できます。
    """
    validator = LeaseValidator("1" * 32, "a" * 64, "b" * 64, 42, 84)
    movement = Lease(
        session_nonce="1" * 32, sequence=1, issued_monotonic_ns=1_000_000_000,
        expires_monotonic_ns=1_075_000_000, target_hash="a" * 64, action_hash="b" * 64,
        action_index=0, target_pid=42, target_hwnd=84,
    )
    validator.accept(movement, 1_001_000_000)
    # 同じsequence=1をUiLeaseで再送しても拒否される(種別を跨いだreplay防止)
    with pytest.raises(ValueError):
        validator.accept(_ui_lease(sequence=1), 1_002_000_000)
    # sequenceを進めればUiLeaseとして受理される
    validator.accept(_ui_lease(sequence=2), 1_002_000_000)
    # 続くmovement Leaseもsequence=2以下は拒否される
    with pytest.raises(ValueError):
        validator.accept(
            Lease(
                session_nonce="1" * 32, sequence=2, issued_monotonic_ns=1_003_000_000,
                expires_monotonic_ns=1_078_000_000, target_hash="a" * 64, action_hash="b" * 64,
                action_index=1, target_pid=42, target_hwnd=84,
            ),
            1_003_000_000,
        )
def test_ui_lease_duration_cap_matches_movement_lease() -> None:
    """UiLeaseのTTL上限もLeaseと同じ150msで、緩いTTLを許容しない。
    150msちょうどは合格、150msを1nsでも超えると拒否されることを確認します。
    """
    ok = _ui_lease(expires_monotonic_ns=1_000_000_000 + 150_000_000)
    assert ok.expires_monotonic_ns - ok.issued_monotonic_ns == 150_000_000
    with pytest.raises(ValueError):
        _ui_lease(expires_monotonic_ns=1_000_000_000 + 150_000_001)

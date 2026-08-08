"""semantic lease と action golden parity の受入テスト。
古い命令や別ターゲットの命令を必ず拒否し、9方向の意味が既存の正解表と完全一致するか確認します。
"""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from survivors.action_semantics import load_action_contract
from survivors.input.lease_protocol import Lease, LeaseValidator, action_golden_bytes
def _lease(**changes: object) -> Lease:
    """正しい lease を作り、異常系で必要な項目だけ差し替える。
    各拒否条件をほかの条件から独立して試せるようにするテスト用の組み立て関数です。
    """
    values: dict[str, object] = {
        "session_nonce": "1" * 32,
        "sequence": 1,
        "issued_monotonic_ns": 1_000_000_000,
        "expires_monotonic_ns": 1_075_000_000,
        "target_hash": "a" * 64,
        "action_hash": "b" * 64,
        "action_index": 0,
        "target_pid": 42,
        "target_hwnd": 84,
    }
    values.update(changes)
    return Lease(**values)
@pytest.mark.parametrize(
    "first, candidate, now_ns",
    [
        (None, {"session_nonce": "2" * 32}, 1_001_000_000),
        (_lease(sequence=3), {"sequence": 2}, 1_001_000_000),
        (None, {"expires_monotonic_ns": 1_000_000_000}, 1_000_000_000),
        (None, {"target_hash": "c" * 64}, 1_001_000_000),
        (None, {"action_hash": "d" * 64}, 1_001_000_000),
        (None, {"target_pid": 43}, 1_001_000_000),
        (None, {"target_hwnd": 85}, 1_001_000_000),
    ],
)
def test_rejects_every_lease_binding_failure(
    first: Lease | None, candidate: dict[str, object], now_ns: int
) -> None:
    """nonce・順序・期限・2種類の hash の不一致を fail-closed にする。
    M1 の5つの反例を個別に与え、どれも検証を通過しないことを確認します。
    """
    validator = LeaseValidator("1" * 32, "a" * 64, "b" * 64, 42, 84)
    if first is not None:
        validator.accept(first, 1_001_000_000)
    with pytest.raises(ValueError):
        validator.accept(_lease(**candidate), now_ns)
def test_wire_schema_is_closed_and_semantic_only() -> None:
    """wire payload に未知 field や任意キー指定を許可しない。
    controller から helper へ渡せるものを action index に限定し、任意 VK の抜け道を防ぎます。
    """
    data = _lease().to_wire()
    assert set(data) == {"kind", "schema_version", "session_nonce", "sequence",
                         "issued_monotonic_ns", "expires_monotonic_ns", "target_hash",
                         "action_hash", "action_index", "target_pid", "target_hwnd"}
    for key in ("vk", "text", "shortcut", "absolute_click"):
        mutated = dict(data)
        mutated[key] = "unsafe"
        with pytest.raises(ValueError):
            Lease.from_wire(mutated)
def test_m1_ac_action_golden_parity_is_byte_identical() -> None:
    """ActionContract から作る9 chord が golden の keys と byte 単位で一致する。
    sim vector の並びも同時に含めることで、方向だけ入れ替わる drift も検知します。
    """
    contract = load_action_contract()
    fixture_path = Path(__file__).parents[2] / "configs" / "golden" / "action_displacement_wasd_v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected = json.dumps(
        [
            {
                "index": sample["index"],
                "sim_vector": sample["sim_delta_sign"],
                "wasd_chord": sample["keys"],
            }
            for sample in fixture["samples"]
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    assert action_golden_bytes(contract) == expected

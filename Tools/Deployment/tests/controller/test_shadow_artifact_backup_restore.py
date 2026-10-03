"""shadow artifact を backup から restore し、stage order・choice support・helper safety refs を再検証する(M13)。

fixture replay(shadow)と短い live run の telemetry を ArtifactStore へ登録し、store を丸ごと
backup へコピーしてから元 store を削除します。backup から開き直した新しい ArtifactStore だけを使って
成果物を読み戻し、次の3点を restore 後の telemetry から計算し直して元の値と一致することを確認します。
stage order は同じ frame の stage 行が controller の書き込み順に並ぶこと、choice support は UI intent の
kind が既知 taxonomy(UiIntentKind)の範囲内であること、helper safety refs は live run の入力解放記録です。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import hashlib
import platform
import shutil
from pathlib import Path
from typing import Any

import pytest

from reinbalance_survivors_contracts.artifact_identity import ArtifactRef
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.ui_intent import UiIntentKind
from survivors.controller.controller import SurvivorsController
from survivors.controller.gate import read_telemetry
from survivors.controller.health_monitor import HealthMonitor
from survivors.controller.state_machine import CampaignRunMode, StateMachine
from survivors.controller.telemetry import TelemetrySessionHeader, TelemetryWriter
from survivors.real_obs_assembler import RealObsAssembler
from survivors.runtime.agent_runtime import AgentRuntime
from survivors.runtime.artifact_bundle import RuntimeBundle
from survivors.vision.entity_tracker import EntityTracker, default_class_map

import replay_controller_fixture as replay

from .test_run_survivors_controller import FakeLease

# 短い台本: gameplay → level_up(choose_card が出る) → gameplay → chest → gameplay。
_CYCLE: tuple[tuple[str, int], ...] = (
    ("gameplay", 40), ("level_up_items", 15), ("gameplay", 20), ("chest", 15), ("gameplay", 20),
)
_TICKS = sum(length for _, length in _CYCLE)
# controller.py の _process_frame / _apply_effect が1 frame 内で書く順序(唯一の正)。
_FRAME_STAGE_ORDER = ("capture", "detector", "tracker", "hud_parser", "obs", "policy", "state_machine", "effect")
_SHUTDOWN_TAIL = ["capture_stop", "queue_drain", "input_release", "shutdown"]
_SUPPORTED_UI_KINDS = frozenset(kind.value for kind in UiIntentKind)


def _stage_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """telemetry 行から stage event だけを取り出す(session header は除く)。"""
    return [row for row in rows if row.get("event") == "stage"]


def stage_order_violations(rows: Iterable[Mapping[str, Any]], control_id: str) -> list[str]:
    """同じ correlation_id の stage 行が controller の書き込み順になっていない箇所を列挙する。

    frame ごとの行は capture で始まり、_FRAME_STAGE_ORDER の順位が減らないこと、effect 以外は
    1回ずつであることを確かめます。controller 用の行は arm で始まり shutdown 手順で終わることを確かめます。
    空リストなら順序に問題はありません。
    """
    rank = {stage: i for i, stage in enumerate(_FRAME_STAGE_ORDER)}
    by_cid: dict[str, list[str]] = {}
    for row in _stage_rows(rows):
        by_cid.setdefault(row["correlation_id"], []).append(row["stage"])
    violations = []
    for cid, stages in by_cid.items():
        if cid == control_id:
            if stages[:1] != ["arm"] or stages[-len(_SHUTDOWN_TAIL):] != _SHUTDOWN_TAIL:
                violations.append(f"{cid}: {stages}")
            continue
        ranks = [rank.get(stage, -1) for stage in stages]
        repeated = [s for s, n in Counter(stages).items() if n > 1 and s != "effect"]
        if stages[0] != "capture" or -1 in ranks or ranks != sorted(ranks) or repeated:
            violations.append(f"{cid}: {stages}")
    return violations


def ui_kinds(rows: Iterable[Mapping[str, Any]]) -> Counter:
    """policy 行の ui_intent.kind と effect 行の ui_action.kind を (stage, kind) で数える。"""
    kinds: Counter = Counter()
    for row in _stage_rows(rows):
        payload = row.get("payload") or {}
        field = {"policy": "ui_intent", "effect": "ui_action"}.get(row["stage"])
        if field and payload.get(field):
            kinds[(row["stage"], payload[field]["kind"])] += 1
    return kinds


def safety_refs(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """input_release 行の payload と shutdown 行の steps/exit_code を取り出す。"""
    rows = _stage_rows(rows)
    shutdown = [row["payload"] for row in rows if row["stage"] == "shutdown"]
    return {
        "input_release": [row["payload"] for row in rows if row["stage"] == "input_release"],
        "shutdown_steps": [p["steps"] for p in shutdown],
        "shutdown_exit_code": [p["exit_code"] for p in shutdown],
    }


def recheck(rows: list[Mapping[str, Any]], session_id: str) -> dict[str, Any]:
    """M13 の3項目を telemetry 行から計算し直した結果をまとめて返す。"""
    kinds = ui_kinds(rows)
    return {
        "stage_order_violations": stage_order_violations(rows, f"{session_id}:controller"),
        "ui_kinds": kinds,
        "unsupported_ui": sorted({kind for _, kind in kinds} - _SUPPORTED_UI_KINDS),
        "safety": safety_refs(rows),
    }


def _run_live(output_dir: Path, session_id: str) -> Path:
    """replay の fake/実部品で live mode の controller を短く流し、telemetry の path を返す。

    shadow mode は入力 adapter を持てないため、入力解放記録(helper safety refs)は live run で作ります。
    入力 adapter は OS へ何も送らない FakeLease です。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = output_dir / "telemetry.jsonl"
    policy = replay.build_combat_policy(0)
    model_hashes = {"combat_policy": hashlib.sha256(b"m13-live-policy").hexdigest()}
    bundle = RuntimeBundle.from_golden_fixture(policy, item_selector=replay.FirstCardItemSelector())
    clock = replay.ReplayClock()
    telemetry = TelemetryWriter(telemetry_path, TelemetrySessionHeader(
        session_id=session_id, mode="live", target_profile_hash="0" * 64, game_build_id="m13-live",
        controller_build_id="m13-live", artifact_hashes=model_hashes,
        host={"platform": platform.platform()}, device={"inference": "cpu"},
        dependency_versions={"python": platform.python_version()},
    ))
    controller = SurvivorsController(
        mode="live", session_id=session_id, capture=replay.ScriptedCapture(clock, sample_every=10**9),
        detector=replay.FixedSceneDetector(), tracker=EntityTracker({i: 5 for i in range(default_class_map().num_classes)}, 0.7, 0.6, 0.9, coarse_by_class_id=default_class_map().coarse_by_class_id()),
        hud_parser=replay.ScriptedHudParser(_CYCLE), assembler=RealObsAssembler(),
        runtime=AgentRuntime(bundle, clock_ns=clock), state_machine=StateMachine(), health=HealthMonitor(),
        telemetry=telemetry, schema=bundle.deploy_schema, model_hashes=model_hashes,
        ui_config=bundle.ui_policy_config, input_controller=FakeLease(), clock_ns=clock, sleep=lambda _: None,
    )
    controller.arm(
        campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART,
        run_id=session_id, gameplay_attempt_id=f"{session_id}-attempt-1",
    )
    assert controller.run(max_frames=_TICKS) == 0, controller.errors
    return telemetry_path


@pytest.fixture(scope="module")
def restored(tmp_path_factory):
    """shadow/live の telemetry を store へ登録 → backup へコピー → 元 store と出力を削除 → backup から開き直す。

    返り値は元 telemetry から計算した再検証結果・登録時の ArtifactRef・restore 後の ArtifactStore です。
    元の store と出力 dir は削除済みなので、restore 後の読み出しは backup からしか行えません。
    """
    base = tmp_path_factory.mktemp("m13")
    store_a, backup_b = base / "store_a", base / "backup_b"
    shadow = replay.run_replay(base / "shadow", ticks=_TICKS, sample_every=50, session_id="m13-shadow",
                               store_root=store_a, cycle=_CYCLE)
    assert shadow.exit_code == 0 and shadow.manifest["restore_verified"] is True
    live_path = _run_live(base / "live", "m13-live")

    store = ArtifactStore(store_a)
    live_ref = store.put_bytes(logical_id="controller_live/m13-live/telemetry.jsonl", data=live_path.read_bytes(),
                               media_type="application/x-ndjson")
    shadow_ref = next(ArtifactRef.from_wire(item["ref"]) for item in shadow.manifest["artifacts"]
                      if item["ref"]["logical_id"].endswith("/telemetry.jsonl"))
    originals = {
        "m13-shadow": recheck(list(read_telemetry(shadow.telemetry_path)), "m13-shadow"),
        "m13-live": recheck(list(read_telemetry(live_path)), "m13-live"),
    }

    shutil.copytree(store_a, backup_b)
    shutil.rmtree(store_a)
    shutil.rmtree(base / "shadow")
    shutil.rmtree(base / "live")
    assert not store_a.exists()
    return originals, {"m13-shadow": shadow_ref, "m13-live": live_ref}, ArtifactStore(backup_b), base


def _restored_rows(store: ArtifactStore, ref: ArtifactRef, scratch: Path) -> list[dict[str, Any]]:
    """backup store から telemetry を verify・resolve して読み戻し、行の list にする。"""
    assert store.verify(ref, expected_size_bytes=ref.size_bytes).ok
    assert store.resolve(ref.logical_id) == ref
    path = scratch / f"{ref.sha256}.jsonl"
    path.write_bytes(store.object_path(ref.store_uri).read_bytes())
    return list(read_telemetry(path))


@pytest.mark.parametrize("session_id", ["m13-shadow", "m13-live"])
def test_restored_telemetry_rechecks_to_original_values(restored, session_id) -> None:
    """backup から restore した telemetry で3項目を計算し直すと、元の値と完全に一致する。"""
    originals, refs, store, base = restored
    assert str(store.root).startswith(str((base / "backup_b").resolve()))
    again = recheck(_restored_rows(store, refs[session_id], base), session_id)
    assert again == originals[session_id]


def test_restored_shadow_passes_stage_order_and_choice_support(restored) -> None:
    """restore 後の shadow telemetry は stage order 違反0件・support 外 UI 0件で、choose_card を実際に含む。

    UI の kind が1件も無いと「support 外0件」が素通りになるため、level_up の choose_card が出ていることも確かめます。
    """
    _, refs, store, base = restored
    result = recheck(_restored_rows(store, refs["m13-shadow"], base), "m13-shadow")
    assert result["stage_order_violations"] == []
    assert result["unsupported_ui"] == []
    assert result["ui_kinds"][("policy", "choose_card")] >= 1
    assert result["ui_kinds"][("effect", "choose_card")] >= 1
    # shadow は入力 adapter を持たないので、解放記録は released=None。
    assert result["safety"]["input_release"] == [{"released": None}]


def test_restored_live_keeps_helper_safety_refs(restored) -> None:
    """restore 後の live telemetry に入力解放成功と shutdown 手順(input_release を含む)が残っている。"""
    _, refs, store, base = restored
    result = recheck(_restored_rows(store, refs["m13-live"], base), "m13-live")
    assert result["stage_order_violations"] == []
    assert result["unsupported_ui"] == []
    (release,) = result["safety"]["input_release"]
    # release_*_ns は helper audit と相関するための実時計値なので、キーの存在だけを固定する。
    assert release["released"] is True
    assert set(release) == {"released", "release_timestamp_ns", "release_monotonic_ns"}
    assert result["safety"]["shutdown_steps"] == [["capture_stop", "queue_drain", "input_release"]]
    assert result["safety"]["shutdown_exit_code"] == [0]


def test_recheck_detects_reordered_stages_and_unsupported_kind() -> None:
    """再検証 helper が、順序の入れ替わり・未知の UI kind・shutdown 手順の欠落を見逃さない。"""
    def row(stage: str, cid: str, payload: dict | None = None) -> dict[str, Any]:
        return {"event": "stage", "stage": stage, "correlation_id": cid, "payload": payload or {}}

    control = [row("arm", "s:controller"), row("capture_stop", "s:controller"), row("queue_drain", "s:controller"),
               row("input_release", "s:controller", {"released": True}),
               row("shutdown", "s:controller", {"steps": ["input_release"], "exit_code": 0})]
    good = [row(s, "s:0") for s in _FRAME_STAGE_ORDER[:5]] + [
        row("policy", "s:0", {"ui_intent": {"kind": "choose_card"}}), row("state_machine", "s:0"),
        row("effect", "s:0", {"ui_action": {"kind": "choose_card"}}), row("effect", "s:0"),
    ]
    assert recheck(good + control, "s") == {
        "stage_order_violations": [],
        "ui_kinds": Counter({("policy", "choose_card"): 1, ("effect", "choose_card"): 1}),
        "unsupported_ui": [],
        "safety": {"input_release": [{"released": True}], "shutdown_steps": [["input_release"]],
                   "shutdown_exit_code": [0]},
    }
    swapped = good[:1] + [good[2], good[1]] + good[3:]
    assert recheck(swapped + control, "s")["stage_order_violations"] == ["s:0: " + str(
        ["capture", "tracker", "detector", "hud_parser", "obs", "policy", "state_machine", "effect", "effect"])]
    unknown = good + [row("policy", "s:1", {"ui_intent": {"kind": "open_shop"}})]
    assert recheck(unknown + control, "s")["unsupported_ui"] == ["open_shop"]
    assert recheck(good + control[:-2], "s")["stage_order_violations"] == ["s:controller: "
                                                                          "['arm', 'capture_stop', 'queue_drain']"]

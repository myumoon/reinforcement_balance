"""``replay_controller_fixture.py`` の smoke test(M9・I4)。

30分スケジュールと同じ1周期(900 tick = 約1分)だけを実 SurvivorsController へ流し、
controller が正常終了すること・gate verdict が PASS になること・report が書かれること・
ArtifactStore へ登録した成果物を別インスタンスから読み戻し検証できること・
manifest が development_only=true で formal shadow の代替にならないことを確認します。
"""

from __future__ import annotations

from collections import Counter
import json

import pytest

from reinbalance_survivors_contracts.artifact_identity import ArtifactRef
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from survivors.controller.gate import read_telemetry

import replay_controller_fixture as replay

_SMOKE_TICKS = sum(length for _, length in replay.DEFAULT_CYCLE)  # 1周期 = 900 tick


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    """1周期分の fixture replay を1回だけ実行して結果を共有する。

    実 runtime/assembler を通すため数十秒かかるので、module 内の test で使い回します。
    """
    out = tmp_path_factory.mktemp("fixture_replay")
    return replay.run_replay(out, ticks=_SMOKE_TICKS, sample_every=150, session_id="smoke-replay")


def test_full_schedule_is_thirty_minutes_by_default() -> None:
    """CLI の既定 tick 数が 30分 x 15 Hz で、台本が必要な画面を1周期に含む。

    smoke は短縮実行なので、既定値側が本当に30分相当であることをここで固定します。
    """
    args = replay.build_parser().parse_args(["--output-dir", "out"])
    assert args.ticks == replay.FULL_SCHEDULE_TICKS == 27_000
    assert args.ticks * replay.TICK_NS >= 30 * 60 * 1_000_000_000
    states = {replay.screen_state_at(tick) for tick in range(_SMOKE_TICKS)}
    assert {"gameplay", "level_up_items", "chest", "death"} <= states


def test_replay_runs_real_pipeline_and_gate_passes(replayed) -> None:
    """実 state machine が level_up・chest・death を経て巡回し、clean run の gate が PASS になる。

    level-up 候補の母数が0だと invalid rate の判定が素通りになるため、母数1以上も確認します。
    """
    assert replayed.exit_code == 0 and replayed.exit_reason == "max_frames"
    assert replayed.errors == []
    verdict = replayed.verdict
    assert verdict["status"] == "PASS", verdict["fail_reasons"]
    assert verdict["formal_shadow_eligible"] is False
    metrics = verdict["metrics"]
    assert metrics["level_up_candidate_eligible"] >= 1
    assert metrics["wrong_state_action_proposal_count"] == 0
    assert metrics["end_to_end_samples"] > 0 and metrics["memory_samples"] >= 2

    transitions, effects = Counter(), Counter()
    for row in read_telemetry(replayed.telemetry_path):
        if row.get("stage") == "state_machine":
            transitions[(row["payload"]["from_state"], row["payload"]["to_state"])] += 1
        elif row.get("stage") == "effect":
            effects[(row["payload"]["kind"], row["payload"]["disposition"])] += 1
    for edge in [("gameplay", "level_up"), ("level_up", "gameplay"), ("gameplay", "chest"),
                 ("chest", "gameplay"), ("gameplay", "death_result"), ("death_result", "run_setup")]:
        assert transitions[edge] >= 1, edge
    assert effects[("move", "proposed")] > 0
    assert effects[("ui_click", "proposed")] >= 1
    assert effects[("combat_reset", "handled")] >= 1
    # shadow なので入力は一度も実行されない(M3)。
    assert not any(disposition == "executed" for _, disposition in effects)


def test_report_and_manifest_mark_development_only(replayed) -> None:
    """verdict の JSON/Markdown report と manifest が書かれ、開発用 smoke であることを明記する(I4)。"""
    out = replayed.manifest_path.parent
    on_disk = json.loads((out / "gate_verdict.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "PASS" and on_disk["formal_shadow_eligible"] is False
    assert "not a formal shadow verdict" in (out / "gate_verdict.md").read_text(encoding="utf-8")
    manifest = json.loads(replayed.manifest_path.read_text(encoding="utf-8"))
    assert manifest == replayed.manifest
    assert manifest["development_only"] is True
    assert manifest["formal_shadow_eligible"] is False
    assert manifest["gate_status"] == "PASS"
    assert manifest["schedule"]["full_schedule_ticks"] == replay.FULL_SCHEDULE_TICKS
    assert manifest["schedule"]["ticks"] == _SMOKE_TICKS


def test_artifacts_restore_from_a_fresh_store(replayed) -> None:
    """ArtifactStore へ登録した4成果物を、新しい store インスタンスから verify して中身も一致する。"""
    out = replayed.manifest_path.parent
    manifest = replayed.manifest
    assert manifest["restore_verified"] is True
    store = ArtifactStore(out / "artifact_store")
    names = set()
    for item in manifest["artifacts"]:
        ref = ArtifactRef.from_wire(item["ref"])
        assert store.verify(ref, expected_size_bytes=ref.size_bytes).ok
        assert store.resolve(ref.logical_id) == ref
        name = ref.logical_id.rsplit("/", 1)[1]
        names.add(name)
        assert store.object_path(ref.store_uri).read_bytes() == (out / name).read_bytes()
    assert names == {"telemetry.jsonl", "gate_verdict.json", "gate_verdict.md", "memory_samples.json"}

"""recorded session E2E replay engine と CLI のテスト(06-01 タスク2)。

synthetic session を実 controller(実 assembler・runtime・state machine・health・telemetry)へ流し、
capture manifest/profile/build/決定性の照合、仮想時計への時刻と correlation id の統一、
effect recorder の semantic JSON、discrete/numeric の分離、同一入力での再現性、
実時計と live input backend に触れないことを確かめます。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

import replay_controller_fixture as fixture
import replay_survivors_session as cli
from survivors.controller import controller as controller_module
from survivors.controller.state_machine import CampaignRunMode
from survivors.input.controller import InputLeaseController
from survivors.replay.e2e_replay import (
    CaptureManifest,
    ReplayIntegrityError,
    formal_parents_eligible,
    run_recorded_replay,
    split_payload,
    verify_virtual_telemetry,
)
from survivors.replay.recorded_frame_source import DeterminismManifest, DeterminismMismatchError, RecordedSession
from survivors.runtime.artifact_bundle import RuntimeBundle
from survivors.target_profile import load_target_profile
from survivors.vision.entity_tracker import EntityTracker

SESSION = "e2e-sess"
PROFILE_HASH = "d" * 64
BUILD_ID = "replay-build"
START_NS = 5_000_000_000
TICK = fixture.TICK_NS
DETERMINISM = DeterminismManifest.from_torch(device="cpu", nms_backend="torchvision.ops.nms")
WALL_CLOCK_APIS = (
    "sleep", "perf_counter", "perf_counter_ns", "monotonic", "monotonic_ns",
    "time", "time_ns", "process_time", "process_time_ns", "thread_time", "thread_time_ns",
)


def _events(frames: int = 24) -> list[dict]:
    """frame 7 を drop、frame 12 を duplicate 再送、frame 20 の後に timeout を挟んだ event 列を作る。"""
    events: list[dict] = []

    def add(kind: str, ts: int, index: int | None = None) -> None:
        events.append({
            "completion_seq": len(events), "kind": kind, "timestamp_ns": ts, "session_frame_index": index,
            "correlation_id": None if index is None else f"{SESSION}:{index}",
        })

    for index in range(frames):
        if index == 7:
            continue
        ts = START_NS + index * TICK
        add("frame", ts, index)
        if index == 12:
            add("duplicate", ts + 1_000_000, index)
        if index == 20:
            add("timeout", ts + 2_000_000)
    return events


def _capture_dict(events: list[dict] | None = None, **overrides) -> dict:
    """synthetic(画素ファイルなし)capture manifest の dict を作る。"""
    data = {
        "schema_version": "survivors.recorded_capture.v1",
        "session": {
            "session_id": SESSION, "target_profile_hash": PROFILE_HASH, "game_build_id": BUILD_ID,
            "client_rect_screen_px": [0, 0, 1920, 1080], "determinism": DETERMINISM.to_dict(),
            "events": _events() if events is None else events,
        },
        "frames_path": None, "frames_sha256": None, "development_only": True,
    }
    data.update(overrides)
    return data


def _write_capture(tmp_path: Path, data: dict) -> Path:
    """capture manifest を JSON ファイルに書いてパスを返す。"""
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _replay(tmp_path: Path, name: str, capture: CaptureManifest | None = None, **overrides):
    """fixture の detector/HUD と実 runtime・controller で replay を1回実行する。"""
    policy = fixture.build_combat_policy(0)
    kwargs = dict(
        detector=fixture.FixedSceneDetector(),
        tracker=EntityTracker({i: 5 for i in range(12)}, 0.7, 0.6, 0.9),
        hud_parser=fixture.ScriptedHudParser(),
        bundle=RuntimeBundle.from_golden_fixture(policy, item_selector=fixture.FirstCardItemSelector()),
        artifact_hashes={"combat_policy": fixture._model_hash(policy)},
        target_profile_hash=PROFILE_HASH, game_build_id=BUILD_ID, replay_determinism=DETERMINISM,
        campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART,
    )
    kwargs.update(overrides)
    if capture is None:
        capture = CaptureManifest.load(_write_capture(tmp_path, _capture_dict()))
    return run_recorded_replay(capture, tmp_path / name, **kwargs)


def _jsonl(path: Path) -> list[dict]:
    """JSONL を dict の list で読む。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _walk(value, visit) -> None:
    """dict/list を再帰的にたどり、各 key と葉の値へ visit を適用する。"""
    if isinstance(value, dict):
        for key, item in value.items():
            visit(key)
            _walk(item, visit)
    elif isinstance(value, list):
        for item in value:
            _walk(item, visit)
    else:
        visit(value)


def test_replay_streams_recorded_frames_through_real_controller_on_virtual_clock(tmp_path: Path) -> None:
    """記録 frame が drop/duplicate/timeout 込みで流れ、全 telemetry 時刻が仮想時計の範囲に収まる。"""
    result = _replay(tmp_path, "run")
    assert result.exit_code == 0, result.errors
    assert result.exit_reason == "stop_requested"
    rows = _jsonl(result.paths["telemetry.jsonl"])
    stages = rows[1:]
    clock = result.manifest["virtual_clock"]
    assert clock["start_ns"] == START_NS
    assert all(clock["start_ns"] <= row["timestamp_ns"] <= clock["end_ns"] for row in stages)
    assert [row["timestamp_ns"] for row in stages] == sorted(row["timestamp_ns"] for row in stages)
    captured = [row for row in stages if row["stage"] == "capture"]
    assert [row["payload"]["frame_index"] for row in captured] == [i for i in range(24) if i != 7]
    assert [row["payload"]["captured_ns"] for row in captured] == [START_NS + i * TICK for i in range(24) if i != 7]
    assert next(row for row in captured if row["payload"]["frame_index"] == 8)["payload"]["dropped_frames"] == 1
    discard = [row for row in stages if row["stage"] == "discard"]
    assert [row["correlation_id"] for row in discard] == [f"{SESSION}:12"]
    assert {row["correlation_id"] for row in stages} <= {f"{SESSION}:controller"} | {f"{SESSION}:{i}" for i in range(24)}
    assert result.manifest["dropped_frame_indices"] == [7]
    assert any(row["stage"] == "policy" for row in stages)


def test_discrete_and_numeric_outputs_are_split_symmetrically(tmp_path: Path) -> None:
    """discrete には float も volatile key も無く、float と latency は全て numeric 側へ移る。"""
    result = _replay(tmp_path, "run")
    discrete = _jsonl(result.paths["discrete.jsonl"])
    numeric = _jsonl(result.paths["numeric.jsonl"])
    telemetry = _jsonl(result.paths["telemetry.jsonl"])[1:]
    assert len(discrete) == len(numeric) == len(telemetry)

    def no_float_or_volatile(value) -> None:
        assert not isinstance(value, float)
        assert value not in ("decision_id", "decision_hash", "obs_hash")

    for row in discrete:
        _walk(row, no_float_or_volatile)
        assert "latency_ns" not in row
    assert all("latency_ns" in row and "timestamp_ns" not in row for row in numeric)
    policy = [row for row in numeric if row["stage"] == "policy"]
    assert policy and all("confidence" in row["values"] for row in policy)
    obs_emitted = [row for row in discrete if row["stage"] == "obs" and row["payload"]["emitted"]]
    with np.load(result.paths["numeric_obs.npz"]) as obs:
        assert obs["values"].shape[0] == obs["validity"].shape[0] == obs["age"].shape[0] == len(obs_emitted)
        assert list(obs["frame_id"]) == [row["payload"]["frame_id"] for row in obs_emitted]


def test_split_payload_moves_every_float_and_drops_volatile_keys() -> None:
    """入れ子の float は path 付きで numeric へ、離散側は None に置き換わる。"""
    discrete, numeric = split_payload({"a": 1, "b": 0.5, "c": {"d": [2, 0.25], "decision_id": "x"}, "e": True})
    assert discrete == {"a": 1, "b": None, "c": {"d": [2, None]}, "e": True}
    assert numeric == {"b": 0.5, "c.d[1]": 0.25}


def test_three_replays_of_same_bundle_are_byte_identical(tmp_path: Path) -> None:
    """同じ session・bundle を3回再生すると discrete/effects/numeric obs が bytes 単位で一致する。"""
    results = [_replay(tmp_path, f"run{i}") for i in range(3)]
    for name in ("discrete.jsonl", "effects.json", "numeric_obs.npz", "numeric.jsonl"):
        assert len({r.paths[name].read_bytes() for r in results}) == 1, name
    # telemetry 生ログだけは uuid の decision_id を含むので比較対象外(discrete/effects では除去済み)
    outputs = [{k: v for k, v in r.manifest["outputs"].items() if k != "telemetry.jsonl"} for r in results]
    assert outputs[0] == outputs[1] == outputs[2]


def test_effect_recorder_saves_semantic_effects_without_executing(tmp_path: Path) -> None:
    """effects.json は意味カテゴリ付きで、入力系 effect は proposed のみ(execute されない)。"""
    result = _replay(tmp_path, "run")
    effects = json.loads(result.paths["effects.json"].read_text(encoding="utf-8"))
    assert effects
    assert {e["category"] for e in effects} <= {"movement", "ui", "release", "control"}
    assert "movement" in {e["category"] for e in effects}
    for effect in effects:
        body = effect["effect"]
        assert body["disposition"] in ("proposed", "handled")
        assert body["ack"] is None
        assert effect["correlation_id"].startswith(f"{SESSION}:")
        assert "decision_id" not in body["source"] and "decision_hash" not in body["source"]
        if body["kind"] == "move":
            assert type(body["action_index"]) is int


def test_replay_never_touches_wall_clock_or_live_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """実時計 API・OS 入力 backend の import・入力 adapter 構築・execute_effect を全て封じても結果が同じ。"""
    baseline = _replay(tmp_path, "baseline").paths["discrete.jsonl"].read_bytes()
    capture = CaptureManifest.load(_write_capture(tmp_path, _capture_dict()))

    def _blocked(*_args, **_kwargs):
        raise AssertionError("replay path touched a wall clock or live input API")

    for name in ("survivors.input.win32_backend", "survivors.input.dry_run_backend"):
        monkeypatch.setitem(sys.modules, name, None)  # import すると ImportError
    monkeypatch.setattr(InputLeaseController, "__init__", _blocked)
    monkeypatch.setattr(controller_module, "execute_effect", _blocked)
    for name in WALL_CLOCK_APIS:
        monkeypatch.setattr(time, name, _blocked)
    result = _replay(tmp_path, "blocked", capture=capture)
    monkeypatch.undo()
    assert result.exit_code == 0, result.errors
    assert result.paths["discrete.jsonl"].read_bytes() == baseline


@pytest.mark.parametrize("field, value", [("target_profile_hash", "e" * 64), ("game_build_id", "other-build")])
def test_profile_or_build_mismatch_is_rejected_before_running(tmp_path: Path, field: str, value: str) -> None:
    """記録と再生で target profile / game build が違えば、telemetry を作る前に拒否する。"""
    with pytest.raises(ReplayIntegrityError, match=field):
        _replay(tmp_path, "run", **{field: value})
    assert not (tmp_path / "run" / "telemetry.jsonl").exists()


def test_determinism_mismatch_is_rejected(tmp_path: Path) -> None:
    """再生環境の決定性設定(device)が記録と違えば1 frame も流さずに拒否する。"""
    other = dataclasses.replace(DETERMINISM, device="cuda:0")
    with pytest.raises(DeterminismMismatchError):
        _replay(tmp_path, "run", replay_determinism=other)
    assert not (tmp_path / "run" / "telemetry.jsonl").exists()


def _frames_npz(tmp_path: Path, indices) -> tuple[str, str]:
    """指定 frame 番号の画素を持つ NPZ を書き、(相対パス, sha256) を返す。"""
    path = tmp_path / "frames.npz"
    np.savez_compressed(path, **{f"frame_{i}": np.full((1080, 1920, 4), i, np.uint8) for i in indices})
    return path.name, hashlib.sha256(path.read_bytes()).hexdigest()


def test_capture_manifest_with_frames_file_streams_recorded_pixels(tmp_path: Path) -> None:
    """画素 NPZ 付き manifest は hash を照合してから記録画素を controller へ渡す。"""
    events = _events(4)
    rel, digest = _frames_npz(tmp_path, range(4))
    capture = CaptureManifest.load(_write_capture(tmp_path, _capture_dict(events, frames_path=rel, frames_sha256=digest)))
    seen: list[int] = []

    class PixelProbe(fixture.FixedSceneDetector):
        """受け取った frame の画素値を記録する detector。"""

        def infer(self, frame_bgr, *, score_threshold):
            seen.append(int(frame_bgr[0, 0, 0]))
            return super().infer(frame_bgr, score_threshold=score_threshold)

    result = _replay(tmp_path, "run", capture=capture, detector=PixelProbe())
    assert result.exit_code == 0, result.errors
    assert seen == [0, 1, 2, 3]
    assert result.manifest["frames_sha256"] == digest
    assert result.manifest["development_only"] is True


@pytest.mark.parametrize("case", ["bad_hash", "missing_frame", "synthetic_formal", "unknown_key", "bad_schema"])
def test_corrupt_capture_manifest_is_rejected(tmp_path: Path, case: str) -> None:
    """画素ファイル hash 不一致・frame 欠落・正式扱いの synthetic・未知キー・schema 違いを拒否する。"""
    events = _events(4)
    rel, digest = _frames_npz(tmp_path, range(3) if case == "missing_frame" else range(4))
    data = _capture_dict(events, frames_path=rel, frames_sha256="0" * 64 if case == "bad_hash" else digest)
    if case == "synthetic_formal":
        data = _capture_dict(events, development_only=False)
    elif case == "unknown_key":
        data["extra"] = 1
    elif case == "bad_schema":
        data["schema_version"] = "survivors.recorded_capture.v0"
    with pytest.raises(ReplayIntegrityError):
        CaptureManifest.load(_write_capture(tmp_path, data))


def test_verify_virtual_telemetry_rejects_wall_clock_and_foreign_ids() -> None:
    """実時計由来の桁違い timestamp、逆行、記録に無い correlation id を検出する。"""
    session = RecordedSession.from_dict(_capture_dict(_events(2))["session"])
    header = {"event": "session_header", "session_id": SESSION}

    def row(ts: int, cid: str = f"{SESSION}:0", latency: int = 0) -> dict:
        return {"stage": "capture", "sequence": 1, "timestamp_ns": ts, "latency_ns": latency, "correlation_id": cid}

    end = START_NS + TICK
    verify_virtual_telemetry([header, row(START_NS), row(end)], session, start_ns=START_NS, end_ns=end)
    for rows in (
        [header, row(10**18)],
        [header, row(end), row(START_NS)],
        [header, row(START_NS, cid=f"{SESSION}:99")],
        [header, row(START_NS, latency=10**12)],
        [{"event": "session_header", "session_id": "other"}, row(START_NS)],
    ):
        with pytest.raises(ReplayIntegrityError):
            verify_virtual_telemetry(rows, session, start_ns=START_NS, end_ns=end)


def test_formal_eligibility_fails_closed() -> None:
    """開発用 parent が1つでも混ざる・detector 正式判定が失敗する・無い場合は formal 不可。"""
    bundle = RuntimeBundle.from_golden_fixture(fixture.build_combat_policy(0))
    formal_bundle = dataclasses.replace(bundle, development_only=False, live_eligible=True)
    session = RecordedSession.from_dict(_capture_dict()["session"])
    formal_capture = CaptureManifest(session, Path("frames.npz"), "0" * 64, False, "1" * 64)
    dev_capture = dataclasses.replace(formal_capture, development_only=True)

    class Formal:
        """正式判定を通る detector manifest の代役。"""

        def assert_formal_eligible(self) -> None:
            return None

    class Rejected:
        """正式判定で拒否される detector manifest の代役。"""

        def assert_formal_eligible(self) -> None:
            raise ValueError("development detector")

    assert formal_parents_eligible(formal_bundle, Formal(), formal_capture) is True
    assert formal_parents_eligible(bundle, Formal(), formal_capture) is False
    assert formal_parents_eligible(formal_bundle, Rejected(), formal_capture) is False
    assert formal_parents_eligible(formal_bundle, None, formal_capture) is False
    assert formal_parents_eligible(formal_bundle, Formal(), dev_capture) is False


def test_synthetic_replay_manifest_is_never_formal(tmp_path: Path) -> None:
    """synthetic session + development bundle の出力 manifest は development_only で formal 不可。"""
    manifest = _replay(tmp_path, "run").manifest
    assert manifest["development_only"] is True
    assert manifest["formal_replay_eligible"] is False
    assert set(manifest["outputs"]) == {"telemetry.jsonl", "discrete.jsonl", "numeric.jsonl", "numeric_obs.npz", "effects.json"}


def test_cli_replays_capture_with_loaded_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI は profile/build を照合し、artifact loader の部品で replay して controller の終了コードを返す。"""
    profile = load_target_profile()
    data = _capture_dict()
    data["session"].update(target_profile_hash=profile.target_hash, game_build_id=str(profile.sections["build"]["build_id"]))
    capture_path = _write_capture(tmp_path, data)
    policy = fixture.build_combat_policy(0)
    parts = type("Parts", (), dict(
        bundle=RuntimeBundle.from_golden_fixture(policy, item_selector=fixture.FirstCardItemSelector()),
        detector=fixture.FixedSceneDetector(), tracker=EntityTracker({i: 5 for i in range(12)}, 0.7, 0.6, 0.9),
        detector_manifest=None, hud_parser=fixture.ScriptedHudParser(),
        artifact_hashes={"combat_policy": fixture._model_hash(policy)},
    ))
    loaded: list = []
    monkeypatch.setattr(cli, "load_runtime_profile", lambda *_a: profile)
    monkeypatch.setattr(cli, "_load_artifacts", lambda args: loaded.append(args) or parts)
    argv = [
        "--capture-manifest", str(capture_path), "--output-dir", str(tmp_path / "out"),
        "--combat-package", "c", "--detector-config", "d",
        "--class-map", str(Path(fixture.__file__).parent / "configs" / "world_class_map_v1.yaml"),
        "--detector-weights", "w", "--detector-manifest", "x", "--campaign-run-mode", "operator_debug_restart",
    ]
    assert cli.main(argv) == 0
    assert loaded and (tmp_path / "out" / "manifest.json").exists()

    data["session"]["game_build_id"] = "other"
    _write_capture(tmp_path, data)
    loaded.clear()
    with pytest.raises(ReplayIntegrityError):
        cli.main([*argv[:3], str(tmp_path / "out2"), *argv[4:]])
    assert not loaded  # artifact を読む前に止まる

"""recorded session E2E replay engine と CLI のテスト(06-01 タスク2)。

synthetic session を実 controller(実 assembler・runtime・state machine・health・telemetry)へ流し、
capture manifest/profile/build/決定性の照合、仮想時計への時刻と correlation id の統一、
effect recorder の semantic JSON、discrete/numeric の分離、同一入力での再現性、
実時計と live input backend に触れないことを確かめます。
後半(タスク3)は golden/diff: quantize と segment tolerance、first divergence の tree report、
golden update の必須項目、安全 fixture の hard assertion、正式 parent 欠落時の formal verdict 拒否です。
最後(タスク4)は configs/e2e_replay_suite_v1.yaml の synthetic suite を標準 pytest で3回ずつ再生する決定性 gate、
CLI サブコマンド(replay/compare/update-golden/publish-formal-verdict)、30分相当の仮想 schedule の benchmark です。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pytest
import yaml

import replay_controller_fixture as fixture
import replay_survivors_session as cli
from survivors.controller import controller as controller_module
from survivors.controller.state_machine import CampaignRunMode
from survivors.input.controller import InputLeaseController
from survivors.replay.e2e_replay import (
    DEFAULT_TOLERANCES,
    FORMAL_REPLAY_RUNS,
    CaptureManifest,
    FormalReplayRejectedError,
    GoldenUpdateError,
    ReplayIntegrityError,
    SafetyAssertionError,
    Tolerance,
    _write_jsonl,
    _write_npz,
    assert_safe_effects,
    compare_replays,
    fault_windows,
    formal_parents_eligible,
    golden_mismatches,
    inference_device,
    publish_formal_replay_verdict,
    quantize,
    quantized_sha256,
    replay_metrics,
    run_recorded_replay,
    split_payload,
    tolerance_for,
    update_golden,
    verify_virtual_telemetry,
)
from survivors.replay.recorded_frame_source import DeterminismManifest, DeterminismMismatchError, RecordedSession
from survivors.runtime.artifact_bundle import RuntimeBundle
from survivors.target_profile import load_target_profile
from survivors.vision.entity_tracker import EntityTracker
from survivors.vision.hud_parser import HudStateV1, ParsedButton, ParsedCard
from survivors.vision.world_detector import DetectionResult

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


@pytest.mark.parametrize("field, value", [("device", "cuda:0"), ("nms_backend", "custom_nms")])
def test_declared_determinism_must_match_loaded_inference_parts(tmp_path: Path, field: str, value: str) -> None:
    """M5: 記録と申告が一致していても、実際に読み込んだ部品の device や NMS 実装と違えば拒否する。"""
    declared = dataclasses.replace(DETERMINISM, **{field: value})
    data = _capture_dict()
    data["session"]["determinism"] = declared.to_dict()
    capture = CaptureManifest.load(_write_capture(tmp_path, data))
    with pytest.raises(DeterminismMismatchError, match="loaded inference parts"):
        _replay(tmp_path, "run", capture, replay_determinism=declared)
    assert not (tmp_path / "run").exists()


def test_inference_device_is_read_from_detector_and_policy_parameters() -> None:
    """M5: device は detector model と combat policy の parameter から読み、混在していれば拒否する。"""
    import torch

    bundle = RuntimeBundle.from_golden_fixture(fixture.build_combat_policy(0))
    assert inference_device(fixture.FixedSceneDetector(), bundle) == "cpu"
    detector = type("MetaDetector", (), {"_model": torch.nn.Linear(1, 1, device="meta")})()
    with pytest.raises(DeterminismMismatchError, match="exactly one device"):
        inference_device(detector, bundle)


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


def test_pixel_override_is_never_formal(tmp_path: Path) -> None:
    """I4: 正式 parent と記録画素が揃っていても、画素ローダーを差し替えた run は formal の材料にしない。"""
    rel, digest = _frames_npz(tmp_path, range(4))
    data = _capture_dict(_events(4), frames_path=rel, frames_sha256=digest, development_only=False)
    capture = CaptureManifest.load(_write_capture(tmp_path, data))
    policy = fixture.build_combat_policy(0)
    formal = dict(
        bundle=dataclasses.replace(
            RuntimeBundle.from_golden_fixture(policy, item_selector=fixture.FirstCardItemSelector()),
            development_only=False, live_eligible=True,
        ),
        detector_manifest=_FormalDetector(),
    )
    assert _replay(tmp_path, "recorded", capture, **formal).manifest["formal_replay_eligible"] is True
    blank = np.zeros((1080, 1920, 4), np.uint8)
    manifest = _replay(tmp_path, "override", capture, pixels=lambda _i: blank, **formal).manifest
    assert manifest["development_only"] is True and manifest["formal_replay_eligible"] is False


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
        "replay", "--capture-manifest", str(capture_path), "--output-dir", str(tmp_path / "out"),
        "--combat-package", "c", "--detector-config", "d",
        "--class-map", str(Path(fixture.__file__).parent / "configs" / "world_class_map_v1.yaml"),
        "--detector-weights", "w", "--detector-manifest", "x", "--campaign-run-mode", "operator_debug_restart",
    ]
    assert cli.main(argv) == 0
    assert loaded and (tmp_path / "out" / "manifest.json").exists()
    assert json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))["determinism"]["device"] == "cpu"

    # M5: 記録も申告も cuda:0 でも、読み込んだ部品は CPU なので拒否する。NMS は実装が使うものしか選べない。
    cuda = {**data, "session": {**data["session"], "determinism": {**data["session"]["determinism"], "device": "cuda:0"}}}
    _write_capture(tmp_path, cuda)
    with pytest.raises(DeterminismMismatchError, match="loaded inference parts"):
        cli.main([*argv[:4], str(tmp_path / "out-cuda"), *argv[5:], "--device", "cuda:0"])
    with pytest.raises(SystemExit):
        cli.main([*argv, "--nms-backend", "custom_nms"])
    _write_capture(tmp_path, data)

    data["session"]["game_build_id"] = "other"
    _write_capture(tmp_path, data)
    loaded.clear()
    with pytest.raises(ReplayIntegrityError):
        cli.main([*argv[:4], str(tmp_path / "out2"), *argv[5:]])
    assert not loaded  # artifact を読む前に止まる


# ---- golden/diff(06-01 タスク3) ----------------------------------------------------------

OBS_LAYOUT = RuntimeBundle.from_golden_fixture(fixture.build_combat_policy(0)).deploy_schema.layout


@pytest.fixture(scope="module")
def base_run(tmp_path_factory: pytest.TempPathFactory):
    """golden/diff テストで共有する基準 replay を1回だけ実行する(各テストは複製を書き換える)。"""
    return _replay(tmp_path_factory.mktemp("base"), "run")


def _copy(run, tmp_path: Path, name: str = "copy") -> dict[str, Path]:
    """基準 replay の出力ディレクトリを複製し、paths dict を返す。"""
    target = tmp_path / name
    shutil.copytree(run.paths["manifest.json"].parent, target)
    return {key: target / path.name for key, path in run.paths.items()}


def _mutate_jsonl(path: Path, mutate) -> None:
    """JSONL を読み、mutate(rows) で書き換えて canonical 形式で書き戻す。"""
    rows = _jsonl(path)
    mutate(rows)
    _write_jsonl(path, rows)


def _run_hashes(run) -> dict[str, str]:
    """run の manifest から golden に記載すべき artifact hashes を組み立てる。"""
    return {**run.manifest["artifact_hashes"], "capture_manifest_sha256": run.manifest["capture_manifest_sha256"]}


def test_quantization_and_segment_tolerance_rules() -> None:
    """quantize は偶数丸め・-0/NaN 正規化、hash は最下位 bit の揺れに鈍感、tolerance は最長前方一致。"""
    q = quantize([0.1, -1e-7, float("nan")])
    assert q[0] == 100000 and q[1] == 0 and not np.signbit(q[1]) and np.isnan(q[2])
    assert quantized_sha256([0.1]) == quantized_sha256([0.1 + 1e-12])
    assert quantized_sha256([0.1]) != quantized_sha256([0.1 + 1e-5])
    assert quantized_sha256([0.1], keys=["a"]) != quantized_sha256([0.1], keys=["b"])
    custom = {"": Tolerance(1.0, 0.0), "obs.values": Tolerance(2.0, 0.0)}
    assert tolerance_for("obs.values.player", custom) == Tolerance(2.0, 0.0)
    assert tolerance_for("obs.age.player", custom) == Tolerance(1.0, 0.0)
    assert tolerance_for("policy.confidence", {}) == Tolerance(0.0, 0.0)  # 未定義は exact


def test_same_bundle_replays_pass_discrete_and_numeric_gate(base_run, tmp_path: Path) -> None:
    """同じ bundle の再生どうしは discrete hash 一致・全 numeric segment quantized 一致で分岐なし。"""
    diff = compare_replays(base_run, _replay(tmp_path, "again"), obs_layout=OBS_LAYOUT)
    assert diff.passed and diff.first_divergence is None
    assert diff.discrete_sha256[0] == diff.discrete_sha256[1]
    assert diff.metrics["discrete_match_rate"] == 1.0 and diff.metrics["numeric_tolerance_pass_rate"] == 1.0
    assert all(s["quantized_equal"] for s in diff.segments.values())
    assert {"policy.confidence", "latency.policy"} <= set(diff.segments)
    assert any(name.startswith("obs.values.") and name != "obs.values.all" for name in diff.segments)


@pytest.mark.parametrize("stage,category", [
    ("hud_parser", "parser_field"), ("obs", "obs"), ("policy", "model_action"),
    ("state_machine", "state"), ("effect", "effect"),
])
def test_first_divergence_tree_names_earliest_frame_stage_and_field(
    base_run, tmp_path: Path, stage: str, category: str
) -> None:
    """最初に分岐した frame → stage(分類)→ field path が tree で報告され、後段の分岐より優先される。"""
    paths = _copy(base_run, tmp_path)
    target: dict = {}

    def inject(rows) -> None:
        target.update(next(row for row in rows if row["stage"] == stage))
        target["index"] = rows.index(next(row for row in rows if row["stage"] == stage))
        rows[target["index"]]["payload"]["injected"] = 1
        rows[-1]["payload"]["later"] = 2  # より後ろの分岐は first にならない

    _mutate_jsonl(paths["discrete.jsonl"], inject)
    diff = compare_replays(base_run, paths)
    assert not diff.passed
    frame = diff.first_divergence["frame"]
    assert frame["correlation_id"] == target["correlation_id"]
    assert frame["stage"]["name"] == stage and frame["stage"]["category"] == category
    assert frame["stage"]["sequence"] == target["sequence"]
    assert frame["stage"]["divergence"] == {"kind": "discrete", "path": "payload.injected", "old": "<missing>", "new": 1}
    rows = diff.metrics["discrete_rows"][0]
    assert diff.metrics["discrete_match_rate"] == (rows - 2) / rows


def test_state_value_change_is_reported_with_old_and_new(base_run, tmp_path: Path) -> None:
    """state 遷移先の値が変われば state 分類で旧値・新値ごと報告される(型違いも exact に区別)。"""
    paths = _copy(base_run, tmp_path)

    def change(rows) -> None:
        next(row for row in rows if row["stage"] == "state_machine")["payload"]["to_state"] = "unknown"

    _mutate_jsonl(paths["discrete.jsonl"], change)
    divergence = compare_replays(base_run, paths).first_divergence["frame"]["stage"]
    assert divergence["category"] == "state"
    assert divergence["divergence"]["path"] == "payload.to_state"
    assert divergence["divergence"]["new"] == "unknown" and divergence["divergence"]["old"] != "unknown"


def test_numeric_stage_values_use_segment_tolerance_not_raw_hash(base_run, tmp_path: Path) -> None:
    """policy 数値は quantized hash が変わっても segment tolerance 内なら合格、超えたら model action で分岐。"""
    tolerances = {**DEFAULT_TOLERANCES, "policy": Tolerance(abs_tol=1e-2, rel_tol=0.0)}

    def shift(delta: float):
        def mutate(rows) -> None:
            row = next(row for row in rows if row["stage"] == "policy")
            row["values"]["confidence"] += delta
            row["latency_ns"] += 10  # latency は 1ms 以内の揺れを許容
        return mutate

    small = _copy(base_run, tmp_path, "small")
    _mutate_jsonl(small["numeric.jsonl"], shift(1e-3))
    diff = compare_replays(base_run, small, tolerances=tolerances)
    assert diff.passed, diff.first_divergence
    assert diff.segments["policy.confidence"]["quantized_equal"] is False
    assert diff.segments["policy.confidence"]["passed"] is True
    assert diff.segments["policy.confidence"]["max_abs_diff"] == pytest.approx(1e-3)

    large = _copy(base_run, tmp_path, "large")
    _mutate_jsonl(large["numeric.jsonl"], shift(0.5))
    diff = compare_replays(base_run, large, tolerances=tolerances)
    stage = diff.first_divergence["frame"]["stage"]
    assert stage["name"] == "policy" and stage["category"] == "model_action"
    assert stage["divergence"]["kind"] == "numeric"
    assert stage["divergence"]["segment"] == "policy.confidence" and stage["divergence"]["path"] == "confidence"
    assert diff.metrics["numeric_segments_failed"] == ["policy.confidence"]
    assert diff.metrics["discrete_equal"] is True and diff.metrics["numeric_tolerance_pass_rate"] < 1.0


def test_obs_divergence_reports_obs_index_and_segment(base_run, tmp_path: Path) -> None:
    """obs 数値の分岐は obs 行の frame・平面・obs index・DeployObs segment 名で報告される。"""
    name, (offset, size) = next((n, span) for n, span in OBS_LAYOUT.items() if span[1] > 1)
    index = offset + size - 1
    paths = _copy(base_run, tmp_path)
    with np.load(paths["numeric_obs.npz"]) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["values"][1, index] += 1.0
    _write_npz(paths["numeric_obs.npz"], arrays)
    diff = compare_replays(base_run, paths, obs_layout=OBS_LAYOUT)
    emitted = [row for row in _jsonl(base_run.paths["discrete.jsonl"]) if row["stage"] == "obs" and row["payload"]["emitted"]]
    frame = diff.first_divergence["frame"]
    assert frame["correlation_id"] == emitted[1]["correlation_id"]
    assert frame["stage"]["name"] == "obs" and frame["stage"]["sequence"] == emitted[1]["sequence"]
    divergence = frame["stage"]["divergence"]
    assert divergence["plane"] == "values" and divergence["obs_row"] == 1
    assert divergence["obs_index"] == index and divergence["obs_segment"] == name
    assert divergence["segment"] == f"obs.values.{name}"
    assert diff.metrics["numeric_segments_failed"] == [f"obs.values.{name}"]

    arrays["values"] = arrays["values"][:-1]  # obs 行数が変わると形の分岐として報告される
    _write_npz(paths["numeric_obs.npz"], arrays)
    divergence = compare_replays(base_run, paths, obs_layout=OBS_LAYOUT).first_divergence["frame"]["stage"]["divergence"]
    assert divergence["kind"] == "numeric_shape" and divergence["plane"] == "values"


def test_golden_update_records_metrics_and_hashes(base_run, tmp_path: Path) -> None:
    """golden update は新旧 metrics・比較 metrics・artifact hashes を記録し、同じ run とは食い違わない。"""
    metrics = replay_metrics(base_run)
    comparison = compare_replays(base_run, base_run).metrics
    golden_path = tmp_path / "golden.json"
    golden = update_golden(
        golden_path, base_run, old_metrics=metrics, new_metrics=metrics, comparison=comparison,
        artifact_hashes=_run_hashes(base_run), obs_layout=OBS_LAYOUT,
    )
    assert json.loads(golden_path.read_text(encoding="utf-8")) == golden
    assert golden["metrics"] == {"old": metrics, "new": metrics, "comparison": comparison}
    assert golden["artifact_hashes"] == _run_hashes(base_run)
    assert golden["development_only"] is True and golden["formal_replay_eligible"] is False
    assert golden_mismatches(golden_path, base_run, obs_layout=OBS_LAYOUT)[0] == []

    mutated = _copy(base_run, tmp_path)
    _mutate_jsonl(mutated["discrete.jsonl"], lambda rows: rows.pop())
    assert "discrete" in golden_mismatches(golden_path, mutated, obs_layout=OBS_LAYOUT)[0]

    # 参照 run 出力が golden に固定した sha256 と合わなければ比較せずに拒否する。
    reference = tmp_path / "golden.reference" / "numeric.jsonl"
    reference.write_text(reference.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ReplayIntegrityError, match="reference"):
        golden_mismatches(golden_path, base_run, obs_layout=OBS_LAYOUT)


def _golden(base_run, tmp_path: Path) -> Path:
    """base_run を golden として書き、そのパスを返す。"""
    metrics = replay_metrics(base_run)
    golden_path = tmp_path / "golden.json"
    update_golden(
        golden_path, base_run, old_metrics=metrics, new_metrics=metrics,
        comparison=compare_replays(base_run, base_run).metrics, artifact_hashes=_run_hashes(base_run), obs_layout=OBS_LAYOUT,
    )
    return golden_path


def _perturb_numeric(paths: dict[str, Path], plane: str, delta: float) -> str:
    """stage 値・latency・obs の各平面の最初の値を delta だけずらし、ずらした segment 名を返す。"""
    if plane == "obs":
        name, (offset, _size) = next(iter(OBS_LAYOUT.items()))
        with np.load(paths["numeric_obs.npz"]) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays["values"][0, offset] += delta
        _write_npz(paths["numeric_obs.npz"], arrays)
        return f"obs.values.{name}"
    segment = {"stage": "hud_parser.capability_confidence", "latency": "latency.policy"}[plane]

    def mutate(rows) -> None:
        if plane == "stage":
            row = next(row for row in rows if "capability_confidence" in row["values"])
            row["values"]["capability_confidence"] += delta
        else:
            next(row for row in rows if row["stage"] == "policy")["latency_ns"] += delta

    _mutate_jsonl(paths["numeric.jsonl"], mutate)
    return segment


@pytest.mark.parametrize("plane, within, beyond", [
    ("stage", 5.01e-7, 0.5), ("latency", 10, 5_000_000), ("obs", 3e-6, 1.0),
])
def test_golden_numeric_uses_segment_tolerance_like_compare_replays(
    base_run, tmp_path: Path, plane: str, within: float, beyond: float
) -> None:
    """I5: golden 比較も quantized hash → segment tolerance の同じ規則で、許容内の差を mismatch にしない。

    quantum の丸め境界をまたぐ許容内の摂動は compare_replays と同じく合格し、
    tolerance を超える摂動は numeric:<segment> と first divergence で報告される。
    """
    golden_path = _golden(base_run, tmp_path)
    small = _copy(base_run, tmp_path, "small")
    segment = _perturb_numeric(small, plane, within)
    direct = compare_replays(base_run, small, obs_layout=OBS_LAYOUT)
    assert direct.passed and direct.segments[segment]["quantized_equal"] is False
    mismatches, diff = golden_mismatches(golden_path, small, obs_layout=OBS_LAYOUT)
    assert mismatches == [] and diff.passed, diff.first_divergence
    assert diff.segments[segment]["quantized_equal"] is False and diff.segments[segment]["passed"] is True
    assert cli.main(["compare", "--golden", str(golden_path), "--new", str(small["manifest.json"].parent)]) == 0

    large = _copy(base_run, tmp_path, "large")
    _perturb_numeric(large, plane, beyond)
    mismatches, diff = golden_mismatches(golden_path, large, obs_layout=OBS_LAYOUT)
    assert mismatches == [f"numeric:{segment}"]
    assert diff.first_divergence["frame"]["stage"]["divergence"]["segment"] == segment
    report = tmp_path / "report.json"
    assert cli.main(["compare", "--golden", str(golden_path), "--new", str(large["manifest.json"].parent),
                     "--report", str(report)]) == 1
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["golden_mismatches"] == [f"numeric:{segment}"]
    assert result["golden_first_divergence"]["frame"]["stage"]["divergence"]["segment"] == segment


@pytest.mark.parametrize("case", [
    "old_metrics", "new_metrics", "comparison", "no_hashes", "missing_capture_hash", "wrong_model_hash", "empty_value",
])
def test_golden_update_rejects_missing_metrics_or_hashes(base_run, tmp_path: Path, case: str) -> None:
    """新旧 metrics・比較 metrics・artifact hashes のどれかが欠ける・食い違うと golden を書かない。"""
    metrics = replay_metrics(base_run)
    kwargs = dict(
        old_metrics=metrics, new_metrics=metrics, comparison=compare_replays(base_run, base_run).metrics,
        artifact_hashes=_run_hashes(base_run),
    )
    if case in ("old_metrics", "new_metrics"):
        kwargs[case] = {k: v for k, v in metrics.items() if k != "effect_counts"}
    elif case == "comparison":
        kwargs[case] = {"discrete_match_rate": 1.0}
    elif case == "no_hashes":
        kwargs["artifact_hashes"] = {}
    elif case == "missing_capture_hash":
        kwargs["artifact_hashes"] = dict(base_run.manifest["artifact_hashes"])
    elif case == "wrong_model_hash":
        kwargs["artifact_hashes"] = {**_run_hashes(base_run), "combat_policy": "0" * 64}
    else:
        kwargs["artifact_hashes"] = {**_run_hashes(base_run), "detector": ""}
    with pytest.raises(GoldenUpdateError):
        update_golden(tmp_path / "golden.json", base_run, **kwargs)
    assert not (tmp_path / "golden.json").exists()


def test_golden_artifact_change_requires_independent_approval(base_run, tmp_path: Path) -> None:
    """前の golden と artifact hashes が変わる更新は approved_by が無ければ拒否される。"""
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps({"artifact_hashes": {"combat_policy": "0" * 64}}), encoding="utf-8")
    metrics = replay_metrics(base_run)
    kwargs = dict(
        old_metrics=metrics, new_metrics=metrics, comparison=compare_replays(base_run, base_run).metrics,
        artifact_hashes=_run_hashes(base_run),
    )
    with pytest.raises(GoldenUpdateError, match="approved_by"):
        update_golden(golden_path, base_run, **kwargs)
    golden = update_golden(golden_path, base_run, approved_by="verifier", **kwargs)
    assert golden["approved_by"] == "verifier" and golden["previous_golden_sha256"]


def test_fault_windows_cover_timeout_focus_loss_and_unknown_state() -> None:
    """timeout/focus_lost は次の frame まで、unknown は unknown を抜けるまでが安全区間になる。"""
    events = _events(6)
    events.insert(3, {"completion_seq": 0, "kind": "focus_lost", "timestamp_ns": START_NS + 2 * TICK + 5,
                      "session_frame_index": None, "correlation_id": None})
    for seq, event in enumerate(events):
        event["completion_seq"] = seq
    session = RecordedSession.from_dict(_capture_dict(events)["session"])
    rows = [
        {"stage": "state_machine", "timestamp_ns": ts, "payload": {"to_state": state}}
        for ts, state in ((10, "gameplay"), (20, "unknown"), (30, "unknown"), (40, "gameplay"), (50, "unknown"))
    ]
    assert fault_windows(session, rows) == [(START_NS + 2 * TICK + 5, START_NS + 3 * TICK), (20, 40), (50, 2**63)]


def test_safety_assertion_hard_fails_on_input_effects_in_fault_window() -> None:
    """区間内の movement/ui は1件でも hard fail、release/control と区間外(終端含まず)は通る。"""

    def effect(ts: int, category: str) -> dict:
        return {"correlation_id": f"{SESSION}:1", "timestamp_ns": ts, "category": category, "effect": {"kind": "x"}}

    windows = [(100, 200)]
    assert_safe_effects([effect(100, "release"), effect(150, "control"), effect(200, "movement"), effect(99, "ui")], windows)
    for category in ("movement", "ui", "unknown"):
        with pytest.raises(SafetyAssertionError):
            assert_safe_effects([effect(150, "release")] * 1000 + [effect(150, category)], windows)
    with pytest.raises(SafetyAssertionError, match="no unknown"):
        assert_safe_effects([], [])


def test_safety_fixture_timeout_and_focus_loss_emit_only_release_or_noop(tmp_path: Path) -> None:
    """実 controller の replay で timeout・focus loss 後(次の frame まで)の effect は release/no-op だけ。"""
    events = _events()
    events.insert(10, {"completion_seq": 0, "kind": "focus_lost", "timestamp_ns": START_NS + 10 * TICK + 3_000_000,
                       "session_frame_index": None, "correlation_id": None})
    for seq, event in enumerate(events):
        event["completion_seq"] = seq
    capture = CaptureManifest.load(_write_capture(tmp_path, _capture_dict(events)))
    result = _replay(tmp_path, "safety", capture)
    effects = json.loads(result.paths["effects.json"].read_text(encoding="utf-8"))
    windows = fault_windows(capture.session, _jsonl(result.paths["discrete.jsonl"]))
    assert len(windows) >= 2  # timeout と focus_lost
    assert_safe_effects(effects, windows)
    injected = [*effects, {**effects[0], "timestamp_ns": windows[0][0], "category": "movement"}]
    with pytest.raises(SafetyAssertionError):
        assert_safe_effects(injected, windows)


class _FormalDetector:
    """正式判定を通る detector manifest の代役。"""

    def assert_formal_eligible(self) -> None:
        return None


class _RejectedDetector:
    """正式判定で拒否される detector manifest の代役。"""

    def assert_formal_eligible(self) -> None:
        raise ValueError("development detector")


def _formal_inputs(base_run):
    """formal verdict を publish できる正式 parent 一式(代役)と、3回分の formal manifest・比較結果を作る。"""
    bundle = dataclasses.replace(
        RuntimeBundle.from_golden_fixture(fixture.build_combat_policy(0)), development_only=False, live_eligible=True,
    )
    session = RecordedSession.from_dict(_capture_dict()["session"])
    capture = CaptureManifest(session, Path("frames.npz"), "0" * 64, False, "1" * 64)
    manifest = {**base_run.manifest, "development_only": False, "formal_replay_eligible": True,
                "capture_manifest_sha256": "1" * 64}
    diff = compare_replays(base_run, base_run)
    return dict(diffs=[diff, diff], manifests=[dict(manifest) for _ in range(3)], bundle=bundle,
                detector_manifest=_FormalDetector(), capture=capture)


def test_formal_verdict_publishes_only_with_formal_parents(base_run, tmp_path: Path) -> None:
    """正式 parent・formal manifest 3回分が揃うときだけ formal verdict を書ける。"""
    kwargs = _formal_inputs(base_run)
    verdict = publish_formal_replay_verdict(tmp_path / "verdict.json", **kwargs)
    assert verdict["formal_replay_eligible"] is True and verdict["passed"] is True and verdict["runs"] == 3
    assert json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8")) == verdict


@pytest.mark.parametrize("case", [
    "synthetic_manifests", "dev_bundle", "rejected_detector", "no_detector", "dev_capture",
    "two_runs", "one_manifest_dev", "different_artifacts", "different_capture", "missing_diffs",
])
def test_formal_verdict_is_refused_when_a_parent_is_development_only(base_run, tmp_path: Path, case: str) -> None:
    """M10: development_only の parent や synthetic run が1つでも混ざれば publish を拒否しファイルも書かない。"""
    kwargs = _formal_inputs(base_run)
    if case == "synthetic_manifests":
        kwargs["manifests"] = [base_run.manifest] * 3
    elif case == "dev_bundle":
        kwargs["bundle"] = RuntimeBundle.from_golden_fixture(fixture.build_combat_policy(0))
    elif case == "rejected_detector":
        kwargs["detector_manifest"] = _RejectedDetector()
    elif case == "no_detector":
        kwargs["detector_manifest"] = None
    elif case == "dev_capture":
        kwargs["capture"] = dataclasses.replace(kwargs["capture"], development_only=True)
    elif case == "two_runs":
        kwargs["manifests"] = kwargs["manifests"][:2]
    elif case == "one_manifest_dev":
        kwargs["manifests"][1]["development_only"] = True
    elif case == "different_artifacts":
        kwargs["manifests"][2]["artifact_hashes"] = {"combat_policy": "0" * 64}
    elif case == "different_capture":
        for manifest in kwargs["manifests"]:
            manifest["capture_manifest_sha256"] = "2" * 64
    else:
        kwargs["diffs"] = [kwargs["diffs"][0]]
    with pytest.raises(FormalReplayRejectedError):
        publish_formal_replay_verdict(tmp_path / "verdict.json", **kwargs)
    assert not (tmp_path / "verdict.json").exists()


# ---- synthetic suite・CLI・30分 benchmark(06-01 タスク4) ---------------------------------------

SUITE = yaml.safe_load(
    (Path(fixture.__file__).parent / "configs" / "e2e_replay_suite_v1.yaml").read_text(encoding="utf-8")
)
SUITE_FIXTURES = {spec["name"]: spec for spec in SUITE["fixtures"]}
SUITE_TOLERANCES = {segment: Tolerance(*values) for segment, values in SUITE["gate"]["tolerances"].items()}


class SceneDetector:
    """suite の detector_scene に応じた場面を返す detector(呼ばれた回数で場面を進める)。

    basic は player anchor と敵1体、gems は gem が player へ近づいて回収され(届いたら消える)、
    dense は 60px の敵を 50px 間隔で並べて重ね、elite・boss・hazard も置いた終盤場面です。
    呼ばれる順番は replay で固定されるので、回数で場面を進めても決定的です。
    """

    def __init__(self, scene: str) -> None:
        """場面名を保持する。"""
        assert scene in {"basic", "gems", "dense"}, scene
        self._scene = scene
        self._calls = 0

    def infer(self, frame_bgr: np.ndarray, *, score_threshold: float) -> DetectionResult:
        """呼び出し回数に応じた DetectionResult を返す(画素は見ない)。"""
        step = self._calls
        self._calls += 1
        boxes = [[940, 520, 980, 560], [1300, 300, 1340, 340]]
        classes = [1, 2]  # player_anchor, enemy_normal
        if self._scene == "gems":
            for k in range(3):  # gem_blue / gem_green / gem_red
                x = 1400 + 120 * k - 12 * step
                if x > 980:
                    boxes.append([x, 530, x + 16, 546])
                    classes.append(5 + k)
        elif self._scene == "dense":
            for k in range(16):
                x, y = 740 + (k % 8) * 50, 380 + (k // 8) * 220 + (step % 5) * 4
                boxes.append([x, y, x + 60, y + 60])
                classes.append(3 if k % 5 == 0 else 2)  # enemy_elite / enemy_normal
            boxes += [[900, 480, 1020, 600], [1000, 540, 1060, 600], [600, 700, 900, 900]]
            classes += [4, 10, 11]  # enemy_boss, hazard_projectile, hazard_area
        return DetectionResult(
            boxes_xyxy=np.array(boxes, np.float32), scores=np.full(len(boxes), .9, np.float32),
            class_ids=np.array(classes, np.int32), image_width=1920, image_height=1080,
        )


class SuiteHudParser:
    """suite の script(画面・confidence・card・ボタン)どおりの HudStateV1 を frame 番号で返す HUD parser。"""

    def __init__(self, spec: dict) -> None:
        """script を frame ごとの区間へ展開し、inventory と timer の初期値を決める。"""
        defaults = SUITE["session"]
        self._frames = [segment for segment in spec["script"] for _ in range(segment["frames"])]
        inventory = list(spec.get("inventory", defaults["default_inventory"]))
        self._inventory = tuple(inventory + [None] * (12 - len(inventory)))
        self._timer = float(spec.get("timer_seconds", defaults["default_timer_seconds"]))

    def reset_temporal_state(self) -> None:
        """arm 時のリセット(台本は時系列状態を持たない)。"""

    def parse(self, frame_bgra, *, session_id: str, frame_index: int, captured_monotonic_ns: int) -> HudStateV1:
        """frame 番号の区間から HudStateV1 を組み立てる。"""
        segment = self._frames[frame_index]
        cards = tuple(
            ParsedCard(slot, item, kind, level, .99, "ok", (100 + 450 * slot, 100, 400 + 450 * slot, 500))
            for slot, (item, kind, level) in enumerate(segment.get("cards", ()))
        )
        names = tuple(segment.get("buttons", ()))
        buttons = tuple(ParsedButton(name, .95, "ok", (860, 900 + 60 * i, 1060, 950 + 60 * i)) for i, name in enumerate(names))
        return HudStateV1(
            "hud_state.v1", session_id, frame_index, captured_monotonic_ns, "a" * 64,
            segment["screen"], segment.get("confidence", .9), "ok",
            self._timer, .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
            self._inventory, .9, "b" * 64, cards, "c" * 64, buttons,
            "reroll" in names, "skip" in names, False, .9, "ok",
        )


def _suite_events(spec: dict) -> list[dict]:
    """suite fixture の script と faults から recorded event 列を作る。

    frame は frame_interval_ns 間隔。drop はその frame を落とし、duplicate/focus_lost はその frame の直後、
    timeout は interval_ns ずつ時計を進めながら count 回積む(後続 frame も後ろへずれる)。
    """
    faults: dict[int, list[dict]] = {}
    for fault in spec.get("faults", ()):
        faults.setdefault(fault["frame"], []).append(fault)
    events: list[dict] = []
    ts = START_NS

    def add(kind: str, at: int, index: int | None = None) -> None:
        events.append({
            "completion_seq": len(events), "kind": kind, "timestamp_ns": at, "session_frame_index": index,
            "correlation_id": None if index is None else f"{SESSION}:{index}",
        })

    for index in range(sum(segment["frames"] for segment in spec["script"])):
        here = faults.get(index, [])
        if all(fault["kind"] != "drop" for fault in here):
            add("frame", ts, index)
        for fault in here:
            if fault["kind"] == "duplicate":
                add("duplicate", ts + 1_000_000, index)
            elif fault["kind"] == "focus_lost":
                add("focus_lost", ts + 2_000_000)
            elif fault["kind"] == "timeout":
                for _ in range(fault.get("count", 1)):
                    ts += fault["interval_ns"]
                    add("timeout", ts)
            else:
                assert fault["kind"] == "drop", fault
        ts += SUITE["session"]["frame_interval_ns"]
    return events


def _suite_replay(tmp_path: Path, spec: dict, name: str):
    """suite fixture を1回 replay し、(capture, result) を返す。"""
    capture = CaptureManifest.load(_write_capture(tmp_path, _capture_dict(_suite_events(spec))))
    scene = spec.get("detector_scene", SUITE["session"]["default_detector_scene"])
    return capture, _replay(tmp_path, name, capture, detector=SceneDetector(scene), hud_parser=SuiteHudParser(spec))


def test_suite_config_lists_every_plan_fixture() -> None:
    """suite は plan の検証一式(画面・card 種別/枚数・fault・場面)を全て含み、run 数は formal と同じ3回。"""
    fixtures = SUITE["fixtures"]
    segments = [segment for spec in fixtures for segment in spec["script"]]
    assert {"gameplay", "level_up_items", "level_up_fallback", "chest", "paused", "death", "result", "unknown"} <= {
        segment["screen"] for segment in segments
    }
    assert {"weapon", "evolved", "fallback"} <= {card[1] for segment in segments for card in segment.get("cards", ())}
    assert {1, 2, 3} <= {len(segment["cards"]) for segment in segments if segment.get("cards")}
    assert {"reroll", "skip"} <= {name for segment in segments for name in segment.get("buttons", ())}
    assert any(segment.get("confidence", 1.0) < 0.5 for segment in segments)
    assert {"timeout", "focus_lost", "drop", "duplicate"} <= {f["kind"] for spec in fixtures for f in spec.get("faults", ())}
    assert {"gems", "dense"} <= {spec.get("detector_scene") for spec in fixtures}
    assert SUITE["gate"]["runs"] == FORMAL_REPLAY_RUNS
    assert len(SUITE_FIXTURES) == len(fixtures)


@pytest.mark.parametrize("name", sorted(SUITE_FIXTURES))
def test_synthetic_suite_fixture_is_deterministic_over_three_runs(tmp_path: Path, name: str) -> None:
    """M11: 同じ bundle で3回 replay し、discrete hash 一致・numeric segment の quantized/tolerance gate を通る。

    safety fixture は unknown/focus loss/timeout 区間の effect が release/no-op だけであることも hard assertion する。
    """
    spec, gate = SUITE_FIXTURES[name], SUITE["gate"]
    runs = [_suite_replay(tmp_path, spec, f"run-{k}") for k in range(gate["runs"])]
    capture, results = runs[0][0], [result for _, result in runs]
    for result in results:
        assert result.manifest["development_only"] is True and result.manifest["formal_replay_eligible"] is False
    diffs = [
        compare_replays(results[0], result, tolerances=SUITE_TOLERANCES, quantum=gate["quantum"], obs_layout=OBS_LAYOUT)
        for result in results[1:]
    ]
    for diff in diffs:
        assert diff.passed, diff.first_divergence
        assert diff.discrete_sha256[0] == diff.discrete_sha256[1]
        assert diff.metrics["numeric_tolerance_pass_rate"] == 1.0
        assert all(s["quantized_equal"] or s["within_tolerance"] for s in diff.segments.values())
    expect, metrics = spec["expect"], replay_metrics(results[0])
    rows = _jsonl(results[0].paths["discrete.jsonl"])
    effects = json.loads(results[0].paths["effects.json"].read_text(encoding="utf-8"))
    assert metrics["exit_reason"] == expect.get("exit_reason", "stop_requested"), metrics
    assert set(expect["states"]) <= {row["payload"]["to_state"] for row in rows if row["stage"] == "state_machine"}
    assert set(expect.get("effects", ())) <= set(metrics["effect_counts"]), metrics
    # parser の confidence が閾値未満の frame からは移動を出さない(低 confidence を行動へ使わない)。
    low, start = set(), 0
    for segment in spec["script"]:
        if segment.get("confidence", 1.0) < 0.5:
            low.update(f"{SESSION}:{i}" for i in range(start, start + segment["frames"]))
        start += segment["frames"]
    assert not [e for e in effects if e["category"] == "movement" and e["correlation_id"] in low]
    if spec.get("safety"):
        assert_safe_effects(effects, fault_windows(capture.session, rows))


def test_cli_suite_replay_compare_golden_and_formal_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI: replay --runs 3 → compare → update-golden → golden 比較、分岐の検出、synthetic からの formal publish 拒否。"""
    spec = SUITE_FIXTURES["card_choice_variants"]
    profile = load_target_profile()
    data = _capture_dict(_suite_events(spec))
    data["session"].update(target_profile_hash=profile.target_hash, game_build_id=str(profile.sections["build"]["build_id"]))
    capture_path = _write_capture(tmp_path, data)
    policy = fixture.build_combat_policy(0)
    loads: list = []

    def load(args):
        """呼ばれるたびに新しい部品一式を返す(run 間で状態を共有しない)。"""
        loads.append(args)
        return type("Parts", (), dict(
            bundle=RuntimeBundle.from_golden_fixture(policy, item_selector=fixture.FirstCardItemSelector()),
            detector=SceneDetector("basic"), tracker=EntityTracker({i: 5 for i in range(12)}, 0.7, 0.6, 0.9),
            detector_manifest=None, hud_parser=SuiteHudParser(spec),
            artifact_hashes={"combat_policy": fixture._model_hash(policy)},
        ))

    monkeypatch.setattr(cli, "load_runtime_profile", lambda *_a: profile)
    monkeypatch.setattr(cli, "_load_artifacts", load)
    artifacts = [
        "--capture-manifest", str(capture_path), "--combat-package", "c", "--detector-config", "d",
        "--class-map", str(Path(fixture.__file__).parent / "configs" / "world_class_map_v1.yaml"),
        "--detector-weights", "w", "--detector-manifest", "x",
    ]
    out = tmp_path / "suite"
    assert cli.main(["replay", *artifacts, "--output-dir", str(out), "--runs", "3",
                     "--campaign-run-mode", "operator_debug_restart"]) == 0
    assert len(loads) == 3
    comparisons = json.loads((out / "comparisons.json").read_text(encoding="utf-8"))
    assert [c["passed"] for c in comparisons] == [True, True]
    runs = [out / f"run-{k}" for k in (1, 2, 3)]
    assert cli.main(["compare", "--old", str(runs[0]), "--new", str(runs[1])]) == 0

    golden, hashes = tmp_path / "golden.json", tmp_path / "hashes.json"
    manifest = json.loads((runs[1] / "manifest.json").read_text(encoding="utf-8"))
    hashes.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(GoldenUpdateError):
        cli.main(["update-golden", "--golden", str(golden), "--old-run", str(runs[0]), "--new-run", str(runs[1]),
                  "--artifact-hashes", str(hashes)])
    hashes.write_text(json.dumps({**manifest["artifact_hashes"],
                                  "capture_manifest_sha256": manifest["capture_manifest_sha256"]}), encoding="utf-8")
    assert cli.main(["update-golden", "--golden", str(golden), "--old-run", str(runs[0]), "--new-run", str(runs[1]),
                     "--artifact-hashes", str(hashes)]) == 0
    assert cli.main(["compare", "--golden", str(golden), "--new", str(runs[2])]) == 0

    def tamper(rows: list[dict]) -> None:
        next(row for row in rows if row["stage"] == "state_machine")["payload"]["to_state"] = "unknown"

    _mutate_jsonl(runs[2] / "discrete.jsonl", tamper)
    report = tmp_path / "report.json"
    assert cli.main(["compare", "--old", str(runs[0]), "--golden", str(golden), "--new", str(runs[2]),
                     "--report", str(report)]) == 1
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["passed"] is False and "discrete" in result["golden_mismatches"]
    assert result["first_divergence"]["frame"]["stage"]["name"] == "state_machine"

    verdict = tmp_path / "verdict.json"
    with pytest.raises(FormalReplayRejectedError):
        cli.main(["publish-formal-verdict", *artifacts, *[a for run in runs for a in ("--run", str(run))],
                  "--verdict", str(verdict)])
    assert not verdict.exists()


def test_thirty_minute_virtual_schedule_replays_within_wall_clock_budget(tmp_path: Path) -> None:
    """M11: 仮想時計で30分進む synthetic schedule を実 controller で再生し、wall-clock 予算(10分)内に終わる。"""
    bench = SUITE["benchmark"]
    interval = bench["frame_interval_ns"]
    frames = bench["virtual_duration_s"] * 1_000_000_000 // interval + 1
    events = [
        {"completion_seq": i, "kind": "frame", "timestamp_ns": START_NS + i * interval,
         "session_frame_index": i, "correlation_id": f"{SESSION}:{i}"}
        for i in range(frames)
    ]
    capture = CaptureManifest.load(_write_capture(tmp_path, _capture_dict(events)))
    started = time.perf_counter()
    result = _replay(tmp_path, "bench", capture, hud_parser=fixture.ScriptedHudParser([tuple(c) for c in bench["cycle"]]))
    elapsed = time.perf_counter() - started
    clock = result.manifest["virtual_clock"]
    assert clock["end_ns"] - clock["start_ns"] >= bench["virtual_duration_s"] * 1_000_000_000
    assert result.exit_reason == "stop_requested"
    assert replay_metrics(result)["stage_counts"]["capture"] == frames
    assert elapsed < bench["wall_clock_budget_s"], f"30-minute virtual replay took {elapsed:.1f}s"

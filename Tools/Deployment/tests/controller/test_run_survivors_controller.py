"""``run_survivors_controller.py`` の起動フラグ・live fail-closed・終了コード伝播を検証する。

実機ウィンドウ・detector 重み・入力 helper だけを fake に差し替え、CLI の ``main()`` を
実際の SurvivorsController・AgentRuntime・RuntimeBundle(development)・CheckpointManifest で動かします。
M8(既定 shadow、live は3点セット必須)と M10(formal verdict が無ければ --live は非0で拒否)を確認します。
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from survivors.capture.captured_frame import CapturedFrame
from survivors.capture.frame_capture import LatestFrameQueue
from survivors.controller.controller import EXIT_ERROR, EXIT_HEALTH_STOP, EXIT_OK
from survivors.runtime.artifact_bundle import (
    REQUIRED_ACTION_DIM,
    BundleLoadError,
    CombatGruPolicy,
    CombatPolicy,
    RuntimeBundle,
)
from survivors.target_profile import load_target_profile
from survivors.vision.entity_tracker import EntityTracker
from survivors.vision.world_detector import CheckpointManifest, DetectionResult, FormalDetectorRejectedError

import run_survivors_controller as cli

_PIXELS = np.zeros((1080, 1920, 4), dtype=np.uint8)
_CLASS_MAP = Path(cli.__file__).parent / "configs" / "world_class_map_v1.yaml"
_REQUIRED = [
    "--combat-package", "c", "--detector-config", "d.yaml", "--class-map", str(_CLASS_MAP),
    "--detector-weights", "w.pt", "--detector-manifest", "manifest.json",
]


def _combat_policy() -> CombatPolicy:
    """runtime 契約どおりの次元を持つ小さな combat policy を作る。"""
    obs_dim = 3 * DeployObsSchema.default_v1().dim
    model = CombatGruPolicy(observation_dim=obs_dim, action_dim=REQUIRED_ACTION_DIM, hidden_dim=8).eval()
    return CombatPolicy(model=model, observation_dim=obs_dim, action_dim=REQUIRED_ACTION_DIM, hidden_dim=8)


def _detector_manifest() -> CheckpointManifest:
    """04-07 の development detector manifest(formal 不可)を作る。"""
    return CheckpointManifest(
        model_hash="1" * 64, data_hash="2" * 64, config_hash="3" * 64, build_hash="4" * 64,
        class_map_hash="5" * 64, formal_detector_eligible=False,
    )


class RealClockCapture:
    """perf_counter_ns 時刻で frame を LatestFrameQueue へ積む capture。

    controller の実時計と同じ時刻源を使い、health の latency 判定を実運用と揃えます。
    """

    def __init__(self) -> None:
        """frame 番号と開始/停止の記録を初期化する。"""
        self.frames = LatestFrameQueue()
        self.index = 0
        self.events: list[str] = []

    def start(self) -> None:
        """開始を記録する。"""
        self.events.append("start")

    def capture_next(self):
        """新しい frame を1枚積んで返す。"""
        frame = CapturedFrame(
            frame_bgra=_PIXELS, captured_monotonic_ns=time.perf_counter_ns(), session_frame_index=self.index,
            client_rect_screen_px=(0, 0, 1920, 1080), foreground=True,
            target_profile_hash="a" * 64, game_build_id="build-1",
        )
        self.index += 1
        self.frames.put_latest(frame)
        return frame

    def close(self) -> None:
        """停止を記録する。"""
        self.events.append("close")


class EmptyDetector:
    """空の検出結果を返し、``fail`` なら例外を出す detector。"""

    def __init__(self, fail: bool = False) -> None:
        """失敗させるかを保持する。"""
        self._fail = fail

    def infer(self, frame_bgr, *, score_threshold):
        """空の DetectionResult を返す。"""
        if self._fail:
            raise RuntimeError("detector exploded")
        return DetectionResult(
            boxes_xyxy=np.zeros((0, 4), np.float32), scores=np.zeros(0, np.float32),
            class_ids=np.zeros(0, np.int32), image_width=1920, image_height=1080,
        )


class FakeLease:
    """InputLeaseController の代わりに構築引数と解放を記録する context manager。"""

    instances: list["FakeLease"] = []

    def __init__(self, **kwargs) -> None:
        """構築引数を記録する。"""
        self.kwargs = kwargs
        self.released = 0
        FakeLease.instances.append(self)

    def __enter__(self):
        """自身を返す。"""
        return self

    def __exit__(self, *_exc) -> None:
        """何もしない。"""

    def send_action(self, action_index: int) -> bool:
        """移動入力を受理する。"""
        return True

    def send_ui_click(self, x: float, y: float) -> bool:
        """UI click を受理する。"""
        return True

    def send_ui_key(self, key: str) -> bool:
        """UI key を受理する。"""
        return True

    def emergency_release(self) -> bool:
        """解放回数を数える。"""
        self.released += 1
        return True


def _forbidden(name: str):
    """呼ばれたら失敗する罠を返す。"""
    def trap(*_args, **_kwargs):
        raise AssertionError(f"{name} must not be called")
    return trap


@pytest.fixture
def env(monkeypatch, tmp_path):
    """実機依存(profile/combat package/detector/capture/入力 helper)だけを差し替える。

    capture と入力 helper は既定で罠にしておき、各テストが必要な分だけ上書きします。
    """
    FakeLease.instances.clear()
    state = SimpleNamespace(detector=EmptyDetector(), capture=RealClockCapture(), telemetry=tmp_path / "t.jsonl")
    monkeypatch.setattr(cli, "load_runtime_profile", lambda *_a: load_target_profile())
    monkeypatch.setattr(cli, "_load_combat_package", lambda _p: (_combat_policy(), {"model_sha256": "6" * 64}))
    monkeypatch.setattr(
        cli, "_load_detector",
        lambda _a: (state.detector, EntityTracker({i: 5 for i in range(12)}, 0.7, 0.6, 0.9), _detector_manifest()),
    )
    monkeypatch.setattr(cli, "_open_capture", _forbidden("_open_capture"))
    monkeypatch.setattr(cli, "InputLeaseController", _forbidden("InputLeaseController"))
    state.use_capture = lambda: monkeypatch.setattr(
        cli, "_open_capture", lambda _p: (state.capture, SimpleNamespace(pid=111, hwnd=222))
    )
    state.monkeypatch = monkeypatch
    return state


def _main(state, *extra: str) -> int:
    """CLI の main() を実行し、SystemExit の終了コードを返す。"""
    with pytest.raises(SystemExit) as exc:
        cli.main(["--telemetry", str(state.telemetry), *_REQUIRED, *extra])
    return exc.value.code


def _rows(path) -> list[dict]:
    """telemetry JSONL を全行読む。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_default_mode_is_shadow():
    """フラグなしは shadow(live=False)で起動する。"""
    args = cli.parse_args(_REQUIRED + ["--telemetry", "t.jsonl"])
    assert args.live is False and args.ack_risk is False


@pytest.mark.parametrize(
    "flags",
    [["--live"], ["--live", "--target-profile", "p.yaml"], ["--live", "--ack-risk"], ["--ack-risk"]],
)
def test_live_requires_all_three_flags(flags):
    """--live/--target-profile/--ack-risk のどれかが欠けると argparse エラー(exit 2)。"""
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(_REQUIRED + ["--telemetry", "t.jsonl", *flags])
    assert exc.value.code == 2


def test_live_accepted_with_three_flags():
    """3点セットがそろえば live として解釈する。"""
    args = cli.parse_args(_REQUIRED + ["--telemetry", "t.jsonl", "--live", "--target-profile", "p.yaml", "--ack-risk"])
    assert args.live and args.ack_risk and str(args.target_profile) == "p.yaml"


def test_live_gate_rejects_development_bundle_and_detector():
    """development bundle も development detector も live gate を通らない(M10)。"""
    with pytest.raises(BundleLoadError):
        cli._live_gate(RuntimeBundle.from_golden_fixture(_combat_policy()), _detector_manifest())
    eligible_bundle = SimpleNamespace(assert_live_eligible=lambda: None)
    with pytest.raises(FormalDetectorRejectedError):
        cli._live_gate(eligible_bundle, _detector_manifest())


def test_live_fails_closed_before_capture_and_input(env):
    """formal verdict が無い --live は capture/入力を作らずに非0終了し、理由を telemetry に残す。"""
    code = _main(env, "--live", "--target-profile", "p.yaml", "--ack-risk")
    assert code == cli.EXIT_LIVE_REJECTED != 0
    rows = _rows(env.telemetry)
    assert rows[0]["mode"] == "live"
    gate = [row for row in rows if row.get("stage") == "live_gate"]
    assert len(gate) == 1 and gate[0]["payload"]["eligible"] is False
    assert gate[0]["payload"]["type"] == "BundleLoadError"
    assert env.capture.events == []


def test_live_runtime_double_gate_blocks_input(env):
    """gate を迂回しても AgentRuntime(require_live) が入力 helper 構築前に止める。"""
    env.monkeypatch.setattr(cli, "_live_gate", lambda *_a: None)
    env.use_capture()
    code = _main(env, "--live", "--target-profile", "p.yaml", "--ack-risk")
    assert code == EXIT_ERROR
    errors = [row for row in _rows(env.telemetry) if row.get("stage") == "error"]
    assert errors[0]["payload"]["type"] == "BundleLoadError"


def test_shadow_runs_real_controller_without_input(env):
    """shadow は入力 helper を一切作らず、実 controller の run() 結果(0)で終了する。"""
    env.use_capture()
    code = _main(env, "--max-frames", "3")
    rows = _rows(env.telemetry)
    assert code == EXIT_OK, [row["payload"] for row in rows if row.get("stage") in ("shutdown", "error")]
    stages = [row.get("stage") for row in rows]
    assert rows[0]["mode"] == "shadow"
    assert "arm" in stages and stages.count("capture") == 3
    shutdown = [row for row in rows if row.get("stage") == "shutdown"][0]
    assert shutdown["payload"]["reason"] == "max_frames"
    assert env.capture.events == ["start", "close"]


def test_header_records_hash_of_class_map_actually_read(env):
    """telemetry header の detector_class_map は manifest の自己申告ではなく、読んだ class map の実 hash。"""
    env.use_capture()
    assert _main(env, "--max-frames", "1") == EXIT_OK
    hashes = _rows(env.telemetry)[0]["artifact_hashes"]
    assert hashes["detector_class_map"] == cli._sha256_file(_CLASS_MAP) != _detector_manifest().class_map_hash


@pytest.mark.parametrize("mismatch", ["weights", "config", "class_map"])
def test_load_detector_rejects_files_not_matching_manifest(tmp_path, monkeypatch, mismatch):
    """weight/config/class map のどれか1つでも manifest の hash と違えば、読み込む前に ValueError で拒否する。"""
    paths = {name: tmp_path / name for name in ("weights", "config", "class_map")}
    for name, path in paths.items():
        path.write_bytes(name.encode())
    manifest = CheckpointManifest(
        model_hash=cli._sha256_file(paths["weights"]), data_hash="2" * 64,
        config_hash=cli._sha256_file(paths["config"]), build_hash="4" * 64,
        class_map_hash=cli._sha256_file(paths["class_map"]), formal_detector_eligible=False,
    )
    paths[mismatch].write_bytes(b"tampered")
    monkeypatch.setattr(cli.CheckpointManifest, "load", staticmethod(lambda _p: manifest))
    monkeypatch.setattr(cli, "load_detector_config", _forbidden("load_detector_config"))
    args = SimpleNamespace(
        detector_manifest=tmp_path / "manifest.json", detector_weights=paths["weights"],
        detector_config=paths["config"], class_map=paths["class_map"],
    )
    expected = {"weights": "model_hash", "config": "config_hash", "class_map": "class_map_hash"}[mismatch]
    with pytest.raises(ValueError, match=expected):
        cli._load_detector(args)


def test_health_stop_exit_code_becomes_process_exit_code(env):
    """health STOP(推論エラー)時の run() 戻り値 2 がそのままプロセス終了コードになる。"""
    env.detector = EmptyDetector(fail=True)
    env.monkeypatch.setattr(
        cli, "_load_detector",
        lambda _a: (env.detector, EntityTracker({i: 5 for i in range(12)}, 0.7, 0.6, 0.9), _detector_manifest()),
    )
    env.use_capture()
    assert _main(env, "--max-frames", "3") == EXIT_HEALTH_STOP


def test_live_with_formal_verdict_binds_input_to_located_window(env):
    """gate を通った live は、特定した pid/hwnd に固定した入力 helper を controller へ渡す。"""
    env.monkeypatch.setattr(cli, "_live_gate", lambda *_a: None)
    real_runtime = cli.AgentRuntime
    env.monkeypatch.setattr(cli, "AgentRuntime", lambda bundle, require_live: real_runtime(bundle))
    env.monkeypatch.setattr(cli, "InputLeaseController", FakeLease)
    env.use_capture()
    assert _main(env, "--live", "--target-profile", "p.yaml", "--ack-risk", "--max-frames", "1") == EXIT_OK
    lease = FakeLease.instances[0]
    assert (lease.kwargs["target_pid"], lease.kwargs["target_hwnd"]) == (111, 222)
    assert lease.released == 1

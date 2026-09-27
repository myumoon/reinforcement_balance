"""30分相当の virtual schedule を実 SurvivorsController へ流す development fixture replay tool(M9)。

実画面キャプチャと学習済み detector が無くても controller 全体を長時間動かせるようにする道具です。
capture・detector・HUD parser だけを台本(script)どおりの fake に差し替え、tracker・obs assembler・
AgentRuntime・state machine・health monitor・telemetry はすべて本物を使います。
実行後は telemetry から M11 gate verdict を計算し、成果物を ArtifactStore へ登録して読み戻し検証します。
結果は development_only=true の開発用 smoke であり、正式30分 replay・formal shadow verdict・
live canary の代わりには使いません(I4)。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import tracemalloc
from typing import Any
import uuid

import numpy as np
import torch

from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.ui_policy import NonModelUiPolicyConfigV1
from survivors.capture.captured_frame import CapturedFrame
from survivors.capture.frame_capture import LatestFrameQueue
from survivors.controller import controller as controller_module
from survivors.controller.controller import SurvivorsController
from survivors.controller.gate import issue_gate_verdict, read_telemetry, write_gate_verdict
from survivors.controller.health_monitor import HealthMonitor
from survivors.controller.state_machine import CampaignRunMode, StateMachine
from survivors.controller.telemetry import TelemetrySessionHeader, TelemetryWriter
from survivors.real_obs_assembler import RealObsAssembler
from survivors.runtime.agent_runtime import AgentRuntime
from survivors.runtime.artifact_bundle import REQUIRED_ACTION_DIM, CombatGruPolicy, CombatPolicy, RuntimeBundle
from survivors.runtime.decision_scheduler import DecisionScheduler
from survivors.vision.entity_tracker import EntityTracker
from survivors.vision.hud_parser import HudStateV1, ParsedCard
from survivors.vision.world_detector import DetectionResult

MANIFEST_SCHEMA_VERSION = "survivors.controller_fixture_replay.v1"
# frame 間隔は 1/15 秒の切り上げ(66,666,667ns)。assembler の tick 間隔(round)と scheduler(floor)の
# どちらも毎 frame を受け付ける最小値で、floor 値だと assembler が1枚おきに snapshot を捨てる。
TICK_NS = -(-1_000_000_000 // DecisionScheduler().hz)
FULL_SCHEDULE_TICKS = 30 * 60 * 15  # 30分 x 15 Hz(TestThirtyMinuteSoak と同じ)
DEFAULT_MEMORY_SAMPLE_EVERY = 300  # 20秒(virtual)ごと
# 1周期 900 tick(約1分)。level_up は 2s timeout、chest は 5s timeout より短く保つ。
DEFAULT_CYCLE: tuple[tuple[str, int], ...] = (
    ("gameplay", 700),
    ("level_up_items", 15),
    ("gameplay", 60),
    ("chest", 15),
    ("gameplay", 100),
    ("death", 10),
)
_PIXELS = np.zeros((1080, 1920, 4), dtype=np.uint8)
_PROFILE_HASH = hashlib.sha256(b"controller-fixture-replay-target").hexdigest()
_BUILD_ID = "fixture-replay"


def screen_state_at(tick: int, cycle: Sequence[tuple[str, int]] = DEFAULT_CYCLE) -> str:
    """tick 番号に対応する台本上の raw screen_state を返す。

    cycle を先頭から順に並べた周期を繰り返し、tick がどの区間に入るかで画面を決めます。
    """
    offset = tick % sum(length for _, length in cycle)
    for state, length in cycle:
        if offset < length:
            return state
        offset -= length
    raise AssertionError("unreachable")


class ReplayClock:
    """実経過時間で進みつつ、frame 予定時刻まで前へ跳べる単調時計。

    frame と frame の間の待ち時間だけを飛ばし、処理にかかった時間は実測のまま残します。
    そのため30分の台本を数十秒で流しても、telemetry の遅延(p99)は本物の処理時間になります。
    """

    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        """virtual 開始時刻と、対応する実時計の基準点を記録する。"""
        self._offset = start_ns - time.perf_counter_ns()

    def __call__(self) -> int:
        """現在の virtual 時刻(ns)を返す。"""
        return time.perf_counter_ns() + self._offset

    def advance_to(self, target_ns: int) -> int:
        """target_ns が未来ならそこまで跳び、跳んだ後の時刻を返す(過去へは戻らない)。"""
        now = self()
        if target_ns > now:
            self._offset += target_ns - now
            return target_ns
        return now


class ScriptedCapture:
    """15 Hz の予定時刻に frame を1枚ずつ積み、一定間隔で memory sample を採る capture。

    frame の中身は空ですが、番号と時刻は本物の CaptureSession と同じ形で LatestFrameQueue へ積みます。
    memory sample は frame 時刻を決める前に採るので、gc の時間は遅延の標本に混ざりません。
    """

    def __init__(self, clock: ReplayClock, *, sample_every: int) -> None:
        """時計・sample 間隔と、frame 番号・memory sample の記録を初期化する。"""
        self.frames = LatestFrameQueue()
        self.memory_samples: list[list[int]] = []
        self._clock = clock
        self._sample_every = sample_every
        self._start_ns = clock()
        self._index = 0

    def start(self) -> None:
        """controller から呼ばれる開始通知(何もしない)。"""

    def capture_next(self) -> CapturedFrame:
        """次の予定時刻まで時計を進め、frame を1枚積んで返す。"""
        if self._index and self._index % self._sample_every == 0:
            gc.collect()
            current, _ = tracemalloc.get_traced_memory()
            self.memory_samples.append([self._clock(), current])
        captured = self._clock.advance_to(self._start_ns + self._index * TICK_NS)
        frame = CapturedFrame(
            frame_bgra=_PIXELS, captured_monotonic_ns=captured, session_frame_index=self._index,
            client_rect_screen_px=(0, 0, 1920, 1080), foreground=True,
            target_profile_hash=_PROFILE_HASH, game_build_id=_BUILD_ID,
        )
        self._index += 1
        self.frames.put_latest(frame)
        return frame

    def close(self) -> None:
        """controller から呼ばれる停止通知(queue の中身は controller が drain する)。"""


class FixedSceneDetector:
    """画面中央の player anchor と敵1体だけを毎 frame 返す detector。

    画面状態は HUD の台本だけで決まるので、world 側は固定の小さな場面で十分です。
    ただし検出0件だと player anchor が fallback になり、combat の validity gate で
    すべて no_op になってしまうため、``_hud_world`` と同じく anchor と敵1体を置きます。
    """

    _RESULT = DetectionResult(
        boxes_xyxy=np.array([[940, 520, 980, 560], [1300, 300, 1340, 340]], np.float32),
        scores=np.array([.9, .9], np.float32), class_ids=np.array([1, 2], np.int32),  # player_anchor, enemy_normal
        image_width=1920, image_height=1080,
    )

    def infer(self, frame_bgr: np.ndarray, *, score_threshold: float) -> DetectionResult:
        """frame を見ずに固定の DetectionResult を返す。"""
        return self._RESULT


class ScriptedHudParser:
    """frame 番号に応じて台本の screen_state を持つ HudStateV1 を返す HUD parser。

    組み立て方は tests/runtime/test_agent_runtime.py の ``_hud_world`` と同じで、
    level-up 画面では有効なカード1枚(whip)を提示し、実 assembler が UI 候補を解決できるようにします。
    """

    def __init__(self, cycle: Sequence[tuple[str, int]] = DEFAULT_CYCLE) -> None:
        """台本の周期を保持する。"""
        self._cycle = tuple(cycle)
        self.resets = 0

    def reset_temporal_state(self) -> None:
        """arm 時のリセットを数える(台本は時系列状態を持たない)。"""
        self.resets += 1

    def parse(self, frame_bgra, *, session_id: str, frame_index: int, captured_monotonic_ns: int) -> HudStateV1:
        """frame 番号の台本画面で HudStateV1 を組み立てる。"""
        card = ParsedCard(0, "whip", "weapon", 2, .99, "ok", (100, 100, 400, 500))
        return HudStateV1(
            "hud_state.v1", session_id, frame_index, captured_monotonic_ns, "a" * 64,
            screen_state_at(frame_index, self._cycle), .9, "ok",
            20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
            ("whip",) + (None,) * 11, .9, "b" * 64, (card,), "c" * 64, (),
            False, False, False, .9, "ok",
        )


class FirstCardItemSelector:
    """level-up 候補の先頭カードを選ぶ固定 logits の item selector。

    学習済み ItemSelector(ONNX)が無い代わりに置く最小の差し替えです。
    入力の組み立て(ItemSession)と候補の UI 解決は本物の経路をそのまま通ります。
    selector が無いと level-up では候補選択(choose_card)が一度も出ず、gate の
    level-up 候補 invalid rate が「母数0」のまま意味を持たなくなるため用意しています。
    """

    nmax = 3
    feature_schema = "context_danger_occupancy_v1"  # RealObsAssembler が出す item context の schema
    temperature = 1.0
    confidence_threshold = 0.0
    ui_policy_config = NonModelUiPolicyConfigV1.load_default()

    def predict(self, context: np.ndarray, candidates: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """先頭候補が最大になる logits を返す。"""
        return np.array([[9.0, 0.0, 0.0]], dtype=np.float32)


def build_combat_policy(seed: int = 0) -> CombatPolicy:
    """runtime 契約どおりの次元を持つ小さな development combat policy を作る。

    重みは乱数の種で固定するので、同じ seed なら毎回同じ policy になります。
    """
    torch.manual_seed(seed)
    obs_dim = 3 * DeployObsSchema.default_v1().dim
    model = CombatGruPolicy(observation_dim=obs_dim, action_dim=REQUIRED_ACTION_DIM, hidden_dim=8).eval()
    return CombatPolicy(model=model, observation_dim=obs_dim, action_dim=REQUIRED_ACTION_DIM, hidden_dim=8)


def _model_hash(policy: CombatPolicy) -> str:
    """combat policy の重みから sha256 を計算する。

    telemetry の artifact_hashes に、どの重みで replay したかを残すために使います。
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class ReplayResult:
    """1回の fixture replay の結果(終了コード・gate verdict・manifest と各出力パス)。

    test と CLI が同じ関数の戻り値から結果を確認できるようにまとめたものです。
    """

    exit_code: int
    exit_reason: str | None
    verdict: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path
    telemetry_path: Path
    errors: list[dict[str, str]] = field(default_factory=list)


def run_replay(
    output_dir: Path | str,
    *,
    ticks: int = FULL_SCHEDULE_TICKS,
    sample_every: int = DEFAULT_MEMORY_SAMPLE_EVERY,
    session_id: str | None = None,
    store_root: Path | str | None = None,
    seed: int = 0,
    cycle: Sequence[tuple[str, int]] = DEFAULT_CYCLE,
) -> ReplayResult:
    """fixture replay を1回実行し、gate 判定と ArtifactStore 登録・読み戻し検証まで行う。

    手順は 1) 実部品で controller を組んで arm する 2) ``run(max_frames=ticks)`` で台本を流す
    3) telemetry から gate verdict を計算して JSON/Markdown に書く 4) telemetry・verdict・
    memory sample を ArtifactStore へ put して verify する 5) development_only=true の manifest を書く、です。
    """
    if ticks < 1 or sample_every < 1:
        raise ValueError("ticks and sample_every must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = session_id or f"fixture-replay-{uuid.uuid4().hex[:12]}"
    telemetry_path = output_dir / "telemetry.jsonl"
    policy = build_combat_policy(seed)
    model_hashes = {"combat_policy": _model_hash(policy)}
    bundle = RuntimeBundle.from_golden_fixture(policy, item_selector=FirstCardItemSelector())
    clock = ReplayClock()
    capture = ScriptedCapture(clock, sample_every=sample_every)
    telemetry = TelemetryWriter(telemetry_path, TelemetrySessionHeader(
        session_id=session_id, mode="shadow", target_profile_hash=_PROFILE_HASH, game_build_id=_BUILD_ID,
        controller_build_id=hashlib.sha256(Path(controller_module.__file__).read_bytes()).hexdigest(),
        artifact_hashes=model_hashes,
        host={"platform": platform.platform(), "node": platform.node()},
        device={"inference": "cpu"},
        dependency_versions={"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        deterministic_replay={"kind": "controller_fixture_replay", "ticks": ticks, "seed": seed,
                              "cycle": [list(item) for item in cycle]},
    ))
    controller = SurvivorsController(
        mode="shadow", session_id=session_id, capture=capture, detector=FixedSceneDetector(),
        tracker=EntityTracker({i: 5 for i in range(12)}, 0.7, 0.6, 0.9), hud_parser=ScriptedHudParser(cycle), assembler=RealObsAssembler(),
        runtime=AgentRuntime(bundle, clock_ns=clock), state_machine=StateMachine(), health=HealthMonitor(),
        telemetry=telemetry, schema=bundle.deploy_schema, model_hashes=model_hashes,
        ui_config=bundle.ui_policy_config, clock_ns=clock, sleep=lambda _: None,
    )
    # ponytail: tracemalloc は Python object しか見えない(torch の native 確保は対象外)。
    # native 側の増加まで gate したくなったら psutil の RSS へ置き換える。
    tracemalloc.start()
    try:
        controller.arm(
            campaign_run_mode=CampaignRunMode.OPERATOR_DEBUG_RESTART,
            run_id=session_id, gameplay_attempt_id=f"{session_id}-attempt-1",
        )
        exit_code = controller.run(max_frames=ticks)
    finally:
        tracemalloc.stop()

    memory_samples = capture.memory_samples
    verdict = issue_gate_verdict(read_telemetry(telemetry_path), memory_samples if len(memory_samples) >= 2 else None)
    telemetry_bytes = telemetry_path.read_bytes()
    verdict["telemetry_sha256"] = hashlib.sha256(telemetry_bytes).hexdigest()
    gate_json, gate_md = output_dir / "gate_verdict.json", output_dir / "gate_verdict.md"
    write_gate_verdict(verdict, gate_json, gate_md)
    memory_json = json.dumps(memory_samples).encode("utf-8")
    (output_dir / "memory_samples.json").write_bytes(memory_json)

    store = ArtifactStore(store_root if store_root is not None else output_dir / "artifact_store")
    artifacts = []
    for name, data, media_type in (
        ("telemetry.jsonl", telemetry_bytes, "application/x-ndjson"),
        ("gate_verdict.json", gate_json.read_bytes(), "application/json"),
        ("gate_verdict.md", gate_md.read_bytes(), "text/markdown"),
        ("memory_samples.json", memory_json, "application/json"),
    ):
        ref = store.put_bytes(logical_id=f"controller_fixture_replay/{session_id}/{name}", data=data,
                              media_type=media_type)
        check = store.verify(ref, expected_size_bytes=ref.size_bytes)
        artifacts.append({"ref": ref.to_wire(), "restore_verified": check.ok, "restore_reason": check.reason})

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        # I4: fixture replay は開発用 smoke。正式30分 replay・formal shadow・live canary の代替ではない。
        "development_only": True,
        "formal_shadow_eligible": False,
        "session_id": session_id,
        "mode": "shadow",
        "schedule": {"ticks": ticks, "tick_ns": TICK_NS, "virtual_duration_ns": ticks * TICK_NS,
                     "full_schedule_ticks": FULL_SCHEDULE_TICKS, "memory_sample_every": sample_every,
                     "cycle": [list(item) for item in cycle]},
        "exit_code": exit_code,
        "exit_reason": controller.exit_reason,
        "gate_status": verdict["status"],
        "gate_fail_reasons": verdict["fail_reasons"],
        "artifacts": artifacts,
        "restore_verified": all(item["restore_verified"] for item in artifacts),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ReplayResult(exit_code, controller.exit_reason, verdict, manifest, manifest_path, telemetry_path,
                        list(controller.errors))


def build_parser() -> argparse.ArgumentParser:
    """CLI の引数解析器を組み立てる。

    既定は30分相当(27,000 tick)。``--ticks`` で短い smoke にも縮められます。
    """
    parser = argparse.ArgumentParser(description="30分相当の fixture schedule を shadow controller へ流して gate を判定する(development only)。")
    parser.add_argument("--output-dir", required=True, type=Path, help="telemetry・verdict・manifest の出力先")
    parser.add_argument("--ticks", type=int, default=FULL_SCHEDULE_TICKS, help="処理する frame 数(既定: 30分 x 15 Hz)")
    parser.add_argument("--memory-sample-every", type=int, default=DEFAULT_MEMORY_SAMPLE_EVERY,
                        help="memory sample を採る frame 間隔")
    parser.add_argument("--session-id", default=None, help="session id(省略時は乱数で生成)")
    parser.add_argument("--store-root", type=Path, default=None, help="ArtifactStore の root(既定: <output-dir>/artifact_store)")
    parser.add_argument("--seed", type=int, default=0, help="development combat policy の乱数 seed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI エントリポイント。

    gate が PASS・restore 検証が全件成功・controller が正常終了のときだけ0、それ以外は1を返します。
    """
    args = build_parser().parse_args(argv)
    result = run_replay(
        args.output_dir, ticks=args.ticks, sample_every=args.memory_sample_every,
        session_id=args.session_id, store_root=args.store_root, seed=args.seed,
    )
    print(f"gate={result.verdict['status']} exit_code={result.exit_code} "
          f"restore_verified={result.manifest['restore_verified']} manifest={result.manifest_path}")
    ok = result.verdict["status"] == "PASS" and result.manifest["restore_verified"] and result.exit_code == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

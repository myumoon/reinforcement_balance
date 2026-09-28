"""recorded capture session を実 SurvivorsController へ仮想時計で流す E2E replay engine(06-01 タスク2)。

capture manifest(記録 session + frame 画素ファイル)と target profile・game build を照合してから、
RecordedFrameSource と VirtualClock を controller の既存注入点(capture/clock_ns/sleep)へ渡して実行します。
controller は常に shadow mode で動かすので ``execute_effect`` は呼ばれず、OS 入力は一切出ません。
実行後は telemetry を読み、全 stage の時刻と correlation id が仮想時計・記録 frame に揃っていることを確かめ、
結果を「exact 比較する離散値(discrete)」と「tolerance 比較する数値(numeric)」に分けて保存します。
後半は golden/diff(06-01 タスク3): numeric の quantize と segment 別 tolerance、2回の replay の比較と
最初の分岐点の tree report、golden 更新、安全 fixture の hard assertion、formal verdict の publish ガードです。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
from typing import Any
import zipfile

import numpy as np
from numpy.typing import NDArray

from ..controller import controller as controller_module
from ..controller.controller import SurvivorsController
from ..controller.health_monitor import HealthMonitor
from ..controller.state_machine import CampaignRunMode, ControllerState, StateMachine
from ..controller.telemetry import TelemetrySessionHeader, TelemetryWriter
from ..real_obs_assembler import RealObsAssembler
from ..runtime.agent_runtime import AgentRuntime
from ..runtime.artifact_bundle import RuntimeBundle
from .recorded_frame_source import DeterminismManifest, RecordedFrameSource, RecordedSession
from .virtual_clock import VirtualClock

CAPTURE_MANIFEST_SCHEMA_VERSION = "survivors.recorded_capture.v1"
OUTPUT_MANIFEST_SCHEMA_VERSION = "survivors.e2e_replay.v1"
_CAPTURE_MANIFEST_KEYS = frozenset({"schema_version", "session", "frames_path", "frames_sha256", "development_only"})
_SHA256_CHARS = frozenset("0123456789abcdef")
_FRAME_SHAPE = (1080, 1920, 4)
# 実行ごとに変わる値(uuid 由来の decision id とその hash、raw float の obs hash)。
# exact 比較にも tolerance 比較にも使えないので discrete/numeric の両方から外す。
VOLATILE_KEYS = frozenset({"decision_id", "decision_hash", "obs_hash"})
# controller の effect 種別 → effect recorder の意味カテゴリ。
EFFECT_CATEGORIES = {
    "move": "movement",
    "ui_click": "ui",
    "ui_key": "ui",
    "release_all": "release",
    "combat_reset": "control",
    "controller_stop": "control",
    "process_terminate": "control",
}


class ReplayIntegrityError(ValueError):
    """capture manifest の照合失敗、または telemetry が仮想時計・記録 frame に揃っていないときに送出する。"""


def _sha256_bytes(data: bytes) -> str:
    """bytes の sha256 hex を返す。"""
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: object) -> bool:
    """value が小文字 64 桁の sha256 hex かどうか。"""
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_CHARS


@dataclass(frozen=True)
class CaptureManifest:
    """記録済み capture session 1本分の manifest(event 列・画素ファイルとその hash・開発用フラグ)。

    画素は ``frames_path`` の NPZ に ``frame_<番号>`` の名前で 1080x1920 BGRA が入っています。
    ``frames_path`` が null の manifest は全 frame を黒画面で代用する synthetic session で、
    ``development_only=true`` のときだけ許します(正式 replay の入力にはなりません)。
    """

    session: RecordedSession
    frames_path: Path | None
    frames_sha256: str | None
    development_only: bool
    manifest_sha256: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, base_dir: Path, manifest_sha256: str) -> "CaptureManifest":
        """JSON 由来の dict を検証して manifest を作る(キーの過不足・hash 不一致・画素欠落は拒否)。

        frames_path は manifest ファイルからの相対パスとして解決し、ファイル全体の sha256 が
        frames_sha256 と一致すること、記録 event が参照する全 frame 番号の画素が入っていることを確かめます。
        """
        if not isinstance(data, Mapping) or set(data) != _CAPTURE_MANIFEST_KEYS:
            raise ReplayIntegrityError(f"capture manifest keys must be {sorted(_CAPTURE_MANIFEST_KEYS)}")
        if data["schema_version"] != CAPTURE_MANIFEST_SCHEMA_VERSION:
            raise ReplayIntegrityError(f"capture manifest schema_version must be {CAPTURE_MANIFEST_SCHEMA_VERSION}")
        if type(data["development_only"]) is not bool:
            raise ReplayIntegrityError("capture manifest development_only must be a bool")
        session = RecordedSession.from_dict(data["session"])
        if data["frames_path"] is None:
            if data["frames_sha256"] is not None or not data["development_only"]:
                raise ReplayIntegrityError("synthetic capture (frames_path=null) must be development_only without frames_sha256")
            return cls(session, None, None, True, manifest_sha256)
        if not isinstance(data["frames_path"], str) or not _is_sha256(data["frames_sha256"]):
            raise ReplayIntegrityError("frames_path must be a string and frames_sha256 a sha256 hex digest")
        frames_path = base_dir / data["frames_path"]
        if _sha256_bytes(frames_path.read_bytes()) != data["frames_sha256"]:
            raise ReplayIntegrityError("recorded frames file hash does not match frames_sha256")
        needed = {f"frame_{event.session_frame_index}" for event in session.events if event.session_frame_index is not None}
        with np.load(frames_path) as archive:
            missing = needed - set(archive.files)
        if missing:
            raise ReplayIntegrityError(f"recorded frames file lacks {sorted(missing)[:5]}")
        return cls(session, frames_path, data["frames_sha256"], data["development_only"], manifest_sha256)

    @classmethod
    def load(cls, path: Path | str) -> "CaptureManifest":
        """manifest JSON ファイルを読み、hash 付きで検証済み manifest を返す。"""
        path = Path(path)
        raw = path.read_bytes()
        return cls.from_dict(json.loads(raw), base_dir=path.parent, manifest_sha256=_sha256_bytes(raw))

    def require_identity(self, *, target_profile_hash: str, game_build_id: str) -> None:
        """記録 session の target profile / game build が再生側の値と一致しなければ拒否する。

        別の画面設定や別のゲーム版で録った frame を今の profile で再生すると、
        回帰ではなく入力の取り違えを検出してしまうためです。
        """
        if self.session.target_profile_hash != target_profile_hash:
            raise ReplayIntegrityError(
                f"target_profile_hash mismatch: recorded={self.session.target_profile_hash} replay={target_profile_hash}"
            )
        if self.session.game_build_id != game_build_id:
            raise ReplayIntegrityError(
                f"game_build_id mismatch: recorded={self.session.game_build_id!r} replay={game_build_id!r}"
            )


class RecordingAssembler:
    """obs assembler を包み、出力された DeployObs の数値配列(values/validity/age)を順に記録する。

    controller の telemetry には obs の要約と raw float hash しか残らないため、
    tolerance 比較用の numeric obs はここで横取りして保存します。組み立て結果は変えません。
    """

    def __init__(self, inner: Any) -> None:
        """包む assembler を受け取り、記録を空にする。"""
        self._inner = inner
        self.frame_ids: list[str] = []
        self.planes: dict[str, list[NDArray[np.float32]]] = {"values": [], "validity": [], "age": []}

    def assemble(self, *args: Any, **kwargs: Any) -> Any:
        """inner.assemble をそのまま呼び、snapshot が出た tick だけ obs の3平面を記録する。"""
        snapshot = self._inner.assemble(*args, **kwargs)
        if snapshot is not None:
            obs = snapshot.deploy_obs
            self.frame_ids.append(str(snapshot.frame_id))
            for name in self.planes:
                self.planes[name].append(np.asarray(getattr(obs, name), dtype=np.float32).reshape(-1))
        return snapshot

    def arrays(self) -> dict[str, NDArray[Any]]:
        """記録を NPZ へ書ける配列の dict にする(snapshot が0件なら長さ0の配列)。"""
        out: dict[str, NDArray[Any]] = {"frame_id": np.asarray(self.frame_ids, dtype=np.str_)}
        for name, rows in self.planes.items():
            out[name] = np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
        return out


def split_payload(value: Any, path: str = "", numeric: dict[str, float] | None = None) -> tuple[Any, dict[str, float]]:
    """payload を「exact 比較する離散部分」と「tolerance 比較する float 部分」に分ける。

    float はどの深さにあっても numeric 側へ ``a.b[0]`` 形式の path で移し、離散側では None に置き換えます
    (フィールドの有無と形は exact に比べ、値だけを tolerance で比べるため)。
    uuid 由来など実行ごとに変わる VOLATILE_KEYS は両方から外します。
    全 stage・全フィールドに同じ規則を当てるので、一方だけ exact/tolerance になることはありません。
    """
    numeric = {} if numeric is None else numeric
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value, numeric
    if isinstance(value, float):
        numeric[path] = value
        return None, numeric
    if isinstance(value, Mapping):
        return {
            key: split_payload(item, f"{path}.{key}" if path else str(key), numeric)[0]
            for key, item in value.items()
            if key not in VOLATILE_KEYS
        }, numeric
    if isinstance(value, (list, tuple)):
        return [split_payload(item, f"{path}[{i}]", numeric)[0] for i, item in enumerate(value)], numeric
    raise ReplayIntegrityError(f"unsupported telemetry value at {path!r}: {type(value).__name__}")


def split_stage_row(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """telemetry の stage 行1つを discrete 行と numeric 行に分ける。

    時刻(timestamp_ns)は仮想時計由来で決定的なので discrete、処理時間(latency_ns)は
    環境で揺れる計測値なので numeric に入れます。
    """
    payload, values = split_payload(row["payload"])
    key = {"sequence": row["sequence"], "stage": row["stage"], "correlation_id": row["correlation_id"]}
    discrete = {**key, "timestamp_ns": row["timestamp_ns"], "queue_depth": row["queue_depth"], "payload": payload}
    return discrete, {**key, "latency_ns": row["latency_ns"], "values": values}


def record_effects(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """telemetry の effect 行を movement/ui/release/control の意味付き JSON に変換する(effect recorder)。

    controller は shadow mode なので入力系 effect は ``disposition="proposed"`` で記録されるだけで、
    実際の入力は出ていません。ここではその提案内容(種別・action・UI 対象・理由・出所)を保存します。
    """
    effects = []
    for row in rows:
        if row.get("event") != "stage" or row["stage"] != "effect":
            continue
        payload, numeric = split_payload(row["payload"])
        effects.append({
            "correlation_id": row["correlation_id"],
            "timestamp_ns": row["timestamp_ns"],
            "category": EFFECT_CATEGORIES.get(payload["kind"], "unknown"),
            "effect": payload,
            "effect_numeric": numeric,
        })
    return effects


def _write_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    """np.savez と同じ形式の NPZ を、zip の日時を固定して書く(同じ配列なら同じ bytes になる)。

    np.savez は zip entry に現在時刻を埋め込むため、実時計を呼ぶうえに出力 hash が毎回変わります。
    """
    with zipfile.ZipFile(path, "w") as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            archive.writestr(zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)), buffer.getvalue())


def verify_virtual_telemetry(
    rows: list[Mapping[str, Any]], session: RecordedSession, *, start_ns: int, end_ns: int
) -> None:
    """telemetry の時刻と correlation id が仮想時計と記録 session に揃っていることを確かめる。

    - header の session_id が記録 session と同じ
    - 全 stage 行の timestamp_ns が仮想時計の範囲 [start_ns, end_ns] に入り、sequence 順に逆行しない
      (実時計の perf_counter_ns が混ざると桁違いの値になり、ここで弾かれます)
    - latency_ns が仮想時計の経過時間を超えない
    - correlation id が ``<session>:controller`` か記録 event にある frame 番号だけ
    1つでも外れたら ReplayIntegrityError を送出します。
    """
    if not rows or rows[0].get("event") != "session_header" or rows[0].get("session_id") != session.session_id:
        raise ReplayIntegrityError("telemetry header does not belong to the recorded session")
    allowed = {f"{session.session_id}:controller"}
    allowed |= {event.correlation_id for event in session.events if event.correlation_id is not None}
    last_ns = start_ns
    for row in rows[1:]:
        ts = row["timestamp_ns"]
        if not last_ns <= ts <= end_ns:
            raise ReplayIntegrityError(
                f"stage {row['stage']} seq={row['sequence']} timestamp {ts} is outside virtual clock [{last_ns}, {end_ns}]"
            )
        last_ns = ts
        if row["latency_ns"] > end_ns - start_ns:
            raise ReplayIntegrityError(f"stage {row['stage']} latency exceeds virtual clock span")
        if row["correlation_id"] not in allowed:
            raise ReplayIntegrityError(f"unexpected correlation_id {row['correlation_id']!r}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL を1行ずつ dict として読む。"""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """canonical な(key 順固定・区切り最小の)JSONL を書く。同じ内容なら同じ bytes になる。"""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def formal_parents_eligible(bundle: RuntimeBundle, detector_manifest: Any, capture: CaptureManifest) -> bool:
    """runtime bundle・detector・capture の全てが正式成果物なら True(1つでも開発用なら False)。

    ``verify_formal_runtime_release`` と同じ fail-closed の考え方で、正式性を確認できない
    (detector manifest が無い・正式判定が例外を出す)ときは必ず False にします。
    """
    if capture.development_only or bundle.development_only or not bundle.live_eligible or detector_manifest is None:
        return False
    try:
        detector_manifest.assert_formal_eligible()
    except Exception:  # noqa: BLE001  # 正式判定に失敗した理由を問わず formal 不可にする
        return False
    return True


@dataclass
class E2EReplayResult:
    """1回の recorded replay の結果(終了コードと各出力ファイルのパス・出力 manifest)。

    次段の golden/diff はこの paths を読みます。discrete は exact hash、numeric は quantize + tolerance の対象です。
    """

    exit_code: int
    exit_reason: str | None
    manifest: dict[str, Any]
    paths: dict[str, Path]
    errors: list[dict[str, str]] = field(default_factory=list)


def run_recorded_replay(
    capture: CaptureManifest,
    output_dir: Path | str,
    *,
    detector: Any,
    tracker: Any,
    hud_parser: Any,
    bundle: RuntimeBundle,
    artifact_hashes: Mapping[str, str],
    target_profile_hash: str,
    game_build_id: str,
    replay_determinism: DeterminismManifest,
    detector_manifest: Any = None,
    class_map_path: Any = None,
    score_threshold: float = 0.5,
    campaign_run_mode: CampaignRunMode = CampaignRunMode.FORMAL_SINGLE_ATTEMPT,
    pixels: Callable[[int], NDArray[np.uint8]] | None = None,
) -> E2EReplayResult:
    """capture manifest を照合し、実 controller を shadow mode・仮想時計で1回再生して結果を保存する。

    手順: 1) profile/build hash と決定性設定を照合 2) 実 assembler・runtime・state machine・health・telemetry で
    controller を組み、RecordedFrameSource と VirtualClock を注入して記録 event を使い切るまで run
    3) telemetry が仮想時計・記録 frame に揃っているか検証 4) discrete/numeric/effect/obs を別ファイルへ保存。
    出力先: telemetry.jsonl / discrete.jsonl / numeric.jsonl / numeric_obs.npz / effects.json / manifest.json。
    ``pixels`` を省くと capture manifest の NPZ(synthetic なら黒画面)から画素を読みます。
    """
    capture.require_identity(target_profile_hash=target_profile_hash, game_build_id=game_build_id)
    session = capture.session
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: output_dir / name for name in (
        "telemetry.jsonl", "discrete.jsonl", "numeric.jsonl", "numeric_obs.npz", "effects.json", "manifest.json",
    )}
    start_ns = session.events[0].timestamp_ns if session.events else 0
    clock = VirtualClock(start_ns)
    archive = np.load(capture.frames_path) if pixels is None and capture.frames_path is not None else None
    if pixels is None:
        blank = np.zeros(_FRAME_SHAPE, dtype=np.uint8)
        pixels = (lambda index: archive[f"frame_{index}"]) if archive is not None else (lambda _index: blank)
    try:
        # determinism の照合は RecordedFrameSource の構築時に行われる(食い違えば1 frame も流さない)
        source = RecordedFrameSource(session, clock, pixels, replay_determinism=replay_determinism)
        assembler = RecordingAssembler(RealObsAssembler())
        telemetry = TelemetryWriter(paths["telemetry.jsonl"], TelemetrySessionHeader(
            session_id=session.session_id, mode="shadow",
            target_profile_hash=session.target_profile_hash, game_build_id=session.game_build_id,
            controller_build_id=_sha256_bytes(Path(controller_module.__file__).read_bytes()),
            artifact_hashes=dict(artifact_hashes),
            host={"platform": platform.platform(), "node": platform.node()},
            device={"inference": replay_determinism.device},
            dependency_versions={"python": platform.python_version(), "numpy": np.__version__},
            deterministic_replay={
                "kind": "recorded_session_replay",
                "capture_manifest_sha256": capture.manifest_sha256,
                "frames_sha256": capture.frames_sha256,
                "virtual_clock_start_ns": start_ns,
                "determinism": replay_determinism.to_dict(),
            },
        ))
        controller = SurvivorsController(
            mode="shadow", session_id=session.session_id, capture=source, detector=detector, tracker=tracker,
            hud_parser=hud_parser, assembler=assembler, runtime=AgentRuntime(bundle, clock_ns=clock),
            state_machine=StateMachine(), health=HealthMonitor(), telemetry=telemetry,
            schema=bundle.deploy_schema, model_hashes=artifact_hashes, ui_config=bundle.ui_policy_config,
            class_map_path=class_map_path, score_threshold=score_threshold, clock_ns=clock, sleep=clock.sleep,
        )
        controller.arm(
            campaign_run_mode=campaign_run_mode,
            run_id=session.session_id, gameplay_attempt_id=f"{session.session_id}:attempt-1",
        )
        exit_code = controller.run(should_stop=lambda: source.exhausted)
    finally:
        if archive is not None:
            archive.close()

    rows = _read_jsonl(paths["telemetry.jsonl"])
    verify_virtual_telemetry(rows, session, start_ns=start_ns, end_ns=clock.now_ns())
    split = [split_stage_row(row) for row in rows[1:]]
    _write_jsonl(paths["discrete.jsonl"], (discrete for discrete, _ in split))
    _write_jsonl(paths["numeric.jsonl"], (numeric for _, numeric in split))
    _write_npz(paths["numeric_obs.npz"], assembler.arrays())
    effects = record_effects(rows)
    paths["effects.json"].write_text(json.dumps(effects, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    development_only = not formal_parents_eligible(bundle, detector_manifest, capture)
    manifest = {
        "schema_version": OUTPUT_MANIFEST_SCHEMA_VERSION,
        # I4: 開発用 parent が1つでも混ざれば formal replay verdict の材料にしない(fail closed)。
        "development_only": development_only,
        "formal_replay_eligible": not development_only,
        "session_id": session.session_id,
        "mode": "shadow",
        "capture_manifest_sha256": capture.manifest_sha256,
        "frames_sha256": capture.frames_sha256,
        "target_profile_hash": session.target_profile_hash,
        "game_build_id": session.game_build_id,
        "artifact_hashes": dict(artifact_hashes),
        "determinism": replay_determinism.to_dict(),
        "virtual_clock": {"start_ns": start_ns, "end_ns": clock.now_ns()},
        "dropped_frame_indices": list(session.dropped_indices),
        "exit_code": exit_code,
        "exit_reason": controller.exit_reason,
        "outputs": {name: _sha256_bytes(path.read_bytes()) for name, path in paths.items() if name != "manifest.json"},
    }
    paths["manifest.json"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return E2EReplayResult(exit_code, controller.exit_reason, manifest, paths, list(controller.errors))


# --------------------------------------------------------------------------------------------
# golden/diff(06-01 タスク3)
# --------------------------------------------------------------------------------------------

GOLDEN_SCHEMA_VERSION = "survivors.e2e_replay_golden.v1"
VERDICT_SCHEMA_VERSION = "survivors.e2e_replay_verdict.v1"
# canonical quantization rule: 値を round(value / quantum)(偶数丸め)の整数値へ丸めてから hash する。
DEFAULT_QUANTUM = 1e-6
# formal verdict に必要な同一 bundle の再生回数。
FORMAL_REPLAY_RUNS = 3
# discrete の stage → first divergence tree の分類(parser field / obs / model action / state / effect)。
STAGE_CATEGORIES = {
    "hud_parser": "parser_field",
    "obs": "obs",
    "policy": "model_action",
    "state_machine": "state",
    "effect": "effect",
}
# unknown/focus loss/timeout の区間で許す effect 分類(OS 入力を出さない release と停止系 control だけ)。
SAFE_EFFECT_CATEGORIES = frozenset({"release", "control"})
# golden 更新で新旧の各 run に必須の集計 metrics と、新旧比較に必須の metrics。
REQUIRED_RUN_METRICS = ("stage_rows", "stage_counts", "effect_counts", "obs_emitted", "exit_code")
REQUIRED_COMPARISON_METRICS = ("discrete_match_rate", "numeric_tolerance_pass_rate")
_OBS_PLANES = ("values", "validity", "age")
_MISSING = "<missing>"
_OPEN_END_NS = 2**63


class SafetyAssertionError(AssertionError):
    """安全 fixture の区間で release/no-op 以外の effect が出たときに送出する(aggregate tolerance で許容しない)。"""


class GoldenUpdateError(ValueError):
    """golden 更新に必須の情報(新旧 metrics・artifact hashes・承認者)が欠けている・食い違うときに送出する。"""


class FormalReplayRejectedError(ValueError):
    """開発用 parent や formal 不可の replay から formal verdict を publish しようとしたときに送出する。"""


@dataclass(frozen=True)
class Tolerance:
    """numeric 1 segment の許容差(``math.isclose`` と同じ対称な abs/rel の意味)。

    |old - new| <= max(abs_tol, rel_tol * max(|old|, |new|)) なら許容します。
    NaN は両方 NaN のとき、inf は同符号どうしのときだけ一致とみなします。
    """

    abs_tol: float
    rel_tol: float


# segment 名(ドット区切り)の最長前方一致で使う tolerance。"" は既定値。
DEFAULT_TOLERANCES: Mapping[str, Tolerance] = {
    "": Tolerance(abs_tol=1e-6, rel_tol=1e-6),
    "obs": Tolerance(abs_tol=1e-5, rel_tol=1e-5),
    "latency": Tolerance(abs_tol=1_000_000.0, rel_tol=0.0),
}


def tolerance_for(segment: str, tolerances: Mapping[str, Tolerance]) -> Tolerance:
    """segment に対しドット区切りの最長前方一致で tolerance を選ぶ(どれにも一致しなければ exact)。

    例えば ``obs.values.player`` は ``obs.values.player`` → ``obs.values`` → ``obs`` → ``""`` の順に探します。
    """
    parts = segment.split(".")
    for n in range(len(parts), -1, -1):
        key = ".".join(parts[:n])
        if key in tolerances:
            return tolerances[key]
    return Tolerance(0.0, 0.0)


def quantize(values: Any, quantum: float = DEFAULT_QUANTUM) -> NDArray[np.float64]:
    """numeric 値を canonical quantization rule で丸めた float64 配列を返す。

    round(value / quantum) を偶数丸めで求め、-0 を 0 に、NaN を単一の bit 表現にそろえます。
    raw float の最下位 bit の揺れで hash が変わらないようにするための規則です。
    """
    quantized = np.rint(np.asarray(values, dtype=np.float64) / quantum) + 0.0
    quantized[np.isnan(quantized)] = np.nan
    return quantized


def quantized_sha256(values: Any, quantum: float = DEFAULT_QUANTUM, keys: Sequence[str] | None = None) -> str:
    """quantize 後の値(と形・key 列)から canonical hash を作る。"""
    quantized = quantize(values, quantum)
    digest = hashlib.sha256(json.dumps([list(quantized.shape), keys], separators=(",", ":")).encode("utf-8"))
    digest.update(quantized.astype("<f8").tobytes())
    return digest.hexdigest()


def _within(old: Any, new: Any, tolerance: Tolerance) -> NDArray[np.bool_]:
    """old と new の各要素が tolerance 内かどうかの bool 配列を返す。"""
    old = np.asarray(old, dtype=np.float64)
    new = np.asarray(new, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        bound = np.maximum(tolerance.abs_tol, tolerance.rel_tol * np.maximum(np.abs(old), np.abs(new)))
        close = np.abs(new - old) <= bound
    return close | (old == new) | (np.isnan(old) & np.isnan(new))


def _canonical_sha256(rows: Iterable[Any]) -> str:
    """rows を _write_jsonl と同じ canonical JSONL にした bytes の sha256(discrete の exact hash)。"""
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return _sha256_bytes(text.encode("utf-8"))


def _paths(run: Any) -> dict[str, Path]:
    """E2EReplayResult か paths dict を受け取り、出力名 → Path の dict にする。"""
    return {name: Path(path) for name, path in getattr(run, "paths", run).items()}


def _column_segments(layout: Mapping[str, tuple[int, int]] | None, width: int) -> list[tuple[str, int, int]]:
    """obs の列範囲を DeployObs schema の segment 名で区切る(layout が無ければ全体で1つ)。"""
    if not layout:
        return [("all", 0, width)] if width else []
    spans = [(name, offset, min(offset + size, width)) for name, (offset, size) in layout.items() if offset < width]
    end = max((hi for _, _, hi in spans), default=0)
    return spans + ([("unassigned", end, width)] if end < width else [])


def _load_numeric(
    paths: Mapping[str, Path], discrete_rows: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, tuple[int, dict[str, Any], float]]], dict[str, NDArray[np.float64]], list[tuple[int, str]]]:
    """numeric.jsonl を segment → key → (sequence, 位置情報, 値) に、numeric_obs.npz を平面ごとの配列に読む。

    stage 値の segment は ``<stage>.<payload の先頭 key>``、処理時間は ``latency.<stage>`` です。
    obs 配列の各行は、discrete の emitted な obs 行(sequence, correlation_id)に順に対応します。
    """
    stage: dict[str, dict[str, tuple[int, dict[str, Any], float]]] = {}
    for row in _read_jsonl(paths["numeric.jsonl"]):
        base = {"sequence": row["sequence"], "stage": row["stage"], "correlation_id": row["correlation_id"]}
        items = [(f"latency.{row['stage']}", "latency_ns", row["latency_ns"])]
        items += [(f"{row['stage']}.{path.split('.')[0].split('[')[0]}", path, value) for path, value in row["values"].items()]
        for segment, path, value in items:
            stage.setdefault(segment, {})[f"{row['sequence']}/{row['stage']}/{path}"] = (
                row["sequence"], {**base, "path": path}, float(value),
            )
    emitted = [
        (row["sequence"], row["correlation_id"])
        for row in discrete_rows if row["stage"] == "obs" and row["payload"].get("emitted")
    ]
    with np.load(paths["numeric_obs.npz"]) as archive:
        planes = {name: np.asarray(archive[name], dtype=np.float64) for name in _OBS_PLANES}
    return stage, planes, emitted


def _stage_segment_hash(entries: Mapping[str, tuple[int, dict[str, Any], float]], quantum: float) -> str:
    """stage 値 segment の quantized hash(key を整列して値と一緒に hash)。"""
    keys = sorted(entries)
    return quantized_sha256([entries[key][2] for key in keys], quantum, keys=keys)


def numeric_segment_hashes(
    run: Any, *, quantum: float = DEFAULT_QUANTUM, obs_layout: Mapping[str, tuple[int, int]] | None = None
) -> dict[str, str]:
    """1回分の replay 出力について numeric segment ごとの quantized hash を返す(golden に保存する値)。"""
    paths = _paths(run)
    stage, planes, _ = _load_numeric(paths, _read_jsonl(paths["discrete.jsonl"]))
    hashes = {segment: _stage_segment_hash(entries, quantum) for segment, entries in stage.items()}
    for plane, array in planes.items():
        for name, lo, hi in _column_segments(obs_layout, array.shape[1]):
            hashes[f"obs.{plane}.{name}"] = quantized_sha256(array[:, lo:hi], quantum)
    return dict(sorted(hashes.items()))


def _first_diff(old: Any, new: Any, path: str = "") -> tuple[str, Any, Any] | None:
    """old と new を型も含めて比べ、最初に食い違う葉の (path, old, new) を返す(一致なら None)。"""
    if type(old) is not type(new):
        return path, old, new
    if isinstance(old, dict):
        for key in sorted(set(old) | set(new)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in old or key not in new:
                return sub, old.get(key, _MISSING), new.get(key, _MISSING)
            found = _first_diff(old[key], new[key], sub)
            if found:
                return found
        return None
    if isinstance(old, list):
        for i in range(max(len(old), len(new))):
            if i >= len(old) or i >= len(new):
                return f"{path}[{i}]", old[i] if i < len(old) else _MISSING, new[i] if i < len(new) else _MISSING
            found = _first_diff(old[i], new[i], f"{path}[{i}]")
            if found:
                return found
        return None
    return None if old == new else (path, old, new)


def _tree(sequence: int, correlation_id: str, stage: str, divergence: dict[str, Any]) -> dict[str, Any]:
    """first divergence を frame → stage → divergence の入れ子(tree)にする。"""
    return {"frame": {"correlation_id": correlation_id, "stage": {
        "name": stage, "sequence": sequence, "category": STAGE_CATEGORIES.get(stage, "stage"),
        "divergence": divergence,
    }}}


@dataclass(frozen=True)
class ReplayDiff:
    """2回の replay 出力の比較結果。

    discrete は canonical hash の exact 一致、numeric は segment ごとに quantized hash 一致か
    tolerance 内かで判定します。どちらかが外れると ``first_divergence`` に、最も早い sequence で
    分岐した frame/stage/field(obs なら平面・index・segment)を tree で入れます。空なら合格です。
    """

    discrete_sha256: tuple[str, str]
    segments: dict[str, dict[str, Any]]
    metrics: dict[str, Any]
    first_divergence: dict[str, Any] | None

    @property
    def passed(self) -> bool:
        """分岐が1つも無ければ True。"""
        return self.first_divergence is None


def compare_replays(
    old: Any,
    new: Any,
    *,
    tolerances: Mapping[str, Tolerance] = DEFAULT_TOLERANCES,
    quantum: float = DEFAULT_QUANTUM,
    obs_layout: Mapping[str, tuple[int, int]] | None = None,
) -> ReplayDiff:
    """旧(golden 側)と新の replay 出力を比べ、metrics と最初の分岐点 tree を返す。

    old/new は E2EReplayResult か、その paths と同じ形の dict です。discrete.jsonl を行ごとに exact 比較し、
    numeric.jsonl の stage 値・latency と numeric_obs.npz の obs 3平面を segment ごとに quantized hash
    → tolerance の順で判定します(全比較対象に同じ2経路を当てる)。effects.json の中身は discrete/numeric の
    effect 行と同じなので別には比べません。obs の segment 名は ``obs_layout``(``bundle.deploy_schema.layout``)で付けます。
    """
    old_paths, new_paths = _paths(old), _paths(new)
    old_rows, new_rows = _read_jsonl(old_paths["discrete.jsonl"]), _read_jsonl(new_paths["discrete.jsonl"])
    hashes = (_canonical_sha256(old_rows), _canonical_sha256(new_rows))
    total_rows = max(len(old_rows), len(new_rows))
    candidates: list[tuple[int, int, int, dict[str, Any]]] = []
    matches = total_rows
    if hashes[0] != hashes[1]:
        matches = 0
        for i in range(total_rows):
            o = old_rows[i] if i < len(old_rows) else _MISSING
            n = new_rows[i] if i < len(new_rows) else _MISSING
            found = _first_diff(o, n)
            if found is None:
                matches += 1
            elif not any(c[1] == 0 for c in candidates):
                row = o if isinstance(o, dict) else n
                candidates.append((row["sequence"], 0, len(candidates), _tree(
                    row["sequence"], row["correlation_id"], row["stage"],
                    {"kind": "discrete", "path": found[0], "old": found[1], "new": found[2]},
                )))

    old_stage, old_planes, emitted = _load_numeric(old_paths, old_rows)
    new_stage, new_planes, _ = _load_numeric(new_paths, new_rows)
    segments: dict[str, dict[str, Any]] = {}

    def record(segment: str, tolerance: Tolerance, values: int, within: int, equal: bool, max_abs: float | None) -> None:
        """segment 1つ分の tolerance metrics を記録する(形が違うときは max_abs_diff=None)。"""
        segments[segment] = {
            "values": values, "within_tolerance": within, "quantized_equal": equal,
            "passed": equal or within == values, "max_abs_diff": max_abs,
            "abs_tol": tolerance.abs_tol, "rel_tol": tolerance.rel_tol,
        }

    for segment in sorted(set(old_stage) | set(new_stage)):
        o, n = old_stage.get(segment, {}), new_stage.get(segment, {})
        tolerance = tolerance_for(segment, tolerances)
        keys = sorted(set(o) | set(n))
        equal = _stage_segment_hash(o, quantum) == _stage_segment_hash(n, quantum)
        within, max_abs, failures = len(keys), 0.0, []
        if not equal:
            within = 0
            for key in keys:
                old_value = o[key][2] if key in o else None
                new_value = n[key][2] if key in n else None
                if old_value is not None and new_value is not None:
                    if math.isfinite(old_value) and math.isfinite(new_value):
                        max_abs = max(max_abs, abs(new_value - old_value))
                    if _within(old_value, new_value, tolerance):
                        within += 1
                        continue
                sequence, locator, _ = o[key] if key in o else n[key]
                failures.append((sequence, key, locator, old_value, new_value))
        record(segment, tolerance, len(keys), within, equal, max_abs)
        if failures:
            sequence, _, locator, old_value, new_value = min(failures, key=lambda f: (f[0], f[1]))
            candidates.append((sequence, 1, len(candidates), _tree(
                sequence, locator["correlation_id"], locator["stage"],
                {"kind": "numeric", "segment": segment, "path": locator["path"],
                 "old": _MISSING if old_value is None else old_value,
                 "new": _MISSING if new_value is None else new_value},
            )))

    def obs_at(row: int) -> tuple[int, str]:
        """obs 配列の行番号から、対応する obs stage 行の (sequence, correlation_id) を返す。"""
        return emitted[row] if row < len(emitted) else (_OPEN_END_NS, _MISSING)

    for plane in _OBS_PLANES:
        o, n = old_planes[plane], new_planes[plane]
        if o.shape != n.shape:
            segment = f"obs.{plane}"
            record(segment, tolerance_for(segment, tolerances), max(o.size, n.size), 0, False, None)
            row = min(o.shape[0], n.shape[0]) if o.shape[1:] == n.shape[1:] else 0
            sequence, correlation_id = obs_at(row)
            candidates.append((sequence, 1, len(candidates), _tree(sequence, correlation_id, "obs", {
                "kind": "numeric_shape", "segment": segment, "plane": plane, "obs_row": row,
                "old": list(o.shape), "new": list(n.shape),
            })))
            continue
        for name, lo, hi in _column_segments(obs_layout, o.shape[1]):
            segment = f"obs.{plane}.{name}"
            tolerance = tolerance_for(segment, tolerances)
            so, sn = o[:, lo:hi], n[:, lo:hi]
            equal = quantized_sha256(so, quantum) == quantized_sha256(sn, quantum)
            ok = np.ones(so.shape, dtype=bool) if equal else _within(so, sn, tolerance)
            finite = np.isfinite(so) & np.isfinite(sn)
            max_abs = float(np.abs(sn - so)[finite].max()) if finite.any() else 0.0
            record(segment, tolerance, int(ok.size), int(ok.sum()), equal, max_abs)
            if not ok.all():
                row, col = (int(x) for x in np.argwhere(~ok)[0])
                sequence, correlation_id = obs_at(row)
                candidates.append((sequence, 1, len(candidates), _tree(sequence, correlation_id, "obs", {
                    "kind": "numeric", "segment": segment, "plane": plane, "obs_row": row,
                    "obs_index": lo + col, "obs_segment": name,
                    "old": float(so[row, col]), "new": float(sn[row, col]),
                })))

    values = sum(s["values"] for s in segments.values())
    within = sum(s["values"] if s["passed"] else s["within_tolerance"] for s in segments.values())
    metrics = {
        "discrete_equal": hashes[0] == hashes[1],
        "discrete_rows": [len(old_rows), len(new_rows)],
        "discrete_match_rate": matches / total_rows if total_rows else 1.0,
        "numeric_values": values,
        "numeric_tolerance_pass_rate": within / values if values else 1.0,
        "numeric_segments_failed": sorted(name for name, s in segments.items() if not s["passed"]),
    }
    first = min(candidates, key=lambda c: c[:3])[3] if candidates else None
    return ReplayDiff(hashes, segments, metrics, first)


def replay_metrics(run: Any) -> dict[str, Any]:
    """1回分の replay 出力の集計 metrics(stage 行数・stage 別件数・effect 分類別件数・obs 件数・終了)を返す。"""
    paths = _paths(run)
    rows = _read_jsonl(paths["discrete.jsonl"])
    effects = json.loads(paths["effects.json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    return {
        "stage_rows": len(rows),
        "stage_counts": dict(sorted(Counter(row["stage"] for row in rows).items())),
        "effect_counts": dict(sorted(Counter(effect["category"] for effect in effects).items())),
        "obs_emitted": sum(1 for row in rows if row["stage"] == "obs" and row["payload"].get("emitted")),
        "exit_code": manifest["exit_code"],
        "exit_reason": manifest["exit_reason"],
    }


def _run_artifact_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    """replay 出力 manifest から、その run を縛る artifact hashes(bundle/detector 等 + capture)を集める。"""
    hashes = {**manifest["artifact_hashes"], "capture_manifest_sha256": manifest["capture_manifest_sha256"]}
    if manifest["frames_sha256"] is not None:
        hashes["frames_sha256"] = manifest["frames_sha256"]
    return hashes


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    """一時ファイルへ書いてから置き換える(途中で落ちても半端な JSON を残さない)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update_golden(
    golden_path: Path | str,
    new_run: Any,
    *,
    old_metrics: Mapping[str, Any],
    new_metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    approved_by: str | None = None,
    quantum: float = DEFAULT_QUANTUM,
    obs_layout: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """expected improvement を golden として書き直す(golden update command)。

    旧 run・新 run の集計 metrics(replay_metrics)、新旧比較の metrics(ReplayDiff.metrics)、
    artifact hashes(新 run の bundle/detector 等と capture の hash を全て)を必須にし、欠けていたり
    新 run の manifest と食い違ったりすれば GoldenUpdateError です。前の golden から artifact hashes が
    変わる(model/parser の版が変わる)更新は、独立検証担当者の ``approved_by`` も必須です。
    golden には discrete の exact hash、numeric の segment 別 quantized hash、metrics、hashes を保存します。
    """
    for label, metrics, required in (
        ("old_metrics", old_metrics, REQUIRED_RUN_METRICS),
        ("new_metrics", new_metrics, REQUIRED_RUN_METRICS),
        ("comparison", comparison, REQUIRED_COMPARISON_METRICS),
    ):
        if not isinstance(metrics, Mapping) or any(key not in metrics for key in required):
            raise GoldenUpdateError(f"{label} must contain {list(required)}")
    paths = _paths(new_run)
    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    expected = _run_artifact_hashes(manifest)
    if not isinstance(artifact_hashes, Mapping) or not manifest["artifact_hashes"]:
        raise GoldenUpdateError("artifact_hashes are required")
    if any(not isinstance(value, str) or not value for value in artifact_hashes.values()):
        raise GoldenUpdateError("artifact_hashes values must be non-empty strings")
    wrong = sorted(key for key, value in expected.items() if artifact_hashes.get(key) != value)
    if wrong:
        raise GoldenUpdateError(f"artifact_hashes missing or not matching the new run: {wrong}")
    golden_path = Path(golden_path)
    previous = golden_path.read_bytes() if golden_path.exists() else None
    if previous is not None and json.loads(previous)["artifact_hashes"] != dict(artifact_hashes):
        if not isinstance(approved_by, str) or not approved_by.strip():
            raise GoldenUpdateError("artifact version change requires approved_by (independent verifier)")
    golden = {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "session_id": manifest["session_id"],
        "development_only": manifest["development_only"],
        "formal_replay_eligible": manifest["formal_replay_eligible"],
        "discrete_sha256": _canonical_sha256(_read_jsonl(paths["discrete.jsonl"])),
        "numeric_quantum": quantum,
        "numeric_quantized_sha256": numeric_segment_hashes(paths, quantum=quantum, obs_layout=obs_layout),
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "metrics": {"old": dict(old_metrics), "new": dict(new_metrics), "comparison": dict(comparison)},
        "approved_by": approved_by,
        "previous_golden_sha256": None if previous is None else _sha256_bytes(previous),
    }
    _atomic_write_json(golden_path, golden)
    return golden


def golden_mismatches(
    golden: Mapping[str, Any], run: Any, *, obs_layout: Mapping[str, tuple[int, int]] | None = None
) -> list[str]:
    """replay 出力が golden と食い違う項目名を返す(空なら一致)。

    discrete hash・numeric segment の quantized hash・artifact hashes を比べます。numeric が quantized hash で
    食い違ったときは、保存済みの旧 run と compare_replays で tolerance 判定と first divergence を確かめます。
    """
    paths = _paths(run)
    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    out = [] if golden["discrete_sha256"] == _canonical_sha256(_read_jsonl(paths["discrete.jsonl"])) else ["discrete"]
    current = numeric_segment_hashes(paths, quantum=golden["numeric_quantum"], obs_layout=obs_layout)
    stored = golden["numeric_quantized_sha256"]
    out += [f"numeric:{name}" for name in sorted(set(current) | set(stored)) if current.get(name) != stored.get(name)]
    hashes = _run_artifact_hashes(manifest)
    out += [
        f"artifact:{name}" for name in sorted(set(hashes) | set(golden["artifact_hashes"]))
        if hashes.get(name) != golden["artifact_hashes"].get(name)
    ]
    return out


def fault_windows(session: RecordedSession, discrete_rows: Iterable[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """unknown state・focus loss・timeout が続いている仮想時刻の区間 [開始, 終了) を列挙する。

    timeout/focus_lost は記録 event の時刻から次の新しい frame が届く時刻まで、
    unknown は state machine が unknown に入った時刻から unknown 以外へ移る時刻までです。
    終端は含まないので、回復後の新しい frame で出た effect は対象外になります。
    """
    windows = []
    events = list(session.events)
    for pos, event in enumerate(events):
        if event.kind in ("timeout", "focus_lost"):
            end = next((later.timestamp_ns for later in events[pos + 1:] if later.kind == "frame"), _OPEN_END_NS)
            windows.append((event.timestamp_ns, end))
    start = None
    for row in discrete_rows:
        if row["stage"] != "state_machine":
            continue
        unknown = row["payload"]["to_state"] == ControllerState.UNKNOWN.value
        if unknown and start is None:
            start = row["timestamp_ns"]
        elif not unknown and start is not None:
            windows.append((start, row["timestamp_ns"]))
            start = None
    if start is not None:
        windows.append((start, _OPEN_END_NS))
    return windows


def assert_safe_effects(effects: Iterable[Mapping[str, Any]], windows: Sequence[tuple[int, int]]) -> None:
    """安全 fixture の hard assertion: 区間内の effect が release/no-op 以外なら SafetyAssertionError。

    effects は record_effects(effects.json)の出力です。movement/ui/unknown は1件でも失敗で、
    aggregate の tolerance では許容しません。区間が1つも無い(fault が起きていない)fixture も
    検証になっていないので失敗にします。
    """
    if not windows:
        raise SafetyAssertionError("safety fixture produced no unknown/focus-loss/timeout window")
    violations = [
        (effect["correlation_id"], effect["timestamp_ns"], effect["category"], effect["effect"].get("kind"))
        for effect in effects
        if effect["category"] not in SAFE_EFFECT_CATEGORIES
        and any(lo <= effect["timestamp_ns"] < hi for lo, hi in windows)
    ]
    if violations:
        raise SafetyAssertionError(f"non release/no-op effects during fault windows: {violations[:5]}")


def publish_formal_replay_verdict(
    path: Path | str,
    *,
    diffs: Sequence[ReplayDiff],
    manifests: Sequence[Mapping[str, Any]],
    bundle: RuntimeBundle,
    detector_manifest: Any,
    capture: CaptureManifest,
) -> dict[str, Any]:
    """同一 bundle を FORMAL_REPLAY_RUNS 回再生した結果から formal replay verdict を書く。

    ``verify_formal_runtime_release`` と同じ fail-closed の考え方で、次のどれかなら
    FormalReplayRejectedError で publish を拒否します: runtime bundle/detector/capture の正式 parent が
    揃わない(development_only を含む)、run 数不足、run の manifest が development_only・formal 不可・
    別 capture・別 artifact、比較結果の不足。合否(passed)は拒否と別で、不合格の verdict も書けます。
    """
    if not formal_parents_eligible(bundle, detector_manifest, capture):
        raise FormalReplayRejectedError("formal parents (runtime bundle/detector/capture) are development_only or unverified")
    if len(manifests) < FORMAL_REPLAY_RUNS or len(diffs) < len(manifests) - 1:
        raise FormalReplayRejectedError(f"formal verdict needs {FORMAL_REPLAY_RUNS} runs and their comparisons")
    identity_keys = ("capture_manifest_sha256", "artifact_hashes", "determinism", "target_profile_hash", "game_build_id")
    for manifest in manifests:
        if manifest.get("development_only") is not False or manifest.get("formal_replay_eligible") is not True:
            raise FormalReplayRejectedError("a replay run is development_only / not formal_replay_eligible")
        if manifest.get("capture_manifest_sha256") != capture.manifest_sha256:
            raise FormalReplayRejectedError("a replay run used a different capture manifest")
    if len({json.dumps({key: m.get(key) for key in identity_keys}, sort_keys=True) for m in manifests}) != 1:
        raise FormalReplayRejectedError("replay runs do not share the same bundle/capture/determinism")
    verdict = {
        "schema_version": VERDICT_SCHEMA_VERSION,
        "development_only": False,
        "formal_replay_eligible": True,
        "passed": all(diff.passed for diff in diffs),
        "runs": len(manifests),
        "capture_manifest_sha256": capture.manifest_sha256,
        "artifact_hashes": dict(manifests[0]["artifact_hashes"]),
        "comparisons": [diff.metrics for diff in diffs],
        "first_divergences": [diff.first_divergence for diff in diffs if not diff.passed],
    }
    _atomic_write_json(Path(path), verdict)
    return verdict

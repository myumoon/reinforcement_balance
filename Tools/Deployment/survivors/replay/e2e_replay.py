"""recorded capture session を実 SurvivorsController へ仮想時計で流す E2E replay engine(06-01 タスク2)。

capture manifest(記録 session + frame 画素ファイル)と target profile・game build を照合してから、
RecordedFrameSource と VirtualClock を controller の既存注入点(capture/clock_ns/sleep)へ渡して実行します。
controller は常に shadow mode で動かすので ``execute_effect`` は呼ばれず、OS 入力は一切出ません。
実行後は telemetry を読み、全 stage の時刻と correlation id が仮想時計・記録 frame に揃っていることを確かめ、
結果を「exact 比較する離散値(discrete)」と「tolerance 比較する数値(numeric)」に分けて保存します。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import platform
from typing import Any
import zipfile

import numpy as np
from numpy.typing import NDArray

from ..controller import controller as controller_module
from ..controller.controller import SurvivorsController
from ..controller.health_monitor import HealthMonitor
from ..controller.state_machine import CampaignRunMode, StateMachine
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

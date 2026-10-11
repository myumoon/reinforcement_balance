"""formal runner と typed UiPresentation replay の synthetic store fixture。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from Tools.Artifacts.artifact_store import ArtifactStore
from build_survivors_formal_annotations import main as build_formal_annotations
from benchmark_survivors_perception import (
    FormalBenchmarkRequest,
    RestoredFormalArtifact,
    _annotation_evidence_index,
    _calibration_descriptor_chain,
    _calibration_provenance_bytes,
    _descriptor_chain,
    _derive_calibration_residuals,
    _ground_truth_semantic_hash,
    _put_descriptor_file,
    _replay_sessions,
    _recover_committed_result,
    _subject_hashes,
    _verified_annotation_payloads,
    run_formal_pipeline,
)
from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor, RestoreTestVerdict
from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from survivors.capture.captured_frame import CapturedFrame
from survivors.capture_dataset import DatasetWriter, SplitFreezer
from survivors.evaluation_windows import EvaluationWindowPlan
from survivors.formal_ground_truth import (
    coco_image_entries_from_wire,
    ground_truth_snapshot,
    hud_state_from_truth,
    normalize_snapshot_ids,
)
from survivors.perception_benchmark import (
    SnapshotReplayTick,
    _foreground_classification_values,
    recompute_gate_from_metrics,
    replay_snapshots,
    run_benchmark,
)
from survivors.perception_error_fit import (
    _FORMAL_FACTORY_TOKEN,
    _HASH_FIELDS,
    CalibrationResidual,
    FinalSessionAlreadyOpenedError,
    FittedPerceptionErrorProfile,
    PerceptionFinalVerdict,
    _create_formal_lineage_seal,
    _load_formal_verdict_from_verified_store,
)
from survivors.vision.entity_tracker import default_class_map
from survivors.vision.hud_truth import read_hud_truth
from survivors.perception_session_split import SessionRecord, SessionSplit, validate_split
from survivors.perception_snapshot import FormalReplayEvidence
from survivors.real_obs_assembler import RealObsAssembler
from survivors.vision.entity_tracker import (
    DEFAULT_CLASS_MAP_PATH,
    PlayerAnchorState,
    Track,
    TrackedWorldState,
    TrackedEntityV2,
    TrackedWorldStateV2,
)
from survivors.vision.hud_parser import (
    HudStateV1,
    ParsedCard,
    _compute_candidate_set_hash,
    _compute_inventory_hash,
)
from survivors.vision.hud_truth import HUD_TRUTH_SCHEMA_VERSION, HudTruthRecord


PROFILE_HASH = "a" * 64
PACKAGE_HASHES = {
    "assembler_schema_hash": "b" * 64,
    "ui_presentation_schema_hash": "c" * 64,
    "atlas_vocabulary_hash": "d" * 64,
    "assembler_impl_hash": "e" * 64,
    "roi_resolver_input_hash": "f" * 64,
    "threshold_hash": "1" * 64,
}


def _captured_frame(session_index: int, frame_index: int, timestamp_ns: int) -> CapturedFrame:
    pixels = np.zeros((1080, 1920, 4), dtype=np.uint8)
    pixels[0, 0] = (session_index, frame_index % 256, 7, 255)
    return CapturedFrame(
        pixels, timestamp_ns, frame_index, (0, 0, 1920, 1080), True,
        PROFILE_HASH, "synthetic-build",
    )


def _capture_fixture(
    tmp_path: Path, parser_artifact_hash: str = "9" * 64,
    include_formal_truth: bool = False,
) -> tuple[Path, object]:
    """formal split session の capture data を作成する。

    formal truth bytes は pipeline integration test が要求した場合だけ追加する。
    """
    capture_root = tmp_path / "captures"
    calibration_ids = [f"cal-{index}" for index in range(3)]
    final_ids = [f"final-{index}" for index in range(3)]
    for session_index, session_id in enumerate((*calibration_ids, *final_ids)):
        with DatasetWriter(capture_root, session_id, PROFILE_HASH, "synthetic-build") as writer:
            for frame_index, timestamp_ns in enumerate(_TIMESTAMPS):
                writer.write_frame(_captured_frame(session_index, frame_index, timestamp_ns))
            published = writer.publish(operator_checkpoint="synthetic-formal-fixture")
        # fixture は実 formal session を開封せず、manifest gate だけを synthetic に再現する。
        manifest_path = published.session_path / "session_manifest.json"
        wire = json.loads(manifest_path.read_text(encoding="utf-8"))
        wire["formal_dataset_eligible"] = True
        manifest_path.write_text(
            json.dumps(wire, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    if include_formal_truth:
        for session_id in (*calibration_ids, *final_ids):
            _write_truth_fixture(
                capture_root / "capture_sessions" / session_id,
                session_id,
                parser_artifact_hash,
            )
    freezer = SplitFreezer(capture_root)
    freezer.assign("error_calibration", calibration_ids)
    freezer.assign("final_e2e_test", final_ids)
    return capture_root, freezer.freeze()


def _truth_record(
    session_id: str, frame_index: int, screen_state: str,
    parser_artifact_hash: str = "9" * 64,
) -> HudTruthRecord:
    """frame index に対応する確定 HUD truth を作成する。

    detector world label と独立した parser artifact hash を保存する。
    """
    return HudTruthRecord(
        HUD_TRUTH_SCHEMA_VERSION, session_id, frame_index, "fixture", True,
        screen_state,
        _ELAPSED_TIMES[frame_index] if frame_index < len(_ELAPSED_TIMES) else 20.0,
        4, 0.75, 0.5,
        ("whip",) + ("empty_slot",) * 11, None,
        ("whip",) if screen_state.startswith("level_up") else None,
        "hud", (0, 32, 1920, 75), "2026-10-11T00:00:00Z",
        {"parser_artifact_hash": parser_artifact_hash, "atlas_content_hash": "d" * 64},
    )


def _write_truth_fixture(
    session_path: Path, session_id: str, parser_artifact_hash: str
) -> None:
    """fixture session に formal truth・frame・COCO を保存する。

    image id と frame id を分け、writer と loader の frame 対応を確かめる。
    """
    frame_wires = [
        json.loads(line)
        for line in (session_path / "frames.jsonl").read_text("utf-8").splitlines()
    ]
    truth_rows = [
        _truth_record(session_id, i, _STATES[i], parser_artifact_hash).to_wire()
        for i in range(len(frame_wires))
    ]
    (session_path / "hud_truth.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in truth_rows),
        encoding="utf-8",
    )
    coco = {
        "images": [
            {
                "id": i + 1, "frame_id": i, "session_id": session_id,
                "object_sha256": row["object_sha256"],
            }
            for i, row in enumerate(frame_wires)
        ],
        "categories": [
            {"id": 1, "name": "player_anchor"},
            {"id": 2, "name": "enemy_boss"},
            {"id": 3, "name": "hazard_area"},
        ],
        "annotations": [
            item
            for i in range(len(frame_wires))
            for item in (
                {"id": 3 * i + 1, "image_id": i + 1, "category_id": 1,
                 "bbox": [910, 490, 100, 100]},
                {"id": 3 * i + 2, "image_id": i + 1, "category_id": 2,
                 "bbox": [1152, 513, 384, 54]},
                {"id": 3 * i + 3, "image_id": i + 1, "category_id": 3,
                 "bbox": [1056, 513, 192, 54]},
            )
        ],
    }
    capture_root = session_path.parent.parent
    coco_dir = capture_root / "fixture-coco"
    session_coco_dir = coco_dir / session_id
    session_coco_dir.mkdir(parents=True, exist_ok=True)
    coco_bytes = canonical_json_bytes(coco)
    (session_coco_dir / "world_coco.json").write_bytes(coco_bytes)
    (session_coco_dir / "ui_coco.json").write_bytes(coco_bytes)
    assert build_formal_annotations([
        "--store-root", str(capture_root), "--session-id", session_id,
        "--coco-dir", str(coco_dir), "--parser-artifact-hash", parser_artifact_hash,
        "--class-map", str(DEFAULT_CLASS_MAP_PATH),
    ]) == 0


def _package_ref(store: ArtifactStore, logical_id: str, payload: dict[str, object]):
    return store.put_bytes(
        logical_id=logical_id,
        data=(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(),
        media_type="application/json",
    )


def _dependency_refs(store: ArtifactStore, split) -> dict[str, object]:
    refs = {
        "capture_dataset": store.put(
            logical_id="fixtures/capture_split.json",
            source_path=split.manifest_path,
            media_type="application/json",
        ),
        "parser_package": _package_ref(store, "fixtures/parser.json", {
            "schema_version": "hud_parser_package.v1", "development_only": False,
            "formal_eligible": True,
            # _hud_world の HudStateV1.parser_artifact_hash と一致させる。
            "artifact_hash": "9" * 64,
        }),
        "detector_package": _package_ref(store, "fixtures/detector.json", {
            "schema_version": "world_detector_package.v1", "development_only": False,
            "formal_eligible": True,
            "artifact_hash": "2" * 64,
        }),
        "assembler_config": _package_ref(store, "fixtures/assembler.json", {
            "schema_version": "assembler_config.v1", "development_only": False,
            "formal_eligible": True, **{key: PACKAGE_HASHES[key] for key in (
                "assembler_schema_hash", "ui_presentation_schema_hash",
                "atlas_vocabulary_hash", "assembler_impl_hash", "roi_resolver_input_hash",
            )},
        }),
        "target_config": _package_ref(store, "fixtures/target.json", {
            "schema_version": "perception_thresholds.v1", "development_only": False,
            "formal_eligible": True, "threshold_hash": PACKAGE_HASHES["threshold_hash"],
        }),
    }
    return refs


def _hud_world(
    session_id: str, frame_index: int, timestamp_ns: int, screen_state: str,
    parser_artifact_hash: str = "9" * 64,
    card_roi: tuple[int, int, int, int] = (100, 100, 400, 500),
):
    card = ParsedCard(0, "whip", "weapon", 2, 1.0, "fixture", card_roi)
    inventory = ("whip",) + (None,) * 11
    # canonical hash を実際の計算関数で生成し、_recomputed_snapshot_hashes の照合に通す。
    real_inventory_hash = _compute_inventory_hash(inventory)
    real_candidate_set_hash = _compute_candidate_set_hash(screen_state, (card,))
    elapsed_time = _ELAPSED_TIMES[frame_index] if frame_index < len(_ELAPSED_TIMES) else 20.0
    hud = HudStateV1(
        "hud_state.v1", session_id, frame_index, timestamp_ns, parser_artifact_hash,
        screen_state, 1.0, "fixture", elapsed_time, 1.0, "fixture", False,
        0.75, 1.0, "fixture", 0.5, 1.0, "fixture", 4, 1.0, "fixture",
        inventory, 1.0, real_inventory_hash, (card,), real_candidate_set_hash,
        (), False, False, False, 1.0, "fixture",
    )
    hud = hud_state_from_truth(
        _truth_record(session_id, frame_index, screen_state, parser_artifact_hash),
        session_id=session_id, frame_index=frame_index, captured_ns=timestamp_ns,
        parser_artifact_hash=parser_artifact_hash,
    )
    if hud.cards and card_roi != hud.cards[0].roi_xyxy:
        cards = tuple(replace(card, roi_xyxy=card_roi) for card in hud.cards)
        hud = replace(
            hud, cards=cards,
            candidate_set_hash=_compute_candidate_set_hash(screen_state, cards),
        )
    boss = TrackedEntityV2(
        1, 2, "enemy_boss", "enemy", 1.0, 1, frame_index,
        0.7, 0.5, 0.2, 0.0, 0.0, 0.0, True, False, 0.05, 0.05, 0,
    )
    hazard = TrackedEntityV2(
        2, 3, "hazard_area", "hazard", 1.0, 1, frame_index,
        0.6, 0.5, 0.1, 0.0, 0.0, 0.0, True, False, 0.05, 0.05, 0,
    )
    world = TrackedWorldStateV2(
        frame_index, timestamp_ns, [boss, hazard], PlayerAnchorState(0.5, 0.5, 1.0, False)
    )
    return hud, world


_CALIBRATION_REQUIRED_FIELDS = frozenset({
    "coord_noise", "hp_ratio", "xp_ratio", "timer_seconds",
    "inventory_hash", "coord_quantization_px",
    "burst_enter", "burst_exit", "burst_dropout",
    "unknown_screen_collapse", "unknown_screen_collapse_duration",
    "item_category", "enemy_category",
})
_CALIBRATION_SAMPLES_PER_FIELD_PER_SESSION = 3  # formal 要件（3 samples × ≥3 sessions）を満たす最小値


def _calibration_provider(manifests, _restored):
    """formal=True で fit_error_profile が必須 field をすべて受理できる残差を返す。

    各 field で 3 サンプル × manifests（cal sessions）分の残差を生成します。
    item_category / enemy_category は confusion matrix 用に category を設定します。
    """
    rows = []
    for manifest in manifests:
        sid = manifest.session_id
        for field in _CALIBRATION_REQUIRED_FIELDS:
            for frame_idx in range(_CALIBRATION_SAMPLES_PER_FIELD_PER_SESSION):
                gt_cat = (frame_idx % 3) if "category" in field else None
                pred_cat = (frame_idx % 3) if "category" in field else None
                rows.append(CalibrationResidual(
                    sid, f"{field}-{frame_idx}", field, 0.01,
                    1.0, 0, 0.0, gt_cat, pred_cat,
                ))
    return rows


# 4 フレーム × 3 final sessions = ≥ 6 records/slice を達成するフィクスチャ
# （_NAMED_SLICE_RECORD_FLOORS のテスト用 floor 値に合わせた最小設定）
# frame_index: 0=gameplay/early, 1=level_up, 2=gameplay/late, 3=level_up → 各スライス 3×2=6 records
_TIMESTAMPS = [1_000_000_000, 2_000_000_000, 1_800_000_000_000, 1_801_000_000_000]
_STATES = ["gameplay", "level_up_items", "gameplay", "level_up_fallback"]
_ELAPSED_TIMES = [20.0, 20.0, 1300.0, 1300.0]


class _SnapshotFixture:
    """テスト用の FormalAssembler：CapturedFrame の session_frame_index を使って合成観測を生成する。

    セッション間でタイムスタンプが逆転するため、セッションごとに独立した RealObsAssembler を保持する。
    FormalReplayEvidence として raw_card_ids / raw_inventory を HUD から抽出して返します。
    """

    def __init__(self, parser_artifact_hash: str = "9" * 64) -> None:
        """test sequence ごとに assembler を分離する。

        frame snapshot は複数 session 間で temporal state を共有しない。
        """
        self._assemblers: dict[str, RealObsAssembler] = {}
        self._screen_overrides: dict[tuple[str, int], str] = {}
        self._schema = DeployObsSchema.default_v2()
        self._parser_artifact_hash = parser_artifact_hash

    def assemble_frame(self, frame, session_id: str, frame_index: int):
        """fixture frame から normalized snapshot と raw UI evidence を返す。

        `predictor` adapter が同じ snapshot を予測専用 API に投影する。
        """
        if frame_index >= len(_TIMESTAMPS):
            return None, None, 0.0, None
        if session_id not in self._assemblers:
            self._assemblers[session_id] = RealObsAssembler()
        hud, world = _hud_world(
            session_id, frame_index, _TIMESTAMPS[frame_index],
            self._screen_overrides.get((session_id, frame_index), _STATES[frame_index]),
            self._parser_artifact_hash,
        )
        snapshot = self._assemblers[session_id].assemble(hud, world, self._schema, (1920, 1080))
        assert snapshot is not None
        filtered_hud = self._assemblers[session_id].temporal.last_hud
        assert filtered_hud is not None
        snapshot = normalize_snapshot_ids(
            snapshot, session_id=session_id, frame_index=frame_index,
            viewport=(1920, 1080), hud=filtered_hud,
        )
        diagnostics = dict(snapshot.diagnostics)
        diagnostics["predicted_boxes"] = [
            {"class": "player_anchor", "xyxy": [910, 490, 1010, 590], "score": 1.0},
            {"class": "enemy_boss", "xyxy": [1152, 513, 1536, 567], "score": 1.0},
            {"class": "hazard_area", "xyxy": [1056, 513, 1248, 567], "score": 1.0},
        ]
        snapshot = replace(snapshot, diagnostics=diagnostics)
        evidence = FormalReplayEvidence(
            raw_card_ids=tuple(c.item_id for c in sorted(hud.cards, key=lambda c: c.slot_index)),
            raw_inventory=hud.inventory,
        )
        # synthetic fixture は正解と予測の内容を一致させる。
        # runner が別入力として扱う契約を守るため、両方を明示的に返す。
        return snapshot, snapshot, 1.0, evidence


def _fixture_predictor_factory(restored):
    """runner に渡す fixture FormalPredictor を作成する。

    snapshot の parser identity を restored package の hash に結び付ける。
    """
    return _FixturePredictor(restored)


def _test_provider_factory(restored):
    """runner unit test 用の FormalPredictor を返す。

    production factory と同じ restored hash と reset 契約を維持する。
    """
    return _FixturePredictor(restored)


def _patch_minimal_test_gates(monkeypatch) -> None:
    """小さな synthetic dataset が持つ slice だけを test gate に残す。

    production threshold は変更せず package hash と test 内定数を同期する。
    """
    import survivors.perception_benchmark as benchmark

    floors = dict.fromkeys(benchmark._FORMAL_SLICE_COUNT_FLOORS, 0)
    floors.update({
        "foreground_class:player_anchor": 2,
        "foreground_class:enemy_boss": 2,
        "foreground_class:hazard_area": 2,
        "event:boss": 2,
        "event:hazard": 2,
        "time_band:early": 2,
        "time_band:late": 2,
    })
    monkeypatch.setattr(benchmark, "_FORMAL_SLICE_COUNT_FLOORS", floors)
    active = {
        "foreground_class:player_anchor", "foreground_class:enemy_boss",
        "foreground_class:hazard_area", "event:boss", "event:hazard",
        "time_band:early", "time_band:late",
    }
    monkeypatch.setattr(benchmark, "_FORMAL_REQUIRED_SLICES", frozenset(active))
    monkeypatch.setattr(
        benchmark, "_FORMAL_SLICE_THRESHOLDS",
        {name: value for name, value in benchmark._FORMAL_SLICE_THRESHOLDS.items()
         if name in active},
    )
    monkeypatch.setattr(
        benchmark, "_FORMAL_SLICE_SESSION_FLOORS",
        {"time_band:early": 1, "time_band:late": 1},
    )
    for name, value in (
        ("THRESHOLD_INVALID_TICK", 1.0),
        ("THRESHOLD_ROI_CENTER_P99", 1.0),
        ("THRESHOLD_ROI_INSIDE", 0.0),
    ):
        monkeypatch.setattr(benchmark, name, value)


def _install_public_formal_factories(monkeypatch, benchmark_module) -> None:
    """public provider と ground-truth loader を pipeline に接続する。

    package IO と detector のみ固定し、assembler・runner・GT loader は実装を使う。
    """
    import survivors.formal_perception_provider as provider
    from survivors.formal_annotation_loader import build_ground_truth_loader
    from survivors.vision.entity_tracker import default_class_map
    from survivors.vision.world_detector import DetectionResult

    class _HudParser:
        """runner fixture と同じ HUD truth を detector から独立して parse する。

        class state は tracker fixture に frame の session context を伝える。
        """

        last_session = ""
        last_frame_index = 0

        def __init__(self, *, parser_artifact_hash: str, icon_matcher=None):
            """package artifact hash を予測 HUD parser に渡す。

            icon matching 自体はこの synthetic runner fixture の対象外にする。
            """
            self._parser_hash = parser_artifact_hash

        def parse(self, _frame_bgra, session_id, frame_index, captured_monotonic_ns):
            """fixture に固定した HUD state を生成する。

            session/frame identity は tracker test double と共有する。
            """
            type(self).last_session = session_id
            type(self).last_frame_index = frame_index
            return _hud_world(
                session_id, frame_index, captured_monotonic_ns,
                _STATES[frame_index], self._parser_hash,
            )[0]

        def reset_temporal_state(self) -> None:
            """frame identity を消去する。

            次 session の detector output に前 session の値を持ち越さない。
            """
            type(self).last_session = ""

    class _WorldDetector:
        """各 fixture frame に同じ formal class box を返す。

        semantic 一致を保ち、runner／factory の実装経路を試す。
        """

        def infer(self, _frame, score_threshold=0.0):
            """synthetic label と同じ位置に detector boxes を返す。

            prediction-only runner が IoU matching を実行できるようにする。
            """
            return DetectionResult(
                np.asarray([[910, 490, 1010, 590], [1152, 513, 1536, 567],
                            [1056, 513, 1248, 567]], dtype=np.float32),
                np.ones(3, dtype=np.float32), np.asarray([1, 4, 11], dtype=np.int64),
                1920, 1080,
            )

    class _EntityTracker:
        """HUD parser の frame identity で world state を返す tracker double。

        prediction fixture が annotation loader の bytes を参照しないことを保つ。
        """

        def __init__(self):
            """formal provider reset が消去する tracker state を初期化する。

            test double も production tracker と同じ `_tracks` field を持つ。
            """
            self._tracks = {}

        def update(self, _detection, frame_index, timestamp_ns):
            """detector tick を canonical low-level world state へ変換する。

            provider 自身が `TrackedWorldStateV2` へ変換する経路を通す。
            """
            session_id = _HudParser.last_session
            return TrackedWorldState(
                frame_index, timestamp_ns,
                [
                    Track(1, 4, np.asarray([1152, 513, 1536, 567]), 1.0,
                          timestamp_ns, age=frame_index, last_seen_frame_index=frame_index),
                    Track(2, 11, np.asarray([1056, 513, 1248, 567]), 1.0,
                          timestamp_ns, age=frame_index, last_seen_frame_index=frame_index),
                ],
                PlayerAnchorState(0.5, 0.5, 1.0, False), 1920, 1080,
            )

    def _load_parser_package(path):
        """envelope hash を package loader の返却型に写す。

        test 用 package payload を通し、provider の hash binding を維持する。
        """
        path = Path(path)
        manifest_path = path / "manifest.json" if path.is_dir() else path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        return SimpleNamespace(
            package_hash=manifest["parser_artifact_hash"],
            meta=SimpleNamespace(
                parser_artifact_hash=manifest["parser_artifact_hash"],
                atlas_content_hash="d" * 64,
            ),
        )

    monkeypatch.setattr(
        provider.PackageLoader, "load_formal", staticmethod(_load_parser_package)
    )
    monkeypatch.setattr(
        provider.IconMatcher, "load_formal",
        staticmethod(lambda _path: SimpleNamespace(
            manifest=SimpleNamespace(atlas_content_hash="d" * 64),
        )),
    )
    monkeypatch.setattr(provider, "HudParser", _HudParser)
    monkeypatch.setattr(provider, "load_class_map", lambda _path: default_class_map())
    monkeypatch.setattr(
        provider, "load_formal_detector",
        lambda path: (
            _WorldDetector(), _EntityTracker(), SimpleNamespace(
                score_threshold=0.0,
                model_hash=json.loads((Path(path) / "manifest.json").read_bytes())["model_hash"],
            ),
        ),
    )
    monkeypatch.setattr(benchmark_module, "_CLI_PROVIDER_FACTORY", provider.build_formal_predictor)
    monkeypatch.setattr(benchmark_module, "_GROUND_TRUTH_FACTORY", build_ground_truth_loader)
    calibration_manifests = [SimpleNamespace(session_id=f"cal-{i}") for i in range(3)]
    monkeypatch.setattr(
        benchmark_module, "_derive_calibration_residuals",
        lambda _ticks, **_kwargs: _calibration_provider(calibration_manifests, None),
    )


class _FixturePredictor:
    """FormalPredictor protocol に沿う runner fixture。

    restored manifest の hash を予測 snapshot に結び付ける。
    """

    def __init__(self, restored):
        """restored package identity と assembler fixture を保存する。

        各予測で parser・detector hash を同じ値に保つ。
        """
        self._parser_hash = restored["parser_package"].manifest["parser_artifact_hash"]
        self._detector_hash = restored["detector_package"].manifest["detector_artifact_hash"]
        self._fixture = _SnapshotFixture(self._parser_hash)

    def predict_frame(self, frame, session_id: str, frame_index: int):
        """unit test の deterministic snapshot を prediction として返す。

        GT loader から値を受け取らず artifact hash だけ restored package に束縛する。
        """
        if session_id.startswith("cal-") and frame_index == 1:
            return None, 1.0
        if session_id.startswith("cal-") and frame_index == 0:
            self._fixture._screen_overrides[(session_id, frame_index)] = "unknown"
        _unused, predicted, latency, _evidence = self._fixture.assemble_frame(
            frame, session_id, frame_index
        )
        if predicted is None:
            return None, latency
        return replace(predicted, detector_artifact_hash=self._detector_hash), latency

    def reset(self) -> None:
        """前 session の fixture assembler を破棄する。

        runner reset 呼び出しが provider interface に届いたことを保証する。
        """
        self._fixture._assemblers.clear()


def _mutating_test_predictor(restored, mutate):
    """snapshot を故意に変える runner regression 用 predictor を作る。

    ground-truth loader には触れず prediction API の挙動だけを変える。
    """
    predictor = _FixturePredictor(restored)
    predict = predictor.predict_frame

    def predict_frame(frame, session_id: str, frame_index: int):
        """予測 snapshot に変更を加えて runner へ返す。

        元 predictor の出力に変異関数を適用し、runner の不変条件検査を試す。
        """
        snapshot, latency = predict(frame, session_id, frame_index)
        return (mutate(snapshot) if snapshot is not None else None), latency

    predictor.predict_frame = predict_frame
    return predictor


def _request(tmp_path: Path, *, formal_inputs: bool = False):
    """formal pipeline request を package と capture descriptor で構成する。

    direct unit test 用には artifact graph を省き、統合 test だけ exact restore を行う。
    """
    parser_bytes = canonical_json_bytes({
        "metadata": {}, "fit_result": {}, "validation_report": {},
    })
    parser_hash = hashlib.sha256(parser_bytes).hexdigest()
    model_bytes = b"fixture model"
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    detector_manifest_bytes = canonical_json_bytes({
        "schema_version": "world_detector_package.v1", "model_hash": model_hash,
    })
    capture_root, split = _capture_fixture(
        tmp_path, parser_hash, include_formal_truth=formal_inputs
    )
    store = ArtifactStore(tmp_path / "artifacts")
    if not formal_inputs:
        return FormalBenchmarkRequest(store=store, capture_store_root=capture_root)
    package_specs = {
        "parser_package": ("hud_parser_package.v1", {"parser_artifact_hash": parser_bytes}),
        "detector_package": ("world_detector_package.v1", {
            "detector_artifact_hash": detector_manifest_bytes,
            "model_hash": model_bytes,
            "build_hash": b"synthetic-build",
        }),
        "assembler_config": ("assembler_config.v1", {
            "assembler_schema_hash": b"fixture assembler schema",
            "ui_presentation_schema_hash": b"fixture ui schema",
            "ui_presentation_golden_fixture_hash": b"fixture ui golden",
            "atlas_vocabulary_hash": b"fixture atlas",
            "assembler_impl_hash": b"fixture assembler",
            "roi_resolver_input_hash": b"fixture roi",
        }),
        "target_config": ("perception_thresholds.v1", {
            "threshold_hash": _formal_threshold_bytes(),
        }),
    }
    descriptors = {}
    restores = {}
    for name, (schema, files) in package_specs.items():
        extras = {}
        if name == "parser_package":
            extras = {
                "fixtures/parser_package/parser_package.json": parser_bytes,
                "fixtures/parser_package/icon_atlas.json": canonical_json_bytes({
                    "schema_version": "icon_atlas.v1", "icons": [],
                }),
            }
        elif name == "detector_package":
            extras = {
                "fixtures/detector_package/manifest.json": detector_manifest_bytes,
                "fixtures/detector_package/model.pt": model_bytes,
                "fixtures/detector_package/world_class_map_v2.yaml":
                    DEFAULT_CLASS_MAP_PATH.read_bytes(),
            }
        descriptors[name], restores[name] = _formal_dependency(
            store, name, schema, files, extras=extras,
            descriptor_manifest_name=("artifact_descriptor.json" if name == "detector_package" else "manifest.json"),
        )

    capture_refs = [store.put(
        logical_id="fixtures/capture/split.json", source_path=split.manifest_path,
        media_type="application/json",
    )]
    bindings = {
        "annotation_bindings": {}, "hud_truth_bindings": {},
        "coco_bindings": {}, "frames_bindings": {},
    }
    session_ids = ("cal-0", "cal-1", "cal-2", "final-0", "final-1", "final-2")
    for session_id in session_ids:
        session_path = capture_root / "capture_sessions" / session_id
        manifest_ref = store.put(
            logical_id=f"fixtures/capture/{session_id}/session_manifest.json",
            source_path=session_path / "session_manifest.json", media_type="application/json",
        )
        annotation_ref = store.put(
            logical_id=f"fixtures/capture/{session_id}/annotations.jsonl",
            source_path=session_path / "annotations.jsonl", media_type="application/x-ndjson",
        )
        capture_refs.extend((manifest_ref, annotation_ref))
        bindings["annotation_bindings"][session_id] = {
            "annotation_logical_id": annotation_ref.logical_id,
            "annotation_sha256": annotation_ref.sha256,
            "session_manifest_logical_id": manifest_ref.logical_id,
            "session_manifest_sha256": manifest_ref.sha256,
        }
        for name, source_path, logical_key, sha_key in (
            ("hud_truth_bindings", session_path / "hud_truth.jsonl", "hud_truth_logical_id", "hud_truth_sha256"),
            ("frames_bindings", session_path / "frames.jsonl", "frames_logical_id", "frames_sha256"),
        ):
            ref = store.put(
                logical_id=f"fixtures/capture/{session_id}/{name}.jsonl",
                source_path=source_path, media_type="application/x-ndjson",
            )
            capture_refs.append(ref)
            bindings[name][session_id] = {logical_key: ref.logical_id, sha_key: ref.sha256}
        world_ref = store.put(
            logical_id=f"fixtures/capture/{session_id}/world_coco.json",
            source_path=capture_root / "fixture-coco" / session_id / "world_coco.json",
            media_type="application/json",
        )
        ui_ref = store.put(
            logical_id=f"fixtures/capture/{session_id}/ui_coco.json",
            source_path=capture_root / "fixture-coco" / session_id / "ui_coco.json",
            media_type="application/json",
        )
        capture_refs.extend((world_ref, ui_ref))
        bindings["coco_bindings"][session_id] = {
            "world_coco_logical_id": world_ref.logical_id,
            "world_coco_sha256": world_ref.sha256,
            "ui_coco_logical_id": ui_ref.logical_id,
            "ui_coco_sha256": ui_ref.sha256,
        }
    model_frames = store.put(
        logical_id="fixtures/capture/model-train/frames.jsonl",
        source_path=capture_root / "capture_sessions" / "cal-0" / "frames.jsonl",
        media_type="application/x-ndjson",
    )
    capture_refs.append(model_frames)
    bindings["frames_bindings"]["model-train"] = {
        "frames_logical_id": model_frames.logical_id,
        "frames_sha256": model_frames.sha256,
    }
    gates_path = Path(__file__).resolve().parents[1] / "configs" / "formal_gates_minimal_v1.yaml"
    gates_values = yaml.safe_load(gates_path.read_text(encoding="utf-8"))
    gates_values["slice_count_floors"] = {
        name: 2 for name in gates_values["slice_count_floors"]
    }
    gates_values["slice_session_floors"] = {
        name: 0 for name in gates_values["slice_session_floors"]
    }
    gates_values["occurrence_dependent_slices"] = sorted(
        name for name in gates_values["slice_count_floors"]
        if name != "foreground_class:enemy_boss"
    )
    gates_bytes = yaml.safe_dump(gates_values, sort_keys=True).encode("utf-8")
    gates_ref = store.put_bytes(
        logical_id="fixtures/capture/formal_gates_test_v1.yaml",
        data=gates_bytes, media_type="application/yaml",
    )
    capture_refs.append(gates_ref)
    windows_sessions = {
        session_id: [[0, DatasetWriter.restore(capture_root, session_id, metadata_only=True).frame_count - 1]]
        for session_id in session_ids
    }
    windows_rule = {
        "fixed_offsets_per_time_band": {
            "window_frames": 900,
            "offsets_seconds": {"early": 60, "mid": 900, "late": 1620},
        },
        "run_tail": {"seconds": 90},
        "event_windows": {
            "detector": "development_hud_parser",
            "states": ["level_up_items", "level_up_fallback", "chest"],
            "pre_frames": 60, "post_frames": 120, "max_per_band": 4,
            "parser_artifact_hash": parser_hash,
        },
        "boss_windows": {
            "detector": "development_world_detector",
            "checkpoint_hash": model_hash, "score_threshold": 0.35,
            "pre_frames": 60, "post_frames": 300, "max_per_session": 8,
        },
    }
    windows_plan = EvaluationWindowPlan.create(
        windows_sessions, rule=windows_rule, stride=1,
        generated_at_utc="2026-10-11T00:00:00Z",
    )
    windows_ref = store.put_bytes(
        logical_id="fixtures/capture/evaluation_windows.json",
        data=canonical_json_bytes(windows_plan.to_wire()), media_type="application/json",
    )
    capture_refs.append(windows_ref)
    capture_manifest = capture_refs[0]
    capture_descriptor = ArtifactDescriptor(
        logical_id="fixtures/capture", node_kind="source_descriptor",
        producer_id="pytest", producer_version="v1",
        identity_metadata={
            "manifest_logical_id": capture_manifest.logical_id, **bindings,
            "formal_gates": {
                "logical_id": gates_ref.logical_id,
                "sha256": gates_ref.sha256,
                "schema_version": "formal_gates.v1",
            },
            "evaluation_windows": {
                "sessions": windows_plan.to_wire()["sessions"],
                "stride": windows_plan.stride,
                "plan_canonical_hash": windows_plan.plan_canonical_hash,
            },
            "evaluation_windows_hash": windows_ref.sha256,
        }, files=tuple(capture_refs),
    )
    descriptors["capture_dataset"] = capture_descriptor
    restores["capture_dataset"] = RestoreTestVerdict(
        logical_id="fixtures/capture.restore", subject=capture_descriptor.node_ref(),
        manifest_hash=capture_manifest.sha256, primary_root="fixture-primary",
        backup_root="fixture-backup", verify_mode="full",
        checked_object_count=len(capture_refs), passed=True, blocking_reasons=(),
    ).to_descriptor()
    return FormalBenchmarkRequest(
        store=store, capture_store_root=capture_root,
        dependency_descriptors=descriptors, restore_verdicts=restores,
    )


def _request_with_capture_metadata(request, identity_metadata):
    """capture descriptor の binding metadata を置き換える。

    restore verdict を新しい descriptor identity に結び直して binding 検証へ進める。
    """
    descriptors = dict(request.dependency_descriptors)
    capture = replace(
        descriptors["capture_dataset"], identity_metadata=identity_metadata,
    )
    descriptors["capture_dataset"] = capture
    restores = dict(request.restore_verdicts)
    old_restore = restores["capture_dataset"]
    metadata = old_restore.identity_metadata
    non_identity = old_restore.non_identity_metadata
    restores["capture_dataset"] = RestoreTestVerdict(
        logical_id=old_restore.logical_id,
        subject=capture.node_ref(),
        manifest_hash=metadata["manifest_hash"],
        primary_root=non_identity["primary_root"],
        backup_root=non_identity["backup_root"],
        verify_mode=metadata["verify_mode"],
        checked_object_count=metadata["checked_object_count"],
        passed=metadata["passed"],
        blocking_reasons=metadata["blocking_reasons"],
        producer_id=old_restore.producer_id,
        producer_version=old_restore.producer_version,
        schema_version=metadata["restore_test_schema_version"],
    ).to_descriptor()
    return replace(request, dependency_descriptors=descriptors, restore_verdicts=restores)


def _formal_threshold_bytes() -> bytes:
    """test threshold 定数の canonical payload を作成する。

    runner の ArtifactStore hash preflight を test gate と一致させる。
    """
    import survivors.perception_benchmark as benchmark
    values = {
            field: getattr(benchmark, constant)
            for field, constant in {
                "screen_state_f1": "THRESHOLD_SCREEN_F1",
                "timer_exact": "THRESHOLD_TIMER_EXACT",
                "level_exact": "THRESHOLD_LEVEL_EXACT",
                "inventory_top1": "THRESHOLD_INVENTORY_TOP1",
                "choice_top1": "THRESHOLD_CHOICE_TOP1",
                "hp_mae": "THRESHOLD_HP_MAE",
                "xp_mae": "THRESHOLD_XP_MAE",
                "density_corr": "THRESHOLD_DENSITY_CORR",
                "nearest_med": "THRESHOLD_NEAREST_MED",
                "lat_p95_ms": "THRESHOLD_LAT_P95_MS",
                "lat_p99_ms": "THRESHOLD_LAT_P99_MS",
                "invalid_tick": "THRESHOLD_INVALID_TICK",
                "levelup_invalid": "THRESHOLD_LEVELUP_INVALID",
                "roi_center_p99": "THRESHOLD_ROI_CENTER_P99",
                "roi_inside": "THRESHOLD_ROI_INSIDE",
                "confidence": "THRESHOLD_CONFIDENCE",
                "cross_frame_equivalence": "THRESHOLD_CROSS_FRAME_EQUIVALENCE",
                "min_screen_state_count": "FORMAL_MIN_SCREEN_STATE_COUNT",
                "min_screen_state_records": "FORMAL_MIN_SCREEN_STATE_RECORDS",
                "foreground_classes": "_FORMAL_FOREGROUND_CLASSES",
                "slice_count_floors": "_FORMAL_SLICE_COUNT_FLOORS",
                "slice_session_floors": "_FORMAL_SLICE_SESSION_FLOORS",
                "slice_thresholds": "_FORMAL_SLICE_THRESHOLDS",
                "time_bands": "_FORMAL_TIME_BANDS",
                "occurrence_dependent_slices": "_FORMAL_OCCURRENCE_DEPENDENT_SLICES",
            }.items()
    }
    values["foreground_classes"] = list(values["foreground_classes"])
    for field in ("slice_count_floors", "slice_session_floors", "slice_thresholds"):
        values[field] = dict(sorted(values[field].items()))
    for field in ("time_bands", "occurrence_dependent_slices"):
        values[field] = sorted(values[field])
    return canonical_json_bytes({"schema_version": "perception_formal_thresholds.v1", **values})


def _formal_dependency(
    store, name: str, schema: str, values: dict[str, bytes], *,
    extras: dict[str, bytes] | None = None,
    descriptor_manifest_name: str = "manifest.json",
):
    """hash-bound package descriptor と full restore verdict を作る。

    role payload と追加 package files を別々に descriptor refs へ登録する。
    """
    refs = {
        field: store.put_bytes(
            logical_id=f"fixtures/{name}/{field}.bin", data=data,
            media_type="application/octet-stream",
        )
        for field, data in values.items()
    }
    payload = {
        "schema_version": schema, "development_only": False, "formal_eligible": True,
        "files": {field: ref.logical_id for field, ref in refs.items()},
        **{field: ref.sha256 for field, ref in refs.items()},
    }
    manifest = store.put_bytes(
        logical_id=f"fixtures/{name}/{descriptor_manifest_name}",
        data=canonical_json_bytes(payload), media_type="application/json",
    )
    extra_refs = tuple(
        store.put_bytes(logical_id=logical_id, data=data, media_type="application/json")
        for logical_id, data in (extras or {}).items()
    )
    files = (manifest, *refs.values(), *extra_refs)
    descriptor = ArtifactDescriptor(
        logical_id=f"fixtures/{name}", node_kind="source_descriptor",
        producer_id="pytest", producer_version="v1",
        identity_metadata={"manifest_logical_id": manifest.logical_id}, files=files,
    )
    restore = RestoreTestVerdict(
        logical_id=f"fixtures/{name}.restore", subject=descriptor.node_ref(),
        manifest_hash=manifest.sha256, primary_root="fixture-primary",
        backup_root="fixture-backup", verify_mode="full",
        checked_object_count=len(files), passed=True, blocking_reasons=(),
    ).to_descriptor()
    return descriptor, restore


def _package_value(request: FormalBenchmarkRequest, package: str, field: str) -> str:
    """restored package manifest が宣言する field の content hash を返す。

    assertion を descriptor の任意 file 順に依存させない。
    """
    return _package_file_ref(request, package, field).sha256


def _package_file_ref(request: FormalBenchmarkRequest, package: str, field: str):
    """role field に結び付いた ArtifactRef を取得する。

    tamper test は manifest が指す ref だけを書き換える。
    """
    descriptor = request.dependency_descriptors[package]
    manifest_ref = next(
        ref for ref in descriptor.files
        if ref.logical_id == descriptor.identity_metadata["manifest_logical_id"]
    )
    manifest = json.loads(request.store.object_path(manifest_ref.store_uri).read_bytes())
    return next(
        ref for ref in descriptor.files
        if ref.logical_id == manifest["files"][field]
    )


def _store_logical_ids(store: ArtifactStore) -> set[str]:
    """ArtifactStore index に commit 済みの logical ID を集める。

    commit 失敗 test で canonical alias が残らないことを確認する。
    """
    return {
        payload["logical_id"]
        for path in store.logical_root.glob("*.json")
        if isinstance((payload := json.loads(path.read_text("utf-8"))), dict)
        and isinstance(payload.get("logical_id"), str)
    }


def test_formal_entry_rejects_ref_only_synthetic_fixture_before_provider(
    tmp_path: Path, monkeypatch
) -> None:
    """ArtifactDescriptor/restore verdict のない synthetic fixture は formal に入れない。"""
    import benchmark_survivors_perception as module

    called = False

    def forbidden_factory(_restored):
        nonlocal called
        called = True
        raise AssertionError("provider must not run")

    monkeypatch.setattr(module, "_CLI_PROVIDER_FACTORY", forbidden_factory)
    request = replace(_request(tmp_path), dependency_descriptors={}, restore_verdicts={})
    before = set(request.store.logical_root.glob("*.json"))
    with pytest.raises(ValueError, match="descriptors/restore verdicts"):
        run_formal_pipeline(request)
    assert called is False
    assert set(request.store.logical_root.glob("*.json")) == before


def test_final_metadata_preflight_does_not_open_or_hash_png(tmp_path: Path) -> None:
    capture_root, _split = _capture_fixture(tmp_path)
    full = DatasetWriter.restore(capture_root, "final-0")
    png_path = full.session_path / full.frame_records[0].object_path
    png_path.unlink()

    metadata = DatasetWriter.restore(capture_root, "final-0", metadata_only=True)

    assert metadata.frame_records == full.frame_records
    assert metadata.frames == ()
    with pytest.raises(ValueError, match="missing object"):
        DatasetWriter.restore(capture_root, "final-0")


def test_formal_predictor_cannot_return_ground_truth_residual_or_evidence(
    tmp_path: Path,
) -> None:
    """旧4-tuple provider output は prediction-only 境界で拒否する。"""
    import benchmark_survivors_perception as module

    with pytest.raises(TypeError, match="prediction, latency_ms"):
        module._unpack_prediction_result((None, None, 1.0, FormalReplayEvidence((), ())))


def test_replay_uses_registered_windows_and_keeps_missing_window_gt_expected(tmp_path: Path) -> None:
    """registered window だけを expected tick にし、窓内 GT 欠損は期待母集団に残す。
    窓外 frame を除外しつつ window ごとの predictor reset を確認する。
    """
    parser_hash = "9" * 64
    capture_root, _split = _capture_fixture(
        tmp_path, parser_hash, include_formal_truth=True,
    )
    manifest = DatasetWriter.restore(capture_root, "cal-0", metadata_only=True)
    truth_rows = {row.frame_id: row for row in read_hud_truth(manifest.session_path)}
    coco = coco_image_entries_from_wire(
        json.loads((capture_root / "fixture-coco" / "cal-0" / "world_coco.json").read_text("utf-8")),
        session_id="cal-0",
    )
    class_map = default_class_map()

    class GroundTruthLoader:
        """frame 1 のみ GT を返す test loader。
        窓内 frame 2 を欠損させ、expected tick の扱いを分けて確認する。
        """

        def load_frame(self, _frame, session_id: str, frame_index: int):
            """選択 frame 1 の GT snapshot だけを生成する。
            ほかの frame は ground truth が無い状態を返す。
            """
            if frame_index != 1:
                return None
            return ground_truth_snapshot(
                truth_rows[frame_index], coco[frame_index], manifest.frame_records[frame_index],
                parser_artifact_hash=parser_hash, session_id=session_id,
                class_map=class_map,
            )

    class Predictor:
        """reset 回数を記録する prediction-only provider。
        全 frame で prediction missing を返し ground truth 混入を防ぐ。
        """

        resets = 0

        def reset(self) -> None:
            """window 開始時の reset 回数を記録する。
            test では provider state 自体は持たない。
            """
            self.resets += 1

        def predict_frame(self, *_args):
            """prediction missing を返し expected tick の母集団だけ確認する。
            GT snapshot を prediction path に渡さない。
            """
            return None, 0.0

    rule = {
        "fixed_offsets_per_time_band": {
            "window_frames": 900,
            "offsets_seconds": {"early": 60, "mid": 900, "late": 1620},
        },
        "run_tail": {"seconds": 90},
        "event_windows": {
            "detector": "development_hud_parser",
            "states": ["level_up_items", "level_up_fallback", "chest"],
            "pre_frames": 60, "post_frames": 120, "max_per_band": 4,
            "parser_artifact_hash": parser_hash,
        },
        "boss_windows": {
            "detector": "development_world_detector",
            "checkpoint_hash": "8" * 64, "score_threshold": 0.5,
            "pre_frames": 60, "post_frames": 300, "max_per_session": 8,
        },
    }
    plan = EvaluationWindowPlan.create(
        {"cal-0": [(1, 2), (3, 3)]}, rule=rule, stride=1,
        generated_at_utc="2026-10-11T00:00:00Z",
    )
    ground = ground_truth_snapshot(
        truth_rows[1], coco[1], manifest.frame_records[1],
        parser_artifact_hash=parser_hash, session_id="cal-0", class_map=class_map,
    )
    assert ground is not None
    predictor = Predictor()
    ticks, expected, _evidence = _replay_sessions(
        (manifest,),
        {"cal-0": SimpleNamespace(kind="error_calibration", source_policy="raw")},
        predictor, GroundTruthLoader(),
        {("cal-0", 1): (object(), _ground_truth_semantic_hash(ground))},
        parser_hash=parser_hash, detector_hash="7" * 64,
        evaluation_plan=plan,
    )

    assert [tick.frame_id for tick in expected] == ["cal-0:1", "cal-0:2", "cal-0:3"]
    assert [tick.frame_id for tick in ticks] == ["cal-0:1"]
    assert predictor.resets == 2


def test_formal_runner_calls_real_pipeline_and_atomic_writers(tmp_path: Path, monkeypatch) -> None:
    """仕様形の descriptor bindings から formal verdict まで実行する。

    model session の frames binding を許容し、runner と writer の出力を確認する。
    """
    import benchmark_survivors_perception as _bsp
    _install_public_formal_factories(monkeypatch, _bsp)
    _patch_minimal_test_gates(monkeypatch)
    request = _request(tmp_path, formal_inputs=True)
    assert "model-train" in request.dependency_descriptors["capture_dataset"].identity_metadata["frames_bindings"]
    result = run_formal_pipeline(request)

    # 3 cal sessions × 3 samples/session × 7 required fields
    assert result.profile.field_sample_counts == {
        field: 9 for field in _CALIBRATION_REQUIRED_FIELDS
    }
    assert result.verdict.development_only is False
    assert result.verdict.formal_perception_verdict_eligible is True
    assert result.verdict.passed is True, result.verdict.blocking_reasons
    assert result.calibration_ref.sha256 == result.descriptors[1].files[0].sha256
    assert result.verdict_ref.sha256 == result.descriptors[2].files[0].sha256
    assert result.descriptors[1].node_kind == "perception_calibration_profile"
    assert result.descriptors[2].node_kind == "perception_final_verdict"
    assert result.verdict.metrics["slice_counts"]["time_band:early"] == 6
    assert result.verdict.metrics["slice_counts"]["event:hazard"] == 3

    # 必須 named slice の欠落と rare slice の最低件数未満を再計算 gate が拒否する。
    # persisted metrics 改変でも同じ blocking 判定になることを確認する。
    missing = dict(result.verdict.metrics)
    missing["slice_counts"] = dict(missing["slice_counts"])
    del missing["slice_counts"]["foreground_class:enemy_boss"]
    passed, reasons = recompute_gate_from_metrics(missing, formal=True)
    assert passed is False
    assert any(
        "foreground_class:enemy_boss" in reason and "0 records" in reason
        for reason in reasons
    )
    underpowered = dict(result.verdict.metrics)
    underpowered["slice_counts"] = dict(underpowered["slice_counts"])
    underpowered["slice_counts"]["foreground_class:enemy_boss"] = 1
    passed, reasons = recompute_gate_from_metrics(underpowered, formal=True)
    assert passed is False
    assert any(
        "foreground_class:enemy_boss" in reason and "1 records" in reason
        for reason in reasons
    )


def test_formal_runner_rejects_missing_formal_capture_binding(tmp_path: Path, monkeypatch) -> None:
    """formal session の必須 binding が欠けた descriptor を拒否する。

    restore verdict を再束縛して、capture binding の fail-closed 検査を通す。
    """
    import benchmark_survivors_perception as _bsp

    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _test_provider_factory)
    request = _request(tmp_path, formal_inputs=True)
    metadata = dict(request.dependency_descriptors["capture_dataset"].identity_metadata)
    frames = dict(metadata["frames_bindings"])
    del frames["cal-0"]
    metadata["frames_bindings"] = frames
    request = _request_with_capture_metadata(request, metadata)

    with pytest.raises(ValueError, match="must cover formal sessions"):
        run_formal_pipeline(request)


@pytest.mark.parametrize(
    ("field", "value"),
    (("ui_coco_sha256", "0" * 64), ("ui_coco_logical_id", None)),
)
def test_formal_runner_rejects_invalid_ui_coco_binding(
    tmp_path: Path, monkeypatch, field: str, value: str | None,
) -> None:
    """UI COCO binding の欠損 key と hash 不一致を拒否する。

    world file だけを検証して ui_coco を取り違える経路を閉じる。
    """
    import benchmark_survivors_perception as _bsp

    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _test_provider_factory)
    request = _request(tmp_path, formal_inputs=True)
    metadata = dict(request.dependency_descriptors["capture_dataset"].identity_metadata)
    coco = {session: dict(entry) for session, entry in metadata["coco_bindings"].items()}
    if value is None:
        del coco["cal-0"][field]
    else:
        coco["cal-0"][field] = value
    metadata["coco_bindings"] = coco
    request = _request_with_capture_metadata(request, metadata)

    expected_error = "invalid schema" if value is None else "not descriptor-backed"
    with pytest.raises(ValueError, match=expected_error):
        run_formal_pipeline(request)


def test_published_calibration_profile_is_consumer_compatible(tmp_path: Path, monkeypatch) -> None:
    """canonical logical_id に保存した profile が PerceptionErrorProfile.from_wire() で読み込める。

    Training consumer は to_artifact_wire() ラッパーではなく to_wire() を期待する（#8 回帰防止）。
    """
    from reinbalance_survivors_contracts.perception_error import PerceptionErrorProfile

    import benchmark_survivors_perception as _bsp
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _test_provider_factory)
    _patch_minimal_test_gates(monkeypatch)
    request = _request(tmp_path, formal_inputs=True)
    result = run_formal_pipeline(request)

    cal_index = request.store._logical_index_path(result.calibration_ref.logical_id)
    from Tools.Artifacts.artifact_store import ArtifactRef
    import json
    cal_ref_wire = json.loads(cal_index.read_text("utf-8"))
    cal_bytes = request.store.object_path(cal_ref_wire["store_uri"]).read_bytes()
    wire = json.loads(cal_bytes)
    # PerceptionErrorProfile.from_wire() が例外を投げないことを確認する（schema互換性）
    profile = PerceptionErrorProfile.from_wire(wire)
    assert profile is not None


def test_committed_batch_second_recovery_uses_profile_artifact_envelope(
    tmp_path: Path,
) -> None:
    """producer descriptorの実payloadから2回復旧してもformal profileを再発行できる。"""
    from test_perception_formal_guard import _passing_formal_verdict_wire

    store = ArtifactStore(tmp_path / "artifacts")
    capture_ref = store.put_bytes(
        logical_id="fixtures/capture.json",
        data=b"{}",
        media_type="application/json",
    )
    target_ref = store.put_bytes(
        logical_id="fixtures/target.json",
        data=b"{}",
        media_type="application/json",
    )

    def source_descriptor(logical_id, ref):
        """fixture refを最小source descriptorへ束縛する。"""
        return ArtifactDescriptor(
            logical_id=logical_id,
            node_kind="source_descriptor",
            producer_id="pytest",
            producer_version="v1",
            identity_metadata={"content_hash": ref.sha256},
            files=(ref,),
        )

    request = FormalBenchmarkRequest(
        store=store,
        capture_store_root=tmp_path / "captures",
        dependency_descriptors={
            "capture_dataset": source_descriptor("capture", capture_ref),
            "target_config": source_descriptor("target", target_ref),
        },
    )
    payloads = {
        "parser_package": {"parser_artifact_hash": "1" * 64},
        "detector_package": {
            "detector_artifact_hash": "2" * 64,
            "model_hash": "3" * 64,
            "build_hash": "4" * 64,
        },
        "assembler_config": {
            "assembler_schema_hash": "5" * 64,
            "ui_presentation_schema_hash": "6" * 64,
            "ui_presentation_golden_fixture_hash": "7" * 64,
            "atlas_vocabulary_hash": "8" * 64,
            "assembler_impl_hash": "9" * 64,
            "roi_resolver_input_hash": "a" * 64,
        },
        "target_config": {"threshold_hash": "b" * 64},
    }
    profile = FittedPerceptionErrorProfile(
        calibration_session_ids=["cal-0"],
        final_e2e_session_ids=["final-0"],
        calibration_session_hashes={"cal-0": "c" * 64},
        field_sample_counts={"hp_ratio": 2},
        fit_code_hash="d" * 64,
        development_only=False,
        _factory_token=_FORMAL_FACTORY_TOKEN,
    )
    run_key = "e" * 64
    provisional = _subject_hashes(
        request,
        payloads,
        calibration_profile_hash="0" * 64,
        lineage_seal_hash="0" * 64,
    )
    provenance_subjects = {
        name: value
        for name, value in provisional.items()
        if name not in {"calibration_profile_hash", "lineage_seal_hash"}
    }
    calibration_descriptors = _calibration_descriptor_chain(
        request,
        profile,
        provenance_subjects,
        request.calibration_logical_id(run_key),
    )
    profile_node = calibration_descriptors[1]
    raw_file = next(ref for ref in profile_node.files if ref.logical_id.endswith("/profile.json"))
    artifact_file = next(
        ref for ref in profile_node.files if ref.logical_id.endswith("/profile.artifact.json")
    )
    provenance_file = next(
        ref for ref in profile_node.files if ref.logical_id.endswith("/provenance.json")
    )
    profile_refs = [
        _put_descriptor_file(store, raw_file, canonical_json_bytes(profile.to_wire())),
        _put_descriptor_file(
            store, artifact_file, canonical_json_bytes(profile.to_artifact_wire())
        ),
        _put_descriptor_file(
            store,
            provenance_file,
            _calibration_provenance_bytes(profile, provenance_subjects),
        ),
    ]

    final_hash = "f" * 64
    preseal_subjects = _subject_hashes(
        request,
        payloads,
        calibration_profile_hash=profile_node.identity_hash,
        lineage_seal_hash="0" * 64,
    )
    seal = _create_formal_lineage_seal(
        final_session_hashes={"final-0": final_hash},
        store=store,
        _publish=False,
        **{
            name: value
            for name, value in preseal_subjects.items()
            if name != "lineage_seal_hash"
        },
    )
    seal_ref = store.put_bytes(
        logical_id=f"perception/package/lineage/{seal.seal_id}/seal.json",
        data=canonical_json_bytes(seal.to_wire()),
        media_type="application/json",
    )
    subjects = _subject_hashes(
        request,
        payloads,
        calibration_profile_hash=profile_node.identity_hash,
        lineage_seal_hash=seal_ref.sha256,
    )
    verdict_wire = _passing_formal_verdict_wire(subjects)
    verdict_wire["seal_id"] = seal.seal_id
    verdict_wire["verdict_id"] = canonical_hash({
        "seal_id": seal.seal_id,
        "final_session_ids": verdict_wire["final_session_ids"],
        "subject_hashes": {name: verdict_wire[name] for name in _HASH_FIELDS},
        "metrics": verdict_wire["metrics"],
    })
    stored_verdict = store.put_bytes(
        logical_id="fixtures/verdict.json",
        data=canonical_json_bytes(verdict_wire),
        media_type="application/json",
    )
    verdict = _load_formal_verdict_from_verified_store(
        verdict_wire,
        store=store,
        ref=stored_verdict,
        current_subject_hashes=subjects,
    )
    assert isinstance(verdict, PerceptionFinalVerdict)
    descriptors = _descriptor_chain(
        calibration_descriptors,
        request,
        verdict,
        request.verdict_logical_id(run_key),
    )
    verdict_ref = _put_descriptor_file(
        store, descriptors[2].files[0], canonical_json_bytes(verdict_wire)
    )
    refs = [capture_ref, *profile_refs, seal_ref, verdict_ref]
    store.put_bytes(
        logical_id=f"perception/batch_commit/{run_key}",
        data=canonical_json_bytes({
            "schema_version": "perception_formal_batch_commit.v2",
            "run_key": run_key,
            "descriptors": [descriptor.to_wire() for descriptor in descriptors],
            "descriptor_hashes": [descriptor.identity_hash for descriptor in descriptors],
            "refs": [ref.to_wire() for ref in refs],
            "profile_artifact": profile.to_artifact_wire(),
            "verdict": verdict_wire,
        }),
        media_type="application/json",
    )
    split = SessionSplit((SessionRecord(
        session_id="final-0",
        session_hash=final_hash,
        build_hash="4" * 64,
        target_profile_hash="0" * 64,
        resolution_wh=(1920, 1080),
        duration_seconds=1800.0,
        kind="final_e2e_test",
        source_policy="lossless",
    ),))

    first = _recover_committed_result(request, split, run_key, payloads)
    second = _recover_committed_result(request, split, run_key, payloads)

    assert first is not None and second is not None
    assert second.profile.to_artifact_wire() == profile.to_artifact_wire()
    assert second.calibration_artifact_ref == first.calibration_artifact_ref
    assert store.resolve(request.calibration_logical_id(run_key)) == second.calibration_ref
    assert json.loads(store.object_path(second.calibration_ref.store_uri).read_bytes()) == (
        profile.to_wire()
    )
    assert json.loads(
        store.object_path(second.calibration_artifact_ref.store_uri).read_bytes()
    ) == profile.to_artifact_wire()


def test_formal_runner_blocks_when_prediction_differs_from_ground_truth(tmp_path: Path, monkeypatch) -> None:
    """予測 HP を大きく外すと自己比較せず formal gate が公開を拒否する。

    UI hash は変更しないため、同じ raw evidence による gt/pred 双方の独立検証と
    HP 精度 gate の失敗を同時に確認できます。
    """
    class _WrongHpAssembler:
        """正解 snapshot を維持し、予測側の HP だけを誤らせる wrapper。"""

        def __init__(self, parser_artifact_hash: str) -> None:
            self._inner = _SnapshotFixture(parser_artifact_hash)

        def assemble_frame(self, frame, session_id: str, frame_index: int):
            ground_truth, predicted, latency, evidence = self._inner.assemble_frame(
                frame, session_id, frame_index
            )
            if predicted is not None and predicted.item_context is not None:
                predicted = replace(
                    predicted,
                    item_context=replace(predicted.item_context, hp_ratio=0.0),
                )
            return ground_truth, predicted, latency, evidence

    def _wrong_hp_factory(restored):
        return _mutating_test_predictor(
            restored,
            lambda snapshot: replace(
                snapshot,
                item_context=replace(snapshot.item_context, hp_ratio=0.0),
            ) if snapshot.item_context is not None else snapshot,
        )

    import benchmark_survivors_perception as _bsp
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _wrong_hp_factory)
    _patch_minimal_test_gates(monkeypatch)
    request = _request(tmp_path, formal_inputs=True)
    with pytest.raises(ValueError, match="HP MAE outside formal threshold"):
        run_formal_pipeline(request)

    # gate 不合格時は canonical verdict を公開しない。
    # 自己比較で pass していた旧 runner への直接的な回帰防止です。
    assert not any(
        logical_id.startswith(request.verdict_namespace + "/")
        and logical_id.endswith("/verdict.json")
        for logical_id in _store_logical_ids(request.store)
    )


def test_replay_excludes_zero_area_ground_rois_from_geometry_records() -> None:
    """幅ゼロ・高さゼロの各 ground ROI を実 replay record 経路で除外する。

    対応 predicted が usable でも geometry 成功値は生成せず、余剰予測として
    ui_false_positive に記録されることを確認します。
    """
    schema = DeployObsSchema.default_v2()
    ticks: list[SnapshotReplayTick] = []
    for index, ground_roi in enumerate(((200, 100, 200, 500), (100, 200, 400, 200))):
        session_id = f"zero-roi-{index}"
        ground_hud, world = _hud_world(
            session_id, 0, 1_000_000_000, "level_up_items", card_roi=ground_roi
        )
        predicted_hud, _ = _hud_world(
            session_id, 0, 1_000_000_000, "level_up_items"
        )
        ground_hud = replace(ground_hud, buttons=())
        predicted_hud = replace(predicted_hud, buttons=())
        ground_truth = RealObsAssembler().assemble(
            ground_hud, world, schema, (1920, 1080)
        )
        predicted = RealObsAssembler().assemble(
            predicted_hud, world, schema, (1920, 1080)
        )
        assert ground_truth is not None and predicted is not None
        ticks.append(SnapshotReplayTick(
            session_id, "final_e2e_test", "lossless",
            ground_truth.frame_id, ground_truth, predicted, 1.0,
        ))

    records = replay_snapshots(ticks)
    geometry = {
        record.field for record in records
        if record.field in {"ui_inside_region", "ui_roi_center_error"}
    }
    false_positives = [
        record for record in records if record.field == "ui_false_positive"
    ]
    assert geometry == set()
    assert len(false_positives) == 2
    assert all(record.predicted is True for record in false_positives)


def test_dependency_tamper_fails_before_any_runner_output(tmp_path: Path, monkeypatch) -> None:
    import benchmark_survivors_perception as _bsp
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _test_provider_factory)
    request = _request(tmp_path, formal_inputs=True)
    before_indexes = set(request.store.logical_root.glob("*.json"))
    parser_ref = _package_file_ref(request, "parser_package", "parser_artifact_hash")
    request.store.object_path(parser_ref.store_uri).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="verification failed"):
        run_formal_pipeline(request)

    assert set(request.store.logical_root.glob("*.json")) == before_indexes


def test_typed_replay_recomputes_ui_hashes_and_rejects_tamper(tmp_path: Path) -> None:
    request = _request(tmp_path)
    manifests = tuple(
        DatasetWriter.restore(request.capture_store_root, f"final-{index}")
        for index in range(3)
    )
    ticks, evidence = _run_assembler_for_test(manifests)
    from survivors.perception_benchmark import ExpectedTick, run_benchmark

    target = ticks[1].predicted.ui_presentation
    bad_ui = replace(target, source_content_hash="0" * 64)
    bad_snapshot = replace(ticks[1].predicted, source_content_hash="0" * 64, ui_presentation=bad_ui)
    ticks[1] = replace(ticks[1], predicted=bad_snapshot)
    expected = [ExpectedTick(tick.session_id, tick.frame_id) for tick in ticks]
    with pytest.raises(ValueError, match="source/content hash"):
        run_benchmark(ticks, expected_ticks=expected)


def _run_assembler_for_test(
    manifests: "Sequence[Any]", detector_hash: str = "2" * 64
) -> "tuple[list[SnapshotReplayTick], dict[str, FormalReplayEvidence]]":
    """unit test 用：mock assembler を使って SnapshotReplayTick リストと evidence を生成する。

    run_formal_pipeline が内部で行うのと同等の tick 生成を再現します。
    """
    assembler = _SnapshotFixture()
    ticks = []
    evidence: dict[str, FormalReplayEvidence] = {}
    for manifest in manifests:
        for frame in manifest.frames:
            ground_truth, predicted, latency, ev = assembler.assemble_frame(
                frame, manifest.session_id, frame.session_frame_index
            )
            if ground_truth is None:
                continue
            ground_truth = replace(
                ground_truth, detector_artifact_hash=detector_hash
            )
            if predicted is not None:
                predicted = replace(predicted, detector_artifact_hash=detector_hash)
            if ev is not None:
                evidence[ground_truth.frame_id] = ev
            ticks.append(SnapshotReplayTick(
                manifest.session_id, "final_e2e_test", "lossless",
                ground_truth.frame_id, ground_truth, predicted, latency,
            ))
    return ticks, evidence


def test_formal_runner_assembler_cannot_override_detector_hash(tmp_path: Path, monkeypatch) -> None:
    """assembler が任意 detector hash を設定しても runner が restored package hash で上書きする。

    verdict の detector_artifact_hash は ArtifactStore 内の detector package ファイル自体の
    SHA-256（_subject_hashes 経由）であり、assembler の自己申告とは無関係です。
    """
    import benchmark_survivors_perception as _bsp
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _test_provider_factory)
    _patch_minimal_test_gates(monkeypatch)
    request = _request(tmp_path, formal_inputs=True)
    result = run_formal_pipeline(request)
    # verdict.detector_artifact_hash = ArtifactStore 内 detector_package JSON の sha256
    expected = _package_value(request, "detector_package", "detector_artifact_hash")
    assert result.verdict.detector_artifact_hash == expected


def test_candidate_set_hash_tamper_caught_by_raw_card_ids(tmp_path: Path) -> None:
    """candidate_set_hash 単独改ざんを raw_card_ids 独立検証で検出する。

    outer hash を更新せずに candidate_set_hash だけ差し替えても、
    raw_card_ids から再計算したハッシュと照合することで改ざんを検出します。
    """
    request = _request(tmp_path)
    manifests = tuple(
        DatasetWriter.restore(request.capture_store_root, f"final-{index}")
        for index in range(3)
    )
    ticks, evidence = _run_assembler_for_test(manifests)
    from dataclasses import replace
    from survivors.perception_benchmark import ExpectedTick, run_benchmark

    target = ticks[0].ground_truth.ui_presentation
    # evidence（raw_card_ids）は実値のまま、candidate_set_hash だけ偽値に差し替える。
    # formal_evidence を渡すと evidence check が先に発火して coherent 偽値でも検出する。
    bad_ui = replace(target, candidate_set_hash="f" * 64)
    bad_snapshot = replace(ticks[0].ground_truth, ui_presentation=bad_ui)
    ticks[0] = replace(ticks[0], ground_truth=bad_snapshot)
    expected = [ExpectedTick(tick.session_id, tick.frame_id) for tick in ticks]
    with pytest.raises(ValueError, match="candidate_set_hash does not match formal evidence card IDs"):
        run_benchmark(ticks, expected_ticks=expected, formal_evidence=evidence)


def test_inventory_hash_tamper_caught_by_raw_inventory(tmp_path: Path) -> None:
    """inventory_hash 単独改ざんを raw_inventory 独立検証で検出する。"""
    request = _request(tmp_path)
    manifests = tuple(
        DatasetWriter.restore(request.capture_store_root, f"final-{index}")
        for index in range(3)
    )
    ticks, evidence = _run_assembler_for_test(manifests)
    from survivors.perception_benchmark import ExpectedTick, run_benchmark

    target = ticks[0].ground_truth.ui_presentation
    bad_ui = replace(target, inventory_hash="e" * 64)
    bad_snapshot = replace(ticks[0].ground_truth, ui_presentation=bad_ui)
    ticks[0] = replace(ticks[0], ground_truth=bad_snapshot)
    expected = [ExpectedTick(tick.session_id, tick.frame_id) for tick in ticks]
    with pytest.raises(ValueError, match="inventory_hash does not match formal evidence inventory"):
        run_benchmark(ticks, expected_ticks=expected, formal_evidence=evidence)


def test_annotation_evidence_is_not_applied_to_prediction(
    tmp_path: Path, monkeypatch
) -> None:
    """annotation raw evidence は ground truth にだけ照合する。"""
    import survivors.perception_benchmark as module

    request = _request(tmp_path)
    manifests = (DatasetWriter.restore(request.capture_store_root, "final-0"),)
    ticks, evidence = _run_assembler_for_test(manifests)
    calls: list[str] = []
    monkeypatch.setattr(
        module,
        "_verify_formal_evidence",
        lambda _ui, _evidence, label: calls.append(label),
    )
    module.replay_snapshots(ticks, formal_evidence=evidence)
    assert calls == [f"ground/{tick.frame_id}" for tick in ticks]


def test_all_foreground_classes_use_entity_semantic_correctness(tmp_path: Path) -> None:
    request = _request(tmp_path)
    manifest = DatasetWriter.restore(request.capture_store_root, "final-0")
    frame = manifest.frames[0]
    ground, predicted, _latency, _evidence = _SnapshotFixture().assemble_frame(
        frame, manifest.session_id, frame.session_frame_index
    )
    assert ground is not None and predicted is not None
    classes = (
        "player_anchor", "enemy_normal", "enemy_elite", "enemy_boss",
        "gem_blue", "gem_green", "gem_red", "pickup_heal",
        "pickup_special", "hazard_projectile", "hazard_area",
    )
    for class_name in classes:
        wrong_class = "enemy_elite" if class_name == "enemy_normal" else "enemy_normal"
        annotated = replace(
            ground,
            diagnostics={
                "foreground_entity_counts": {class_name: 1},
                "foreground_entity_classes": {"entity-0": class_name},
            },
        )
        wrong = replace(
            predicted,
            diagnostics={"foreground_entity_classes": {"entity-0": wrong_class}},
        )
        values = _foreground_classification_values(annotated, wrong)
        assert values[f"foreground_class:{class_name}"] == [0.0]


def test_loader_cannot_self_authenticate_different_ground_truth(tmp_path: Path) -> None:
    """loader が evidence を返せず、immutable annotation の GT hash と不一致なら拒否する。"""
    request = _request(tmp_path)
    manifest = DatasetWriter.restore(request.capture_store_root, "final-0")
    frame = manifest.frames[0]
    original_ground, predicted, _latency, evidence = _SnapshotFixture().assemble_frame(
        frame, manifest.session_id, frame.session_frame_index
    )
    assert original_ground is not None and predicted is not None and evidence is not None
    bound_frame_id = f"{manifest.session_id}:{frame.session_frame_index}"
    original_ground = replace(
        original_ground,
        frame_id=bound_frame_id,
        ui_presentation=replace(
            original_ground.ui_presentation, frame_id=bound_frame_id
        ),
    )
    predicted = replace(
        predicted,
        frame_id=bound_frame_id,
        ui_presentation=replace(predicted.ui_presentation, frame_id=bound_frame_id),
    )
    forged_ui = replace(
        original_ground.ui_presentation, screen_state="forged-screen-state"
    )
    forged_ground = replace(
        original_ground,
        screen_state="forged-screen-state",
        ui_presentation=forged_ui,
    )

    class ForgedLoader:
        def load_frame(self, _frame, _session_id, _frame_index):
            return forged_ground

    class Predictor:
        def reset(self) -> None:
            """fake predictor の reset 要求を no-op で受ける。

            fixture に session state がなく、予測値へ影響しない。
            """

        def predict_frame(self, _frame, _session_id, _frame_index):
            """固定した tampered prediction を返す。

            detector hash の改変が runner の immutable annotation 検査を通らないことを試す。
            """
            return replace(predicted, detector_artifact_hash="2" * 64), 1.0

    annotation = {
        (manifest.session_id, frame.session_frame_index): (
            evidence,
            _ground_truth_semantic_hash(original_ground),
        )
    }
    with pytest.raises(ValueError, match="immutable annotation"):
        _replay_sessions(
            (replace(manifest, frames=(frame,)),),
            {
                manifest.session_id: SimpleNamespace(
                    kind="final_e2e_test", source_policy="lossless"
                )
            },
            Predictor(),
            ForgedLoader(),
            annotation,
            parser_hash="9" * 64,
            detector_hash="2" * 64,
        )


def test_annotation_payload_must_match_descriptor_and_session_manifest(
    tmp_path: Path,
) -> None:
    capture_root, split = _capture_fixture(tmp_path)
    store = ArtifactStore(tmp_path / "annotation-artifacts")
    split_ref = store.put(
        logical_id="capture/split.json",
        source_path=split.manifest_path,
        media_type="application/json",
    )
    refs = [split_ref]
    bindings = {}
    manifests = {}
    for units in split.splits.values():
        for unit in units:
            session_id = unit.session_id
            session_root = capture_root / "capture_sessions" / session_id
            annotation_path = session_root / "annotations.jsonl"
            annotation_path.write_text(
                json.dumps({
                    "schema_version": "perception_formal_annotation.v1",
                    "session_id": session_id,
                    "frame_index": 0,
                    "raw_card_ids": ["whip"],
                    "raw_inventory": ["whip"],
                    "ground_truth_semantic_hash": "1" * 64,
                }, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            annotation_ref = store.put(
                logical_id=f"capture/{session_id}/annotations.jsonl",
                source_path=annotation_path,
                media_type="application/x-ndjson",
            )
            manifest_ref = store.put(
                logical_id=f"capture/{session_id}/session_manifest.json",
                source_path=session_root / "session_manifest.json",
                media_type="application/json",
            )
            refs.extend((annotation_ref, manifest_ref))
            bindings[session_id] = {
                "annotation_logical_id": annotation_ref.logical_id,
                "annotation_sha256": annotation_ref.sha256,
                "session_manifest_logical_id": manifest_ref.logical_id,
                "session_manifest_sha256": manifest_ref.sha256,
            }
            manifests[session_id] = DatasetWriter.restore(
                capture_root, session_id, metadata_only=True
            )
    descriptor = ArtifactDescriptor(
        logical_id="capture/formal-source",
        node_kind="source_descriptor",
        producer_id="test",
        producer_version="v1",
        identity_metadata={
            "manifest_logical_id": split_ref.logical_id,
            "annotation_bindings": bindings,
        },
        parents=(),
        files=tuple(refs),
    )
    restored = {
        "capture_dataset": RestoredFormalArtifact(
            descriptor,
            descriptor,
            {
                ref.logical_id: store.object_path(ref.store_uri).read_bytes()
                for ref in refs
            },
            None,
        )
    }
    validated = validate_split(split, manifests)
    request = FormalBenchmarkRequest(
        store=store,
        capture_store_root=capture_root,
        dependency_descriptors={"capture_dataset": descriptor},
    )
    selected = {"cal-0"}
    payloads = _verified_annotation_payloads(
        request, restored, validated, selected
    )
    assert set(_annotation_evidence_index(payloads)) == {("cal-0", 0)}

    annotation_path = capture_root / "capture_sessions/cal-0/annotations.jsonl"
    annotation_path.write_bytes(annotation_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="immutable descriptor"):
        _verified_annotation_payloads(request, restored, validated, selected)


def test_subject_hashes_use_current_fit_file_hash(monkeypatch, tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "subject-artifacts")
    capture_ref = _package_ref(store, "capture.json", {"capture": True})
    target_ref = _package_ref(store, "target.json", {"target": True})

    def descriptor(logical_id, ref):
        return ArtifactDescriptor(
            logical_id=logical_id,
            node_kind="source_descriptor",
            producer_id="test",
            producer_version="v1",
            identity_metadata={"manifest_logical_id": ref.logical_id},
            parents=(),
            files=(ref,),
        )

    request = FormalBenchmarkRequest(
        store=store,
        capture_store_root=tmp_path,
        dependency_descriptors={
            "capture_dataset": descriptor("capture", capture_ref),
            "target_config": descriptor("target", target_ref),
        },
    )
    payloads = {
        "parser_package": {"parser_artifact_hash": "1" * 64},
        "detector_package": {
            "detector_artifact_hash": "2" * 64,
            "model_hash": "3" * 64,
            "build_hash": "4" * 64,
        },
        "assembler_config": {
            "assembler_schema_hash": "5" * 64,
            "ui_presentation_schema_hash": "6" * 64,
            "ui_presentation_golden_fixture_hash": "7" * 64,
            "atlas_vocabulary_hash": "8" * 64,
            "assembler_impl_hash": "9" * 64,
            "roi_resolver_input_hash": "a" * 64,
        },
        "target_config": {"threshold_hash": "b" * 64},
    }
    monkeypatch.setattr(
        "benchmark_survivors_perception._current_fit_code_hash",
        lambda: "f" * 64,
    )
    subjects = _subject_hashes(
        request,
        payloads,
        calibration_profile_hash="c" * 64,
        lineage_seal_hash="d" * 64,
    )
    assert subjects["benchmark_fit_code_hash"] == "f" * 64
    assert set(subjects) == {
        "parser_artifact_hash", "detector_artifact_hash", "model_hash", "build_hash",
        "assembler_schema_hash", "ui_presentation_schema_hash",
        "ui_presentation_golden_fixture_hash", "config_hash", "capture_dataset_hash",
        "calibration_profile_hash", "threshold_hash", "atlas_vocabulary_hash",
        "assembler_impl_hash", "roi_resolver_input_hash", "benchmark_fit_code_hash",
        "lineage_seal_hash",
    }


def test_runner_rejects_prediction_with_unbound_detector_hash(
    tmp_path: Path, monkeypatch
) -> None:
    """restored detector package と不一致の prediction hash を拒否する。

    formal provider が返す hash を package envelope と照合し、別 artifact の値が
    prediction snapshot へ混入したときに benchmark の replay 前に fail-closed にする。
    """
    def _malicious_factory(restored):
        return _mutating_test_predictor(
            restored,
            lambda snapshot: replace(snapshot, detector_artifact_hash="b" * 64),
        )

    import benchmark_survivors_perception as _bsp
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _malicious_factory)
    _patch_minimal_test_gates(monkeypatch)
    request = _request(tmp_path, formal_inputs=True)
    with pytest.raises(ValueError, match="prediction detector hash is not bound"):
        run_formal_pipeline(request)


def test_canonical_logical_ids_not_set_when_calibration_commit_fails(tmp_path: Path, monkeypatch) -> None:
    """calibration batch commit 失敗時に canonical logical_id が設定されないことを確認する。

    staging → batch commit → seal → canonical の順序で publish するため、
    batch commit が失敗しても consumer から見える canonical logical_id が残りません。
    calibration alias は calibration commit marker を永続化した後だけ公開されます。
    """
    from unittest.mock import patch

    from Tools.Artifacts.artifact_store import ArtifactStoreError

    import benchmark_survivors_perception as _bsp
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _test_provider_factory)
    _patch_minimal_test_gates(monkeypatch)
    request = _request(tmp_path, formal_inputs=True)
    canonical_cal_prefix = request.calibration_namespace + "/"
    canonical_ver_prefix = request.verdict_namespace + "/"

    original_put_bytes = request.store.put_bytes

    def _fail_on_batch_commit(*, logical_id, data, media_type):
        if logical_id.startswith("perception/calibration_commit/"):
            raise ArtifactStoreError("simulated calibration commit failure")
        return original_put_bytes(logical_id=logical_id, data=data, media_type=media_type)

    with patch.object(request.store, "put_bytes", side_effect=_fail_on_batch_commit):
        with pytest.raises(ArtifactStoreError, match="simulated calibration commit failure"):
            run_formal_pipeline(request)

    # canonical logical_id の index が存在しないことを確認する。
    # ArtifactStore は logical_id を logical_root 以下のパスへ記録する。
    committed_ids = _store_logical_ids(request.store)
    assert not any(
        item.startswith(canonical_cal_prefix) and item.endswith("/profile.json")
        for item in committed_ids
    ), "calibration canonical logical_id must not be set before batch commit"
    assert not any(
        item.startswith(canonical_ver_prefix) and item.endswith("/verdict.json")
        for item in committed_ids
    ), "verdict canonical logical_id must not be set before batch commit"


def test_final_sessions_are_reserved_before_provider_and_remain_consumed(tmp_path: Path, monkeypatch) -> None:
    """final 集合を provider/publish 前に予約し、後続失敗でも消費済みにする。"""
    import benchmark_survivors_perception as _bsp

    _patch_minimal_test_gates(monkeypatch)
    request = _request(tmp_path, formal_inputs=True)

    def _failing_factory(restored):
        predictor = _FixturePredictor(restored)
        predict = predictor.predict_frame

        def fail_during_final(frame, session_id: str, frame_index: int):
            """予約済み final session の prediction を失敗させる。

            provider 失敗後も session lease が消費済みであることを確認する。
            """
            if session_id.startswith("final-"):
                for index in range(3):
                    assert request.store._logical_index_path(
                        f"perception/lineage/opened/final-{index}.json"
                    ).exists()
                assert not any(
                    item.startswith(request.verdict_namespace + "/")
                    and item.endswith("/verdict.json")
                    for item in _store_logical_ids(request.store)
                )
                raise RuntimeError("provider failed after reservation")
            return predict(frame, session_id, frame_index)

        predictor.predict_frame = fail_during_final
        return predictor

    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _failing_factory)
    with pytest.raises(RuntimeError, match="provider failed after reservation"):
        run_formal_pipeline(request)
    monkeypatch.setattr(_bsp, "_CLI_PROVIDER_FACTORY", _test_provider_factory)
    with pytest.raises(FinalSessionAlreadyOpenedError):
        run_formal_pipeline(request)


def _make_tick(session_id: str, frame_idx: int, screen_state: str) -> SnapshotReplayTick:
    """指定 screen_state の最小 SnapshotReplayTick を生成するヘルパー（回帰テスト用）。"""
    schema = DeployObsSchema.default_v2()
    assembler = RealObsAssembler()
    hud, world = _hud_world(session_id, frame_idx, frame_idx * 1_000_000_000, screen_state)
    snapshot = assembler.assemble(hud, world, schema, (1920, 1080))
    assert snapshot is not None
    return SnapshotReplayTick(
        session_id, "final_e2e_test", "lossless",
        snapshot.frame_id, snapshot, snapshot, 0.0,
    )


def test_event_rising_edge_counts_two_occurrences_not_four() -> None:
    """event:level_up は rising-edge (False→True) のみ計上し、継続 True は重複しない。

    gameplay→level_up→gameplay→level_up→gameplay の 2 occurrence 入力に対して
    count=4 でなく count=2 になることを確認する（P1-1 regression）。
    """
    states = ["gameplay", "level_up_items", "gameplay", "level_up_items", "gameplay"]
    ticks = [_make_tick("sess", i, s) for i, s in enumerate(states)]
    report = run_benchmark(ticks)
    assert report.slice_counts.get("event:level_up", 0) == 2


def test_coord_quantization_uses_half_dimension_for_normalized_domain() -> None:
    """coord_quantization_px は [-1,1] domain なので 1 unit = dimension/2 px で変換する。

    正規化差 0.02、幅 1920 の場合、19.2px（38.4px ではない）になることを確認する（P1-4 regression）。
    """
    schema = DeployObsSchema.default_v2()
    assembler = RealObsAssembler()
    width, height = 1920, 1080
    session_id = "coord-test"
    hud_g, world = _hud_world(session_id, 0, 1_000_000_000, "gameplay")
    ground = assembler.assemble(hud_g, world, schema, (width, height))
    assert ground is not None
    ps_off, ps_sz = schema.layout["player_screen_pos"]
    # x 方向に 0.02 の正規化差を注入する（[−1,1] domain → 1920/2 * 0.02 = 19.2 px 期待値）。
    pred_values = np.array(ground.deploy_obs.values, dtype=float)
    pred_values[ps_off] += 0.02
    pred_deploy = replace(ground.deploy_obs, values=pred_values)
    predicted = replace(ground, deploy_obs=pred_deploy)
    tick = SnapshotReplayTick(session_id, "error_calibration", "lossless", ground.frame_id,
                              ground, predicted, 0.0)
    residuals = _derive_calibration_residuals([tick], resolution_wh=(width, height))
    quant_residuals = [r for r in residuals if r.field == "coord_quantization_px"]
    assert quant_residuals, "coord_quantization_px residual が生成されていない"
    assert any(abs(r.residual - 19.2) < 0.5 for r in quant_residuals), (
        f"expected ~19.2 px but got {[r.residual for r in quant_residuals]}"
    )


def test_invalid_predicted_validity_excludes_coord_residual() -> None:
    """predicted validity=0 の座標は coord residual を生成しない（P1-5 regression）。

    invalid な predicted が noise sample として混入しないことを確認する。
    """
    schema = DeployObsSchema.default_v2()
    assembler = RealObsAssembler()
    width, height = 1920, 1080
    session_id = "validity-test"
    hud_g, world = _hud_world(session_id, 0, 1_000_000_000, "gameplay")
    ground = assembler.assemble(hud_g, world, schema, (width, height))
    assert ground is not None
    # predicted validity を全 coord segment で 0 にして invalid にする。
    pred_validity = np.zeros_like(ground.deploy_obs.validity)
    pred_deploy = replace(ground.deploy_obs, validity=pred_validity)
    predicted = replace(ground, deploy_obs=pred_deploy)
    tick = SnapshotReplayTick(session_id, "error_calibration", "lossless", ground.frame_id,
                              ground, predicted, 0.0)
    residuals = _derive_calibration_residuals([tick], resolution_wh=(width, height))
    coord_residuals = [
        r for r in residuals
        if r.field in {"coord_noise", "coord_quantization_px"}
        and r.session_id == session_id and r.frame_id == ground.frame_id
    ]
    assert coord_residuals == [], (
        f"invalid predicted validity でも {len(coord_residuals)} 件の coord residual が生成された"
    )


def test_predicted_none_skips_all_continuous_residuals() -> None:
    """predicted=None のフレームは coord/item_category/enemy_category/hp/xp/timer/inventory を生成しない。

    欠測フレームが ground truth を prediction として代入したゼロ残差で calibration を
    汚染しないことを確認する（P1 #2 regression）。
    """
    schema = DeployObsSchema.default_v2()
    assembler = RealObsAssembler()
    width, height = 1920, 1080
    session_id = "none-predicted-test"
    hud_g, world = _hud_world(session_id, 0, 1_000_000_000, "gameplay")
    ground = assembler.assemble(hud_g, world, schema, (width, height))
    assert ground is not None
    # predicted=None: detector/parser が出力できなかったフレームを模擬する。
    tick = SnapshotReplayTick(
        session_id, "error_calibration", "lossless",
        ground.frame_id, ground, None, 0.0,
    )
    residuals = _derive_calibration_residuals([tick], resolution_wh=(width, height))
    continuous_fields = {
        "coord_noise", "coord_quantization_px",
        "hp_ratio", "xp_ratio", "timer_seconds", "inventory_hash",
        "item_category", "enemy_category",
    }
    bad = [r for r in residuals if r.field in continuous_fields]
    assert bad == [], (
        f"predicted=None なのに {len(bad)} 件の連続 residual が生成された: "
        f"{[r.field for r in bad]}"
    )


def test_age_frames_propagated_from_ui_state_age() -> None:
    """hp_ratio/xp_ratio/timer の age_frames は predicted の ui_state_age から変換する。

    age_frames が常に 0 に固定されると fit_error_profile の confidence weighting が
    完全に無効になる（P1 #3 regression）。item_context を合成して直接注入する。
    """
    from reinbalance_survivors_contracts.item_decision import CandidateFeatures, ItemDecisionFeatures
    schema = DeployObsSchema.default_v2()
    assembler = RealObsAssembler()
    width, height = 1920, 1080
    session_id = "age-test"
    from benchmark_survivors_perception import _CONSUMER_FRAME_PERIOD_MS
    frame_period_ms = _CONSUMER_FRAME_PERIOD_MS
    age_ms = 4.0 * frame_period_ms  # 4 フレーム分の遅延
    hud_g, world = _hud_world(session_id, 0, 1_000_000_000, "gameplay")
    ground = assembler.assemble(hud_g, world, schema, (width, height))
    assert ground is not None
    # ItemDecisionFeatures を直接構築して ui_state_age を制御する。
    # CandidateFeatures は padding 1 枚で最小有効な choice_count=0 にする。
    real_cand = CandidateFeatures(
        kind="weapon", item_id="whip", new_level=1, owned=False,
        is_new=True, is_evolve=False, is_union=False, has_prerequisite=False, slot_capacity=0,
    )
    padding_cand = CandidateFeatures(
        kind="padding", item_id="__padding__", new_level=0, owned=False,
        is_new=False, is_evolve=False, is_union=False, has_prerequisite=False, slot_capacity=0,
    )
    synthetic_context = ItemDecisionFeatures(
        decision_id="age-test-decision",
        feature_schema="context_only_v1",
        elapsed_time=60.0, level=2, hp_ratio=1.0, xp_ratio=0.5,
        weapon_slots=(1, 0, 0, 0, 0, 0), passive_slots=(0, 0, 0, 0, 0, 0),
        empty_slot_count=11, evolution_readiness=0.0, choice_count=1, card_mask=(True, False),
        fallback_kind="none",
        ui_state_validity=0.7,
        ui_state_age=age_ms / 1000.0,
        candidates=(real_cand,), max_item_cards=2,
    )
    # ground も item_context を持つ必要がある（guard: predicted_context is not None and ground_context is not None）。
    # age は predicted の ui_state_age から取るので ground の値は 0 にする。
    ground_context = replace(synthetic_context, ui_state_age=0.0, ui_state_validity=1.0, decision_id="age-test-ground")
    ground_with_ctx = replace(ground, item_context=ground_context)
    predicted = replace(ground_with_ctx, item_context=synthetic_context)
    tick = SnapshotReplayTick(
        session_id, "error_calibration", "lossless",
        ground.frame_id, ground_with_ctx, predicted, 0.0,
    )
    residuals = _derive_calibration_residuals([tick], resolution_wh=(width, height))
    hp_residuals = [r for r in residuals if r.field == "hp_ratio"]
    expected_age = round(age_ms / frame_period_ms)
    assert hp_residuals, "hp_ratio residual が生成されていない"
    assert all(r.age_frames == expected_age for r in hp_residuals), (
        f"expected age_frames={expected_age}, got {[r.age_frames for r in hp_residuals]}"
    )
    assert all(r.confidence == pytest.approx(0.7) for r in hp_residuals), (
        f"expected confidence=0.7, got {[r.confidence for r in hp_residuals]}"
    )


def test_coord_age_comes_from_deploy_obs_age_not_item_context() -> None:
    """coord residual の age_frames は deploy_obs.age から取得する（P1-4 regression）。

    item_context が None（ゲームプレイ中の標準状態）でも age_frames は 0 にならず、
    deploy_obs.age 平面から正しく変換されることを確認する。
    """
    schema = DeployObsSchema.default_v2()
    assembler = RealObsAssembler()
    width, height = 1920, 1080
    session_id = "coord-age-test"
    from benchmark_survivors_perception import _CONSUMER_FRAME_PERIOD_MS
    frame_period_ms = _CONSUMER_FRAME_PERIOD_MS
    # player_screen_pos の max_age_ms = 500.0（schema.default_v1 による）
    ps_max_age_ms = 500.0
    # age = 0.5 → 0.5 * 500 / frame_period_ms フレーム分の遅延
    age_normalized = 0.5
    expected_age = round(age_normalized * ps_max_age_ms / frame_period_ms)
    hud_g, world = _hud_world(session_id, 0, 1_000_000_000, "gameplay")
    ground = assembler.assemble(hud_g, world, schema, (width, height))
    assert ground is not None
    ps_off, ps_sz = schema.layout["player_screen_pos"]
    ne_off, ne_sz = schema.layout["nearest_enemy_offset"]
    # predicted の coord segment age 平面を両方とも age_normalized に設定する。
    # 値は ground と同一（residual=0）にして coord_noise residual のみを取得する。
    pred_age = np.array(ground.deploy_obs.age, dtype=float)
    pred_age[ps_off:ps_off + ps_sz] = age_normalized
    pred_age[ne_off:ne_off + ne_sz] = age_normalized
    # validity は有効値（>0）にする。
    pred_validity = np.array(ground.deploy_obs.validity, dtype=float)
    pred_validity[ps_off:ps_off + ps_sz] = 1.0
    pred_validity[ne_off:ne_off + ne_sz] = 1.0
    pred_deploy = replace(ground.deploy_obs, age=pred_age, validity=pred_validity)
    # item_context は None のまま（ゲームプレイ中の標準状態）。
    predicted = replace(ground, deploy_obs=pred_deploy, item_context=None)
    # ground の validity も有効にする。
    gnd_validity = np.array(ground.deploy_obs.validity, dtype=float)
    gnd_validity[ps_off:ps_off + ps_sz] = 1.0
    gnd_validity[ne_off:ne_off + ne_sz] = 1.0
    gnd_deploy = replace(ground.deploy_obs, validity=gnd_validity)
    ground_with_valid = replace(ground, deploy_obs=gnd_deploy, item_context=None)
    tick = SnapshotReplayTick(
        session_id, "error_calibration", "lossless",
        ground.frame_id, ground_with_valid, predicted, 0.0,
    )
    residuals = _derive_calibration_residuals([tick], resolution_wh=(width, height))
    coord_residuals = [
        r for r in residuals
        if r.field in {"coord_noise", "coord_quantization_px"}
    ]
    assert coord_residuals, "coord residual が生成されていない"
    assert all(r.age_frames == expected_age for r in coord_residuals), (
        f"expected age_frames={expected_age} (from deploy_obs.age), "
        f"got {[r.age_frames for r in coord_residuals]}. "
        "item_context=None でも age_frames は 0 であってはならない。"
    )

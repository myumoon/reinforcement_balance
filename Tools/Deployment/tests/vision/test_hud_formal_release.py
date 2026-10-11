"""formal HUD tooling の synthetic regression test。

実画面を使わず、atlas の formal entry と crop 不足の fail-closed を確認します。
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import survivors.vision.hud_formal_release as formal_release
from survivors.vision.hud_formal_release import FormalAtlasError, FormalValidationReport, build_formal_atlas
from survivors.vision.hud_formal_release import PreflightReport, run_formal_preflight, validate_formal
from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor, RestoreTestVerdict
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from survivors.capture.captured_frame import CapturedFrame
from survivors.capture_dataset import DatasetWriter, SPLIT_NAMES, SplitFreezer
from survivors.vision.hud_formal_release import FormalPreflightError
from survivors.vision.hud_truth import expected_roi_for_state
from calibrate_survivors_hud_parser import main as calibration_cli_main
from survivors.vision.icon_matcher import AtlasManifest, IconMatcher, TemplateEntry
from survivors.vision.hud_calibration import (
    FormalPackageMixingError,
    PackageHashMismatch,
    PackageLoader,
    PackageWriter,
    fit,
)


def _frame_record(frame: np.ndarray, session_path: Path):
    """合成 BGRA frame を hash 付き PNG record にする。

    実 loader に実ファイルと object SHA を渡し、atlas builder の遅延 decode を通します。
    """
    ok, encoded = cv2.imencode(".png", frame)
    assert ok
    payload = encoded.tobytes()
    (session_path / "frame.png").write_bytes(payload)
    return SimpleNamespace(
        frame_id=0,
        object_path="frame.png",
        object_sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_formal_atlas_is_loadable_and_binds_real_crops(tmp_path: Path, monkeypatch) -> None:
    """inventory と card crop から作った atlas が formal loader で照合できる。

    surface ごとの crop 数と遅延 decode を記録し、未知 item は拒否します。
    """
    frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
    frame[..., :3] = (20, 80, 220)
    frame[..., 3] = 255
    for index in range(3, 6):
        x = 100 + 46 * index
        frame[86:128, x:x + 42, :3] = (220, 40, 10)
    session_path = tmp_path / "session"
    session_path.mkdir()
    frame_record = _frame_record(frame, session_path)
    frame_records = tuple(
        SimpleNamespace(
            frame_id=index,
            object_path=frame_record.object_path,
            object_sha256=frame_record.object_sha256,
        )
        for index in range(3)
    )
    manifest = SimpleNamespace(session_id="train-a", session_path=session_path, frame_records=frame_records)
    monkeypatch.setattr(formal_release, "_detect_card_rows", lambda _frame: [0])
    monkeypatch.setattr(formal_release, "_card_count", lambda rows: len(rows))
    truth = [
        {
            "session_id": "train-a",
            "frame_id": index,
            "confirmed": True,
            "expected_screen_state": "level_up_items",
            "expected_items": ["whip"] * 9 + ["empty_slot"] * 3,
            "expected_choice": ["whip"],
        }
        for index in range(3)
    ]
    vocabulary = tmp_path / "vocabulary.yaml"
    vocabulary.write_text("schema_version: test\nweapons:\n  whip: Whip\npassives: {}\nnon_items: []\nempty_slot: empty_slot\n", encoding="utf-8")
    atlas_path = tmp_path / "atlas.json"
    decode_calls = 0
    original_loader = formal_release.load_frame_pixels

    def counted_loader(path, frame_record):
        nonlocal decode_calls
        decode_calls += 1
        return original_loader(path, frame_record)

    monkeypatch.setattr(formal_release, "load_frame_pixels", counted_loader)

    atlas, provenance = build_formal_atlas(
        train_frames_by_session={"train-a": manifest},
        train_truth=truth,
        vocabulary_path=vocabulary,
        profile_hash="a" * 64,
        build_hash="b" * 64,
        min_crops=3,
        output_path=atlas_path,
    )

    matcher = IconMatcher.load_formal(atlas_path)
    assert matcher.match(frame[40:82, 100:142]).item_id == "whip"
    assert matcher.match(formal_release.card_icon_roi(0).crop(frame), surface="card").item_id == "whip"
    assert atlas.formal_parser_eligible is True
    assert provenance["entries"]["whip@inventory"]["crop_count"] == 27
    assert provenance["entries"]["whip@card"]["crop_count"] == 3
    assert decode_calls == 3

    repeated_atlas, repeated_provenance = build_formal_atlas(
        train_frames_by_session={"train-a": manifest},
        train_truth=truth,
        vocabulary_path=vocabulary,
        profile_hash="a" * 64,
        build_hash="b" * 64,
        min_crops=3,
    )
    assert repeated_atlas.atlas_content_hash == atlas.atlas_content_hash
    assert repeated_provenance == provenance
    with pytest.raises(FormalAtlasError, match="need 31"):
        build_formal_atlas(
            train_frames_by_session={"train-a": manifest},
            train_truth=truth,
            vocabulary_path=vocabulary,
            profile_hash="a" * 64,
            build_hash="b" * 64,
            min_crops=31,
        )

    with pytest.raises(FormalAtlasError):
        build_formal_atlas(
            train_frames_by_session={"train-a": manifest},
            train_truth=[{**truth[0], "expected_items": ["missing"] * 12}],
            vocabulary_path=vocabulary,
            profile_hash="a" * 64,
            build_hash="b" * 64,
            min_crops=3,
        )


def test_formal_atlas_rejects_frame_object_hash_mismatch(tmp_path: Path) -> None:
    """frame PNG の object sha mismatch は crop 前に拒否する。

壊れた dataset pixel が formal template に紛れ込まないことを確認します。
    """
    frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
    frame[..., 3] = 255
    session_path = tmp_path / "bad-session"
    session_path.mkdir()
    real_record = _frame_record(frame, session_path)
    bad_record = SimpleNamespace(
        frame_id=real_record.frame_id,
        object_path=real_record.object_path,
        object_sha256="0" * 64,
    )
    manifest = SimpleNamespace(session_id="train-bad", session_path=session_path, frame_records=(bad_record,))
    vocabulary = tmp_path / "vocabulary.yaml"
    vocabulary.write_text("schema_version: test\nweapons:\n  whip: Whip\npassives: {}\nnon_items: []\nempty_slot: empty_slot\n", encoding="utf-8")

    with pytest.raises(ValueError, match="object hash mismatch"):
        build_formal_atlas(
            train_frames_by_session={"train-bad": manifest},
            train_truth=[{
                "session_id": "train-bad",
                "frame_id": 0,
                "confirmed": True,
                "expected_items": ["whip"] * 12,
            }],
            vocabulary_path=vocabulary,
            profile_hash="a" * 64,
            build_hash="b" * 64,
            min_crops=3,
        )


def test_validation_prediction_keeps_truth_separate_and_resets_sessions(tmp_path: Path, monkeypatch) -> None:
    """parser prediction は HUD truth の正解 field を上書きしない。

    session ごとの temporal reset と frame 順も fake parser の呼び出しで確認します。
    """
    feature = np.zeros(88, dtype=np.float32)
    atlas_hash = hashlib.sha256(feature.tobytes()).hexdigest()
    manifest = AtlasManifest(
        "icon_atlas.v1", "a" * 64, "b" * 64, False, True, atlas_hash,
        (TemplateEntry("predicted-item", "weapon", 1, 1, feature),),
    )
    atlas_path = tmp_path / "prediction-atlas.json"
    atlas_path.write_bytes(formal_release.serialize_manifest(manifest))
    atlas = IconMatcher.load_formal(atlas_path)
    frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
    monkeypatch.setattr(formal_release, "load_frame_pixels", lambda _path, _record: frame)
    parser_instances = []

    class FakeHudParser:
        def __init__(self, *, parser_artifact_hash: str, icon_matcher: IconMatcher) -> None:
            self.reset_calls = 0
            self.parsed_indexes = []
            parser_instances.append(self)

        def reset_temporal_state(self) -> None:
            self.reset_calls += 1

        def parse(self, _frame, *, session_id, frame_index, captured_monotonic_ns):
            self.parsed_indexes.append(frame_index)
            return SimpleNamespace(
                timer_seconds=10.0 + frame_index,
                level=2,
                hp_ratio=0.75,
                xp_ratio=0.25,
                inventory=("predicted-item",) * 12,
                cards=(SimpleNamespace(item_id="predicted-card"),),
                screen_state="gameplay",
            )

    monkeypatch.setattr(formal_release, "HudParser", FakeHudParser)
    sessions = {
        session_id: SimpleNamespace(
            session_path=tmp_path / session_id,
            frame_records=tuple(
                SimpleNamespace(frame_id=index, captured_monotonic_ns=index + 1)
                for index in range(2)
            ),
        )
        for session_id in ("validation-a", "validation-b")
    }
    truth = [
        {
            "session_id": session_id,
            "frame_id": index,
            "confirmed": True,
            "expected_timer_seconds": 999.0,
            "expected_level": 99,
            "expected_hp_ratio": 0.1,
            "expected_xp_ratio": 0.9,
            "expected_items": ["truth-item"] * 12,
            "expected_choice": ["truth-card"],
            "expected_screen_state": "unknown",
            "expected_roi": None,
        }
        for session_id in ("validation-b", "validation-a")
        for index in (1, 0)
    ]

    predictions = formal_release.predict_validation_records(
        validation_frames_by_session=sessions,
        validation_truth=truth,
        atlas=atlas,
    )

    assert [record["frame_id"] for record in predictions] == [0, 1, 0, 1]
    assert all(record["expected_timer_seconds"] == 999.0 for record in predictions)
    assert all(record["predicted_timer_seconds"] == 10.0 + record["frame_id"] for record in predictions)
    assert all(record["predicted_items"] == ["predicted-item"] * 12 for record in predictions)
    assert all(record["predicted_choice"] == "predicted-card" for record in predictions)
    assert [parser.reset_calls for parser in parser_instances] == [1, 1]
    assert [parser.parsed_indexes for parser in parser_instances] == [[0, 1], [0, 1]]


def test_formal_preflight_rejects_non_source_descriptor(tmp_path: Path) -> None:
    """development producer node は formal dataset として開かない。

preflight の入口で node kind を拒否し、capture path を読み始めないことを確認します。
    """
    from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor

    descriptor = ArtifactDescriptor(
        logical_id="test/development",
        node_kind="perception_calibration_profile",
        producer_id="test",
        producer_version="1",
    )
    with pytest.raises(FormalPreflightError, match="source_descriptor"):
        run_formal_preflight(
            capture_store_root=tmp_path,
            dataset_descriptor=descriptor,
            restore_verdict_descriptor=descriptor,
            target_profile_hash="a" * 64,
            minimum_counts={},
        )


@pytest.mark.parametrize(
    ("screen_state", "items", "confirmed", "frame_exists", "sealed", "binding_mismatch", "message"),
    [
        ("unlisted_state", None, True, True, False, False, "unknown screen state"),
        ("gameplay", ["whip"] * 11 + [None], True, True, False, False, "confirmed item slots contain null"),
        ("gameplay", None, False, True, False, False, "session has no confirmed HUD truth"),
        ("gameplay", None, True, False, False, False, "HUD truth frame does not exist"),
        ("gameplay", None, True, True, True, False, "references a sealed split session"),
        ("gameplay", None, True, True, False, True, "descriptor file binding mismatch"),
    ],
)
def test_confirmed_truth_rejects_unknown_state_and_partial_slots(
    tmp_path: Path,
    monkeypatch,
    screen_state: str,
    items: list[str | None] | None,
    confirmed: bool,
    frame_exists: bool,
    sealed: bool,
    binding_mismatch: bool,
    message: str,
) -> None:
    """confirmed truth は未知 state と slot 単位 null を受理しない。

    bound file の sha256 が一致していても schema と item gate の境界で止めます。
    """
    session_id = "train-a"
    capture_root = tmp_path / "capture"
    session_path = capture_root / "capture_sessions" / session_id
    session_path.mkdir(parents=True)
    truth_path = session_path / "hud_truth.jsonl"
    truth_path.write_bytes(b"synthetic truth binding\n")
    store = ArtifactStore(tmp_path / "truth-store")
    truth_ref = store.put(
        logical_id="dataset/train-a/hud_truth.jsonl",
        source_path=truth_path,
        media_type="application/jsonl",
    )
    dataset = ArtifactDescriptor(
        logical_id="dataset/source",
        node_kind="source_descriptor",
        producer_id="hud-test",
        producer_version="1",
        files=(truth_ref,),
    )
    truth = SimpleNamespace(
        schema_version="hud_truth.v1",
        session_id=session_id,
        frame_id=0,
        annotator_id="annotator",
        confirmed=confirmed,
        expected_screen_state=screen_state,
        expected_timer_seconds=None,
        expected_level=None,
        expected_hp_ratio=None,
        expected_xp_ratio=None,
        expected_items=items,
        expected_choice=None,
        roi_name="hud",
        expected_roi=expected_roi_for_state(screen_state) if screen_state == "gameplay" else None,
        confirmed_at="2026-10-10T00:00:00Z",
        draft_source={"parser_artifact_hash": "parser", "atlas_content_hash": "atlas"},
    )
    monkeypatch.setattr(formal_release, "read_hud_truth", lambda _path: [truth])

    with pytest.raises(FormalPreflightError, match=message):
        formal_release._confirmed_truth(
            capture_root,
            dataset,
            "model_train",
            session_id,
            {session_id: {
                "hud_truth_logical_id": truth_ref.logical_id,
                "hud_truth_sha256": "0" * 64 if binding_mismatch else truth_ref.sha256,
            }},
            {0} if frame_exists else set(),
            {session_id} if sealed else set(),
        )


def test_formal_preflight_accepts_bound_frozen_model_sessions(tmp_path: Path) -> None:
    """model train/validation の全 binding が一致した formal source を受け入れる。

split manifest、session、frame metadata、annotations、HUD truth と restore verdict を実 hash で束縛します。
    """
    from survivors.vision.hud_formal_release import run_formal_preflight

    profile_hash, build_id = "a" * 64, "b" * 64
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    all_sessions = [f"train-{i}" for i in range(3)] + [f"validation-{i}" for i in range(3)]
    truth_by_state = ("gameplay", "level_up_items", "level_up_fallback", "unknown")
    for session_id in all_sessions:
        writer = DatasetWriter(capture_root, session_id, profile_hash, build_id)
        pixels = np.zeros((1080, 1920, 4), dtype=np.uint8)
        pixels[..., 3] = 255
        for frame_id in range(4):
            writer.write_frame(CapturedFrame(
                frame_bgra=pixels,
                captured_monotonic_ns=frame_id + 1,
                session_frame_index=frame_id,
                client_rect_screen_px=(0, 0, 1920, 1080),
                foreground=True,
                target_profile_hash=profile_hash,
                game_build_id=build_id,
            ))
        writer.publish("synthetic-test")
        session_path = capture_root / "capture_sessions" / session_id
        manifest_path = session_path / "session_manifest.json"
        manifest_wire = json.loads(manifest_path.read_bytes())
        manifest_wire["formal_dataset_eligible"] = True
        manifest_path.write_bytes(json.dumps(manifest_wire, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        (session_path / "annotations.jsonl").write_bytes(b"")
        truth_lines = []
        for frame_id, state in enumerate(truth_by_state):
            truth = {
                "schema_version": "hud_truth.v1",
                "session_id": session_id,
                "frame_id": frame_id,
                "annotator_id": "test-annotator",
                "confirmed": True,
                "expected_screen_state": state,
                "expected_timer_seconds": 10.0,
                "expected_level": 1,
                "expected_hp_ratio": 0.5,
                "expected_xp_ratio": 0.25,
                "expected_items": ["empty_slot"] * 12,
                "expected_slot_levels": [None] * 12,
                "expected_choice": ["whip"] if state in {"level_up_items", "level_up_fallback"} else None,
                "expected_buttons": ["reroll"] if state == "level_up_items" else [],
                "roi_name": "hud",
                "expected_roi": expected_roi_for_state(state),
                "confirmed_at": "2026-10-10T00:00:00Z",
                "draft_source": {"parser_artifact_hash": "parser", "atlas_content_hash": "atlas"},
            }
            truth_lines.append(json.dumps(truth, sort_keys=True, separators=(",", ":")))
        (session_path / "hud_truth.jsonl").write_text("\n".join(truth_lines) + "\n", encoding="utf-8")

    freezer = SplitFreezer(capture_root)
    freezer.assign("model_train", all_sessions[:3])
    freezer.assign("model_validation", all_sessions[3:])
    split = freezer.freeze()
    artifact_store = ArtifactStore(tmp_path / "artifact-store")
    refs = [artifact_store.put(
        logical_id="dataset/split-manifest.json",
        source_path=split.manifest_path,
        media_type="application/json",
    )]
    model_bindings = {}
    frames_bindings = {}
    truth_bindings = {}
    for session_id in all_sessions:
        session_path = capture_root / "capture_sessions" / session_id
        session_manifest = session_path / "session_manifest.json"
        annotations = session_path / "annotations.jsonl"
        frames = session_path / "frames.jsonl"
        truth = session_path / "hud_truth.jsonl"
        refs.extend([
            artifact_store.put(logical_id=f"dataset/{session_id}/session_manifest.json", source_path=session_manifest, media_type="application/json"),
            artifact_store.put(logical_id=f"dataset/{session_id}/annotations.jsonl", source_path=annotations, media_type="application/json"),
            artifact_store.put(logical_id=f"dataset/{session_id}/frames.jsonl", source_path=frames, media_type="application/json"),
            artifact_store.put(logical_id=f"dataset/{session_id}/hud_truth.jsonl", source_path=truth, media_type="application/json"),
        ])
        hashes = {ref.logical_id.split("/")[-1]: ref for ref in refs if ref.logical_id.startswith(f"dataset/{session_id}/")}
        model_bindings[session_id] = {
            "session_manifest_logical_id": hashes["session_manifest.json"].logical_id,
            "session_manifest_sha256": hashes["session_manifest.json"].sha256,
            "annotation_logical_id": hashes["annotations.jsonl"].logical_id,
            "annotation_sha256": hashes["annotations.jsonl"].sha256,
        }
        frames_bindings[session_id] = {"frames_logical_id": hashes["frames.jsonl"].logical_id, "frames_sha256": hashes["frames.jsonl"].sha256}
        truth_bindings[session_id] = {"hud_truth_logical_id": hashes["hud_truth.jsonl"].logical_id, "hud_truth_sha256": hashes["hud_truth.jsonl"].sha256}
    dataset = ArtifactDescriptor(
        logical_id="dataset/formal-source",
        node_kind="source_descriptor",
        producer_id="hud-test",
        producer_version="1",
        identity_metadata={
            "manifest_logical_id": refs[0].logical_id,
            "target_profile_hash": profile_hash,
            "game_build_id": build_id,
            "dataset_identity_hash": "d" * 64,
            "model_session_bindings": model_bindings,
            "frames_bindings": frames_bindings,
            "hud_truth_bindings": truth_bindings,
        },
        files=tuple(refs),
    )
    verdict = RestoreTestVerdict(
        logical_id="dataset/restore-verdict",
        subject=dataset.node_ref(),
        manifest_hash=refs[0].sha256,
        primary_root="primary",
        backup_root="backup",
        verify_mode="full",
        checked_object_count=len(refs),
        passed=True,
        blocking_reasons=(),
    ).to_descriptor()
    minimum_counts = {
        "time_bands": {"0-20": 1},
        "screen_states": {state: 1 for state in truth_by_state},
        "card_counts": {"1": 1},
        "item_classes": {"empty_slot": 1},
        "buttons": {"reroll": 1},
        "fallback": 1,
        "roi_negative": 1,
    }

    report = run_formal_preflight(
        capture_store_root=capture_root,
        dataset_descriptor=dataset,
        restore_verdict_descriptor=verdict,
        target_profile_hash=profile_hash,
        minimum_counts=minimum_counts,
    )

    assert report.train_record_count == 12
    assert report.validation_record_count == 12
    assert len(report.code_hashes) >= 12
    assert PreflightReport.from_wire(report.to_wire()).preflight_canonical_hash == report.preflight_canonical_hash


def test_formal_cli_requires_release_inputs() -> None:
    """formal CLI は全ての lineage input が揃うまで exit 2 にする。

空引数で development path へ流れず、formal contract の必須 option を表示します。
    """
    with pytest.raises(SystemExit) as error:
        calibration_cli_main(["--formal"])
    assert error.value.code == 2


def test_formal_validation_uses_truth_and_session_cluster_ci(tmp_path: Path) -> None:
    """完全一致する三 session は pooled gate と全 session CI に合格する。

negative ROI を各 session に置き、presence gate の分母と session binding を検査します。
    """
    truth_wires = []
    records = []
    for session_index in range(3):
        session_id = f"validation-{session_index}"
        for frame_id, state in ((0, "gameplay"), (1, "unknown")):
            positive = state == "gameplay"
            truth = {
                "schema_version": "hud_truth.v1",
                "session_id": session_id,
                "frame_id": frame_id,
                "confirmed": True,
                "expected_screen_state": state,
                "expected_timer_seconds": 10.0 if positive else None,
                "expected_level": 1 if positive else None,
                "expected_hp_ratio": 0.5 if positive else None,
                "expected_xp_ratio": 0.25 if positive else None,
                "expected_items": ["whip"] * 12 if positive else None,
                "expected_choice": ["whip"] if positive else None,
                "roi_name": "hud",
                "expected_roi": [0, 32, 1920, 75] if positive else None,
                "confirmed_at": "2026-10-10T00:00:00Z",
                "draft_source": {"parser_artifact_hash": "p", "atlas_content_hash": "a"},
            }
            truth_wires.append(truth)
            records.append({
                **truth,
                "sample_id": f"{session_id}:{frame_id}",
                "split": "model_validation",
                "predicted_timer_seconds": 10.0 if positive else None,
                "predicted_level": 1 if positive else None,
                "predicted_hp_ratio": 0.5 if positive else None,
                "predicted_xp_ratio": 0.25 if positive else None,
                "predicted_items": ["whip"] * 12 if positive else None,
                "predicted_choice": "whip" if positive else None,
                "predicted_screen_state": state,
                "predicted_roi": [0, 32, 1920, 75] if positive else None,
                "latency_ms": 1.0,
                "_hud_truth_wire": truth,
            })
    truth_hash = canonical_hash(truth_wires)
    payload = {
        "schema_version": "hud_formal_preflight.v1",
        "dataset_logical_id": "dataset/source",
        "dataset_identity_hash": "d" * 64,
        "restore_verdict_logical_id": "dataset/restore",
        "restore_verdict_identity_hash": "e" * 64,
        "target_profile_hash": "a" * 64,
        "game_build_id": "b" * 64,
        "train_sessions": ["train-a", "train-b", "train-c"],
        "validation_sessions": ["validation-0", "validation-1", "validation-2"],
        "train_record_count": 3,
        "validation_record_count": 6,
        "train_truth_canonical_hash": "f" * 64,
        "validation_truth_canonical_hash": truth_hash,
        "slice_counts": {},
        "code_hashes": {},
    }
    preflight = PreflightReport(
        payload["dataset_logical_id"], payload["dataset_identity_hash"],
        payload["restore_verdict_logical_id"], payload["restore_verdict_identity_hash"],
        payload["target_profile_hash"], payload["game_build_id"],
        tuple(payload["train_sessions"]), tuple(payload["validation_sessions"]),
        payload["train_record_count"], payload["validation_record_count"],
        payload["train_truth_canonical_hash"], truth_hash, {}, {}, canonical_hash(payload),
    )

    fit_result = fit([{
        "sample_id": "train:0",
        "split": "model_train",
        "expected_roi": [0, 32, 1920, 75],
    }])
    feature = np.zeros(88, dtype=np.float32)
    empty_feature = np.ones(88, dtype=np.float32)
    atlas_content_hash = hashlib.sha256(np.concatenate([empty_feature, feature]).tobytes()).hexdigest()
    atlas_manifest = AtlasManifest(
        "icon_atlas.v1", "a" * 64, "b" * 64, False, True,
        atlas_content_hash,
        (
            TemplateEntry("empty_slot", "unknown", 1, 1, empty_feature),
            TemplateEntry("whip", "weapon", 1, 1, feature),
        ),
    )
    report = validate_formal(
        records,
        preflight=preflight,
        atlas_content_hash=atlas_content_hash,
        fit_canonical_hash=fit_result.fit_canonical_hash,
    )

    assert report.all_passed is True
    assert len(report.ci_bounds) == 10
    assert {gate.name: gate.value for gate in report.pooled_report.gates}["roi_false_positive"] == 0.0
    assert FormalValidationReport.from_wire(report.to_wire()).all_passed is True
    development_atlas = AtlasManifest(
        "icon_atlas.v1", "a" * 64, "b" * 64, True, False,
        atlas_content_hash,
        (
            TemplateEntry("empty_slot", "unknown", 1, 1, empty_feature),
            TemplateEntry("whip", "weapon", 1, 1, feature),
        ),
    )
    rejected_path = tmp_path / "rejected-formal-package.json"
    with pytest.raises(FormalPackageMixingError):
        PackageWriter.publish_formal(
            rejected_path,
            fit_result,
            report,
            preflight=preflight,
            atlas_manifest=development_atlas,
        )
    assert not rejected_path.exists()
    package_path = tmp_path / "formal-package.json"
    PackageWriter.publish_formal(
        package_path,
        fit_result,
        report,
        preflight=preflight,
        atlas_manifest=atlas_manifest,
    )
    loaded = PackageLoader.load_formal(package_path)
    assert loaded.package_hash == loaded.meta.package_hash
    with pytest.raises(FormalPackageMixingError):
        PackageLoader.load_development(package_path)
    artifact_store = ArtifactStore(tmp_path / "release-store")
    dataset_descriptor = ArtifactDescriptor(
        logical_id="dataset/source",
        node_kind="source_descriptor",
        producer_id="hud-test",
        producer_version="1",
        identity_metadata={
            "dataset_identity_hash": preflight.dataset_identity_hash,
            "target_profile_hash": preflight.target_profile_hash,
            "game_build_id": preflight.game_build_id,
        },
        files=(),
    )
    atlas_path = tmp_path / "atlas.json"
    atlas_path.write_bytes(formal_release.serialize_manifest(atlas_manifest))
    provenance_path = tmp_path / "atlas-provenance.json"
    provenance_path.write_text(json.dumps({
        "schema_version": "hud_atlas_provenance.v1",
        "entries": {
            "empty_slot@inventory": {"crop_count": 3, "source_frame_hashes": ["1" * 64]},
            "whip@inventory": {"crop_count": 3, "source_frame_hashes": ["1" * 64]},
        },
        "atlas_content_hash": atlas_content_hash,
    }), encoding="utf-8")
    release_descriptor = formal_release.publish_formal_parser_artifact(
        artifact_store,
        package_path=package_path,
        atlas_manifest_path=atlas_path,
        atlas_provenance_path=provenance_path,
        preflight=preflight,
        formal_report=report,
        dataset_descriptor=dataset_descriptor,
        logical_prefix="hud/formal",
    )
    from benchmark_survivors_perception import (
        _PACKAGE_REQUIRED_FIELDS,
        _manifest_ref,
        _validate_restore_verdict,
    )

    envelope_ref = _manifest_ref(release_descriptor)
    envelope = json.loads(artifact_store.object_path(envelope_ref.store_uri).read_bytes())
    assert set(envelope) == _PACKAGE_REQUIRED_FIELDS["parser_package"]
    assert envelope["files"]["parser_artifact_hash"] == release_descriptor.files[0].logical_id
    from survivors.artifact_restore_test import run_restore_test

    verdict, verdict_descriptor = run_restore_test(
        artifact_store,
        release_descriptor,
        backup_root=tmp_path / "backup-store",
        restore_root=tmp_path / "restore-store",
        verdict_logical_id="hud/formal/restore-verdict",
    )
    assert verdict.passed is True
    _validate_restore_verdict(release_descriptor, verdict_descriptor)
    resolved_path = formal_release.emit_resolved_layout_yaml(
        fit_result,
        report,
        preflight,
        release_descriptor,
        verdict_descriptor,
        tmp_path / "hud_layout_1080p_ja_v1.yaml",
    )
    resolved = formal_release.yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert resolved["schema_version"] == "hud_layout.v1"
    assert resolved["source"]["package_logical_id"] == release_descriptor.files[0].logical_id
    package_wire = json.loads(package_path.read_text(encoding="utf-8"))
    package_wire["metadata"]["atlas_content_hash"] = "0" * 64
    package_path.write_text(json.dumps(package_wire), encoding="utf-8")
    with pytest.raises(PackageHashMismatch):
        PackageLoader.load_formal(package_path)
    package_wire["metadata"]["schema_version"] = "hud_parser_package.unknown.v9"
    package_path.write_text(json.dumps(package_wire), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported parser package schema"):
        PackageLoader.load_formal(package_path)
    tampered = [dict(record) for record in records]
    tampered[0]["_hud_truth_wire"] = {**tampered[0]["_hud_truth_wire"], "expected_level": 2}
    with pytest.raises(FormalPreflightError, match="changed after preflight"):
        validate_formal(
            tampered,
            preflight=preflight,
            atlas_content_hash="c" * 64,
            fit_canonical_hash="f" * 64,
        )

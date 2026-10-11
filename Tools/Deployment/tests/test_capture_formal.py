"""capture session の formal 昇格と dataset descriptor の契約を検証する。

synthetic store だけを使い、producer から release preflight までの hash binding を固定する。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from survivors.capture.captured_frame import CapturedFrame
from survivors.capture_dataset import (
    AnnotationWriter,
    DatasetWriter,
    SplitFreezer,
    load_split_manifest_by_hash,
)
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from survivors.capture_formal import (
    ModelExportPaths,
    PromotionError,
    PublishError,
    assign_from_manifests,
    build_annotation_agreement,
    promote_session,
    publish_capture_dataset,
    verify_capture_dataset_descriptor,
)
from survivors.artifact_restore_test import run_restore_test


PROFILE_HASH = "a" * 64
BUILD_ID = "survivors-live-build-001"


def _publish_session(root, session_id="session-001", *, build_id=BUILD_ID, checkpoint="operator checkpoint"):
    """小さな lossless capture session を作成する。

    昇格前の frame metadata と PNG を実装依存の mock なしで用意する。
    """
    writer = DatasetWriter(root, session_id, PROFILE_HASH, build_id)
    frame = CapturedFrame(
        frame_bgra=np.zeros((1080, 1920, 4), dtype=np.uint8),
        captured_monotonic_ns=100,
        session_frame_index=0,
        client_rect_screen_px=(0, 0, 1920, 1080),
        foreground=True,
        target_profile_hash=PROFILE_HASH,
        game_build_id=build_id,
    )
    writer.write_frame(frame)
    manifest = writer.publish(operator_checkpoint=checkpoint)
    return manifest


def test_promotion_changes_only_manifest_and_is_idempotent(tmp_path):
    """昇格が manifest だけを変え、再実行で同じ結果になることを確認する。

    pixel と annotation の hash／bytes が昇格処理から保護される契約を固定する。
    """
    published = _publish_session(tmp_path)
    session_dir = published.session_path
    frames_before = (session_dir / "frames.jsonl").read_bytes()
    png_path = session_dir / published.frame_records[0].object_path
    png_before = hashlib.sha256(png_path.read_bytes()).hexdigest()
    annotations = session_dir / "annotations.jsonl"
    annotations.write_bytes(b'{"old":"annotation"}\n')

    promoted = promote_session(
        tmp_path,
        published.session_id,
        operator_id="operator-01",
        attestation_note="reviewed build",
        expected_target_profile_hash=PROFILE_HASH,
        expected_game_build_id=BUILD_ID,
    )
    again = promote_session(
        tmp_path,
        published.session_id,
        operator_id="operator-01",
        attestation_note="reviewed build",
        expected_target_profile_hash=PROFILE_HASH,
        expected_game_build_id=BUILD_ID,
    )

    assert promoted.formal_dataset_eligible is True
    assert promoted.manifest_sha256 == again.manifest_sha256
    assert promoted.operator_checkpoint.startswith("operator checkpoint|formal:operator-01:")
    assert (session_dir / "frames.jsonl").read_bytes() == frames_before
    assert hashlib.sha256(png_path.read_bytes()).hexdigest() == png_before
    assert annotations.read_bytes() == b'{"old":"annotation"}\n'


def test_promotion_rejects_session_already_in_frozen_split(tmp_path):
    """freeze 済み split に含まれる session の昇格を拒否する。

    session manifest hash の固定後に split の参照を壊す回帰を捕まえる。
    """
    published = _publish_session(tmp_path)
    freezer = SplitFreezer(tmp_path)
    freezer.assign("model_train", [published.session_id])
    freezer.freeze()

    with pytest.raises(PromotionError, match="frozen split contains this session"):
        promote_session(
            tmp_path,
            published.session_id,
            operator_id="operator-01",
            attestation_note="reviewed build",
            expected_target_profile_hash=PROFILE_HASH,
            expected_game_build_id=BUILD_ID,
        )


def test_promotion_rejects_every_frozen_generation_and_invalid_freezer_path(tmp_path):
    """日付付き freeze 後の用途変更と consumer 非対応 path を拒否する。

    root 直下に複数 generation があっても session manifest hash を保持する。
    """
    published = _publish_session(tmp_path)
    promote_session(
        tmp_path,
        published.session_id,
        operator_id="operator-01",
        attestation_note="reviewed build",
        expected_target_profile_hash=PROFILE_HASH,
        expected_game_build_id=BUILD_ID,
        planned_split="model_train",
    )
    session_manifest_path = published.session_path / "session_manifest.json"
    session_bytes = session_manifest_path.read_bytes()
    split_path = tmp_path / "capture_split_manifest.20261011.json"
    freezer = SplitFreezer(tmp_path, manifest_path=split_path)
    freezer.assign("model_train", [published.session_id])
    freezer.freeze()
    split_bytes = split_path.read_bytes()

    with pytest.raises(PromotionError, match="frozen split contains this session"):
        promote_session(
            tmp_path,
            published.session_id,
            operator_id="operator-01",
            attestation_note="reviewed build",
            expected_target_profile_hash=PROFILE_HASH,
            expected_game_build_id=BUILD_ID,
            planned_split="model_validation",
        )

    assert session_manifest_path.read_bytes() == session_bytes
    assert split_path.read_bytes() == split_bytes
    with pytest.raises(ValueError, match="root-level capture split manifest pattern"):
        SplitFreezer(tmp_path, manifest_path=tmp_path / "splits" / "gen2.json")


def test_all_planned_promotes_new_session_after_dated_freeze_without_changing_frozen_sessions(
    tmp_path, monkeypatch, capsys
):
    """日付付き freeze 済み session を no-op にして新規 planned session を昇格する。

    all-planned が既存の frozen manifest を変更せず処理を完了することを確認する。
    """
    from types import SimpleNamespace

    import promote_survivors_capture_session as promote_cli

    monkeypatch.setattr(
        promote_cli,
        "load_runtime_profile",
        lambda _: SimpleNamespace(target_hash=PROFILE_HASH, sections={"build": {"build_id": BUILD_ID}}),
    )
    frozen_ids = ("a-model", "b-model")
    frozen_paths = []
    for session_id in frozen_ids:
        published = _publish_session(
            tmp_path,
            session_id,
            checkpoint="planned_split:model_train|planned_at:2026-10-11T00:00:00Z|capture",
        )
        promote_session(
            tmp_path,
            session_id,
            operator_id="operator-01",
            attestation_note="reviewed build",
            expected_target_profile_hash=PROFILE_HASH,
            expected_game_build_id=BUILD_ID,
        )
        frozen_paths.append(published.session_path / "session_manifest.json")
    frozen_bytes = [path.read_bytes() for path in frozen_paths]

    split_path = tmp_path / "capture_split_manifest.20261011.json"
    freezer = SplitFreezer(tmp_path, manifest_path=split_path)
    freezer.assign("model_train", frozen_ids)
    freezer.freeze()
    _publish_session(
        tmp_path,
        "c-sealed",
        checkpoint="planned_split:final_e2e_test|planned_at:2026-10-11T00:00:00Z|capture",
    )

    assert promote_cli.main([
        "--store-root", str(tmp_path), "--all-planned",
        "--operator-id", "operator-01", "--note", "reviewed build",
        "--target-profile", str(tmp_path / "target-profile.yaml"),
    ]) == 0
    promoted = json.loads(capsys.readouterr().out)["promoted"]

    assert {item["session_id"] for item in promoted} == {*frozen_ids, "c-sealed"}
    assert DatasetWriter.restore(tmp_path, "c-sealed", metadata_only=True).formal_dataset_eligible
    assert [path.read_bytes() for path in frozen_paths] == frozen_bytes


def test_frozen_eligible_noop_matches_direct_and_session_id_calls(tmp_path, monkeypatch, capsys):
    """frozen eligible session への direct と session-id の no-op が一致する。

    同じ planned split の再指定を含め、session manifest bytes が固定されることを確認する。
    """
    from types import SimpleNamespace

    import promote_survivors_capture_session as promote_cli

    monkeypatch.setattr(
        promote_cli,
        "load_runtime_profile",
        lambda _: SimpleNamespace(target_hash=PROFILE_HASH, sections={"build": {"build_id": BUILD_ID}}),
    )
    published = _publish_session(
        tmp_path,
        checkpoint="planned_split:model_train|planned_at:2026-10-11T00:00:00Z|capture",
    )
    eligible = promote_session(
        tmp_path,
        published.session_id,
        operator_id="operator-01",
        attestation_note="reviewed build",
        expected_target_profile_hash=PROFILE_HASH,
        expected_game_build_id=BUILD_ID,
    )
    freezer = SplitFreezer(tmp_path, manifest_path=tmp_path / "capture_split_manifest.20261011.json")
    freezer.assign("model_train", [published.session_id])
    freezer.freeze()
    manifest_path = published.session_path / "session_manifest.json"
    frozen_bytes = manifest_path.read_bytes()

    direct = promote_session(
        tmp_path,
        published.session_id,
        operator_id="operator-01",
        attestation_note="reviewed build",
        expected_target_profile_hash=PROFILE_HASH,
        expected_game_build_id=BUILD_ID,
    )
    assert direct.manifest_sha256 == eligible.manifest_sha256
    assert manifest_path.read_bytes() == frozen_bytes

    assert promote_cli.main([
        "--store-root", str(tmp_path), "--session-id", published.session_id,
        "--operator-id", "operator-01", "--note", "reviewed build",
        "--target-profile", str(tmp_path / "target-profile.yaml"),
        "--planned-split", "model_train",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["promoted"][0]["manifest_sha256"] == eligible.manifest_sha256
    assert manifest_path.read_bytes() == frozen_bytes


def test_promotion_rejects_fixture_build_id(tmp_path):
    """fixture 専用 build id を formal 昇格の trust boundary で拒否する。

    fixture artifact が実 build と同じ形式で release 経路へ入る誤りを検出する。
    """
    published = _publish_session(tmp_path, build_id="TEST_FIXTURE_ONLY")

    with pytest.raises(PromotionError, match="build id"):
        promote_session(
            tmp_path,
            published.session_id,
            operator_id="operator-01",
            attestation_note="reviewed build",
            expected_target_profile_hash=PROFILE_HASH,
            expected_game_build_id="TEST_FIXTURE_ONLY",
        )


def test_planned_split_can_be_corrected_before_freeze_without_losing_attestation(tmp_path):
    """昇格済み checkpoint の split prefix を差し替えて後続を保持する。

    planned split の修正で operator attestation が消えたり二重化したりしない。
    """
    published = _publish_session(tmp_path)
    first = promote_session(
        tmp_path,
        published.session_id,
        operator_id="operator-01",
        attestation_note="reviewed",
        expected_target_profile_hash=PROFILE_HASH,
        expected_game_build_id=BUILD_ID,
        planned_split="model_validation",
    )
    corrected = promote_session(
        tmp_path,
        published.session_id,
        operator_id="operator-01",
        attestation_note="reviewed",
        expected_target_profile_hash=PROFILE_HASH,
        expected_game_build_id=BUILD_ID,
        planned_split="model_train",
    )

    assert corrected.operator_checkpoint.startswith("planned_split:model_train|planned_at:")
    assert corrected.operator_checkpoint.count("|formal:operator-01:") == 1
    assert corrected.operator_checkpoint.endswith("reviewed")
    assert corrected.manifest_sha256 != first.manifest_sha256


def test_capture_cli_persists_planned_split_and_rejects_unknown_choice(tmp_path, capsys):
    """capture CLI が split 用途を先頭 segment に書き、未知値を parser error にする。

    command line の choices と manifest checkpoint の結線を実動作で検査する。
    """
    from capture_survivors import _parser, main as capture_main

    assert capture_main([
        "--store-root", str(tmp_path), "--session-id", "planned-session",
        "--duration-sec", "0.1", "--synthetic", "--planned-split", "model_train",
    ]) == 0
    manifest = DatasetWriter.restore(tmp_path, "planned-session", metadata_only=True)
    assert manifest.operator_checkpoint.startswith("planned_split:model_train|planned_at:")
    with pytest.raises(SystemExit):
        _parser().parse_args(["--planned-split", "unknown"])


def test_assign_from_manifests_includes_only_formal_sessions_and_keeps_generation(tmp_path):
    """manifest に保存した用途だけで昇格済み session を再構成する。

    明示 path に作る split 世代と昇格後 session hash の対応を検査する。
    """
    for session_id, split_name in (
        ("train-session", "model_train"),
        ("validation-session", "model_validation"),
        ("draft-session", "model_train"),
    ):
        checkpoint = f"planned_split:{split_name}|planned_at:2026-10-11T00:00:00Z|capture"
        published = _publish_session(tmp_path, session_id)
        if session_id != "draft-session":
            wire = json.loads((published.session_path / "session_manifest.json").read_text())
            wire["operator_checkpoint"] = checkpoint
            (published.session_path / "session_manifest.json").write_text(
                json.dumps(wire, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            promote_session(
                tmp_path,
                session_id,
                operator_id="operator-01",
                attestation_note="reviewed",
                expected_target_profile_hash=PROFILE_HASH,
                expected_game_build_id=BUILD_ID,
            )

    manifest_path = tmp_path / "capture_split_manifest.20261011.json"
    freezer = assign_from_manifests(tmp_path, manifest_path=manifest_path)
    manifest = freezer.freeze()

    assert manifest.manifest_path == manifest_path
    assert [unit.session_id for unit in manifest.splits["model_train"]] == ["train-session"]
    assert [unit.session_id for unit in manifest.splits["model_validation"]] == ["validation-session"]
    assert all("draft-session" != unit.session_id for units in manifest.splits.values() for unit in units)
    assert manifest.splits["model_train"][0].session_manifest_sha256 == hashlib.sha256(
        (tmp_path / "capture_sessions/train-session/session_manifest.json").read_bytes()
    ).hexdigest()


def test_legacy_annotation_reader_rejects_formal_schema(tmp_path):
    """旧 AnnotationRecord reader は formal annotation schema を扱わない。

    別 semantic contract の行を誤って legacy record に変換する回帰を拒む。
    """
    published = _publish_session(tmp_path)
    (published.session_path / "annotations.jsonl").write_text(
        '{"schema_version":"perception_formal_annotation.v1"}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="formal annotation schema; use formal loader"):
        AnnotationWriter.read_annotations(tmp_path, published.session_id)


def test_split_manifest_hash_selects_the_requested_frozen_generation(tmp_path):
    """preflight 用 manifest lookup は hash に一致する日付付き世代を選ぶ。

    default model split と sealed generation が共存する store で誤選択しない。
    """
    splits = {name: [] for name in ("model_train", "model_validation", "error_calibration", "final_e2e_test")}
    default_wire = {"schema_version": 1, "frozen": True, "splits": splits}
    dated_splits = {name: [] for name in splits}
    dated_splits["final_e2e_test"] = [{
        "session_id": "sealed-session",
        "game_build_id": BUILD_ID,
        "session_manifest_sha256": "d" * 64,
    }]
    dated_wire = {"schema_version": 1, "frozen": True, "splits": dated_splits}
    default_bytes = (json.dumps(default_wire, sort_keys=True, separators=(",", ":")) + "\n").encode()
    dated_bytes = (json.dumps(dated_wire, sort_keys=True, separators=(",", ":")) + "\n").encode()
    root = tmp_path / "store"
    root.mkdir()
    (root / "capture_split_manifest.json").write_bytes(default_bytes)
    dated_path = root / "capture_split_manifest.20261011.json"
    dated_path.write_bytes(dated_bytes)

    selected = load_split_manifest_by_hash(root, hashlib.sha256(dated_bytes).hexdigest())

    assert selected.manifest_path == dated_path
    assert [unit.session_id for unit in selected.splits["final_e2e_test"]] == ["sealed-session"]


def _write_label(path, labels, *, checked=True):
    """fixture 用 LabelMe JSON を保存する。

    box 座標と class を固定値にし、agreement の期待値を手で確定できるようにする。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    shapes = [
        {"label": label, "shape_type": "rectangle", "points": [[left, top], [right, bottom]]}
        for label, left, top, right, bottom in labels
    ]
    path.write_text(
        json.dumps({"shapes": shapes, "imagePath": "", "imageData": None, "checked": checked}),
        encoding="utf-8",
    )


def test_annotation_agreement_matches_boxes_and_ignores_unshared_frames(tmp_path):
    """agreement は共有 checked frame だけを IoU で対応付ける。

    class mismatch と class match を別々に数え、片側だけの frame を母集団から外す。
    """
    first, second = tmp_path / "first", tmp_path / "second"
    _write_label(first / "session-a" / "00000001.json", [
        ("enemy_normal", 0, 0, 10, 10),
        ("enemy_normal", 20, 20, 30, 30),
        ("weapon_target", 40, 40, 50, 50),
    ])
    _write_label(first / "session-a" / "00000002.json", [("enemy_normal", 0, 0, 10, 10)], checked=False)
    _write_label(first / "session-a" / "00000004.json", [
        ("enemy_normal", 0, 0, 10, 10),
        ("labeled_region", 0, 0, 10, 10),
    ])
    _write_label(second / "session-a" / "00000001.json", [
        ("enemy_boss", 0, 0, 10, 10),
        ("enemy_normal", 20, 20, 30, 30),
    ])
    _write_label(second / "session-a" / "00000003.json", [("enemy_normal", 0, 0, 10, 10)])
    _write_label(second / "session-a" / "00000004.json", [("enemy_normal", 0, 0, 10, 10)])

    report = build_annotation_agreement(first, second, ["session-a"])

    assert report["schema_version"] == "annotation_agreement.v1"
    assert report["pair_count"] == 2
    assert report["iou_mean"] == 1.0
    assert report["class_agreement"] == 0.5
    assert report["unmatched_first"] == report["unmatched_second"] == 0
    assert report["per_class"]["enemy_normal"]["pair_count"] == 2


def test_annotation_agreement_empty_second_work_root_returns_zero_pairs(tmp_path):
    """second annotator に共有 checked frame が無い場合は zero pair を出す。

    preflight が空の double-review を数値で fail できる形を保つ。
    """
    first, second = tmp_path / "first", tmp_path / "second"
    _write_label(first / "session-a" / "00000001.json", [("enemy_normal", 0, 0, 10, 10)])
    (second / "session-a").mkdir(parents=True)

    report = build_annotation_agreement(first, second, ["session-a"])

    assert report["pair_count"] == 0
    assert report["iou_mean"] == 0.0
    assert report["class_agreement"] == 0.0


def test_coco_export_binds_capture_frame_hash(tmp_path):
    """store root 指定時 COCO image は frames.jsonl の object hash を持つ。

    export 済み image と artifact session の pixel identity が分離しないことを確認する。
    """
    from export_survivors_annotations_coco import main as export_main

    published = _publish_session(tmp_path / "store", "session-a")
    work_root = tmp_path / "work"
    label_dir = work_root / "session-a"
    label_dir.mkdir(parents=True)
    image_path = published.session_path / published.frame_records[0].object_path
    shutil.copyfile(image_path, label_dir / "00000000.png")
    _write_label(label_dir / "00000000.json", [("enemy_normal", 0, 0, 20, 20)])
    output = tmp_path / "output"

    assert export_main([
        "--work-root", str(work_root), "--output-dir", str(output),
        "--store-root", str(tmp_path / "store"),
    ]) == 0
    image = json.loads((output / "world_coco.json").read_text(encoding="utf-8"))["images"][0]

    assert image["session_id"] == "session-a"
    assert image["frame_id"] == 0
    assert image["object_sha256"] == published.frame_records[0].object_sha256


def test_publish_emits_consumer_bindings_and_passed_restore_verdict(tmp_path):
    """formal model dataset を artifact descriptor へ束縛し全 object を復元する。

    model binding shape、split manifest ref、export links が producer/consumer 契約に沿う。
    """
    root = tmp_path / "capture"
    train = _publish_session(root, "train-session", checkpoint="planned_split:model_train|planned_at:2026-10-11T00:00:00Z|capture")
    validation = _publish_session(root, "validation-session", checkpoint="planned_split:model_validation|planned_at:2026-10-11T00:00:00Z|capture")
    validation_unknown = _publish_session(root, "validation-unknown", checkpoint="planned_split:model_validation|planned_at:2026-10-11T00:00:00Z|capture")
    from survivors.vision.hud_truth import expected_roi_for_state

    for session, screen_state in (
        (train, "gameplay"),
        (validation, "level_up_fallback"),
        (validation_unknown, "unknown"),
    ):
        (session.session_path / "annotations.jsonl").write_bytes(b"")
        (session.session_path / "hud_truth.jsonl").write_text(json.dumps({
            "schema_version": "hud_truth.v1",
            "session_id": session.session_id,
            "frame_id": 0,
            "annotator_id": "test-annotator",
            "confirmed": True,
            "expected_screen_state": screen_state,
            "expected_timer_seconds": 10.0,
            "expected_level": 1,
            "expected_hp_ratio": 0.5,
            "expected_xp_ratio": 0.25,
            "expected_items": ["empty_slot"] * 12,
            "expected_slot_levels": [None] * 12,
            "expected_choice": ["whip"] if screen_state == "level_up_fallback" else None,
            "roi_name": "hud",
            "expected_roi": expected_roi_for_state(screen_state),
            "confirmed_at": "2026-10-10T00:00:00Z",
            "draft_source": {"parser_artifact_hash": "parser", "atlas_content_hash": "atlas"},
        }) + "\n", encoding="utf-8")
        promote_session(
            root,
            session.session_id,
            operator_id="operator-01",
            attestation_note="reviewed",
            expected_target_profile_hash=PROFILE_HASH,
            expected_game_build_id=BUILD_ID,
        )
    split_path = root / "capture_split_manifest.20261011.json"
    freezer = SplitFreezer(root, manifest_path=split_path)
    freezer.assign("model_train", ["train-session"])
    freezer.assign("model_validation", ["validation-session", "validation-unknown"])
    freezer.freeze()
    from export_survivors_annotations_coco import main as export_main
    from merge_survivors_coco import main as merge_main
    from benchmark_survivors_perception import _manifest_ref, _validate_restore_verdict

    work_a, work_b = tmp_path / "work-a", tmp_path / "work-b"
    for session in (train, validation, validation_unknown):
        for work in (work_a, work_b):
            label_dir = work / session.session_id
            _write_label(label_dir / "00000000.json", [
                ("enemy_normal", 0, 0, 20, 20),
                ("weapon_aura", 30, 30, 50, 50),
            ])
        shutil.copyfile(
            session.session_path / session.frame_records[0].object_path,
            work_a / session.session_id / "00000000.png",
        )
    train_export, validation_export = tmp_path / "train-export", tmp_path / "validation-export"
    for split_name, output_dir in (("model_train", train_export), ("model_validation", validation_export)):
        assert export_main([
            "--work-root", str(work_a), "--output-dir", str(output_dir),
            "--store-root", str(root), "--split-manifest", str(split_path), "--split", split_name,
        ]) == 0
    model_dir = tmp_path / "model-export"
    assert merge_main([
        "--world-train", str(train_export / "world_coco.json"),
        "--world-validation", str(validation_export / "world_coco.json"),
        "--ui-train", str(train_export / "ui_coco.json"),
        "--ui-validation", str(validation_export / "ui_coco.json"),
        "--output-dir", str(model_dir),
    ]) == 0
    agreement = tmp_path / "agreement.json"
    agreement.write_text(
        json.dumps(build_annotation_agreement(
            work_a, work_b, ["train-session", "validation-session", "validation-unknown"]
        )),
        encoding="utf-8",
    )
    artifact_store = ArtifactStore(tmp_path / "artifact-store")

    descriptor = publish_capture_dataset(
        artifact_store,
        store_root=root,
        split_manifest_path=split_path,
        logical_prefix="survivors/capture/model-v1",
        model_exports=ModelExportPaths(model_dir / "world_coco_model.json", model_dir / "ui_coco_model.json"),
        agreement_path=agreement,
    )
    verdict, verdict_descriptor = run_restore_test(
        artifact_store,
        descriptor,
        backup_root=tmp_path / "backup",
        restore_root=tmp_path / "restore",
        verdict_logical_id="survivors/capture/model-v1/restore_test.json",
    )
    identity = descriptor.identity_metadata

    assert _manifest_ref(descriptor).logical_id == identity["manifest_logical_id"]
    _validate_restore_verdict(descriptor, verdict_descriptor)
    assert identity["manifest_logical_id"] == next(
        ref.logical_id for ref in descriptor.files if ref.logical_id.endswith("/capture_split_manifest.json")
    )
    assert identity["annotation_bindings"] == {}
    assert set(identity["model_session_bindings"]) == {
        "train-session", "validation-session", "validation-unknown"
    }
    assert set(identity["dataset_exports"]) == {"world_coco_model", "ui_coco_model", "annotation_agreement"}
    assert verdict.passed is True
    assert verdict_descriptor.identity_metadata["subject_identity_hash"] == descriptor.identity_hash

    from pathlib import Path
    import yaml
    from survivors.vision.hud_formal_release import run_formal_preflight as run_hud_preflight
    from survivors.vision.world_formal_release import run_formal_preflight as run_world_preflight

    hud_report = run_hud_preflight(
        capture_store_root=root,
        dataset_descriptor=descriptor,
        restore_verdict_descriptor=verdict_descriptor,
        target_profile_hash=PROFILE_HASH,
        minimum_counts={
            "time_bands": {"0-20": 1},
            "screen_states": {"level_up_fallback": 1, "unknown": 1},
            "card_counts": {"1": 1},
            "item_classes": {"empty_slot": 1},
            "buttons": {},
            "fallback": 1,
            "roi_negative": 1,
        },
    )
    assert hud_report.validation_record_count == 2

    deployment_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((deployment_root / "configs" / "world_detector_v2.yaml").read_text(encoding="utf-8"))
    config["training"]["preflight"] = {
        "min_frames": 1,
        "min_entities": 1,
        "min_classes": 1,
        "min_time_bands": 1,
        "min_frames_per_split": 1,
        "min_entities_per_split": 1,
        "min_classes_per_split": 1,
    }
    config_path = tmp_path / "world_detector.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    feasibility_path = tmp_path / "feasibility.json"
    feasibility_path.write_text(json.dumps({
        "schema_version": "perception_feasibility_verdict.v1",
        "status": "PASS",
        "selected_architecture": "ssdlite320",
    }), encoding="utf-8")
    world_report = run_world_preflight(
        feasibility_verdict_path=feasibility_path,
        capture_store_root=root,
        dataset_descriptor=descriptor,
        restore_verdict_descriptor=verdict_descriptor,
        annotations_path=model_dir / "world_coco_model.json",
        class_map_path=deployment_root / "configs" / "world_class_map_v2.yaml",
        config_path=config_path,
        cli_overrides={},
        target_profile_hash=PROFILE_HASH,
        hardware_id="synthetic-test",
        annotation_agreement_path=agreement,
        minimum_counts={
            "min_frames": 1,
            "min_entities": 1,
            "min_classes": 1,
            "min_time_bands": 1,
            "min_frames_per_split": 1,
            "min_entities_per_split": 1,
            "min_classes_per_split": 1,
            "heavy_effect_slice": 1,
            "agreement": {"pair_count": 1, "iou_mean": 0.9, "class_agreement": 0.9},
        },
    )
    assert world_report.train_sessions == ["train-session"]
    assert world_report.validation_sessions == ["validation-session", "validation-unknown"]

    (model_dir / "world_coco_model.json").write_text(
        '{"images":[],"annotations":[],"categories":[],"revision":2}\n', encoding="utf-8"
    )
    next_generation = publish_capture_dataset(
        artifact_store,
        store_root=root,
        split_manifest_path=split_path,
        logical_prefix="survivors/capture/model-v1",
        model_exports=ModelExportPaths(model_dir / "world_coco_model.json", model_dir / "ui_coco_model.json"),
        agreement_path=agreement,
    )
    assert next_generation.logical_id != descriptor.logical_id
    assert next_generation.identity_metadata["dataset_identity_hash"] != identity["dataset_identity_hash"]


def test_capture_publish_cli_retries_restore_with_fresh_roots(tmp_path, monkeypatch, capsys):
    """failed restore verdict の後に別 root で CLI を再実行する。

    logical id の root binding と ArtifactStoreError の CLI boundary を確認する。
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    import publish_survivors_capture_dataset as publisher_module
    from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor
    from reinbalance_survivors_contracts.artifact_store import ArtifactStoreError

    root = tmp_path / "capture"
    sessions = [
        _publish_session(root, "train-session"),
        _publish_session(root, "validation-session"),
    ]
    for session in sessions:
        (session.session_path / "annotations.jsonl").write_bytes(b"")
        promote_session(
            root,
            session.session_id,
            operator_id="operator-01",
            attestation_note="reviewed",
            expected_target_profile_hash=PROFILE_HASH,
            expected_game_build_id=BUILD_ID,
        )
    freezer = SplitFreezer(root)
    freezer.assign("model_train", ["train-session"])
    freezer.assign("model_validation", ["validation-session"])
    split = freezer.freeze()
    artifact_root = tmp_path / "artifact-store"

    def arguments(name):
        """各試行に分離した空 restore root を指定する。

        同一 capture generation の publish 入力はそのまま再利用する。
        """
        return [
            "--store-root", str(root),
            "--split-manifest", str(split.manifest_path),
            "--artifact-store", str(artifact_root),
            "--backup-root", str(tmp_path / f"backup-{name}"),
            "--restore-root", str(tmp_path / f"restore-{name}"),
            "--logical-prefix", "survivors/capture/retry-v1",
        ]

    original_verify = ArtifactStore.verify

    def fail_restore_after_publish(store, descriptor, **kwargs):
        """restore helper が失敗 verdict を発行する状態を作る。

        publish 側の descriptor verification を終えた後だけ object verify を落とす。
        """
        with monkeypatch.context() as restore_patch:
            restore_patch.setattr(
                ArtifactStore,
                "verify",
                lambda _store, _ref: (
                    original_verify(_store, _ref)
                    if _ref.logical_id == kwargs["verdict_logical_id"]
                    else SimpleNamespace(ok=False, reason="fixture failure")
                ),
            )
            return run_restore_test(store, descriptor, **kwargs)

    monkeypatch.setattr(publisher_module, "run_restore_test", fail_restore_after_publish)
    assert publisher_module.main(arguments("failed")) == 1
    failed_output = capsys.readouterr()
    assert not failed_output.err, failed_output.err
    failed = json.loads(failed_output.out)
    assert failed["passed"] is False

    monkeypatch.setattr(publisher_module, "run_restore_test", run_restore_test)
    assert publisher_module.main(arguments("retry")) == 0
    retried = json.loads(capsys.readouterr().out)
    assert retried["passed"] is True
    assert retried["restore_verdict_logical_id"] != failed["restore_verdict_logical_id"]

    store = ArtifactStore(artifact_root)
    dataset_ref = store.resolve(retried["descriptor_logical_id"])
    verdict_ref = store.resolve(retried["restore_verdict_logical_id"])
    dataset = ArtifactDescriptor.from_wire(json.loads(store.object_path(dataset_ref.store_uri).read_bytes()))
    verdict = ArtifactDescriptor.from_wire(json.loads(store.object_path(verdict_ref.store_uri).read_bytes()))
    from benchmark_survivors_perception import _validate_restore_verdict

    _validate_restore_verdict(dataset, verdict)

    monkeypatch.setattr(
        publisher_module,
        "run_restore_test",
        Mock(side_effect=ArtifactStoreError("fixture store failure")),
    )
    assert publisher_module.main(arguments("store-error")) == 1
    assert "fixture store failure" in capsys.readouterr().err
    assert ArtifactStore.verify is original_verify


def test_publish_binds_formal_annotation_gates_coco_and_evaluation_windows(tmp_path, monkeypatch):
    """multi-session plan を publish と formal input restore まで通す。

    sealed split 全件を一度に評価 identity へ結ぶ手順を検証する。
    """
    root = tmp_path / "capture"
    model = _publish_session(root, "model-session")
    sealed = _publish_session(root, "sealed-session")
    sealed_second = _publish_session(root, "sealed-session-2")
    (model.session_path / "annotations.jsonl").write_bytes(b"")
    (model.session_path / "hud_truth.jsonl").write_text('{"fixture":true}\n', encoding="utf-8")
    sealed_sessions = (sealed, sealed_second)
    for session in (model, *sealed_sessions):
        promote_session(
            root,
            session.session_id,
            operator_id="operator-01",
            attestation_note="reviewed",
            expected_target_profile_hash=PROFILE_HASH,
            expected_game_build_id=BUILD_ID,
        )
    freezer = SplitFreezer(root)
    freezer.assign("model_train", ["model-session"])
    freezer.assign("final_e2e_test", [session.session_id for session in sealed_sessions])
    split = freezer.freeze()
    gates = Path(__file__).resolve().parents[1] / "configs" / "formal_gates_minimal_v1.yaml"
    store = ArtifactStore(tmp_path / "artifact-store")
    with pytest.warns(RuntimeWarning, match="evaluation windows are absent"), pytest.raises(
        PublishError, match="formal annotations"
    ):
        publish_capture_dataset(
            store,
            store_root=root,
            split_manifest_path=split.manifest_path,
            logical_prefix="survivors/capture/sealed-v1",
            formal_gates_path=gates,
        )
    with pytest.raises(PublishError, match="only valid when sealed splits are empty"):
        publish_capture_dataset(
            store,
            store_root=root,
            split_manifest_path=split.manifest_path,
            logical_prefix="survivors/capture/sealed-v1",
            formal_gates_path=gates,
            allow_missing_formal_annotations=True,
        )
    for session in sealed_sessions:
        formal_row = {
            "schema_version": "perception_formal_annotation.v1",
            "session_id": session.session_id,
            "frame_index": 0,
            "raw_card_ids": [],
            "raw_inventory": [],
            "ground_truth_semantic_hash": "b" * 64,
        }
        (session.session_path / "annotations.jsonl").write_text(
            json.dumps(formal_row) + "\n", encoding="utf-8"
        )
        coco_dir = root / "coco" / session.session_id
        coco_dir.mkdir(parents=True)
        (coco_dir / "world_coco.json").write_text(
            '{"images":[],"annotations":[],"categories":[]}\n', encoding="utf-8"
        )
        (coco_dir / "ui_coco.json").write_text(
            '{"images":[],"annotations":[],"categories":[]}\n', encoding="utf-8"
        )

    import plan_survivors_evaluation_windows as window_plan_module

    parser_hash = hashlib.sha256(Path(window_plan_module.hud_parser_module.__file__).read_bytes()).hexdigest()
    rule_path = tmp_path / "window_rule.yaml"
    rule_path.write_text(json.dumps({
        "fixed_offsets_per_time_band": {
            "window_frames": 4, "offsets_seconds": {"early": 2, "mid": 8, "late": 14},
        },
        "run_tail": {"seconds": 3},
        "event_windows": {
            "detector": "development_hud_parser",
            "states": ["level_up_items", "level_up_fallback", "chest"],
            "pre_frames": 1, "post_frames": 1, "max_per_band": 4,
            "parser_artifact_hash": parser_hash,
        },
        "boss_windows": {
            "detector": "development_world_detector", "checkpoint_hash": "b" * 64,
            "score_threshold": 0.5, "pre_frames": 1, "post_frames": 2,
            "max_per_session": 8,
        },
    }), encoding="utf-8")
    monkeypatch.setattr(window_plan_module, "_load_development_detector", lambda _args, _rule: (None, None))
    monkeypatch.setattr(window_plan_module, "HudParser", lambda **_kwargs: object())
    monkeypatch.setattr(window_plan_module, "generate_windows", lambda _manifest, _rule, frame_loader: [(0, 0)])
    windows = tmp_path / "evaluation_windows.json"
    assert window_plan_module.main([
        "--store-root", str(root),
        "--session-id", "sealed-session",
        "--session-id", "sealed-session-2",
        "--rule", str(rule_path),
        "--output", str(windows),
        "--detector-manifest", str(tmp_path / "detector.json"),
        "--detector-weights", str(tmp_path / "weights.pt"),
        "--detector-config", str(tmp_path / "detector.yaml"),
        "--class-map", str(tmp_path / "class_map.yaml"),
    ]) == 0
    descriptor = publish_capture_dataset(
        store,
        store_root=root,
        split_manifest_path=split.manifest_path,
        logical_prefix="survivors/capture/sealed-v1",
        evaluation_windows_path=windows,
        formal_gates_path=gates,
    )
    verify_capture_dataset_descriptor(store, descriptor, root)
    from types import SimpleNamespace
    from benchmark_survivors_perception import _capture_evaluation_inputs

    restored_capture = SimpleNamespace(
        descriptor=descriptor,
        files={ref.logical_id: store.object_path(ref.store_uri).read_bytes() for ref in descriptor.files},
    )
    restored_gates, restored_windows = _capture_evaluation_inputs(restored_capture)
    assert restored_gates.min_session_seconds == 300
    assert set(restored_windows.sessions) == {session.session_id for session in sealed_sessions}
    from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor

    invalid_identity = dict(descriptor.identity_metadata)
    invalid_evaluation_windows = dict(invalid_identity["evaluation_windows"])
    invalid_sessions = dict(invalid_evaluation_windows["sessions"])
    invalid_sessions["sealed-session"] = {"start_frame": 0, "end_frame": 0}
    invalid_evaluation_windows["sessions"] = invalid_sessions
    invalid_identity["evaluation_windows"] = invalid_evaluation_windows
    invalid_window_descriptor = ArtifactDescriptor(
        logical_id=descriptor.logical_id,
        node_kind=descriptor.node_kind,
        producer_id=descriptor.producer_id,
        producer_version=descriptor.producer_version,
        identity_metadata=invalid_identity,
        parents=descriptor.parents,
        files=descriptor.files,
        non_identity_metadata=descriptor.non_identity_metadata,
        schema_version=descriptor.schema_version,
    )
    with pytest.raises(PublishError, match="invalid evaluation window: sealed-session"):
        verify_capture_dataset_descriptor(store, invalid_window_descriptor, root)

    identity = descriptor.identity_metadata
    assert set(identity["annotation_bindings"]) == {session.session_id for session in sealed_sessions}
    assert set(identity["model_session_bindings"]) == {"model-session"}
    assert set(identity["annotation_bindings"]["sealed-session"]) == {
        "annotation_logical_id", "annotation_sha256", "session_manifest_logical_id", "session_manifest_sha256"
    }
    assert set(identity["hud_truth_bindings"]["model-session"]) == {
        "hud_truth_logical_id", "hud_truth_sha256"
    }
    for session in sealed_sessions:
        assert set(identity["frames_bindings"][session.session_id]) == {
            "frames_logical_id", "frames_sha256"
        }
    assert set(identity["coco_bindings"]) == {session.session_id for session in sealed_sessions}
    for session in sealed_sessions:
        assert set(identity["coco_bindings"][session.session_id]) == {
            "world_coco_logical_id", "world_coco_sha256", "ui_coco_logical_id", "ui_coco_sha256"
        }
    assert identity["formal_gates"]["schema_version"] == "formal_gates.v1"
    assert identity["evaluation_windows"] == {
        "sessions": {session.session_id: [[0, 0]] for session in sealed_sessions},
        "stride": 3,
        "plan_canonical_hash": json.loads(windows.read_text(encoding="utf-8"))["plan_canonical_hash"],
    }

    windows.write_text(json.dumps({
        "schema_version": "evaluation_windows.v1",
        "rule": {},
        "sessions": {"sealed-session": {"start_frame": 0, "end_frame": 0}},
        "stride": 1,
        "plan_canonical_hash": "c" * 64,
    }), encoding="utf-8")
    with pytest.raises(PublishError, match="invalid evaluation window: sealed-session"):
        publish_capture_dataset(
            store,
            store_root=root,
            split_manifest_path=split.manifest_path,
            logical_prefix="survivors/capture/sealed-v1",
            evaluation_windows_path=windows,
            formal_gates_path=gates,
        )

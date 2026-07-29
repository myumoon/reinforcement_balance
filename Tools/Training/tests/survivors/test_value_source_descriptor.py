"""Survivors value source descriptor の不変 identity と gate を検証する。

初心者向け:
モデル一式を後工程へ渡す前に、内容が同じなら同じ ID、内容が変われば別 ID になることを
確認し、検証結果やローカル環境だけでは ID が揺れないことを保証します。
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

_TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.value_source_descriptor import (
    ValueSourceDescriptorError,
    build_value_source_descriptor,
    finalize_value_source_descriptor,
    validate_value_source_descriptor,
    write_value_source_descriptor,
)


def _write_inputs(root: Path) -> tuple[Path, dict, dict]:
    """ready descriptor に必要な最小 fixture を作成する。

    初心者向け:
    実運用の run と source tree を小さな一時ディレクトリで再現します。
    """
    run_dir = root / "runs" / "source-run"
    source_root = root / "checkout"
    (run_dir / "result").mkdir(parents=True)
    (run_dir / "log").mkdir()
    source_root.mkdir()
    (run_dir / "result" / "model.zip").write_bytes(b"model-v1")
    (run_dir / "result" / "vecnormalize.pkl").write_bytes(b"vec-v1")
    (run_dir / "log" / "config_resolved.yaml").write_text(
        "frame_skip: 2\n", encoding="utf-8"
    )
    (run_dir / "log" / "package_freeze.txt").write_text(
        "numpy==1.26.4\npytest==9.0.3\n", encoding="utf-8"
    )
    code_paths = {}
    for name in (
        "cpp_logic",
        "cpp_base_reward",
        "python_reward",
        "hp_penalty",
        "noveld_config",
        "noveld_callback",
    ):
        path = source_root / f"{name}.txt"
        path.write_text(f"{name}-v1\n", encoding="utf-8")
        code_paths[name] = {"path": path.name}

    provenance = {
        "source_root": str(source_root),
        "dirty": False,
        "allow_dirty": False,
        "artifacts": {
            "model": {"path": "result/model.zip"},
            "vecnormalize": {"path": "result/vecnormalize.pkl"},
            "package_freeze": {"path": "log/package_freeze.txt"},
        },
        "model_spec": {
            "algorithm": "PPO",
            "policy": "MlpPolicy",
            "recurrent": False,
            "settings": {"frame_stack": 1},
        },
        "resolved_config": {"path": "log/config_resolved.yaml"},
        "code": code_paths,
        "runtime": {
            "action_semantics_version": "action_semantics.v1",
            "physics_dt": 1.0 / 60.0,
            "frame_skip": 2,
            "decision_hz": 30.0,
            "ordered_action_map": [
                "move_dx-1_dy-1",
                "move_dx0_dy-1",
                "move_dx1_dy-1",
            ],
        },
    }
    completion = {
        "item_stage_key": "IS2",
        "is2_complete": True,
        "weapon_coverage_count": 14,
        "passive_coverage_count": 17,
        "evolution_coverage_count": 14,
        "union_coverage_count": 1,
    }
    return run_dir, completion, provenance


def _obs_schema() -> dict:
    """順序付き observation schema fixture を返す。

    初心者向け:
    segment 順序もモデル入力契約なので identity の一部として扱います。
    """
    return {
        "obs_schema_hash": "ue5-schema-v1",
        "total_dim": 3,
        "segments": [
            {"name": "player", "dim": 1},
            {"name": "enemies", "dim": 2},
        ],
    }


def _build(
    run_dir: Path,
    completion: dict,
    provenance: dict,
    *,
    created_at_utc: str = "2026-07-29T00:00:00Z",
    obs_schema: dict | None = None,
) -> dict:
    """共通の固定入力で descriptor を構築する。

    初心者向け:
    テストごとの差分を一項目だけにして、何が identity を変えたか明確にします。
    """
    return build_value_source_descriptor(
        run_dir=run_dir,
        completion=completion,
        obs_schema=_obs_schema() if obs_schema is None else obs_schema,
        git_commit="a" * 40,
        created_at_utc=created_at_utc,
        source_provenance=provenance,
    )


def test_ready_descriptor_is_strict_and_contains_no_label_or_verdict_fields(
    tmp_path: Path,
) -> None:
    """全 artifact と provenance が揃った IS2 descriptor だけを ready にする。

    初心者向け:
    teacher 判定は後工程の子 artifact であり、source 自身へ逆参照させません。
    """
    run_dir, completion, provenance = _write_inputs(tmp_path)
    descriptor = _build(run_dir, completion, provenance)

    assert descriptor["schema_version"] == "survivors.value_source_descriptor.v1"
    assert descriptor["source_run"] == "source-run"
    assert descriptor["ready_for_probe"] is True
    assert descriptor["blocking_reasons"] == []
    assert descriptor["artifacts"]["model"]["path_relative"] == "result/model.zip"
    assert descriptor["observation_schema"]["total_dim"] == 3
    assert descriptor["runtime"]["ordered_action_map_sha256"]
    assert "ready_for_labels" not in json.dumps(descriptor)
    assert "teacher_validation" not in json.dumps(descriptor)
    assert "verdict" not in json.dumps(descriptor)

    invalid = copy.deepcopy(descriptor)
    invalid["teacher_verdict"] = {"passed": True}
    with pytest.raises(ValueSourceDescriptorError, match="unknown"):
        validate_value_source_descriptor(invalid)


def test_identity_ignores_clock_paths_and_gate_results_but_binds_content(
    tmp_path: Path,
) -> None:
    """volatile metadata を除外し、全 immutable content を identity に束縛する。

    初心者向け:
    コピー先や監査時刻は同一性を変えず、モデル・設定・コード・action の変更は必ず
    identity を変えることを反例で確認します。
    """
    run_a, completion_a, provenance_a = _write_inputs(tmp_path / "a")
    first = _build(run_a, completion_a, provenance_a)
    run_b, completion_b, provenance_b = _write_inputs(tmp_path / "b")
    second = _build(
        run_b,
        completion_b,
        provenance_b,
        created_at_utc="2030-01-01T00:00:00Z",
    )
    assert first["identity_sha256"] == second["identity_sha256"]

    run_b.joinpath("result/model.zip").write_bytes(b"model-v2")
    assert _build(run_b, completion_b, provenance_b)["identity_sha256"] != first["identity_sha256"]
    run_b.joinpath("result/model.zip").write_bytes(b"model-v1")

    run_b.joinpath("log/config_resolved.yaml").write_text("frame_skip: 3\n", encoding="utf-8")
    assert _build(run_b, completion_b, provenance_b)["identity_sha256"] != first["identity_sha256"]
    run_b.joinpath("log/config_resolved.yaml").write_text("frame_skip: 2\n", encoding="utf-8")

    source_root = Path(provenance_b["source_root"])
    source_root.joinpath("cpp_logic.txt").write_text("cpp_logic-v2\n", encoding="utf-8")
    assert _build(run_b, completion_b, provenance_b)["identity_sha256"] != first["identity_sha256"]
    source_root.joinpath("cpp_logic.txt").write_text("cpp_logic-v1\n", encoding="utf-8")

    changed_obs = _obs_schema()
    changed_obs["segments"][0]["name"] = "player_v2"
    assert (
        _build(
            run_b,
            completion_b,
            provenance_b,
            obs_schema=changed_obs,
        )["identity_sha256"]
        != first["identity_sha256"]
    )

    provenance_b["runtime"]["ordered_action_map"].append("move_dx-1_dy0")
    assert _build(run_b, completion_b, provenance_b)["identity_sha256"] != first["identity_sha256"]


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (lambda run, done, source: done.update(is2_complete=False), "is2_incomplete"),
        (lambda run, done, source: run.joinpath("result/model.zip").unlink(), "model_missing"),
        (
            lambda run, done, source: run.joinpath("result/vecnormalize.pkl").unlink(),
            "vecnormalize_missing",
        ),
        (
            lambda run, done, source: run.joinpath(
                "log/config_resolved.yaml"
            ).unlink(),
            "resolved_config_hash_unknown",
        ),
        (lambda run, done, source: source.update(code={}), "code_hash_unknown:"),
        (
            lambda run, done, source: source.update(runtime={}),
            "action_semantics_version_unknown",
        ),
    ],
)
def test_missing_completion_artifacts_or_provenance_is_not_ready(
    tmp_path: Path,
    mutation,
    expected_reason: str,
) -> None:
    """probe gate の全 blocking sibling を fail-closed にする。

    初心者向け:
    一項目でも証明できなければ ready と推測せず、理由を機械可読に残します。
    """
    run_dir, completion, provenance = _write_inputs(tmp_path)
    mutation(run_dir, completion, provenance)
    descriptor = _build(run_dir, completion, provenance)
    assert descriptor["ready_for_probe"] is False
    assert any(reason.startswith(expected_reason) for reason in descriptor["blocking_reasons"])


def test_unknown_observation_and_action_hashes_are_both_blocking(
    tmp_path: Path,
) -> None:
    """schema と action hash の両契約を対称に fail-closed にする。

    初心者向け:
    model 入力か出力のどちらか一方だけが不明でも probe ready にはしません。
    """
    run_dir, completion, provenance = _write_inputs(tmp_path)
    provenance["runtime"]["ordered_action_map"] = []
    descriptor = _build(
        run_dir,
        completion,
        provenance,
        obs_schema={},
    )
    assert descriptor["ready_for_probe"] is False
    assert "obs_schema_hash_unknown" in descriptor["blocking_reasons"]
    assert "ordered_action_map_hash_unknown" in descriptor["blocking_reasons"]


def test_dirty_source_is_rejected_by_default_and_patch_is_stored_when_allowed(
    tmp_path: Path,
) -> None:
    """dirty source の黙認を禁止し、許可時は patch hash を identity に含める。

    初心者向け:
    commit にない修正も patch artifact として固定し、同じ commit の別内容を区別します。
    """
    run_dir, completion, provenance = _write_inputs(tmp_path)
    provenance.update(
        dirty=True,
        patch_text="diff --git a/source.py b/source.py\n+change\n",
        artifact_store_root=str(tmp_path / "artifact-store"),
    )
    with pytest.raises(ValueSourceDescriptorError, match="dirty"):
        _build(run_dir, completion, provenance)

    provenance["allow_dirty"] = True
    descriptor = _build(run_dir, completion, provenance)
    patch_ref = descriptor["patch_artifact_ref"]
    assert patch_ref["sha256"]
    assert patch_ref["store_uri"].endswith(patch_ref["sha256"])
    assert descriptor["ready_for_probe"] is True

    changed = copy.deepcopy(provenance)
    changed["patch_text"] += "+another-change\n"
    assert _build(run_dir, completion, changed)["identity_sha256"] != descriptor["identity_sha256"]


def test_write_validates_then_atomically_replaces_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """temp file から os.replace する atomic publish を検証する。

    初心者向け:
    書き込み途中の JSON が result に見える時間を作らないようにします。
    """
    run_dir, completion, provenance = _write_inputs(tmp_path)
    descriptor = _build(run_dir, completion, provenance)
    calls: list[tuple[Path, Path]] = []

    import games.survivors.value_source_descriptor as module

    real_replace = module.os.replace

    def recording_replace(source, destination):
        calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", recording_replace)
    destination = write_value_source_descriptor(run_dir, descriptor)
    assert destination == run_dir / "result" / "value_source_descriptor.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == descriptor
    assert calls and calls[-1][0] != destination and calls[-1][1] == destination
    assert not list(destination.parent.glob(".value_source_descriptor.*.tmp"))

    changed_metadata = copy.deepcopy(descriptor)
    changed_metadata["created_at_utc"] = "2030-01-01T00:00:00Z"
    validate_value_source_descriptor(changed_metadata)
    with pytest.raises(ValueSourceDescriptorError, match="different immutable"):
        write_value_source_descriptor(run_dir, changed_metadata)
    assert json.loads(destination.read_text(encoding="utf-8")) == descriptor


@pytest.mark.parametrize("exit_reason", ["keyboard_interrupt", "exception"])
def test_finalize_keeps_incomplete_out_of_result_on_abnormal_exit(
    tmp_path: Path,
    exit_reason: str,
) -> None:
    """SIGINT / 例外の全 sibling 経路で descriptor 昇格を禁止する。

    初心者向け:
    model が保存できていても正常な IS2 完了でなければ log marker だけを残します。
    """
    run_dir, completion, provenance = _write_inputs(tmp_path)
    destination = finalize_value_source_descriptor(
        run_dir=run_dir,
        exit_reason=exit_reason,
        final_model_zip=run_dir / "result" / "model.zip",
        completion=completion,
        obs_schema=_obs_schema(),
        git_commit="a" * 40,
        created_at_utc="2026-07-29T00:00:00Z",
        source_provenance=provenance,
    )
    assert destination is None
    assert not (run_dir / "result" / "value_source_descriptor.json").exists()
    incomplete = json.loads(
        (run_dir / "log" / "value_source_descriptor.incomplete.json").read_text(
            encoding="utf-8"
        )
    )
    assert incomplete["reason"] == exit_reason


def test_finalize_publishes_only_curriculum_complete_with_model(
    tmp_path: Path,
) -> None:
    """IS2 正常終了と model 存在を同時に要求する。

    初心者向け:
    同じ completion 入力でも model 欠落は incomplete、存在時だけ result publish になります。
    """
    missing_run, completion, provenance = _write_inputs(tmp_path / "missing")
    missing_model = missing_run / "result" / "model.zip"
    missing_model.unlink()
    assert finalize_value_source_descriptor(
        run_dir=missing_run,
        exit_reason="curriculum_complete",
        final_model_zip=missing_model,
        completion=completion,
        obs_schema=_obs_schema(),
        git_commit="a" * 40,
        created_at_utc="2026-07-29T00:00:00Z",
        source_provenance=provenance,
    ) is None
    assert not (
        missing_run / "result" / "value_source_descriptor.json"
    ).exists()

    ready_run, ready_completion, ready_provenance = _write_inputs(
        tmp_path / "ready"
    )
    destination = finalize_value_source_descriptor(
        run_dir=ready_run,
        exit_reason="curriculum_complete",
        final_model_zip=ready_run / "result" / "model.zip",
        completion=ready_completion,
        obs_schema=_obs_schema(),
        git_commit="a" * 40,
        created_at_utc="2026-07-29T00:00:00Z",
        source_provenance=ready_provenance,
    )
    assert destination == ready_run / "result" / "value_source_descriptor.json"
    assert not (
        ready_run / "log" / "value_source_descriptor.incomplete.json"
    ).exists()

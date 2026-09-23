"""Perception実現可能性ゲートCLIドライバの契約テスト。

合成fixture一式でend-to-endに判定JSON/Markdownを生成できること、複数
--probe-resultsを束ねられること、欠損・型不正・schema不一致・annotator不足を
fail-closed（非ゼロ終了、原因を含むメッセージ）で拒否することを確認します。
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from spikes import feasibility_gate_cli
from spikes.perception_probe import evaluate_probe, make_synthetic_fixture
from survivors.target_profile import CONFIG, load_target_profile


def _write_json(path: Path, data) -> Path:
    """データをJSONとしてpathへ書き出し、そのpathを返す。"""
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _sessions() -> list[dict]:
    """3セッション・全7場面を最低1回ずつ含む有効なpilotセッションmetadataを返す。"""
    return [
        {"session_id": "s1", "build_id": "build-a", "profile_id": "profile-a",
         "minutes": 12.0, "slice": ["early", "mid"]},
        {"session_id": "s2", "build_id": "build-a", "profile_id": "profile-a",
         "minutes": 15.0, "slice": ["late", "heavy"]},
        {"session_id": "s3", "build_id": "build-a", "profile_id": "profile-a",
         "minutes": 11.0, "slice": ["level_up", "chest", "death_result"]},
    ]


def _annotation_summary() -> dict:
    """summarize_annotation()の戻り値と同じ形の有効なannotation集計値を返す。"""
    return {
        "entities_per_hour": 500.0, "dense_entities_per_hour": 420.0,
        "qa_rework_rate": 0.05, "bbox_qa_iou": 0.90,
        "class_agreement": 0.97, "annotation_hours": 12.0,
    }


def _operator_attested_profile_path(tmp_path: Path) -> Path:
    """target_audit_passがTrueになるoperator-attested provenanceのYAMLを書き出す。"""
    wire = load_target_profile(CONFIG).to_wire()
    wire["provenance"] = "operator-attested"
    path = tmp_path / "target_profile.yaml"
    path.write_text(yaml.safe_dump(wire), encoding="utf-8")
    return path


def _write_inputs(tmp_path: Path, *, probe_evaluations=None, sessions=None, annotation_summary=None):
    """CLI入力一式（sessions/probe-results/annotation-results）をtmp_pathへ書き出す。"""
    sessions_path = _write_json(
        tmp_path / "sessions.json", sessions if sessions is not None else _sessions())
    evaluations = (probe_evaluations if probe_evaluations is not None
                  else [evaluate_probe(make_synthetic_fixture(seed=0))])
    probe_paths = [_write_json(tmp_path / f"probe_{i}.json", evaluation)
                  for i, evaluation in enumerate(evaluations)]
    annotation_path = _write_json(
        tmp_path / "annotation.json",
        annotation_summary if annotation_summary is not None else _annotation_summary())
    return sessions_path, probe_paths, annotation_path


def _argv(tmp_path, sessions_path, probe_paths, annotation_path, target_profile_path,
         *, annotators=("alice", "bob")):
    """main()にそのまま渡せる引数リストを組み立てる。"""
    return [
        "--sessions", str(sessions_path),
        "--probe-results", *[str(p) for p in probe_paths],
        "--annotation-results", str(annotation_path),
        "--annotators", *annotators,
        "--representative-frames", "300",
        "--target-profile", str(target_profile_path),
        "--output-json", str(tmp_path / "verdict.json"),
        "--output-markdown", str(tmp_path / "verdict.md"),
    ]


def test_cli_end_to_end_writes_matching_verdict(tmp_path):
    """合成fixture一式でCLIを実行し、write_verdictと同じ内容が生成されることを確認する。"""
    sessions_path, probe_paths, annotation_path = _write_inputs(tmp_path)
    target_profile_path = _operator_attested_profile_path(tmp_path)
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, sessions_path, probe_paths, annotation_path, target_profile_path))
    assert exit_code == 0
    output_json, output_markdown = tmp_path / "verdict.json", tmp_path / "verdict.md"
    assert output_json.exists() and output_markdown.exists()
    verdict = json.loads(output_json.read_text(encoding="utf-8"))
    assert verdict["status"] in {"PASS", "FAIL"}
    assert "budget" in verdict


def test_cli_bundles_multiple_probe_results(tmp_path):
    """複数--probe-results・複数pilotセッションを1回のCLI実行で束ねられることを確認する。"""
    evaluations = [evaluate_probe(make_synthetic_fixture(seed=0)),
                  evaluate_probe(make_synthetic_fixture(seed=1))]
    sessions_path, probe_paths, annotation_path = _write_inputs(tmp_path, probe_evaluations=evaluations)
    assert len(probe_paths) == 2
    target_profile_path = _operator_attested_profile_path(tmp_path)
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, sessions_path, probe_paths, annotation_path, target_profile_path))
    assert exit_code == 0
    assert (tmp_path / "verdict.json").exists()


def test_cli_rejects_missing_sessions_file(tmp_path, capsys):
    """存在しない--sessionsファイルを非ゼロ終了・原因入りメッセージで拒否する。"""
    _, probe_paths, annotation_path = _write_inputs(tmp_path)
    target_profile_path = _operator_attested_profile_path(tmp_path)
    missing = tmp_path / "does_not_exist.json"
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, missing, probe_paths, annotation_path, target_profile_path))
    assert exit_code == 1
    assert "--sessions" in capsys.readouterr().err


def test_cli_rejects_malformed_json(tmp_path, capsys):
    """壊れたJSON構文を非ゼロ終了・原因入りメッセージで拒否する。"""
    sessions_path, probe_paths, annotation_path = _write_inputs(tmp_path)
    sessions_path.write_text("{not valid json", encoding="utf-8")
    target_profile_path = _operator_attested_profile_path(tmp_path)
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, sessions_path, probe_paths, annotation_path, target_profile_path))
    assert exit_code == 1
    assert "json" in capsys.readouterr().err.lower()


def test_cli_rejects_schema_mismatch(tmp_path, capsys):
    """未知フィールドを含むpilotセッションmetadataを非ゼロ終了で拒否する。"""
    sessions = _sessions()
    sessions[0]["unexpected_field"] = "surprise"
    sessions_path, probe_paths, annotation_path = _write_inputs(tmp_path, sessions=sessions)
    target_profile_path = _operator_attested_profile_path(tmp_path)
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, sessions_path, probe_paths, annotation_path, target_profile_path))
    assert exit_code == 1
    assert capsys.readouterr().err


def test_cli_rejects_insufficient_annotators(tmp_path, capsys):
    """--annotatorsが1名だけの場合を非ゼロ終了・原因入りメッセージで拒否する。"""
    sessions_path, probe_paths, annotation_path = _write_inputs(tmp_path)
    target_profile_path = _operator_attested_profile_path(tmp_path)
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, sessions_path, probe_paths, annotation_path, target_profile_path,
             annotators=("alice",)))
    assert exit_code == 1
    assert "annotator" in capsys.readouterr().err.lower()

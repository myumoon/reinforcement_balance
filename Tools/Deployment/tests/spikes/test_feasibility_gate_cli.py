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
from spikes.pilot_evidence import build_gate_evidence
from spikes.survivors_vertical_feasibility import _REQUIRED_SLICES, write_verdict
from survivors.target_profile import CONFIG, load_target_profile


def _write_json(path: Path, data) -> Path:
    """データをJSONとしてpathへ書き出し、そのpathを返す。

    CLI入力ファイル一式を組み立てる各ヘルパーが共通で使う。
    """
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _sessions() -> list[dict]:
    """3セッション・全7場面を毎回含む有効なpilotセッションmetadataを返す。

    ゲート閾値min_sessions_per_slice(=2)を満たすには各場面が最低2セッションに
    現れる必要があるため、単純化のため全セッションが全7場面を含む形にしている。
    """
    return [
        {"session_id": "s1", "build_id": "build-a", "profile_id": "profile-a",
         "minutes": 12.0, "slice": list(_REQUIRED_SLICES)},
        {"session_id": "s2", "build_id": "build-a", "profile_id": "profile-a",
         "minutes": 15.0, "slice": list(_REQUIRED_SLICES)},
        {"session_id": "s3", "build_id": "build-a", "profile_id": "profile-a",
         "minutes": 11.0, "slice": list(_REQUIRED_SLICES)},
    ]


def _annotation_summary() -> dict:
    """summarize_annotation()の戻り値と同じ形の有効なannotation集計値を返す。

    entities_per_hour等6項目すべてを含む、閾値をすべて満たす値にしている。
    """
    return {
        "entities_per_hour": 500.0, "dense_entities_per_hour": 420.0,
        "qa_rework_rate": 0.05, "bbox_qa_iou": 0.90,
        "class_agreement": 0.97, "annotation_hours": 12.0,
    }


def _operator_attested_profile_path(tmp_path: Path) -> Path:
    """target_audit_passがTrueになるoperator-attested provenanceのYAMLを書き出す。

    tracked設定ファイルのprovenanceだけをその場で上書きした複製を書き出す。
    """
    wire = load_target_profile(CONFIG).to_wire()
    wire["provenance"] = "operator-attested"
    path = tmp_path / "target_profile.yaml"
    path.write_text(yaml.safe_dump(wire), encoding="utf-8")
    return path


def _write_inputs(tmp_path: Path, *, probe_evaluations=None, sessions=None, annotation_summary=None):
    """CLI入力一式（sessions/probe-results/annotation-results）をtmp_pathへ書き出す。

    各引数を渡さなければ、有効な既定fixtureがそのまま使われる。
    """
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
    """main()にそのまま渡せる引数リストを組み立てる。

    出力先は毎回tmp_path直下のverdict.json/verdict.mdに固定する。
    """
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
    """合成fixture一式でCLIを実行し、判定JSON/Markdownが生成されることを確認する。

    全7場面を含む現在のfixtureは真にPASS経路を通るため、statusをPASSまで確認する。
    """
    sessions_path, probe_paths, annotation_path = _write_inputs(tmp_path)
    target_profile_path = _operator_attested_profile_path(tmp_path)
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, sessions_path, probe_paths, annotation_path, target_profile_path))
    assert exit_code == 0
    output_json, output_markdown = tmp_path / "verdict.json", tmp_path / "verdict.md"
    assert output_json.exists() and output_markdown.exists()
    verdict = json.loads(output_json.read_text(encoding="utf-8"))
    assert verdict["status"] == "PASS"
    assert "budget" in verdict


def test_cli_output_matches_direct_write_verdict_call(tmp_path):
    """CLI出力が、同じ入力でwrite_verdict()を直接呼んだ場合と同一内容になることを確認する。

    CLIは判定ロジックに一切手を加えない薄いドライバであることを、出力内容の
    完全一致（JSON構造とMarkdown本文の両方）で裏付ける回帰テスト。
    """
    sessions_path, probe_paths, annotation_path = _write_inputs(tmp_path)
    target_profile_path = _operator_attested_profile_path(tmp_path)
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, sessions_path, probe_paths, annotation_path, target_profile_path))
    assert exit_code == 0

    evidence = build_gate_evidence(
        sessions=_sessions(),
        probe_evaluations=[evaluate_probe(make_synthetic_fixture(seed=0))],
        annotation_summary=_annotation_summary(),
        annotators=("alice", "bob"),
        representative_frames=300,
        target_profile=load_target_profile(target_profile_path),
    )
    direct_json, direct_markdown = tmp_path / "direct.json", tmp_path / "direct.md"
    write_verdict(evidence, direct_json, direct_markdown)

    cli_output = json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))
    direct_output = json.loads(direct_json.read_text(encoding="utf-8"))
    assert cli_output == direct_output
    assert (tmp_path / "verdict.md").read_text(encoding="utf-8") == direct_markdown.read_text(encoding="utf-8")


def test_cli_bundles_multiple_probe_results(tmp_path):
    """複数--probe-results・複数pilotセッションを1回のCLI実行で束ねられることを確認する。

    seedの異なる2件のprobe結果を渡し、正常に統合されて終了コード0になることを確認する。
    """
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
    """存在しない--sessionsファイルを非ゼロ終了・原因入りメッセージで拒否する。

    生のFileNotFoundErrorではなく、原因のわかるメッセージへ変換されることを確認する。
    """
    _, probe_paths, annotation_path = _write_inputs(tmp_path)
    target_profile_path = _operator_attested_profile_path(tmp_path)
    missing = tmp_path / "does_not_exist.json"
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, missing, probe_paths, annotation_path, target_profile_path))
    assert exit_code == 1
    assert "--sessions" in capsys.readouterr().err


def test_cli_rejects_malformed_json(tmp_path, capsys):
    """壊れたJSON構文を非ゼロ終了・原因入りメッセージで拒否する。

    json.JSONDecodeErrorが生のまま伝播せず、ContractValidationErrorへ変換されることを確認する。
    """
    sessions_path, probe_paths, annotation_path = _write_inputs(tmp_path)
    sessions_path.write_text("{not valid json", encoding="utf-8")
    target_profile_path = _operator_attested_profile_path(tmp_path)
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, sessions_path, probe_paths, annotation_path, target_profile_path))
    assert exit_code == 1
    assert "json" in capsys.readouterr().err.lower()


def test_cli_rejects_schema_mismatch(tmp_path, capsys):
    """未知フィールドを含むpilotセッションmetadataを非ゼロ終了で拒否する。

    --sessions側の閉じたschema検査がCLI経由でも効いていることを確認する。
    """
    sessions = _sessions()
    sessions[0]["unexpected_field"] = "surprise"
    sessions_path, probe_paths, annotation_path = _write_inputs(tmp_path, sessions=sessions)
    target_profile_path = _operator_attested_profile_path(tmp_path)
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, sessions_path, probe_paths, annotation_path, target_profile_path))
    assert exit_code == 1
    assert capsys.readouterr().err


def test_cli_rejects_probe_results_schema_mismatch(tmp_path, capsys):
    """未知フィールドを含む--probe-resultsを非ゼロ終了で拒否する。

    これまでCLIレベルでは--sessionsのschema不一致しか確認していなかったため、
    他の入力ファイルでも同様にfail-closedであることを確認する。
    """
    evaluation = evaluate_probe(make_synthetic_fixture(seed=0))
    evaluation["unexpected_field"] = "surprise"
    sessions_path, probe_paths, annotation_path = _write_inputs(
        tmp_path, probe_evaluations=[evaluation])
    target_profile_path = _operator_attested_profile_path(tmp_path)
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, sessions_path, probe_paths, annotation_path, target_profile_path))
    assert exit_code == 1
    assert capsys.readouterr().err


def test_cli_rejects_annotation_results_schema_mismatch(tmp_path, capsys):
    """未知フィールドを含む--annotation-resultsを非ゼロ終了で拒否する。

    sessions以外の入力でもfail-closedにschema検証が効いていることを確認する。
    """
    annotation_summary = _annotation_summary()
    annotation_summary["unexpected_field"] = "surprise"
    sessions_path, probe_paths, annotation_path = _write_inputs(
        tmp_path, annotation_summary=annotation_summary)
    target_profile_path = _operator_attested_profile_path(tmp_path)
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, sessions_path, probe_paths, annotation_path, target_profile_path))
    assert exit_code == 1
    assert capsys.readouterr().err


def test_cli_rejects_malformed_target_profile_yaml(tmp_path, capsys):
    """--target-profileのYAML構文が壊れている場合を非ゼロ終了で拒否する。

    以前はyaml.YAMLErrorが生のトレースバックとして伝播していた回帰を防ぐ。
    """
    sessions_path, probe_paths, annotation_path = _write_inputs(tmp_path)
    target_profile_path = tmp_path / "broken_target_profile.yaml"
    target_profile_path.write_text("key: [unclosed", encoding="utf-8")
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, sessions_path, probe_paths, annotation_path, target_profile_path))
    assert exit_code == 1
    assert capsys.readouterr().err


def test_cli_rejects_insufficient_annotators(tmp_path, capsys):
    """--annotatorsが1名だけの場合を非ゼロ終了・原因入りメッセージで拒否する。

    2名以上必須という入力形状の不備が、CLI経由でも拒否されることを確認する。
    """
    sessions_path, probe_paths, annotation_path = _write_inputs(tmp_path)
    target_profile_path = _operator_attested_profile_path(tmp_path)
    exit_code = feasibility_gate_cli.main(
        _argv(tmp_path, sessions_path, probe_paths, annotation_path, target_profile_path,
             annotators=("alice",)))
    assert exit_code == 1
    assert "annotator" in capsys.readouterr().err.lower()

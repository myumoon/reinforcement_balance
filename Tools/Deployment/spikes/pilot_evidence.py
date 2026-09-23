"""実測pilot入力からPerception実現可能性ゲートの判定証拠を組み立てる変換層。

`perception_probe.evaluate_probe()` や `annotation_throughput.summarize_annotation()` の
戻り値は、判定本体が要求する `GateEvidence` の形（フラットな `p10_short_side_px` や
`architecture_metrics[name]["utility"]` など）とキー名・ネスト構造が一致しません。
このモジュールは、そのキー名の付け替えと複数pilotセッションの集約だけを行う薄い層です。
`GateEvidence` / `issue_verdict` / `write_verdict` 側の判定ロジックには一切手を加えません。
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from reinbalance_survivors_contracts.ui_intent import ensure, is_strict_number
from spikes.survivors_vertical_feasibility import GateEvidence, _REQUIRED_SLICES
from survivors.target_profile import TargetProfile

# GateEvidence.validate() が要求する既定の方式集合と同じ4方式（既存コードの
# ハードコードされた既定値に合わせており、perception_probe._ARCH_KEYS を
# import はしない。既存モジュール同士も同じ値を別々に定数化しているため、
# 既存の慣習に合わせる）。
_ARCHITECTURES = frozenset({"ssdlite320", "ssdlite640_multiscale", "tile_2x2", "coarse_density"})
# GateEvidence.single_pass_p95_ms は「単発・非タイル分割」検出方式の遅延を表す。
# 4方式のうちタイル分割しないのは ssdlite320 だけなので、その遅延をそのまま使う。
_SINGLE_PASS_ARCHITECTURE = "ssdlite320"

_SESSION_KEYS = frozenset({"session_id", "build_id", "profile_id", "minutes", "slice"})
_ANNOTATION_SUMMARY_KEYS = frozenset({
    "entities_per_hour", "dense_entities_per_hour", "qa_rework_rate",
    "bbox_qa_iou", "class_agreement", "annotation_hours",
})
_PROBE_EVAL_KEYS = frozenset({
    "pixel_size", "recall_upper_bound", "latency_ms", "annotation_seconds", "architectures",
})
_PIXEL_SIZE_KEYS = frozenset({"p10_short_side", "median_short_side"})
# evaluate_probe()がrecall_upper_bound（trunk）へ常に返す場面名の全集合。関数内部の
# ローカル変数として定義されているため公開importができず、_ARCHITECTURESと同様に
# ここで別途定数化する。
_PROBE_SLICE_NAMES = frozenset({"small", "occluded", "late", "heavy", "boss", "gem"})
# evaluate_probe()がarchitectures[name]へ常に返すキー全集合。
_ARCHITECTURE_METRIC_KEYS = frozenset({
    "latency_p95_ms", "utility_per_latency", "recall_upper_bound", "slice_metrics",
})


def _closed(mapping: Any, keys: frozenset, label: str) -> Mapping[str, Any]:
    """マッピングのキー集合が期待どおり（過不足なし）であることを検証する。

    JSONから読み込んだ値は型もキーも保証されないため、想定外のキーや
    欠落したキーを、原因のわかるメッセージ付きで早期に拒否します。
    """
    ensure(isinstance(mapping, Mapping), f"{label} must be a JSON object")
    missing, unknown = keys - set(mapping), set(mapping) - keys
    ensure(not missing, f"{label} is missing fields: {sorted(missing)}")
    ensure(not unknown, f"{label} has unknown fields: {sorted(unknown)}")
    return mapping


def _validate_session(session: Any, index: int) -> Mapping[str, Any]:
    """pilotセッションmetadata1件分の形と値を検証する。

    session_id等の文字列項目が空でないこと、minutesが有限非負数であること、
    sliceが既知の場面名を1つ以上持つことを確認します。
    """
    label = f"pilot session[{index}]"
    _closed(session, _SESSION_KEYS, label)
    for key in ("session_id", "build_id", "profile_id"):
        ensure(isinstance(session[key], str) and session[key], f"{label}.{key} must be non-empty")
    minutes = session["minutes"]
    ensure(is_strict_number(minutes) and math.isfinite(minutes) and minutes >= 0,
           f"{label}.minutes must be a finite non-negative number")
    slices = session["slice"]
    ensure(isinstance(slices, list) and slices and all(isinstance(s, str) for s in slices),
           f"{label}.slice must be a non-empty list of strings")
    ensure(all(s in _REQUIRED_SLICES for s in slices),
           f"{label}.slice contains unknown slice name (known: {sorted(_REQUIRED_SLICES)})")
    return session


def _finite_number(value: Any, label: str) -> float:
    """値が真の有限数（int/float、boolは不可）であることを検証し、そのまま返す。

    boolはPythonではintのサブクラスなので、この検証を経ずにsum()/mean()へ渡すと
    True/Falseが1.0/0.0へ黙って変換され、GateEvidence.validate()側のbool拒否を
    すり抜けてしまう。算術に使う前に必ずこの関数を通すことでそれを防ぐ。
    """
    ensure(is_strict_number(value) and math.isfinite(value), f"{label} must be a finite number")
    return value


def merge_probe_evaluations(evaluations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """複数の`evaluate_probe()`結果をGateEvidence用のフラットな指標へ変換・統合する。

    pilotは複数セッションに分けて実施されうるため、`--probe-results`は複数渡せます。
    各ファイルはevaluate_probe()の集計済み出力（生の観測ではない）なので、標本へ
    戻しての再集計はできません。ここでは対応する数値指標の単純平均を統合方針とします
    （ponytail: セッション時間や標本数による重み付けはしない。将来必要になれば
    各評価にサンプル数を持たせて加重平均へ拡張する）。
    utilityへの変換は、evaluate_probe()自身の定義
    `utility_per_latency = mean(oracle_detectable) / latency_p95_ms` の逆算として
    `utility = utility_per_latency * latency_p95_ms` を採用する
    （これは常に[0,1]範囲の割合に戻るはずの値で、範囲外ならGateEvidence.validate()が
    fail-closedで拒否する）。
    """
    values = tuple(evaluations)
    ensure(values, "at least one probe evaluation is required")
    for index, evaluation in enumerate(values):
        label = f"probe evaluation[{index}]"
        _closed(evaluation, _PROBE_EVAL_KEYS, label)
        _closed(evaluation["pixel_size"], _PIXEL_SIZE_KEYS, f"{label}.pixel_size")
        _finite_number(evaluation["pixel_size"]["p10_short_side"],
                       f"{label}.pixel_size.p10_short_side")
        _closed(evaluation["recall_upper_bound"], _PROBE_SLICE_NAMES, f"{label}.recall_upper_bound")
        for slice_name in _PROBE_SLICE_NAMES:
            _finite_number(evaluation["recall_upper_bound"][slice_name],
                           f"{label}.recall_upper_bound.{slice_name}")
        ensure(set(evaluation["architectures"]) == _ARCHITECTURES,
               f"{label}.architectures must contain exactly {sorted(_ARCHITECTURES)}")
        for name in _ARCHITECTURES:
            metrics = evaluation["architectures"][name]
            _closed(metrics, _ARCHITECTURE_METRIC_KEYS, f"{label}.architectures[{name!r}]")
            _finite_number(metrics["latency_p95_ms"],
                           f"{label}.architectures[{name!r}].latency_p95_ms")
            _finite_number(metrics["utility_per_latency"],
                           f"{label}.architectures[{name!r}].utility_per_latency")

    def mean(numbers: Sequence[float]) -> float:
        return sum(numbers) / len(numbers)

    architecture_metrics = {}
    for name in _ARCHITECTURES:
        latencies = [ev["architectures"][name]["latency_p95_ms"] for ev in values]
        utilities = [ev["architectures"][name]["utility_per_latency"] * latency
                     for ev, latency in zip(values, latencies)]
        architecture_metrics[name] = {
            "utility": mean(utilities),
            "latency_p95_ms": mean(latencies),
        }

    return {
        "p10_short_side_px": mean([ev["pixel_size"]["p10_short_side"] for ev in values]),
        "late_recall": mean([ev["recall_upper_bound"]["late"] for ev in values]),
        "heavy_recall": mean([ev["recall_upper_bound"]["heavy"] for ev in values]),
        "single_pass_p95_ms": mean(
            [ev["architectures"][_SINGLE_PASS_ARCHITECTURE]["latency_p95_ms"] for ev in values]),
        "architecture_metrics": architecture_metrics,
    }


def build_gate_evidence(
    *,
    sessions: Sequence[Mapping[str, Any]],
    probe_evaluations: Sequence[Mapping[str, Any]],
    annotation_summary: Mapping[str, Any],
    annotators: Sequence[str],
    representative_frames: int,
    target_profile: TargetProfile,
    unresolved_risks: Sequence[str] = (),
) -> GateEvidence:
    """pilotの実測入力一式を検証し、実行可能な`GateEvidence`を1つに組み立てる。

    pilotセッションmetadata・probe評価・annotation集計・TargetProfileの4種類の
    入力から、GateEvidenceが要求する全フィールドを導出します。導出したGateEvidenceは
    構築時に自身の`.validate()`を通るため、ここでは重複するチェックを増やさず、
    GateEvidence自身では検証できない入力形状（annotator数など）だけを追加で検証します。
    """
    ensure(isinstance(sessions, Sequence) and not isinstance(sessions, (str, bytes)) and sessions,
           "at least one pilot session is required")
    for index, session in enumerate(sessions):
        _validate_session(session, index)
    ensure(isinstance(target_profile, TargetProfile), "target_profile must be a TargetProfile")
    # independent_annotators の重複（例: alice, alice）はGateEvidence構築後にissue_verdict()が
    # 判定FAILとして扱う既存の業務仕様（test_perception_probe.pyで確認済み）なのでここでは
    # 弾かない。ここで弾くのは「そもそも人数が足りない」という入力形状の不備だけ。
    ensure(len(annotators) >= 2, "at least two independent annotators are required")
    _closed(annotation_summary, _ANNOTATION_SUMMARY_KEYS, "annotation summary")

    probe_metrics = merge_probe_evaluations(probe_evaluations)
    session_ids = tuple(s["session_id"] for s in sessions)
    build_ids = tuple(sorted({s["build_id"] for s in sessions}))
    profile_ids = tuple(sorted({s["profile_id"] for s in sessions}))
    session_minutes = {s["session_id"]: float(s["minutes"]) for s in sessions}
    slice_counts = {name: sum(1 for s in sessions if name in s["slice"]) for name in _REQUIRED_SLICES}

    return GateEvidence(
        session_ids=session_ids,
        build_ids=build_ids,
        profile_ids=profile_ids,
        target_audit_pass=(target_profile.provenance == "operator-attested"),
        session_minutes=session_minutes,
        slice_counts=slice_counts,
        representative_frames=representative_frames,
        independent_annotators=tuple(annotators),
        p10_short_side_px=probe_metrics["p10_short_side_px"],
        late_recall=probe_metrics["late_recall"],
        heavy_recall=probe_metrics["heavy_recall"],
        single_pass_p95_ms=probe_metrics["single_pass_p95_ms"],
        bbox_qa_iou=annotation_summary["bbox_qa_iou"],
        class_agreement=annotation_summary["class_agreement"],
        dense_entities_per_hour=annotation_summary["dense_entities_per_hour"],
        architecture_metrics=probe_metrics["architecture_metrics"],
        unresolved_risks=tuple(unresolved_risks),
    )

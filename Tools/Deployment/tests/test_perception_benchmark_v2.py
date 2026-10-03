"""perception benchmark の DeployObs v2 対応（04-13）を検証する。

weapon 4クラスの formal slice、出現数依存 slice の「該当なし」規則と final verdict への記録、
v2 新 segment の誤差指標と calibration profile への反映を確かめます。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.deploy_obs_v2_features import HudSlot, TrackPx, build_deploy_obs_v2
from reinbalance_survivors_contracts.perception_profile import (
    CALIBRATION_ARTIFACT_SCHEMA_VERSION, OBS_V2_RESIDUAL_FIELDS, CalibrationResidual, FittedPerceptionErrorProfile,
)

from survivors.perception_benchmark import (
    THRESHOLD_CHOICE_TOP1, THRESHOLD_CONFIDENCE, THRESHOLD_DENSITY_CORR, THRESHOLD_INVENTORY_TOP1,
    THRESHOLD_LEVEL_EXACT, THRESHOLD_SCREEN_F1, THRESHOLD_TIMER_EXACT,
    _FORMAL_FOREGROUND_CLASSES, _FORMAL_OCCURRENCE_DEPENDENT_SLICES, _FORMAL_REQUIRED_SLICES,
    _FORMAL_SLICE_COUNT_FLOORS, _FORMAL_SLICE_SESSION_FLOORS, _FORMAL_SLICE_THRESHOLDS,
    _empty_report, formal_absent_slices, obs_v2_residuals, recompute_gate_from_metrics,
)
from survivors.perception_error_fit import fit_error_profile

WEAPONS = ("weapon_projectile", "weapon_zone", "weapon_aura", "weapon_orbit")


def _formal_metrics(**count_overrides: int) -> dict[str, Any]:
    """全 formal 条件を満たす metrics を作り、指定 slice の出現数だけ上書きする。

    absent_slices は上書き後の slice_counts から再計算した値を入れます。
    """
    metrics: dict[str, Any] = dict(_empty_report(development_only=True, formal_eligible=False).metrics_wire())
    metrics.update({
        "total_records": 1000, "screen_state_f1": THRESHOLD_SCREEN_F1, "timer_exact_rate": THRESHOLD_TIMER_EXACT,
        "level_exact_rate": THRESHOLD_LEVEL_EXACT, "inventory_top1_rate": THRESHOLD_INVENTORY_TOP1,
        "choice_top1_rate": THRESHOLD_CHOICE_TOP1, "hp_mae": 0.0, "xp_mae": 0.0,
        "density_correlation": THRESHOLD_DENSITY_CORR, "nearest_normalized_median_error": 0.0,
        "latency_p95_ms": 0.0, "latency_p99_ms": 0.0, "invalid_tick_rate": 0.0,
        "levelup_invalid_choice_rate": 0.0, "roi_center_p99": 0.0, "roi_inside_region_rate": 1.0,
        "roi_false_positive_count": 0, "confidence_mean": THRESHOLD_CONFIDENCE,
        "expected_tick_count": 10, "observed_tick_count": 10, "latency_tick_count": 10,
    })
    counts = {name: 1 for name in (
        "screen_state", "timer_seconds", "level", "hp_ratio", "xp_ratio", "inventory_top1", "choice_top1",
        "entity_density", "nearest_distance", "ui_roi_center_error", "ui_inside_region", "ui_false_positive",
        "confidence",
    )}
    counts.update(_FORMAL_SLICE_COUNT_FLOORS)
    counts.update(count_overrides)
    metrics["slice_counts"] = counts
    metrics["slice_session_counts"] = dict(_FORMAL_SLICE_SESSION_FLOORS)
    metrics["slices"] = [
        {"name": name, "count": max(counts.get(name, 1), 1), "session_count": 2,
         "metric_value": _FORMAL_SLICE_THRESHOLDS[name], "threshold": _FORMAL_SLICE_THRESHOLDS[name],
         "ci_lower": _FORMAL_SLICE_THRESHOLDS[name]}
        for name in sorted(_FORMAL_REQUIRED_SLICES)
    ]
    metrics["absent_slices"] = formal_absent_slices(counts)
    return metrics


def test_weapon_classes_are_formal_foreground_with_floors():
    """weapon 4クラスが formal foreground class で、下限は orbit 50・他 200。

    King Bible は Mad Forest 標準ビルドで出現が少ないため orbit だけ下限を下げています。
    """
    assert set(WEAPONS) <= set(_FORMAL_FOREGROUND_CLASSES)
    floors = {name: _FORMAL_SLICE_COUNT_FLOORS[f"foreground_class:{name}"] for name in WEAPONS}
    assert floors == {"weapon_projectile": 200, "weapon_zone": 200, "weapon_aura": 200, "weapon_orbit": 50}
    assert {f"foreground_class:{name}" for name in WEAPONS} <= _FORMAL_OCCURRENCE_DEPENDENT_SLICES
    assert {"event:hazard", "foreground_class:hazard_projectile", "foreground_class:hazard_area"} <= _FORMAL_OCCURRENCE_DEPENDENT_SLICES


def test_absent_occurrence_slices_are_recorded_and_excluded():
    """出現数依存 slice が下限未満なら実測値とともに「該当なし」となり、正式判定を止めない。

    hazard が 7 件・event:hazard が 3 件・orbit が 0 件でも、absent_slices に実測値が残れば通過します。
    """
    metrics = _formal_metrics(**{
        "foreground_class:hazard_projectile": 7, "event:hazard": 3, "foreground_class:weapon_orbit": 0,
    })
    assert metrics["absent_slices"] == {
        "event:hazard": 3, "foreground_class:hazard_projectile": 7, "foreground_class:weapon_orbit": 0,
    }
    passed, reasons = recompute_gate_from_metrics(metrics, formal=True)
    assert passed and not reasons


def test_non_occurrence_slice_below_floor_still_fails():
    """出現数依存でない slice（enemy_elite・event:boss 等）は下限未満なら従来どおり失敗する。

    「該当なし」規則で外せるのは決められた slice だけで、黙って下限を下げません。
    """
    metrics = _formal_metrics(**{"foreground_class:enemy_elite": 10, "event:boss": 2})
    assert metrics["absent_slices"] == {}
    passed, reasons = recompute_gate_from_metrics(metrics, formal=True)
    assert not passed
    assert any("enemy_elite" in r and "200" in r for r in reasons)
    assert any("event:boss" in r and "100" in r for r in reasons)


def test_tampered_absent_slices_cannot_drop_requirements():
    """absent_slices を実測値と食い違うよう書き換えると正式判定は失敗する。

    非依存 slice を書き足すと読み込み時に拒否、依存 slice の値を変えると gate が止めます。
    """
    metrics = _formal_metrics(**{"foreground_class:hazard_area": 5})
    stale = dict(metrics, absent_slices={})
    passed, reasons = recompute_gate_from_metrics(stale, formal=True)
    assert not passed and any("absent_slices" in r for r in reasons)
    forged = dict(metrics, absent_slices={**metrics["absent_slices"], "foreground_class:enemy_elite": 0})
    with pytest.raises(ValueError):
        recompute_gate_from_metrics(forged, formal=True)
    hidden = dict(metrics, absent_slices={"foreground_class:hazard_area": 500})
    assert not recompute_gate_from_metrics(hidden, formal=True)[0]


def test_final_verdict_keeps_absent_slices_with_counts():
    """final verdict の metrics に absent_slices（slice 名 → 実測出現数）が残る。

    benchmark が実データから求めた absent_slices が metrics_wire 経由で verdict へ渡ることを確かめます。
    """
    from survivors.perception_benchmark import BenchmarkRecord, run_benchmark
    from survivors.perception_error_fit import create_synthetic_final_verdict, _HASH_FIELDS

    rows = [
        BenchmarkRecord(f"f{i}", f"s{i % 2}", "error_calibration", "raw", "screen_state", "gameplay", "gameplay", 1.0, 1.0)
        for i in range(4)
    ]
    report = run_benchmark(rows, n_bootstrap=20)
    for name, value in report.metrics_wire().items():
        if isinstance(value, float) and not np.isfinite(value):
            setattr(report, name, 0.0)  # 記録の無い指標の inf を verdict 再計算できる値へ置き換える
    report.passed, report.blocking_reasons = recompute_gate_from_metrics(report.metrics_wire())
    assert report.absent_slices == formal_absent_slices(report.slice_counts)
    assert report.absent_slices["foreground_class:weapon_zone"] == 0 and "event:hazard" in report.absent_slices
    verdict = create_synthetic_final_verdict(
        report, seal_id="a" * 64, final_session_ids=["f0"], **{name: "b" * 64 for name in _HASH_FIELDS},
    )
    wire = verdict.to_wire()
    assert wire["metrics"]["absent_slices"] == report.absent_slices
    assert "obs_v2_errors" in wire["metrics"]


def _obs(*, zone_cx=600., zone_r=20., zone_first=4.0, weapon="SantaWater", now=5.0):
    """zone 1つ・敵1体の v2 観測を Common ビルダーで作る。

    残差の計算に使う正解・予測の組を、入力を少しずつ変えて作るための補助です。
    """
    slots = [HudSlot("weapon", i, weapon if i == 0 else None, 1 if i == 0 else None) for i in range(6)]
    slots += [HudSlot("passive", i, None, None) for i in range(6)]
    tracks = [
        TrackPx("weapon_zone", zone_cx, 500., zone_r, 1, zone_first, False),
        TrackPx("enemy_normal", 700., 400., 10., 2, 0., False),
    ]
    return build_deploy_obs_v2(
        viewport_wh=(1000, 1000), player_px=(500., 500.), tracks=tracks, hud_slots=slots,
        now_s=now, duration_mult=1.0, world_valid=True,
    )


def test_obs_v2_residuals_measure_each_new_segment():
    """v2 残差は16方向 L1・zone 位置/半径・スロット不一致・残り時間（初観測ずれ）を測る。

    予測の zone が 0.1 ずれ・半径 +0.02 なら位置・半径の誤差になり、初観測が 0.1 秒遅いと
    経過時間が短く見えるので残り時間は +0.1 秒の過大評価（初観測ずれ）になります。
    """
    ground = _obs()
    predicted = _obs(zone_cx=650., zone_r=30., zone_first=4.1, weapon="LaBorra")
    residuals = obs_v2_residuals(ground, predicted)
    by_field: dict[str, list[float]] = {}
    for name, value in residuals:
        by_field.setdefault(name, []).append(value)
    assert set(by_field) <= set(OBS_V2_RESIDUAL_FIELDS)
    assert by_field["obs_v2_zone_position"][0] == pytest.approx(.1, abs=1e-5)
    assert by_field["obs_v2_zone_radius"][0] == pytest.approx(.02, abs=1e-5)
    assert sum(by_field["obs_v2_slot_id_mismatch"]) == 1.0 and len(by_field["obs_v2_slot_id_mismatch"]) == 12
    # SantaWater Lv1 と LaBorra Lv1 で持続時間が違うので ttl には持続時間差 + 初観測ずれが入る
    assert len(by_field["obs_v2_ttl_first_seen_offset_s"]) == 1
    assert by_field["obs_v2_enemy_dir16_l1"] == [0.0, 0.0, 0.0]
    same = obs_v2_residuals(ground, _obs(zone_first=4.1))
    ttl = [v for n, v in same if n == "obs_v2_ttl_first_seen_offset_s"]
    assert ttl == [pytest.approx(0.1, abs=1e-5)]


def test_obs_v2_residuals_skip_v1_and_invalid_elements():
    """v1 観測や validity 0 の要素は比べない。

    無効な要素を 0 誤差として数えると誤差が小さく見えるためです。
    """
    ground = _obs()
    v1 = replace(ground, schema_hash=DeployObsSchema.default_v1().schema_hash)
    assert obs_v2_residuals(v1, ground) == []
    invalid = replace(ground, validity=np.zeros_like(ground.validity), values=ground.values.copy(), age=np.ones_like(ground.age))
    assert obs_v2_residuals(ground, invalid) == []


def test_v2_residuals_reach_calibration_profile_segment_error_stats():
    """v2 残差を fit すると calibration profile の segment_error_stats（mean/std/count）に載る。

    ttl の符号付き平均が初観測ずれの calibration 項目で、artifact の v2 schema で往復できます。
    1件しかない field は統計を出さず、fit 全体は止めません。
    """
    rows = []
    for session in ("c0", "c1"):
        for frame, offset in (("f0", .1), ("f1", .3)):
            rows.append(CalibrationResidual(session, frame, "hp_ratio", .01, 1.0, 0))
            rows.append(CalibrationResidual(session, frame, "obs_v2_ttl_first_seen_offset_s", offset, 1.0, 0))
    rows.append(CalibrationResidual("c0", "f0", "obs_v2_zone_radius", .5, 1.0, 0))
    profile = fit_error_profile(rows, ["c0", "c1"], ["final"])
    stats = profile.segment_error_stats
    assert set(stats) == {"obs_v2_ttl_first_seen_offset_s"}
    assert stats["obs_v2_ttl_first_seen_offset_s"]["mean"] == pytest.approx(.2)
    assert stats["obs_v2_ttl_first_seen_offset_s"]["count"] == 4
    wire = profile.to_artifact_wire()
    assert wire["schema_version"] == CALIBRATION_ARTIFACT_SCHEMA_VERSION == "perception_calibration_profile.v2"
    assert FittedPerceptionErrorProfile.from_artifact_wire(wire).segment_error_stats == stats
    with pytest.raises(ValueError):
        FittedPerceptionErrorProfile.from_artifact_wire({**wire, "segment_error_stats": {"hp_ratio": {"mean": 0., "std": 0., "count": 4}}})


def test_obs_v2_errors_schema_is_validated_on_recompute():
    """保存済み metrics の obs_v2_errors は決まった field・キー・値域だけを受け付ける。

    未知の field や負の平均誤差を含む metrics は読み込み時に拒否します。
    """
    metrics = _formal_metrics()
    metrics["obs_v2_errors"] = {"obs_v2_zone_position": {"mean_abs": .05, "count": 3}}
    assert recompute_gate_from_metrics(metrics, formal=True)[0]
    for bad in ({"unknown": {"mean_abs": .1, "count": 1}}, {"obs_v2_zone_position": {"mean_abs": -1., "count": 1}},
                {"obs_v2_zone_position": {"mean_abs": .1, "count": 0}}):
        with pytest.raises(ValueError):
            recompute_gate_from_metrics(dict(metrics, obs_v2_errors=bad), formal=True)


def test_calibration_residual_derivation_includes_v2_segments():
    """formal runner の残差導出（_derive_calibration_residuals）が v2 segment の残差を含む。

    実 assembler の snapshot を正解とし、予測の zone 半径だけをずらすと obs_v2_zone_radius が出ます。
    """
    from benchmark_survivors_perception import _derive_calibration_residuals
    from survivors.perception_benchmark import SnapshotReplayTick
    from survivors.real_obs_assembler import RealObsAssembler
    from survivors.vision.entity_tracker import PlayerAnchorState, TrackedEntityV2, TrackedWorldStateV2
    from survivors.vision.hud_parser import HudStateV1

    schema = DeployObsSchema.default_v2()
    hud = HudStateV1(
        "hud_state.v1", "cal", 1, 1_000_000_000, "a" * 64, "gameplay", .9, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
        ("santa_water",) + (None,) * 11, .9, "b" * 64, (), "c" * 64, (),
        False, False, False, .9, "ok",
    )
    zone = TrackedEntityV2(1, 0, "weapon_zone", "weapon", .9, 2, 1, .6, .5, .1, 0., 0., 0., True, False, .04, .04, 500_000_000)
    world = TrackedWorldStateV2(1, 1_000_000_000, [zone], PlayerAnchorState(.5, .5, .9, False))
    ground = RealObsAssembler().assemble(hud, world, schema, (1920, 1080))
    offset, _ = schema.layout["weapon_zone_geometry"]
    values = np.array(ground.deploy_obs.values, dtype=float)
    values[offset + 2] += .01
    predicted = replace(ground, deploy_obs=replace(ground.deploy_obs, values=values))
    tick = SnapshotReplayTick("cal", "error_calibration", "lossless", ground.frame_id, ground, predicted, 0.0)
    residuals = _derive_calibration_residuals([tick], resolution_wh=(1920, 1080))
    radius = [r.residual for r in residuals if r.field == "obs_v2_zone_radius"]
    assert radius == [pytest.approx(.01, abs=1e-5)]
    assert {r.field for r in residuals} & set(OBS_V2_RESIDUAL_FIELDS) >= {"obs_v2_slot_id_mismatch", "obs_v2_enemy_dir16_l1"}

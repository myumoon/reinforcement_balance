"""gameplay slice 集計と event 正誤の回帰を検証する。

item context がない GT frame でも time band と boss/hazard を数える。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from benchmark_survivors_perception import _assign_foreground_entities
from survivors.perception_benchmark import event_flag_correct, slice_counts_for_snapshot


@dataclass(frozen=True)
class _Snapshot:
    """runner の diagnostics 再構築を確認する最小 snapshot 形。

    dataclasses.replace を実データと同じ frozen な形で通す。
    """

    diagnostics: dict


def test_gameplay_diagnostics_supply_named_slices_without_item_context() -> None:
    """gameplay diagnostics が time band と boss/hazard occurrence を供給する。

    entity class 集計は別の diagnostics が無くても event slice を落とさない。
    """
    snapshot = SimpleNamespace(
        screen_state="gameplay", item_context=None,
        diagnostics={
            "slice_elapsed_time": 300.0,
            "slice_boss_present": True,
            "slice_hazard_present": True,
        },
    )
    counts = slice_counts_for_snapshot(snapshot)
    assert counts["time_band:early"] == 1
    assert counts["event:boss"] == 1
    assert counts["event:hazard"] == 1


def test_event_flags_match_gameplay_boxes_and_keep_level_up_behavior() -> None:
    """gameplay box presence and level-up item flags use their intended branches.

    boss IoU の精度は foreground class slice が扱い、event 値は visible presence を比べる。
    """
    ground = SimpleNamespace(
        item_context=None,
        diagnostics={"slice_boss_present": True, "slice_hazard_present": True},
    )
    predicted = SimpleNamespace(diagnostics={"predicted_boxes": [
        {"class": "enemy_boss", "xyxy": [100, 100, 150, 150], "score": 0.9},
        {"class": "hazard_area", "xyxy": [200, 200, 250, 250], "score": 0.9},
    ]})
    assert event_flag_correct(ground, predicted, "event:boss") == 1.0
    assert event_flag_correct(ground, predicted, "event:hazard") == 1.0
    assert event_flag_correct(ground, SimpleNamespace(diagnostics={"predicted_boxes": []}), "event:boss") == 0.0

    old_ground = SimpleNamespace(item_context=SimpleNamespace(boss_flag=True, hazard_flag=False), diagnostics={})
    old_predicted = SimpleNamespace(item_context=SimpleNamespace(boss_flag=True, hazard_flag=True), diagnostics={})
    assert event_flag_correct(old_ground, old_predicted, "event:boss") == 1.0
    assert event_flag_correct(old_ground, old_predicted, "event:hazard") == 0.0


def test_runner_assigns_predicted_boxes_to_ground_truth_ids_by_greedy_iou() -> None:
    """runner が score 順 IoU で予測 class を GT entity identity に割り当てる。

    低 IoU の別 box は frame-local ID を持つ false positive として残す。
    """
    ground = _Snapshot({
        "foreground_entity_boxes": {"s:1:0": [10, 10, 30, 30]},
        "foreground_entity_classes": {"s:1:0": "enemy_boss"},
    })
    predicted = _Snapshot({"predicted_boxes": [
        {"class": "enemy_boss", "xyxy": [10, 10, 30, 30], "score": 0.9},
        {"class": "enemy_normal", "xyxy": [100, 100, 120, 120], "score": 0.8},
        {"class": "unknown", "xyxy": [10, 10, 30, 30], "score": 1.0},
    ]})
    replay = _assign_foreground_entities(ground, predicted)
    assert replay.diagnostics["foreground_entity_classes"] == {
        "s:1:0": "enemy_boss", "pred:1": "enemy_normal",
    }
    assert replay.diagnostics["foreground_entity_counts"] == {"enemy_boss": 1, "enemy_normal": 1}

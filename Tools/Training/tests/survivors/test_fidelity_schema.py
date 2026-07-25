"""C++ content schema からの一方向 annotation を検証する。

Python が id/level/pair 集合を別管理しないことを fixture で固定します。
"""

import pytest

from games.survivors.fidelity_schema import annotate_cpp_content_schema


def _export() -> dict:
    """完全な C++ export fixture を返す。

    必須構造を一つに集約し、各テストでは検証したい差だけを変更します。
    """
    return {
        "schema_version": "survivors.content_schema.v1",
        "content": {
            "max_level": 2,
            "weapons": [
                {"id": "future", "max_level": 8},
                {"id": "1", "max_level": 8},
                {"id": "2", "max_level": 1},
            ],
            "passives": [{"id": "1", "max_level": 5}],
            "gems": [{"id": "blue", "xp": 2}, {"id": "green", "xp": 9}, {"id": "red", "xp": 10}],
            "xp_curve": [5, 10],
            "level_cadence": "xp_threshold",
            "offer": {"count": 3, "fallback": "none_when_pool_empty"},
            "slots": {"weapon": 6, "passive": 6},
            "evolutions": [{"base_id": "1", "evolved_id": "2", "passive_id": "1", "union_partner_id": "0"}],
            "chest": {"boss_drop": True, "evolution_enabled_by_config": True},
        },
        "action_time": {
            "physics_dt": 1 / 60,
            "decision_steps": 1,
            "directions": [
                {"id": 0, "x": 0.0, "y": 1.0},
                {"id": 1, "x": 2 ** -0.5, "y": 2 ** -0.5},
                {"id": 2, "x": 1.0, "y": 0.0},
                {"id": 3, "x": 2 ** -0.5, "y": -(2 ** -0.5)},
                {"id": 4, "x": 0.0, "y": -1.0},
                {"id": 5, "x": -(2 ** -0.5), "y": -(2 ** -0.5)},
                {"id": 6, "x": -1.0, "y": 0.0},
                {"id": 7, "x": -(2 ** -0.5), "y": 2 ** -0.5},
                {"id": 8, "x": 0.0, "y": 0.0},
            ],
            "move_speed": 100.0,
            "screen_displacement_per_step": 100 / 60,
            "pause_during_level_up": False,
            "level_up_timing": "same_physics_step",
        },
    }


def test_schema_ids_are_taken_only_from_cpp_export() -> None:
    """入力に追加した id をコード変更なしで annotation できる。

    Python 内に content mirror がないことを振る舞いで確認します。
    """
    export = _export()
    result = annotate_cpp_content_schema(export, {"weapons": {"future": {"target_relevant": True}}})
    assert result.cpp_schema["content"]["weapons"][0]["id"] == "future"
    with pytest.raises(ValueError):
        annotate_cpp_content_schema(export, {"weapons": {"manual_mirror": {"target_relevant": True}}})


def test_annotation_ids_are_namespaced_by_collection() -> None:
    """同じ ID の weapon/passive annotation を独立して保持する。

    collection namespace を潰さず、対象 content の曖昧な上書きを防ぎます。
    """
    export = _export()
    export["content"]["weapons"] = [{"id": "1", "max_level": 8}, {"id": "2", "max_level": 1}]
    result = annotate_cpp_content_schema(export, {
        "weapons": {"1": {"target_relevant": True}},
        "passives": {"1": {"target_relevant": False}},
    })
    assert result.annotations["weapons"]["1"]["target_relevant"] is True
    assert result.annotations["passives"]["1"]["target_relevant"] is False


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value["content"].pop("xp_curve"), "content keys mismatch"),
        (lambda value: value["content"]["weapons"][0].pop("max_level"), "weapons row keys mismatch"),
        (lambda value: value["content"]["gems"][0].update(xp=float("nan")), "gems.xp"),
        (lambda value: value["content"]["evolutions"][0].pop("passive_id"), "evolution row keys mismatch"),
        (lambda value: value["action_time"].pop("physics_dt"), "action_time keys mismatch"),
        (lambda value: value["action_time"].update(directions=[]), "exactly nine"),
        (lambda value: value["action_time"].update(move_speed="fast"), "move_speed"),
    ],
)
def test_cpp_export_missing_invalid_and_nonfinite_fields_fail_closed(mutate, match: str) -> None:
    """content/action の兄弟経路を同じ fail-closed 境界で拒否する。

    欠落、型不一致、非 finite、要素数不足の export を annotation より前に止めます。
    """
    export = _export()
    mutate(export)
    with pytest.raises(ValueError, match=match):
        annotate_cpp_content_schema(export, {})


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value["action_time"]["directions"][0].update(x=1.1), "direction"),
        (lambda value: value["content"].update(xp_curve=[5, 4]), "non-decreasing"),
        (lambda value: value["content"].update(level_cadence="per_frame"), "level_cadence"),
        (
            lambda value: value["content"]["evolutions"][0].update(evolved_id="missing"),
            "evolved_id",
        ),
    ],
)
def test_cpp_export_semantic_violations_fail_before_annotation(mutate, match: str) -> None:
    """意味的に範囲外の C++ export を注釈前に拒否する。

    方向、XP、cadence、進化参照の代表違反を固定し、構造だけ正しい入力も
    fail-closed 境界を通過できないことを確認します。
    """
    export = _export()
    mutate(export)
    with pytest.raises(ValueError, match=match):
        annotate_cpp_content_schema(export, {})

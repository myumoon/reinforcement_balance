"""C++ content schema からの一方向 annotation を検証する。

Python が id/level/pair 集合を別管理しないことを fixture で固定します。
"""

import pytest

from games.survivors.fidelity_schema import annotate_cpp_content_schema


def test_schema_ids_are_taken_only_from_cpp_export() -> None:
    """入力に追加した id をコード変更なしで annotation できる。

    Python 内に content mirror がないことを振る舞いで確認します。
    """
    export = {"schema_version": "survivors.content_schema.v1", "content": {"weapons": [{"id": "future", "max_level": 8}], "passives": [], "gems": []}, "action_time": {}}
    result = annotate_cpp_content_schema(export, {"weapons": {"future": {"target_relevant": True}}})
    assert result.cpp_schema["content"]["weapons"][0]["id"] == "future"
    with pytest.raises(ValueError):
        annotate_cpp_content_schema(export, {"weapons": {"manual_mirror": {"target_relevant": True}}})


def test_annotation_ids_are_namespaced_by_collection() -> None:
    """同じ ID の weapon/passive annotation を独立して保持する。

    collection namespace を潰さず、対象 content の曖昧な上書きを防ぎます。
    """
    export = {"schema_version": "survivors.content_schema.v1", "content": {
        "weapons": [{"id": "1"}], "passives": [{"id": "1"}], "gems": [],
    }, "action_time": {}}
    result = annotate_cpp_content_schema(export, {
        "weapons": {"1": {"target_relevant": True}},
        "passives": {"1": {"target_relevant": False}},
    })
    assert result.annotations["weapons"]["1"]["target_relevant"] is True
    assert result.annotations["passives"]["1"]["target_relevant"] is False

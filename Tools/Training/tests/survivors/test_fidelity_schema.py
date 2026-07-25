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
    result = annotate_cpp_content_schema(export, {"future": {"target_relevant": True}})
    assert result.cpp_schema["content"]["weapons"][0]["id"] == "future"
    with pytest.raises(ValueError):
        annotate_cpp_content_schema(export, {"manual_mirror": {"target_relevant": True}})

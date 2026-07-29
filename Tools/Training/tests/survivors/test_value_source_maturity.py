"""旧 Value Source maturity helper の互換出力を検証する。

初心者向け:
古い呼び出し元の情報は維持しつつ、廃止した label-ready 判定を新しい probe gate へ
混入させないことを確認します。
"""

from __future__ import annotations

import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.value_source_maturity import make_value_source_maturity_record


def test_value_source_maturity_is_compatibility_data_without_label_gate():
    """互換 record から deprecated label-ready field を除外する。

    初心者向け:
    label の可否は後続 teacher validation の責務なので、この helper は判定しません。
    """
    record = make_value_source_maturity_record(
        run_name="13_is2_evolution_bootstrap",
        item_stage_key="IS2",
        bootstrap_complete=True,
        passive_coverage_count=10,
        evolution_coverage_count=8,
        union_coverage_count=1,
        model_path="result/model.zip",
        vecnormalize_path="result/vecnormalize.pkl",
    )

    assert record["item_stage_key"] == "IS2"
    assert "ready_for_value_labels" not in record
    assert "ready_for_probe" not in record

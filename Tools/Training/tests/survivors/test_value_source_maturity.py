from __future__ import annotations

import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.value_source_maturity import make_value_source_maturity_record


def test_value_source_maturity_requires_is2_for_ready():
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

    assert record["ready_for_value_labels"] is True
    assert record["item_stage_key"] == "IS2"

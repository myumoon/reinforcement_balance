"""Value choice ranking v3 の schema と append gate を検証する。

非有限値や構造不正を JSONL へ一 byte も追加せず、tie と source/context binding を
machine-readable な固定 schema で保持する。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.value_choice_schema import (
    ValueChoiceSchemaError,
    append_value_choice_ranking,
    build_value_choice_ranking,
    validate_value_choice_ranking,
)
from games.survivors.value_scorer import CandidateValue


def _ranking() -> dict:
    """正常な二候補 ranking row を生成する。

    hash は検証対象の field shape を満たす固定値とし、値の ordering と tie 計算に集中する。
    """
    return build_value_choice_ranking(
        source_identity_sha256="a" * 64,
        manifest_sha256="b" * 64,
        model_sha256="c" * 64,
        vecnormalize_sha256="d" * 64,
        observation_schema_sha256="e" * 64,
        policy_state_schema_sha256="f" * 64,
        context_sha256="1" * 64,
        context_mode="captured",
        environment_step=7,
        decision_id="decision-7",
        pending_obs_sha256="2" * 64,
        values=[
            CandidateValue("choice-z", 1.0, 2.5),
            CandidateValue("choice-a", 1.0 - 0.5e-5, 2.49),
        ],
        zero_state_smoke=False,
    )


def test_ranking_tie_preserves_input_order_and_never_claims_label_ready() -> None:
    """tie 候補を choice ID 順に並べ替えないことを確認する。

    ranking は label release verdict ではないため、formal context でも label ready を主張しない。
    """
    row = _ranking()
    validate_value_choice_ranking(row)

    assert row["tie"] is True
    assert row["tie_epsilon"] == 1e-5
    assert [item["choice_id"] for item in row["ordered_candidates"]] == [
        "choice-z",
        "choice-a",
    ]
    assert row["ready_for_training_label"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(extra=True),
        lambda row: row["ordered_candidates"][0].update(
            value_normalized_return=float("nan")
        ),
        lambda row: row["ordered_candidates"][0].update(
            value_unscaled_return=float("inf")
        ),
        lambda row: row.update(tie_epsilon=1e-4),
    ],
)
def test_schema_and_finite_checks_run_before_jsonl_append(
    tmp_path: Path,
    mutation,
) -> None:
    """未知 field・NaN・Inf・epsilon 改変の全 sibling を append 前に拒否する。

    既存 JSONL の末尾は不正 row によって変化せず、後続 reader が部分データを読まない。
    """
    output = tmp_path / "ranking.jsonl"
    output.write_bytes(b'{"existing":true}\n')
    before = output.read_bytes()
    row = copy.deepcopy(_ranking())
    mutation(row)

    with pytest.raises(ValueChoiceSchemaError):
        append_value_choice_ranking(output, row)
    assert output.read_bytes() == before


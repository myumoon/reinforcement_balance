"""development score-difference 専用 scale の fit・commit・binding を検証する。

raw critic と overlay の identity を混ぜず、final ref や scale 改変後の calibration reuse を
fail-closed で拒否する。
"""

from __future__ import annotations

import copy

import pytest

from games.survivors.teacher_score_scale import (
    ScoreScaleContractError,
    assert_calibration_scale_binding,
    fit_teacher_score_scale,
    transform_teacher_scores,
    validate_teacher_score_scale,
)


def _refs() -> list[dict]:
    """development train/validation の score-difference refs を返す。

    candidate raw scores ではなく監査可能な差分 ref を scale fitting の親にする。
    """
    return [
        {
            "ref_id": f"score-diff-{index}",
            "partition": (
                "development_train" if index % 2 == 0 else "development_validation"
            ),
            "difference": float(index - 10) / 3.0,
        }
        for index in range(21)
    ]


def test_scale_fit_uses_development_refs_and_transform_contract() -> None:
    """q05/q95 と sigma を development refs だけから固定する。

    transform は max valid score を 0 とし、tie epsilon z を schema に保存する。
    """
    scale = fit_teacher_score_scale(
        teacher_type="raw_critic",
        teacher_identity="a" * 64,
        development_fit_partition_id="b" * 64,
        score_difference_refs=_refs(),
    )
    validate_teacher_score_scale(scale)
    assert scale["schema_version"] == "survivors.teacher_score_scale.v1"
    assert scale["sigma"] == pytest.approx(
        max((scale["q95"] - scale["q05"]) / 3.29, 1e-6)
    )
    assert scale["tie_epsilon_z"] == 0.02
    transformed = transform_teacher_scores([2.0, 1.0, None], scale)
    assert transformed[0] == 0.0
    assert transformed[1] < 0.0
    assert transformed[2] is None


def test_final_score_difference_ref_is_rejected() -> None:
    """untouched final test の score difference を fit に使用させない。

    calibration と同じ sealing を scale fitting 経路にも適用する。
    """
    refs = _refs()
    refs[0]["partition"] = "final_test"
    with pytest.raises(ScoreScaleContractError, match="final_test"):
        fit_teacher_score_scale(
            teacher_type="raw_critic",
            teacher_identity="a" * 64,
            development_fit_partition_id="b" * 64,
            score_difference_refs=refs,
        )


def test_raw_and_overlay_get_distinct_scale_identity() -> None:
    """teacher type/identity が違う artifact を別 scale identity にする。

    overlay scale に raw critic calibration を再利用できない境界を確認する。
    """
    raw = fit_teacher_score_scale(
        teacher_type="raw_critic",
        teacher_identity="a" * 64,
        development_fit_partition_id="b" * 64,
        score_difference_refs=_refs(),
    )
    overlay = fit_teacher_score_scale(
        teacher_type="overlay",
        teacher_identity="c" * 64,
        development_fit_partition_id="b" * 64,
        score_difference_refs=_refs(),
    )
    assert raw["scale_identity"] != overlay["scale_identity"]
    with pytest.raises(ScoreScaleContractError, match="scale"):
        assert_calibration_scale_binding(
            {"teacher_score_scale_id": raw["scale_identity"]},
            overlay,
        )


def test_identity_tampering_is_rejected() -> None:
    """fit 後の sigma や refs 改変を identity mismatch として拒否する。

    committed scale の意味を後から変えて calibration weight を流用させない。
    """
    scale = fit_teacher_score_scale(
        teacher_type="raw_critic",
        teacher_identity="a" * 64,
        development_fit_partition_id="b" * 64,
        score_difference_refs=_refs(),
    )
    tampered = copy.deepcopy(scale)
    tampered["sigma"] *= 2.0
    with pytest.raises(ScoreScaleContractError, match="identity"):
        validate_teacher_score_scale(tampered)

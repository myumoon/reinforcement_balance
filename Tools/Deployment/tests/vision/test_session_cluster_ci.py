"""session cluster CI の contract test。

session 単位の比率再標本化が分母ゼロと不正値を安全に扱うことを確認します。
"""

import math

import pytest

from survivors.vision.session_cluster_ci import cluster_ci_bounds


def test_equal_session_ratios_have_exact_bounds() -> None:
    """同じ比率の cluster は再標本化後も同じ区間になる。

分母ゼロの session は最小 applicable 数へ数えず、CI の幅にも影響させません。
    """
    assert cluster_ci_bounds(
        {"a": (2.0, 4.0), "b": (3.0, 6.0), "c": (1.0, 2.0), "ignored": (0.0, 0.0)},
        resamples=200,
        seed=17,
    ) == (0.5, 0.5)


@pytest.mark.parametrize(
    "counts",
    [
        {"a": (0.0, -1.0), "b": (1.0, 1.0), "c": (1.0, 1.0)},
        {"a": (math.nan, 1.0), "b": (1.0, 1.0), "c": (1.0, 1.0)},
        {"a": (1.0, 1.0), "b": (1.0, 1.0)},
    ],
)
def test_invalid_or_insufficient_clusters_fail_closed(counts) -> None:
    """不正数値と applicable session 不足を評価前に拒否する。

閾値未満の session 数から信頼区間を捏造しないことを固定します。
    """
    with pytest.raises(ValueError):
        cluster_ci_bounds(counts, resamples=100)

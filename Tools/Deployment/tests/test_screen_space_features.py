"""画面内 world track から作る release feature を検証する。

C++ と同じ方位規則と、画面外情報を捨てる境界を固定します。
"""

import pytest

from survivors.screen_space_features import build_screen_space_estimates, directional_bin
from survivors.vision.entity_tracker import PlayerAnchorState, TrackedEntityV1, TrackedWorldStateV1


def _track(track_id: int, coarse: str, x: float, y: float, *, on_screen=True, clipped=False) -> TrackedEntityV1:
    """指定した player-relative 座標の track を作る。

    detector を介さず feature の可視性規則だけを試します。
    """
    return TrackedEntityV1(
        track_id, 2, coarse, coarse, .9, 2, 9, .5 + x, .5 + y, x, y,
        0., 0., on_screen, clipped,
    )


def test_directional_bin_matches_cpp_atan2_plus_pi_mapping() -> None:
    """八方位 bin を C++ の atan2 + PI 式へ一致させる。

    +X を bin 0 とする一般的な実装へ誤って戻らないよう軸方向を固定します。
    """
    assert directional_bin(1., 0., 8) == 4
    assert directional_bin(0., 1., 8) == 6
    assert directional_bin(0., -1., 8) == 2
    assert directional_bin(-1., 0., 8) in {0, 7}


def test_only_unclipped_on_screen_tracks_affect_counts_and_nearest() -> None:
    """release estimate から画面外・clipped track を除く。

    tracker の全状態件数ではなく、現在見えている敵と gem だけを数えます。
    """
    world = TrackedWorldStateV1(
        frame_index=10, timestamp_ns=1_000,
        tracks=[
            _track(1, "enemy", .2, 0.),
            _track(2, "enemy", .01, 0., on_screen=False),
            _track(3, "enemy", .02, 0., clipped=True),
            _track(4, "gem", 0., .3),
        ],
        player_anchor=PlayerAnchorState(.5, .5, .9, False),
    )
    estimates = build_screen_space_estimates(world)
    assert estimates["visible_enemy_count"].value == pytest.approx((.05,))
    assert estimates["enemy_density"].value == pytest.approx((.05,))
    assert estimates["gem_density"].value == pytest.approx((.05,))
    assert estimates["nearest_enemy_offset"].value == pytest.approx((.2, 0.))
    assert sum(estimates["enemy_directional_density"].value) == pytest.approx(.05)


def test_no_visible_enemy_omits_nearest_but_returns_zero_densities() -> None:
    """敵が見えない正常場面を neutral estimate にする。

    欠損 nearest とゼロ件 density を区別し、過去の位置を再利用しません。
    """
    world = TrackedWorldStateV1(0, 10, [], PlayerAnchorState(.5, .5, .1, True))
    estimates = build_screen_space_estimates(world)
    assert "nearest_enemy_offset" not in estimates
    assert estimates["visible_enemy_count"].value == (0.,)
    assert estimates["gem_density"].value == (0.,)


"""実 HUD/world からの atomic observation assembly を検証する。

release tensor、item context、操作用 UI snapshot の境界をまとめて確認します。
"""

import dataclasses

import numpy as np
import pytest

from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from survivors.deploy_obs_adapter import build_deploy_observation
from survivors.perception_snapshot import UiPresentationSnapshotV1
from survivors.real_obs_assembler import RealObsAssembler
from survivors.vision.entity_tracker import PlayerAnchorState, TrackedEntityV1, TrackedWorldStateV1
from survivors.vision.hud_parser import HudStateV1, ParsedCard


def _inputs(state="gameplay") -> tuple[HudStateV1, TrackedWorldStateV1]:
    """assembler test 用の同時刻 HUD/world を返す。

    visible と off-screen の敵を混ぜ、release 境界を一度に検証します。
    """
    card = ParsedCard(0, "whip", "weapon", 2, .99, "ok", (100, 100, 400, 500))
    hud = HudStateV1(
        "hud_state.v1", "session", 4, 1_000_000_000, "a" * 64, state, .9, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
        ("whip",) + (None,) * 11, .9, "b" * 64, (card,), "c" * 64, (),
        False, False, False, .9, "ok",
    )
    visible = TrackedEntityV1(1, 2, "enemy_normal", "enemy", .9, 1, 4, .7, .5, .2, 0., 0., 0., True, False)
    leaked = TrackedEntityV1(2, 2, "enemy_normal", "enemy", .99, 1, 4, .51, .5, .01, 0., 0., 0., False, False)
    world = TrackedWorldStateV1(4, 1_000_000_000, [visible, leaked], PlayerAnchorState(.5, .5, .9, False))
    return hud, world


def test_assemble_builds_release_observation_without_offscreen_leak() -> None:
    """visible track だけを DeployObservation へ組み立てる。

    全 tracker 件数や画面外の最近傍が policy tensor に現れないことを確認します。
    """
    schema = DeployObsSchema.default_v1()
    snapshot = RealObsAssembler().assemble(*_inputs(), schema, (1000, 1000))
    count_offset, _ = schema.layout["visible_enemy_count"]
    nearest_offset, _ = schema.layout["nearest_enemy_offset"]
    assert snapshot.deploy_obs.values[count_offset] == pytest.approx(.05)
    assert snapshot.deploy_obs.values[nearest_offset:nearest_offset + 2] == pytest.approx((.2, 0.))
    assert snapshot.deploy_obs.provenance == "release"
    assert np.isfinite(snapshot.deploy_obs.as_policy_tensor(schema)).all()


def test_ui_snapshot_is_atomic_and_diagnostics_have_no_roi() -> None:
    """操作用 presentation を PerceptionSnapshot 内へ一体化する。

    HudState side channel や ROI diagnostics を作らず、identity を同じ snapshot に束縛します。
    """
    snapshot = RealObsAssembler().assemble(*_inputs("level_up_items"), DeployObsSchema.default_v1(), (1000, 1000))
    assert snapshot.ui_presentation.snapshot_id == snapshot.snapshot_id
    assert snapshot.ui_presentation.frame_id == snapshot.frame_id
    assert snapshot.ui_presentation.source_content_hash == snapshot.source_content_hash
    assert snapshot.ui_presentation.ui_state_key == snapshot.ui_state_key
    assert "hud" not in {field.name for field in dataclasses.fields(type(snapshot))}
    assert all("roi" not in str(key).lower() for key in snapshot.diagnostics)
    assert snapshot.item_context is not None and len(snapshot.choices) == 1


def test_ui_presentation_cannot_enter_tensor_builder() -> None:
    """ROI を持つ UI presentation を model builder が拒否する。

    型の境界を誤って跨いでも Mapping estimate として解釈されないことを確認します。
    """
    schema = DeployObsSchema.default_v1()
    snapshot = RealObsAssembler().assemble(*_inputs(), schema, (1000, 1000))
    assert isinstance(snapshot.ui_presentation, UiPresentationSnapshotV1)
    with pytest.raises(ValueError):
        build_deploy_observation(schema, snapshot.ui_presentation, snapshot.captured_ns)


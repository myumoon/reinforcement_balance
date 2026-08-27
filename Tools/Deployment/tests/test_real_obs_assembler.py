"""実 HUD/world からの atomic observation assembly を検証する。

release tensor、item context、操作用 UI snapshot の境界をまとめて確認します。
"""
import dataclasses
import json
from pathlib import Path
import numpy as np
import pytest
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.item_decision import ItemDecisionFeatures
from survivors.deploy_obs_adapter import build_deploy_observation
from survivors.perception_snapshot import UiPresentationSnapshotV1
from survivors.real_obs_assembler import RealObsAssembler
from survivors.vision.entity_tracker import PlayerAnchorState, TrackedEntityV1, TrackedWorldStateV1
from survivors.vision.hud_parser import HudStateV1, ParsedCard

def _inputs(state="gameplay", *, ts=1_000_000_000) -> tuple[HudStateV1, TrackedWorldStateV1]:
    """assembler test 用の同時刻 HUD/world を返す。

    visible と off-screen の敵を混ぜ、release 境界を一度に検証します。
    ts を変えると別フレームとして tick 制限を超えられます。
    """
    card = ParsedCard(0, "whip", "weapon", 2, .99, "ok", (100, 100, 400, 500))
    hud = HudStateV1(
        "hud_state.v1", "session", 4, ts, "a" * 64, state, .9, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
        ("whip",) + (None,) * 11, .9, "b" * 64, (card,), "c" * 64, (),
        False, False, False, .9, "ok",
    )
    visible = TrackedEntityV1(1, 2, "enemy_normal", "enemy", .9, 1, 4, .7, .5, .2, 0., 0., 0., True, False)
    leaked = TrackedEntityV1(2, 2, "enemy_normal", "enemy", .99, 1, 4, .51, .5, .01, 0., 0., 0., False, False)
    world = TrackedWorldStateV1(4, ts, [visible, leaked], PlayerAnchorState(.5, .5, .9, False))
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
    assembler = RealObsAssembler()
    schema = DeployObsSchema.default_v1()
    assembler.assemble(*_inputs("gameplay"), schema, (1000, 1000))
    snapshot = assembler.assemble(*_inputs("level_up_items", ts=2_000_000_000), schema, (1000, 1000))
    assert snapshot is not None
    assert snapshot.ui_presentation.snapshot_id == snapshot.snapshot_id
    assert snapshot.ui_presentation.frame_id == snapshot.frame_id
    assert snapshot.ui_presentation.source_content_hash == snapshot.source_content_hash
    assert snapshot.ui_presentation.ui_state_key == snapshot.ui_state_key
    assert "hud" not in {field.name for field in dataclasses.fields(type(snapshot))}
    assert all("roi" not in str(key).lower() for key in snapshot.diagnostics)
    assert snapshot.item_context is not None and len(snapshot.choices) == 1
    with pytest.raises(ValueError):
        dataclasses.replace(snapshot, diagnostics={"nested": {"roi": [0., 0., 1., 1.]}})
    fixture = json.loads((Path(__file__).parent / "fixtures" / "ui_presentation_v1.json").read_text(encoding="utf-8"))
    assert fixture["fixture_scope"] == "development-only" and fixture["formal_perception_verdict_eligible"] is False

def test_ui_presentation_cannot_enter_tensor_builder() -> None:
    """ROI を持つ UI presentation を model builder が拒否する。

    型の境界を誤って跨いでも Mapping estimate として解釈されないことを確認します。
    """
    schema = DeployObsSchema.default_v1()
    snapshot = RealObsAssembler().assemble(*_inputs(), schema, (1000, 1000))
    assert isinstance(snapshot.ui_presentation, UiPresentationSnapshotV1)
    with pytest.raises(ValueError):
        build_deploy_observation(schema, snapshot.ui_presentation, snapshot.captured_ns)
    with pytest.raises(ValueError):
        ItemDecisionFeatures.from_wire(snapshot.ui_presentation)

def test_item_context_uses_cached_gameplay_danger() -> None:
    """level_up 画面では直前の gameplay world の enemy_density を保持する。

    level_up overlay は enemy tracks が空なので、gameplay snapshot を参照しないと density=0 になります。
    """
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    hud_gp, world_gp = _inputs("gameplay")
    assembler.assemble(hud_gp, world_gp, schema, (1000, 1000))
    card = ParsedCard(0, "knife", "weapon", 2, .99, "ok", (100, 100, 400, 500))
    hud_lu = HudStateV1(
        "hud_state.v1", "session", 5, 2_000_000_000, "a" * 64, "level_up_items", .9, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
        ("whip",) + (None,) * 11, .9, "b" * 64, (card,), "c" * 64, (),
        False, False, False, .9, "ok",
    )
    world_lu = TrackedWorldStateV1(5, 2_000_000_000, [], PlayerAnchorState(.5, .5, .9, False))
    snapshot = assembler.assemble(hud_lu, world_lu, schema, (1000, 1000))
    assert snapshot is not None
    assert snapshot.item_context is not None
    assert snapshot.item_context.enemy_density > 0.
    assert snapshot.item_context.world_age > 0.

def test_assemble_returns_none_for_old_frame() -> None:
    """時刻が前回 tick より古いフレームは snapshot を発行しない。

    monotonic tick に基づき、now_ns が _last_tick_ns 未満のフレームを抑制します。
    """
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    card = ParsedCard(0, "whip", "weapon", 2, .99, "ok", (100, 100, 400, 500))
    def _make_hud(frame_idx: int, ts: int) -> HudStateV1:
        return HudStateV1(
            "hud_state.v1", "session", frame_idx, ts, "a" * 64, "gameplay", .9, "ok",
            20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
            ("whip",) + (None,) * 11, .9, "b" * 64, (card,), "c" * 64, (),
            False, False, False, .9, "ok",
        )
    snap1 = assembler.assemble(_make_hud(5, 2_000_000_000), TrackedWorldStateV1(5, 2_000_000_000, [], PlayerAnchorState(.5, .5, .9, False)), schema, (1000, 1000))
    assert snap1 is not None
    snap2 = assembler.assemble(_make_hud(3, 1_500_000_000), TrackedWorldStateV1(3, 1_500_000_000, [], PlayerAnchorState(.5, .5, .9, False)), schema, (1000, 1000))
    assert snap2 is None

def test_gameplay_cache_cleared_on_session_change() -> None:
    """session 変更後に gameplay キャッシュが破棄され、旧 session の danger が引き継がれない。

    s1 で danger を蓄積後、s2 の item overlay では item_context が None になります。
    """
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    hud_gp, world_gp = _inputs("gameplay")
    assembler.assemble(hud_gp, world_gp, schema, (1000, 1000))
    card = ParsedCard(0, "knife", "weapon", 2, .99, "ok", (100, 100, 400, 500))
    hud_lu = HudStateV1(
        "hud_state.v1", "session2", 5, 2_000_000_000, "a" * 64, "level_up_items", .9, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
        ("whip",) + (None,) * 11, .9, "b" * 64, (card,), "c" * 64, (),
        False, False, False, .9, "ok",
    )
    world_lu = TrackedWorldStateV1(5, 2_000_000_000, [], PlayerAnchorState(.5, .5, .9, False))
    snapshot = assembler.assemble(hud_lu, world_lu, schema, (1000, 1000))
    assert snapshot is not None
    assert snapshot.item_context is None
    assert snapshot.choices == ()

def test_low_combat_validity_does_not_update_danger_cache() -> None:
    """combat_validity=0 の gameplay フレームは danger キャッシュを上書きしない。

    信頼度不足の低品質フレームで敵密度ゼロに更新されると次の item 選択精度が落ちます。
    """
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    hud_gp, world_gp = _inputs("gameplay")
    assembler.assemble(hud_gp, world_gp, schema, (1000, 1000))
    hud_low = dataclasses.replace(hud_gp, frame_index=5, captured_monotonic_ns=2_000_000_000, screen_state_confidence=.3)
    world_empty = TrackedWorldStateV1(5, 2_000_000_000, [], PlayerAnchorState(.5, .5, .9, False))
    assembler.assemble(hud_low, world_empty, schema, (1000, 1000))
    card = ParsedCard(0, "knife", "weapon", 2, .99, "ok", (100, 100, 400, 500))
    hud_lu = HudStateV1(
        "hud_state.v1", "session", 6, 3_000_000_000, "a" * 64, "level_up_items", .9, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
        ("whip",) + (None,) * 11, .9, "b" * 64, (card,), "c" * 64, (),
        False, False, False, .9, "ok",
    )
    world_lu = TrackedWorldStateV1(6, 3_000_000_000, [], PlayerAnchorState(.5, .5, .9, False))
    snapshot = assembler.assemble(hud_lu, world_lu, schema, (1000, 1000))
    assert snapshot is not None
    assert snapshot.item_context is not None
    assert snapshot.item_context.enemy_density > 0.

def test_item_context_fail_closed_without_gameplay_cache() -> None:
    """gameplay キャッシュ未取得時に item overlay を gameplay として流さない。

    初回が item 画面の場合、空の world が新鮮な danger 特徴として学習に入ることを防ぎます。
    """
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    card = ParsedCard(0, "knife", "weapon", 2, .99, "ok", (100, 100, 400, 500))
    hud_lu = HudStateV1(
        "hud_state.v1", "session", 4, 1_000_000_000, "a" * 64, "level_up_items", .9, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
        ("whip",) + (None,) * 11, .9, "b" * 64, (card,), "c" * 64, (),
        False, False, False, .9, "ok",
    )
    world_lu = TrackedWorldStateV1(4, 1_000_000_000, [], PlayerAnchorState(.5, .5, .9, False))
    snapshot = assembler.assemble(hud_lu, world_lu, schema, (1000, 1000))
    assert snapshot is not None
    assert snapshot.item_context is None
    assert snapshot.choices == ()

def test_duplicate_card_ids_fail_closed() -> None:
    """重複 choice_id を持つカードは presentation 生成で ValueError を上げる。

    model と UI で identity が対応しない状態を UiPresentationSnapshotV1 の検証で排除します。
    """
    schema = DeployObsSchema.default_v1()
    card_a = ParsedCard(0, "whip", "weapon", 2, .99, "ok", (100, 100, 400, 500))
    card_b = ParsedCard(1, "whip", "weapon", 3, .99, "ok", (500, 100, 800, 500))
    hud = HudStateV1(
        "hud_state.v1", "session", 4, 1_000_000_000, "a" * 64, "level_up_items", .9, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
        ("whip",) + (None,) * 11, .9, "b" * 64, (card_a, card_b), "c" * 64, (),
        False, False, False, .9, "ok",
    )
    world = TrackedWorldStateV1(4, 1_000_000_000, [], PlayerAnchorState(.5, .5, .9, False))
    with pytest.raises(ValueError):
        RealObsAssembler().assemble(hud, world, schema, (1000, 1000))

def test_assemble_respects_policy_hz() -> None:
    """tick_interval 未満の入力では None を返し、15 Hz 発行制限を守る。

    60 Hz 等の高頻度入力でも snapshot 発行は 15 Hz に抑制されます。
    """
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    hud_gp, world_gp = _inputs("gameplay")
    snap1 = assembler.assemble(hud_gp, world_gp, schema, (1000, 1000))
    assert snap1 is not None
    hud2 = dataclasses.replace(hud_gp, frame_index=5, captured_monotonic_ns=hud_gp.captured_monotonic_ns + 1_000_000)
    world2 = TrackedWorldStateV1(5, world_gp.timestamp_ns + 1_000_000, [], PlayerAnchorState(.5, .5, .9, False))
    snap2 = assembler.assemble(hud2, world2, schema, (1000, 1000))
    assert snap2 is None  # 1ms は 15 Hz interval (66.7ms) 未満

def test_item_context_world_age_and_snapshot_age_computed_independently() -> None:
    """world_age は cached world timestamp、snapshot_age は tick time から個別に計算される。

    world が tick より 40ms 古い場合に両者が異なる値になることを確認します。
    """
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    hud_gp, _ = _inputs("gameplay")  # ts=1B
    visible = TrackedEntityV1(1, 2, "enemy_normal", "enemy", .9, 1, 4, .7, .5, .2, 0., 0., 0., True, False)
    world_skewed = TrackedWorldStateV1(4, 960_000_000, [visible], PlayerAnchorState(.5, .5, .9, False))
    assembler.assemble(hud_gp, world_skewed, schema, (1000, 1000))
    card = ParsedCard(0, "knife", "weapon", 2, .99, "ok", (100, 100, 400, 500))
    hud_lu = HudStateV1(
        "hud_state.v1", "session", 5, 2_000_000_000, "a" * 64, "level_up_items", .9, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
        ("whip",) + (None,) * 11, .9, "b" * 64, (card,), "c" * 64, (),
        False, False, False, .9, "ok",
    )
    world_lu = TrackedWorldStateV1(5, 2_000_000_000, [], PlayerAnchorState(.5, .5, .9, False))
    snapshot = assembler.assemble(hud_lu, world_lu, schema, (1000, 1000))
    assert snapshot is not None and snapshot.item_context is not None
    assert snapshot.item_context.world_age == pytest.approx(1.04)  # (2B - 960M) / 1e9
    assert snapshot.item_context.last_gameplay_snapshot_age == pytest.approx(1.00)  # (2B - 1B) / 1e9


def test_fallback_card_excluded_from_model_choices() -> None:
    """fallback/gold/chicken カードは model 候補に含めず fallback_kind にのみ反映する。

    ItemSelector が非選択 UI policy の所有物を選択しないよう、choices から除きます。
    """
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    hud_gp, world_gp = _inputs("gameplay")
    assembler.assemble(hud_gp, world_gp, schema, (1000, 1000))
    card_item = ParsedCard(0, "knife", "weapon", 2, .99, "ok", (100, 100, 400, 500))
    card_fallback = ParsedCard(1, "gold_bag", "fallback", 0, .99, "ok", (500, 100, 700, 400))
    hud_lu = HudStateV1(
        "hud_state.v1", "session", 5, 2_000_000_000, "a" * 64, "level_up_fallback", .9, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
        ("whip",) + (None,) * 11, .9, "b" * 64, (card_item, card_fallback), "c" * 64, (),
        False, False, False, .9, "ok",
    )
    world_lu = TrackedWorldStateV1(5, 2_000_000_000, [], PlayerAnchorState(.5, .5, .9, False))
    snapshot = assembler.assemble(hud_lu, world_lu, schema, (1000, 1000))
    assert snapshot is not None and snapshot.item_context is not None
    assert all(c.item_id != "gold_bag" for c in snapshot.choices)
    assert snapshot.item_context.fallback_kind == "gold_bag"
    assert len(snapshot.choices) == 1


def test_item_context_excludes_low_confidence_ui_targets() -> None:
    """UI confidence < 0.35 のカードは model choices から除く。

    クリック不能な候補を ItemSelector が選択するとレベルアップ画面で進行不能になります。
    """
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    hud_gp, world_gp = _inputs("gameplay")
    assembler.assemble(hud_gp, world_gp, schema, (1000, 1000))
    card = ParsedCard(0, "knife", "weapon", 2, .1, "ok", (100, 100, 400, 500))  # confidence=0.1 < 0.35
    hud_lu = HudStateV1(
        "hud_state.v1", "session", 5, 2_000_000_000, "a" * 64, "level_up_items", .9, "ok",
        20., .9, "ok", False, .75, .9, "ok", .5, .9, "ok", 4, .9, "ok",
        ("whip",) + (None,) * 11, .9, "b" * 64, (card,), "c" * 64, (),
        False, False, False, .9, "ok",
    )
    world_lu = TrackedWorldStateV1(5, 2_000_000_000, [], PlayerAnchorState(.5, .5, .9, False))
    snapshot = assembler.assemble(hud_lu, world_lu, schema, (1000, 1000))
    assert snapshot is not None
    assert snapshot.item_context is None
    assert snapshot.choices == ()


def test_skew_ms_uses_joined_timestamps() -> None:
    """hud_world_skew_ms は joined 後の HUD/world タイムスタンプから計算される。

    out-of-order 入力を片側だけ破棄した場合、raw args ではなく joined state のズレを示します。
    """
    schema = DeployObsSchema.default_v1()
    assembler = RealObsAssembler()
    # HUD ts=2B, world ts=1B → joined 後は両者が採用されるので skew = |2B - 1B| = 1B ns = 1000 ms
    hud, _ = _inputs(ts=2_000_000_000)
    world = TrackedWorldStateV1(4, 1_000_000_000, [], PlayerAnchorState(.5, .5, .9, False))
    snapshot = assembler.assemble(hud, world, schema, (1000, 1000))
    assert snapshot is not None
    assert snapshot.diagnostics["hud_world_skew_ms"] == pytest.approx(1000.)

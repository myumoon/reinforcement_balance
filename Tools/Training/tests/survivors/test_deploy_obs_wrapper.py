"""Simulator DeployObs wrapper の release/oracle 分離と Gym 互換動作を検証する。

SB3・torch を使わず、synthetic raw state だけで projection と leakage gate を確認します。
"""

from pathlib import Path

import numpy as np
import pytest

from survivors.deploy_obs_adapter import (
    NamedEstimate, build_deploy_observation, build_oracle_diagnostic_observation, load_schema,
)
from games.survivors.deploy_obs_wrapper import DeployObsWrapper, fresh_vecnormalize

CONFIG = Path(__file__).parents[3] / "Deployment" / "configs" / "deploy_obs_v1.yaml"


def _raw(privileged=None, camera_half_width=10):
    """同じ world state を camera scale だけ変更できる raw state を作る。

    wrapper 自身の projection・visibility・occlusion・clipping 経路を通します。
    """
    return {
        "timestamp_ns": 1_000_000_000,
        "viewport": (100, 100),
        "target_camera": {"center_x": 0., "center_y": 0., "half_width": camera_half_width, "half_height": 10.},
        "hud": {"player_hp": .75, "level": .2},
        "player_world": {"x": 0., "y": 0.},
        "world_entities": [
            {"world_x": 7.5, "world_y": 0., "occluded": False, "timestamp_ns": 1_000_000_000},
            {"world_x": 2., "world_y": 2., "occluded": True, "timestamp_ns": 1_000_000_000},
        ],
        "temporal": {"movement_direction": (.5, 0.), "timestamp_ns": 1_000_000_000},
        "inventory": {"weapon_category": "aura"},
        "privileged": privileged or {"player_pos": [999, 999], "enemy_hp": .9, "cooldown": .8, "all_entity_count": 200, "density": 1.0},
    }


class FakeEnv:
    """Gymnasium の reset/step 戻り値だけを模倣する環境。

    wrapper の契約 test を重い学習依存なしで実行できるようにします。
    """

    def reset(self, **kwargs):
        """同じ synthetic state と空 info を返す。

        seed 等の引数は受け取りますが test fixture 自体は決定的です。
        """
        return _raw(), {}

    def step(self, action):
        """同じ synthetic state と固定遷移情報を返す。

        observation 変換以外の戻り値が維持されることを確認できます。
        """
        return _raw(), 1.0, False, False, {"action": action}


def test_release_projection_and_privileged_leakage():
    """release tensor が画面 semantics のみで決まることを検証する。

    privileged truth を変更しても同じ camera の release tensor は変化しません。
    """
    schema = load_schema(CONFIG)
    wrapper = DeployObsWrapper.release(FakeEnv(), schema)
    first = wrapper.observation(_raw())
    second = wrapper.observation(_raw({"player_pos": [-1, -2], "enemy_hp": .1, "cooldown": .1, "all_entity_count": 1, "density": 0.0}))
    assert np.array_equal(first, second)
    assert wrapper.run_manifest["deploy_obs_mode"] == "release"


@pytest.mark.parametrize("viewport", [(100, 100), [100, 100]])
def test_wrapper_accepts_tuple_and_json_list_viewport(viewport):
    """tuple と JSON 由来 list の正常 viewport を等しく受理する。

    wire decode でコンテナ型だけが変わっても、同じ画面寸法なら
    release projection と policy tensor が一致することを確認します。
    """
    schema = load_schema(CONFIG)
    wrapper = DeployObsWrapper.release(FakeEnv(), schema)
    raw = _raw()
    raw["viewport"] = viewport
    assert np.array_equal(wrapper.observation(raw), wrapper.observation(_raw()))


def test_modes_are_separate_and_oracle_artifact_is_forbidden():
    """release と oracle diagnostic の constructor/gate を分離する。

    oracle は診断用 state を受けられても release artifact を生成できません。
    """
    schema = load_schema(CONFIG)
    release = DeployObsWrapper.release(FakeEnv(), schema)
    oracle = DeployObsWrapper.oracle_diagnostic(FakeEnv(), schema)
    release.assert_release_artifact_allowed()
    with pytest.raises(ValueError):
        oracle.assert_release_artifact_allowed()
    oracle_observation = build_oracle_diagnostic_observation(
        schema, {"enemy_hp": NamedEstimate((.9,), 1_000_000_000)}, 1_000_000_000,
    )
    release_observation = build_deploy_observation(schema, {}, 1_000_000_000)
    release.assert_release_artifact_allowed(release_observation)
    with pytest.raises(ValueError):
        release.assert_release_artifact_allowed(oracle_observation)
    assert oracle.run_manifest["deploy_obs_mode"] == "oracle_diagnostic"
    assert not np.array_equal(release.observation(_raw()), oracle.observation(_raw()))


def test_release_constructor_and_builder_have_no_oracle_switch():
    """release constructor と公開 builder の両方から oracle 切替を除外する。

    wrapper の release 経路は privileged truth を変えても不変であり、
    builder へ旧 bool capability を渡す呼び方も拒否されます。
    """
    schema = load_schema(CONFIG)
    release = DeployObsWrapper.release(FakeEnv(), schema)
    assert np.array_equal(
        release.observation(_raw()),
        release.observation(_raw({"player_pos": [0, 0], "enemy_hp": 0., "cooldown": 0., "all_entity_count": 0, "density": 0.})),
    )
    with pytest.raises(TypeError):
        build_deploy_observation(schema, {}, 0, oracle_diagnostic=True)
    with pytest.raises(ValueError):
        DeployObsWrapper(FakeEnv(), schema, "oracle_diagnostic")


def test_reset_step_and_fresh_vecnormalize_outside():
    """Gym 戻り値と deploy tensor 外側の新規 normalization を検証する。

    source の統計を渡さず factory が wrapper を直接包む構造を確認します。
    """
    schema = load_schema(CONFIG)
    wrapper = DeployObsWrapper.release(FakeEnv(), schema)
    obs, _ = wrapper.reset()
    stepped = wrapper.step(3)
    assert obs.shape == (schema.dim * 3,) and stepped[1:] == (1.0, False, False, {"action": 3})
    calls = []
    normalized = fresh_vecnormalize(wrapper, lambda env, **kwargs: calls.append((env, kwargs)) or "new")
    assert normalized == "new"
    assert calls == [(wrapper, {"norm_obs": True, "training": True})]


def test_wrapper_rejects_unknown_nested_input():
    """raw state の未知 nested key を入力境界で拒否する。

    parser typo や将来 field が release 経路へ黙って入ることを防ぎます。
    """
    schema = load_schema(CONFIG)
    raw = _raw()
    raw["hud"]["hidden_hp"] = 1
    with pytest.raises(ValueError):
        DeployObsWrapper.release(FakeEnv(), schema).observation(raw)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.update(viewport=("bad", "bad")),
        lambda raw: raw.update(viewport=["bad", "bad"]),
        lambda raw: raw.update(viewport=(0, 0)),
        lambda raw: raw.update(viewport=[0, 0]),
        lambda raw: raw.update(viewport=(-1, 5)),
        lambda raw: raw.update(viewport=[-1, 5]),
        lambda raw: raw.update(viewport=(1.5, 2)),
        lambda raw: raw.update(viewport=[1.5, 2]),
        lambda raw: raw.update(viewport=(True, 2)),
        lambda raw: raw.update(viewport=[True, 2]),
        lambda raw: raw.update(viewport=(1,)),
        lambda raw: raw.update(viewport=[1]),
        lambda raw: raw.update(viewport="10"),
        lambda raw: raw["privileged"].update(enemy_hp="bad"),
        lambda raw: raw["world_entities"][0].update(unused=1),
        lambda raw: raw["target_camera"].update(half_width=0),
        lambda raw: raw.update(world_entities="not-a-sequence"),
    ],
)
def test_wrapper_rejects_invalid_nested_values_even_when_release_unused(mutation):
    """全 nested 型の未知・非finite・型・範囲違反を入口で拒否する。

    release が値を特徴へ使わない場合も、壊れた raw payload を黙認しません。
    """
    schema = load_schema(CONFIG)
    raw = _raw()
    mutation(raw)
    with pytest.raises(ValueError):
        DeployObsWrapper.release(FakeEnv(), schema).observation(raw)


def test_camera_scale_changes_projection_clipping_and_visible_count_without_leakage():
    """camera zoom 差で同一 world entity の画面内外と count が変わることを検証する。

    狭いcameraでは敵をclipし、広いcameraでは可視化しますが、off-screen位置や
    privileged count・HP・cooldown はどちらのrelease tensorにも現れません。
    """
    schema = load_schema(CONFIG)
    wrapper = DeployObsWrapper.release(FakeEnv(), schema)
    narrow = wrapper.observation(_raw(camera_half_width=5))
    wide = wrapper.observation(_raw(camera_half_width=10))
    value = slice(0, schema.dim)
    count_offset, _ = schema.layout["visible_enemy_count"]
    nearest_offset, nearest_size = schema.layout["nearest_enemy_offset"]
    assert narrow[value][count_offset] == 0
    assert wide[value][count_offset] == pytest.approx(.05)
    assert np.all(narrow[value][nearest_offset:nearest_offset + nearest_size] == 0)
    changed_truth = _raw(
        {"player_pos": [999, 999], "enemy_hp": 0., "cooldown": 0., "all_entity_count": 999, "density": 1.},
        camera_half_width=5,
    )
    assert np.array_equal(narrow, wrapper.observation(changed_truth))


# ---- DeployObs v2（03-07）----

import json
import math
from copy import deepcopy

from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.deploy_obs_v2_features import effect_duration_s, load_deploy_obs_v2_feature_params
from reinbalance_survivors_contracts.ui_intent import ContractValidationError

V2 = DeployObsSchema.default_v2()
V2_PARAMS = load_deploy_obs_v2_feature_params()
GOLDEN_V2 = Path(__file__).parents[3] / "Common" / "tests" / "fixtures" / "deploy_obs_v2_golden_v1.json"
_EFFECT_SLOT = {"weapon_zone": 0, "weapon_orbit": 1, "weapon_aura": 2, "weapon_projectile": 3}


def _v2_entity(entity_id, class_name, x, y, t_s, *, ttl_true_s=1.0, warning=False, radius=10.0):
    """v2 raw の entity 1 行を作る（武器エフェクトだけ slot・真の残り時間・warning を持つ）。

    slot は _v2_raw の武器スロット（SantaWater / KingBible / Garlic / Knife）に合わせます。
    """
    effect = class_name in _EFFECT_SLOT
    return {
        "entity_id": entity_id, "class_name": class_name, "world_x": x, "world_y": y, "radius_world": radius,
        "occluded": False, "timestamp_ns": int(round(t_s * 1e9)),
        "slot": _EFFECT_SLOT[class_name] if effect else None,
        "ttl_true_s": ttl_true_s if effect else None, "warning": warning if effect else False,
    }


def _v2_raw(entities, t_s=1.0):
    """sim カメラ（800u × 450u）と 1920×1080 viewport の v2 raw state を作る。

    武器は zone=SantaWater・orbit=KingBible・aura=Garlic・projectile=Knife の各1つで、
    どのエフェクト種類も出しうる武器が一意に決まります。
    """
    ts = int(round(t_s * 1e9))
    weapons = [("SantaWater", 1), ("KingBible", 2), ("Garlic", 1), ("Knife", 3), (None, None), (None, None)]
    return {
        "timestamp_ns": ts, "viewport": (1920, 1080),
        "target_camera": {"center_x": 0., "center_y": 0., "half_width": 400., "half_height": 225.},
        "hud": {"player_hp": .5, "level": 3}, "player_world": {"x": 0., "y": 0.},
        "world_entities": entities,
        "temporal": {"movement_direction": (0., 0.), "timestamp_ns": ts},
        "inventory": {
            "weapon_slots": [{"index": i, "type_name": n, "level": lv} for i, (n, lv) in enumerate(weapons)],
            "passive_slots": [{"index": i, "type_name": None, "level": None} for i in range(6)],
            "duration_mult": 1.0,
        },
    }


def _segment(tensor, name, plane=0):
    """policy tensor の value / validity / age 平面から segment を切り出す。

    plane は 0=value, 1=validity, 2=age です。
    """
    offset, size = V2.layout[name]
    return tensor[plane * V2.dim + offset:plane * V2.dim + offset + size]


def test_v2_release_estimates_zone_ttl_from_first_seen_not_true_ttl():
    """release の zone 残り時間は初観測時刻＋持続時間表から推定し、sim の真値を読まない。

    0.5 秒後の推定値が (持続時間 − 0.5) / MaxTtl になり、真の残り時間を変えても tensor が同じことを確かめます。
    """
    wrapper = DeployObsWrapper.release(None, V2)
    wrapper.observation(_v2_raw([_v2_entity(7, "weapon_zone", 50., 0., 1.0)], 1.0))
    tensor = wrapper.observation(_v2_raw([_v2_entity(7, "weapon_zone", 50., 0., 1.5, ttl_true_s=0.01)], 1.5))
    expected = (effect_duration_s("SantaWater", 1, 1.0) - 0.5) / V2_PARAMS["max_projectile_obs_ttl_s"]
    assert _segment(tensor, "weapon_zone_ttl")[0] == pytest.approx(expected, abs=1e-6)
    other = DeployObsWrapper.release(None, V2)
    other.observation(_v2_raw([_v2_entity(7, "weapon_zone", 50., 0., 1.0, ttl_true_s=99.)], 1.0))
    assert np.array_equal(tensor, other.observation(_v2_raw([_v2_entity(7, "weapon_zone", 50., 0., 1.5, ttl_true_s=7.)], 1.5)))


@pytest.mark.parametrize("class_name", sorted(_EFFECT_SLOT))
def test_v2_first_seen_is_dropped_after_max_age_unseen_frames_and_on_reset(class_name):
    """4種類の武器エフェクトとも、max_age フレーム続けて見えなければ初観測時刻を捨てる。

    max_age−1 フレームの欠落なら初観測時刻を保ち、max_age フレームの欠落後は新しい時刻になります。
    reset ではすべての記録を捨てます。画面外（x=1000u）を「見えない」として使います。
    """
    max_age = V2_PARAMS["track_max_age_frames"][class_name]

    class StaticEnv:
        """reset で可視の entity 1つを返すだけの環境。

        reset 後の記録が新しい episode の1フレーム分だけになることを確かめるために使います。
        """

        def reset(self, **kwargs):
            """t=50 s の可視フレームと空 info を返す。

            引数は受け取るだけで使いません。
            """
            return _v2_raw([_v2_entity(9, class_name, 10., 10., 50.)], 50.), {}

    wrapper = DeployObsWrapper.release(StaticEnv(), V2)
    t = 1.0

    def frame(visible):
        """1 フレーム進め、entity を画面内（visible）か画面外に置く。

        時刻は 0.1 秒ずつ進めます。
        """
        nonlocal t
        t += 0.1
        wrapper.observation(_v2_raw([_v2_entity(9, class_name, 10. if visible else 1000., 10., t)], t))

    frame(True)
    first = wrapper._tracks[9][0]
    for _ in range(max_age - 1):
        frame(False)
    frame(True)
    assert wrapper._tracks[9][0] == first
    for _ in range(max_age):
        frame(False)
    assert 9 not in wrapper._tracks
    frame(True)
    assert wrapper._tracks[9][0] == pytest.approx(t) != first
    wrapper.reset()
    assert wrapper._tracks == {9: (50., 1, class_name)}


def test_v2_release_ignores_privileged_entity_fields_and_offscreen_positions():
    """release は slot 以外の sim 専用欄（真の残り時間・warning）と画面外 entity の位置・数を読まない。

    これらを変えても release tensor は同じで、画面内の敵を動かすと変わる（比較が有効な）ことも確かめます。
    """
    def build(ttl, warning, offscreen):
        """条件を変えた同じ画面内 state の release tensor を作る。

        offscreen は画面外に置く boss の x 座標の列です。
        """
        entities = [
            _v2_entity(1, "enemy_normal", 100., 50., 1.0),
            _v2_entity(2, "weapon_zone", -80., 20., 1.0, ttl_true_s=ttl, warning=warning),
            _v2_entity(3, "weapon_orbit", 0., 60., 1.0, ttl_true_s=ttl, warning=warning),
            _v2_entity(4, "weapon_projectile", 30., -30., 1.0, ttl_true_s=ttl, warning=warning),
            _v2_entity(5, "weapon_aura", 0., 0., 1.0, ttl_true_s=ttl, warning=warning, radius=40.),
        ] + [_v2_entity(100 + i, "enemy_boss", x, 0., 1.0) for i, x in enumerate(offscreen)]
        return DeployObsWrapper.release(None, V2).observation(_v2_raw(entities))

    base = build(1.0, False, [900.])
    assert np.array_equal(base, build(7.5, True, [950., -1200., 3000.]))
    moved = DeployObsWrapper.release(None, V2).observation(_v2_raw([_v2_entity(1, "enemy_normal", 10., 50., 1.0)]))
    assert not np.array_equal(base, moved)


def test_v2_release_ignores_effect_slot_field():
    """release は武器エフェクトの sim 上の slot 欄を読まず、HUD のスロットだけから emitter を決める。

    同じ zone の slot 欄を 0 と 5 に変えても release tensor が同じことを確かめます。
    """
    def build(slot):
        """zone の slot 欄だけを変えた release tensor を作る。

        slot 5 は空き枠ですが、release はこの欄を参照しないので結果は変わりません。
        """
        entity = _v2_entity(2, "weapon_zone", -80., 20., 1.0)
        entity["slot"] = slot
        return DeployObsWrapper.release(None, V2).observation(_v2_raw([entity]))

    assert np.array_equal(build(0), build(5))


def test_v2_oracle_reports_ttl_error_against_true_ttl():
    """oracle_diagnostic は zone / orbit の推定残り時間と真値の差を info に出す。

    release は同じ入力で誤差を出さず、projectile（持続時間表が無い）は対象外です。
    """
    duration = effect_duration_s("SantaWater", 1, 1.0)

    class TwoFrameEnv:
        """reset と step で同じ zone を 0.25 秒ずらして返す環境。

        真の残り時間は推定値より 0.1 秒短くしてあります。
        """

        def reset(self, **kwargs):
            """t=1.0 の初観測フレームを返す。

            projectile も1つ置き、誤差の対象外になることを確かめます。
            """
            return _v2_raw([_v2_entity(7, "weapon_zone", 50., 0., 1.0, ttl_true_s=duration - 0.1),
                            _v2_entity(8, "weapon_projectile", 0., 50., 1.0)], 1.0), {"k": 1}

        def step(self, action):
            """t=1.25 のフレームと固定 reward を返す。

            info の既存キーは wrapper がそのまま残します。
            """
            return _v2_raw([_v2_entity(7, "weapon_zone", 50., 0., 1.25, ttl_true_s=duration - 0.35)], 1.25), 0.5, False, False, {"k": 2}

    oracle = DeployObsWrapper.oracle_diagnostic(TwoFrameEnv(), V2)
    _, info = oracle.reset()
    assert info["k"] == 1 and info["deploy_ttl_error_s"] == {7: pytest.approx(0.1, abs=1e-5)}
    _, reward, _, _, info = oracle.step(0)
    assert reward == 0.5 and info["k"] == 2 and info["deploy_ttl_error_s"] == {7: pytest.approx(0.1, abs=1e-5)}
    release = DeployObsWrapper.release(TwoFrameEnv(), V2)
    assert "deploy_ttl_error_s" not in release.reset()[1]


def test_v2_oracle_ttl_error_follows_release_builder_output():
    """oracle の残り時間誤差は、release ビルダーが実際に出した値（clip・無効化込み）と真値の差になる。

    orbit は最古の初観測の本が対象になり、zone は emitter が一意に決まらない構成（FireWand + SantaWater）では
    ビルダーが残り時間を無効にするので誤差を出しません。真値は 0..MaxProjectileObsTtl に clip して比べます。
    """
    max_ttl = V2_PARAMS["max_projectile_obs_ttl_s"]
    orbit_duration = effect_duration_s("KingBible", 2, 1.0)
    oracle = DeployObsWrapper.oracle_diagnostic(None, V2)
    oracle.observation(_v2_raw([_v2_entity(20, "weapon_orbit", 0., 60., 1.0, ttl_true_s=orbit_duration)], 1.0))
    oracle.observation(_v2_raw([
        _v2_entity(20, "weapon_orbit", 0., 60., 1.5, ttl_true_s=orbit_duration - 0.7),
        _v2_entity(21, "weapon_orbit", 60., 0., 1.5, ttl_true_s=orbit_duration - 0.7),
        _v2_entity(7, "weapon_zone", 50., 0., 1.5, ttl_true_s=max_ttl + 5.0),
    ], 1.5))
    # orbit: 推定 = duration − 0.5、真値 = duration − 0.7 → +0.2。zone は SantaWater Lv1 の推定と clip 済み真値 8 の差
    zone_expected = min(effect_duration_s("SantaWater", 1, 1.0), max_ttl) - max_ttl
    assert oracle.last_ttl_error_s == {20: pytest.approx(0.2, abs=1e-5), 7: pytest.approx(zone_expected, abs=1e-5)}

    ambiguous = _v2_raw([_v2_entity(7, "weapon_zone", 50., 0., 1.0, ttl_true_s=1.0)], 1.0)
    ambiguous["inventory"]["weapon_slots"][4] = {"index": 4, "type_name": "FireWand", "level": 1}
    oracle = DeployObsWrapper.oracle_diagnostic(None, V2)
    oracle.observation(ambiguous)
    assert oracle.last_ttl_error_s == {}


def test_v2_rejects_viewport_aspect_mismatching_camera():
    """v2 は viewport と camera の縦横比が違う raw を拒否する（等方座標が歪むため）。

    sim カメラ 800u × 450u に 1000×1000 の viewport を組み合わせると ContractValidationError になります。
    """
    raw = _v2_raw([_v2_entity(1, "enemy_normal", 100., 50., 1.0)])
    DeployObsWrapper.release(None, V2).observation(raw)
    raw["viewport"] = (1000, 1000)
    with pytest.raises(ContractValidationError, match="aspect"):
        DeployObsWrapper.release(None, V2).observation(raw)


def _mutate(path, value=None, *, drop=False):
    """raw の入れ子の位置 path を value にする（drop なら消す）mutation を作る。

    path の最後が新しいキーなら未知キーの追加になります。
    """
    def apply(raw):
        """raw をその場で書き換える。

        deepcopy した raw に対して呼び出します。
        """
        target = raw
        for key in path[:-1]:
            target = target[key]
        if drop:
            del target[path[-1]]
        else:
            target[path[-1]] = value
    return apply


@pytest.mark.parametrize("mutation", [
    _mutate(("privileged",), {}),
    _mutate(("inventory",), drop=True),
    _mutate(("hud", "elapsed_s"), 1.0),
    _mutate(("hud", "level"), .5),
    _mutate(("inventory", "duration_mult"), 0.),
    _mutate(("inventory", "weapon_slots", 0, "type_name"), "Sword"),
    _mutate(("inventory", "weapon_slots", 0, "level"), None),
    _mutate(("inventory", "weapon_slots", 0, "level"), 9),
    _mutate(("inventory", "passive_slots"), []),
    _mutate(("world_entities", 0, "extra"), 1),
    _mutate(("world_entities", 0, "radius_world"), drop=True),
    _mutate(("world_entities", 0, "radius_world"), -1.),
    _mutate(("world_entities", 0, "class_name"), "player_anchor"),
    _mutate(("world_entities", 0, "entity_id"), "1"),
    _mutate(("world_entities", 1, "entity_id"), 1),
    _mutate(("world_entities", 0, "slot"), 0),
    _mutate(("world_entities", 0, "warning"), True),
    _mutate(("world_entities", 1, "slot"), None),
    _mutate(("world_entities", 1, "slot"), 6),
    _mutate(("world_entities", 1, "ttl_true_s"), math.nan),
    _mutate(("world_entities", 1, "warning"), 0),
])
def test_v2_wrapper_rejects_invalid_raw(mutation):
    """v2 raw の未知キー・欠損キー・非数・語彙外・id 重複・欄の矛盾を拒否する。

    release で読まない欄（slot・ttl_true_s・warning）も入口で同じく検証します。
    """
    raw = _v2_raw([_v2_entity(1, "enemy_normal", 10., 0., 1.0), _v2_entity(2, "weapon_zone", 20., 0., 1.0)])
    DeployObsWrapper.release(None, V2).observation(deepcopy(raw))
    mutation(raw)
    with pytest.raises(ContractValidationError):
        DeployObsWrapper.release(None, V2).observation(raw)


def test_v1_and_v2_raw_contracts_are_not_interchangeable():
    """v1 schema の wrapper は v2 raw を、v2 schema の wrapper は v1 raw を拒否する。

    schema version で分岐し、どちらの版も相手の raw を黙って受け取らないことを確かめます。
    """
    with pytest.raises(ContractValidationError):
        DeployObsWrapper.release(None, V2).observation(_raw())
    with pytest.raises(ContractValidationError):
        DeployObsWrapper.release(None, load_schema(CONFIG)).observation(_v2_raw([]))


def _golden_frames(case, scale=0.5):
    """golden の px 入力を、wrapper へ渡す world 座標の v2 raw の時系列へ逆変換する。

    camera 半幅 = viewport 幅 × scale / 2（縦も同じ縮尺）として x = (px/W*2−1)*半幅 で world に戻し、
    各 track が first_seen_s の時刻に初めて現れるフレーム列を作ります。Common 語彙外の class は除きます。
    """
    data = case["input"]
    width, height = data["viewport_wh"]
    half_w, half_h = width * scale / 2, height * scale / 2
    classes = V2_PARAMS["entity_classes"]
    vocabulary = set(classes["enemy"]) | set(classes["gem"]) | set(classes["effect"])
    tracks = [t for t in data["tracks"] if t["class_name"] in vocabulary]

    def world(px, py):
        """px 座標を world 座標へ戻す。

        _project_point の逆変換です。
        """
        return (px / width * 2 - 1) * half_w, (py / height * 2 - 1) * half_h

    slots = {(s["kind"], s["index"]): s for s in data["hud_slots"]}
    for now in sorted({t["first_seen_s"] for t in tracks} | {data["now_s"]}):
        ts = int(round(now * 1e9))
        entities = []
        for t in tracks:
            if t["first_seen_s"] > now:
                continue
            x, y = world(t["cx_px"], t["cy_px"])
            effect = t["class_name"] in classes["effect"]
            entities.append({
                "entity_id": t["track_id"], "class_name": t["class_name"], "world_x": x, "world_y": y,
                "radius_world": t["radius_px"] * 2 * half_w / width, "occluded": t["occluded"], "timestamp_ns": ts,
                "slot": 0 if effect else None, "ttl_true_s": 0.0 if effect else None, "warning": False,
            })
        player = world(*data["player_px"])
        yield {
            "timestamp_ns": ts, "viewport": (width, height),
            "target_camera": {"center_x": 0., "center_y": 0., "half_width": half_w, "half_height": half_h},
            "hud": {"player_hp": data.get("hp_ratio", .5), "level": data.get("player_level", 1)},
            "player_world": {"x": player[0], "y": player[1]}, "world_entities": entities,
            "temporal": {"movement_direction": tuple(data.get("movement_direction") or (0., 0.)), "timestamp_ns": ts},
            "inventory": {
                "weapon_slots": [{k: slots[("weapon", i)][k] for k in ("index", "type_name", "level")} for i in range(6)],
                "passive_slots": [{k: slots[("passive", i)][k] for k in ("index", "type_name", "level")} for i in range(6)],
                "duration_mult": data["duration_mult"],
            },
        }


def test_v2_wrapper_matches_common_golden_through_projection():
    """wrapper の投影・半径 px 変換・初観測時刻の追跡を通した結果が Common golden と一致する。

    mixed_combat は全平面が一致、fire_wand_and_santa_water_ambiguous は sim が常に持つ
    HP・レベル・移動方向の3 segment を除いて一致します。world_invalid_partial_hud は
    sim が常に全 HUD・world を知っている（world_valid=False や欠けた HUD を作らない）ので対象外です。
    """
    golden = json.loads(GOLDEN_V2.read_text(encoding="utf-8"))
    assert golden["schema_hash"] == V2.schema_hash
    cases = {case["name"]: case for case in golden["cases"]}
    assert set(cases) == {"mixed_combat", "fire_wand_and_santa_water_ambiguous", "world_invalid_partial_hud"}
    for name, sim_supplied in (("mixed_combat", ()), ("fire_wand_and_santa_water_ambiguous", ("player_hp", "level", "movement_direction"))):
        wrapper = DeployObsWrapper.release(None, V2)
        for raw in _golden_frames(cases[name]):
            tensor = wrapper.observation(raw)
        expected = np.concatenate([np.asarray(cases[name]["expected"][p], np.float32) for p in ("values", "validity", "age")])
        mask = np.ones(3 * V2.dim, bool)
        for segment in sim_supplied:
            offset, size = V2.layout[segment]
            for plane in range(3):
                mask[plane * V2.dim + offset:plane * V2.dim + offset + size] = False
        np.testing.assert_allclose(tensor[mask], expected[mask], atol=1e-6, err_msg=name)
        assert all(_segment(tensor, s, 1)[0] == 1.0 for s in sim_supplied)

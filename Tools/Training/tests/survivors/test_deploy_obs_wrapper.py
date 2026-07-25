"""Simulator DeployObs wrapper の release/oracle 分離と Gym 互換動作を検証する。

SB3・torch を使わず、synthetic raw state だけで projection と leakage gate を確認します。
"""

from pathlib import Path

import numpy as np
import pytest

from survivors.deploy_obs_adapter import build_deploy_observation, load_schema
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

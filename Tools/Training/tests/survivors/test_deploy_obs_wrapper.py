"""Simulator DeployObs wrapper の release/oracle 分離と Gym 互換動作を検証する。

SB3・torch を使わず、synthetic raw state だけで projection と leakage gate を確認します。
"""

from pathlib import Path

import numpy as np
import pytest

from survivors.deploy_obs_adapter import load_schema
from games.survivors.deploy_obs_wrapper import DeployObsWrapper, fresh_vecnormalize

CONFIG = Path(__file__).parents[3] / "Deployment" / "configs" / "deploy_obs_v1.yaml"


def _raw(privileged=None, scale=1):
    """画面内外 track を含む最小 raw state を作る。

    privileged 値を変えて release tensor が不変か比較できる fixture です。
    """
    return {
        "timestamp_ns": 1_000_000_000,
        "viewport": (100 * scale, 100 * scale),
        "hud": {"player_hp": .75, "level": .2},
        "player_screen": (50 * scale, 50 * scale),
        "tracks": [
            {"screen_x": 75 * scale, "screen_y": 50 * scale, "visible": True, "occluded": False, "clipped": False, "timestamp_ns": 1_000_000_000},
            {"screen_x": 1 * scale, "screen_y": 1 * scale, "visible": False, "occluded": False, "clipped": False, "timestamp_ns": 1_000_000_000},
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

    player_pos・hidden HP/cooldown・全 entity 情報や camera scale を変えても一致します。
    """
    schema = load_schema(CONFIG)
    wrapper = DeployObsWrapper.release(FakeEnv(), schema)
    first = wrapper.observation(_raw())
    second = wrapper.observation(_raw({"player_pos": [-1, -2], "enemy_hp": .1, "cooldown": .1, "all_entity_count": 1, "density": 0.0}, scale=2))
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

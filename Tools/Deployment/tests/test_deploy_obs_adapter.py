"""DeployObs release adapter の可視性・鮮度・leakage 防止を検証する。

synthetic named estimates と early/mid/late fixture だけで parser 未実装時も smoke できます。
"""

from pathlib import Path

import numpy as np
import pytest

from survivors.deploy_obs_adapter import (
    NamedEstimate, build_deploy_observation, deploy_obs_producer_hash, load_schema, normalized_category,
    screen_to_centered, visible_track_estimates,
)

CONFIG = Path(__file__).parents[1] / "configs" / "deploy_obs_v1.yaml"
FIXTURES = {
    "early": {"player_hp": (1.0,), "level": (0.0,)},
    "mid": {"player_hp": (.6,), "level": (.4,), "movement_direction": (.5, -.5)},
    "late": {"player_hp": (.2,), "level": (.9,), "movement_direction": (-1., 1.)},
}


def _estimates(values, now=1_000_000_000):
    """fixture 値へ source timestamp を付ける。

    flat array ではなく feature 名ごとの estimate として adapter に渡します。
    """
    return {name: NamedEstimate(tuple(value), now) for name, value in values.items()}


@pytest.mark.parametrize("stage", FIXTURES)
def test_synthetic_early_mid_late_smoke(stage):
    """三段階 fixture から有限 tensor を生成する。

    parser が無くても共有 hash・dim・range の契約を end-to-end で確認します。
    """
    schema = load_schema(CONFIG)
    obs = build_deploy_observation(schema, _estimates(FIXTURES[stage]), 1_000_000_000)
    assert obs.schema_hash == schema.schema_hash
    assert obs.as_policy_tensor().shape == (schema.dim * 3,)
    assert np.all(np.isfinite(obs.as_policy_tensor()))


def test_missing_and_unobservable_are_neutral_invalid_old():
    """欠損と unobservable の表現を全 resolver で統一する。

    privileged enemy_hp/cooldown を入力しても release schema は neutral・0・1 を返します。
    """
    schema = load_schema(CONFIG)
    obs = build_deploy_observation(schema, _estimates({"enemy_hp": (.9,), "cooldown": (.8,)}), 1_000_000_000)
    for name in ("player_screen_pos", "enemy_hp", "cooldown"):
        offset, size = schema.layout[name]
        assert np.all(obs.values[offset:offset + size] == 0)
        assert np.all(obs.validity[offset:offset + size] == 0)
        assert np.all(obs.age[offset:offset + size] == 1)


def test_age_and_stale_validity_decay():
    """source timestamp から age と stale decay を算出する。

    threshold 直後は信頼度が段階的に下がり、max age では欠損表現になります。
    """
    schema = load_schema(CONFIG)
    now = 1_000_000_000
    obs = build_deploy_observation(schema, {"player_hp": NamedEstimate((.5,), now - 500_000_000)}, now)
    offset, _ = schema.layout["player_hp"]
    assert obs.age[offset] == pytest.approx(.5)
    assert 0 < obs.validity[offset] < 1
    old = build_deploy_observation(schema, {"player_hp": NamedEstimate((.5,), 0)}, now)
    assert old.values[offset] == 0 and old.validity[offset] == 0 and old.age[offset] == 1


def test_visible_tracks_only_clipping_occlusion_and_camera_scale():
    """on-screen track だけで count と中心相対 offset を作る。

    off-screen truth・occlusion・clipping を除外し、解像度差でも正規化座標を一致させます。
    """
    now = 10
    visible = {"screen_x": 75., "screen_y": 25., "visible": True, "occluded": False, "clipped": False, "timestamp_ns": now}
    hidden = [
        {"screen_x": 10., "screen_y": 10., "visible": False, "occluded": False, "clipped": False, "timestamp_ns": now},
        {"screen_x": 10., "screen_y": 10., "visible": True, "occluded": True, "clipped": False, "timestamp_ns": now},
        {"screen_x": 10., "screen_y": 10., "visible": True, "occluded": False, "clipped": True, "timestamp_ns": now},
    ]
    out = visible_track_estimates([visible, *hidden], (100, 100), now)
    assert out["visible_enemy_count"].value == (.05,)
    assert out["nearest_enemy_offset"].value == (.5, -.5)
    assert screen_to_centered(150, 50, 200, 200) == (.5, -.5)


def test_category_unknown_and_invalid_adapter_inputs():
    """categorical unknown と adapter 入力境界を検証する。

    未知 enum は予約 id に正規化し、未知 feature や未来 timestamp は拒否します。
    """
    assert normalized_category("new_weapon") == 1
    schema = load_schema(CONFIG)
    with pytest.raises(ValueError):
        build_deploy_observation(schema, {"privileged": NamedEstimate((1.,), 0)}, 0)
    with pytest.raises(ValueError):
        build_deploy_observation(schema, {"player_hp": NamedEstimate((1.,), 2)}, 1)


def test_schema_change_invalidates_producer_identity():
    """schema producer hash の変更が既存 identity と一致しないことを示す。

    fidelity gating 本体を変更せず、DeployObs 更新時の baseline 失効を明示します。
    """
    schema = load_schema(CONFIG)
    wire = schema.to_wire()
    wire["fields"][0]["stale_after_ms"] += 1
    from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
    changed = DeployObsSchema.from_wire(wire)
    adapter_hash = "a" * 64
    baseline = deploy_obs_producer_hash(schema, adapter_hash)
    assert deploy_obs_producer_hash(changed, adapter_hash) != baseline
    assert deploy_obs_producer_hash(schema, "b" * 64) != baseline

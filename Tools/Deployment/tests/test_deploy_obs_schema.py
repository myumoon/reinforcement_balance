"""DeployObsV1 schema の決定性と fail-closed 境界を検証する。

設定の欠落・重複・未知分類・不正範囲・観測三平面のずれを小さな反例で確認します。
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from reinbalance_survivors_contracts.deploy_obs import DeployObservation, DeployObsSchema
from survivors.deploy_obs_adapter import load_schema

CONFIG = Path(__file__).parents[1] / "configs" / "deploy_obs_v1.yaml"
EXPECTED_DEPLOY_HASH = "0f758972a905e49d19a0b8ba0dbec6bdfe0ff6ae405b5b6a76f270db496a327c"


def _wire():
    """検証用 config の独立コピーを返す。

    各 reject test が他の test の入力を書き換えないようにします。
    """
    return deepcopy(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))


def test_stable_hash_dim_and_ordered_layout():
    """同じ YAML から hash・dim・layout を決定的に再生成する。

    offset は field の順序だけから積算されることも合わせて確認します。
    """
    first, second = load_schema(CONFIG), load_schema(CONFIG)
    assert first.schema_hash == second.schema_hash == EXPECTED_DEPLOY_HASH
    assert first.schema_hash == DeployObsSchema.default_v1().schema_hash
    assert first.dim == 13
    assert list(first.layout) == [field["name"] for field in _wire()["fields"]]
    assert [offset for offset, _ in first.layout.values()] == [0, 1, 2, 4, 6, 7, 9, 10, 11, 12]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "neutral", "source"])
def test_schema_rejects_invalid_segments(mutation):
    """代表的な schema 破損を全て拒否する。

    silent default を許さず producer 間の layout 差を起動時に止めます。
    """
    wire = _wire()
    if mutation == "missing":
        wire["fields"].pop()
    elif mutation == "duplicate":
        wire["fields"][-1]["name"] = wire["fields"][0]["name"]
    elif mutation == "neutral":
        wire["fields"][0]["neutral"] = 2.0
    else:
        wire["fields"][0]["source_class"] = "privileged_truth"
    with pytest.raises(ValueError):
        DeployObsSchema.from_wire(wire)


def test_wire_unknown_and_missing_keys_rejected():
    """schema と field の未知・欠落キーを拒否する。

    typo や将来 field を旧 consumer が黙認しないことを保証します。
    """
    wire = _wire()
    wire["surprise"] = True
    with pytest.raises(ValueError):
        DeployObsSchema.from_wire(wire)
    wire = _wire()
    del wire["fields"][0]["neutral"]
    with pytest.raises(ValueError):
        DeployObsSchema.from_wire(wire)


def test_observation_planes_are_immutable_and_shape_checked():
    """三平面の shape と防御的コピーを検証する。

    policy tensor が value・validity・age の全てを含み、検証後に変更されないことを示します。
    """
    schema = load_schema(CONFIG)
    values = np.zeros(schema.dim, np.float32)
    validity = np.ones(schema.dim, np.float32)
    age = np.full(schema.dim, .25, np.float32)
    obs = DeployObservation(values, validity, age, schema.schema_hash, 1)
    assert obs.as_policy_tensor().shape == (schema.dim * 3,)
    assert np.all(obs.as_policy_tensor()[schema.dim:2 * schema.dim] == 1)
    values[0] = .9
    assert obs.values[0] == 0 and not obs.values.flags.writeable
    with pytest.raises(ValueError):
        DeployObservation(values, validity[:-1], age, schema.schema_hash, 1)


def test_nonfinite_and_missing_representation_rejected_at_use():
    """NaN と不統一な欠損表現を利用直前 gate で拒否する。

    欠損は必ず neutral・validity 0・age 1 の組として扱います。
    """
    schema = load_schema(CONFIG)
    with pytest.raises(ValueError):
        DeployObservation(np.full(schema.dim, np.nan), np.ones(schema.dim), np.zeros(schema.dim), schema.schema_hash, 1)
    obs = DeployObservation(np.zeros(schema.dim), np.zeros(schema.dim), np.zeros(schema.dim), schema.schema_hash, 1)
    with pytest.raises(ValueError):
        obs.validate_for(schema)

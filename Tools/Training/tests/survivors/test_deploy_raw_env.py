"""実 UE5 PIE の deploy_raw と DeployRawEnv の fail-closed 動作を検証する。

Python テストでは deploy_raw_pie_v1.json を正として読み、raw dict が DeployObsWrapper の v2 tensor になることを確かめます。
LLT fixture との producer 共通部分も比較します。deploy_raw_llt_v1.json は C++ LLT の [fixture] テスト専用です。
"""

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from games.survivors.deploy_obs_wrapper import DeployObsWrapper
from games.survivors.deploy_raw_env import DeployRawEnv, deploy_raw_to_raw
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.ui_intent import ContractValidationError as ContractError

FIXTURE = Path(__file__).parent / "fixtures" / "deploy_raw_pie_v1.json"
LLT_FIXTURE = Path(__file__).parent / "fixtures" / "deploy_raw_llt_v1.json"
V2 = DeployObsSchema.default_v2()


def _payloads():
    """fixture の deploy_raw を応答順（reset → step 60, 61, ...）に並べて返す。

    /reset はトップレベル、/step は info の下にあるので、それぞれから取り出します。
    """
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [r["deploy_raw"] if r["endpoint"] == "/reset" else r["info"]["deploy_raw"] for r in data["responses"]]


class FakeSurvivorsEnv:
    """SurvivorsEnv の set_params / reset / step / last_reset_response だけを模倣する。

    reset は fixture の /reset 応答を last_reset_response に置き、step は以降の deploy_raw を info で返します。
    """

    def __init__(self, params_ok=True, drop_deploy_raw=False):
        """fixture を読み込み、/params の成否と deploy_raw の欠落を切り替えられるようにする。

        drop_deploy_raw=True は deploy_raw を出さない（無効のままの）UE5 を表します。
        """
        self.payloads, self.params_ok, self.drop = _payloads(), params_ok, drop_deploy_raw
        self.params_calls, self.cursor = [], 0
        self.last_reset_response = None

    def set_params(self, **kwargs):
        """/params 呼び出しを記録し、設定した成否を返す。

        SurvivorsEnv.set_params は失敗時に例外ではなく False を返すので同じ形にします。
        """
        self.params_calls.append(kwargs)
        return self.params_ok

    def reset(self, *, seed=None, options=None):
        """/reset 応答を last_reset_response に置き、flat obs と空 info を返す。

        BaseUE5Env.reset と同じく info は空で、deploy_raw は応答全体からしか読めません。
        """
        self.cursor = 1
        self.last_reset_response = {"obs": [0.0], "obs_schema_hash": "h"}
        if not self.drop:
            self.last_reset_response["deploy_raw"] = self.payloads[0]
        return np.zeros(1, np.float32), {}

    def step(self, action):
        """次の deploy_raw を info に入れて返す。

        既存の info キー（base_reward）も残し、DeployRawEnv が deploy_raw だけを取り除くことを確かめます。
        """
        info = {"base_reward": 1.0}
        if not self.drop:
            info["deploy_raw"] = self.payloads[self.cursor]
        self.cursor += 1
        return np.zeros(1, np.float32), 1.0, False, False, info


def test_fixture_reset_and_steps_become_v2_tensors_through_release_wrapper():
    """fixture の reset と連続 step が DeployRawEnv → release wrapper で v2 tensor になる。

    reset 前に deploy_raw=True を /params へ送り、step の info から deploy_raw が取り除かれ、
    敵・ジェム・武器エフェクトの class 名と id がそのまま raw dict に入ることを確認します。
    """
    fake = FakeSurvivorsEnv()
    env = DeployRawEnv(fake)
    raw, info = env.reset(seed=73013)
    assert fake.params_calls == [{"deploy_raw": True}] and info == {}
    assert raw["world_entities"][0]["class_name"] == "weapon_aura" and raw["hud"] == {"player_hp": 1.0, "level": 1}
    wrapper = DeployObsWrapper.release(DeployRawEnv(FakeSurvivorsEnv()), V2)
    tensor, _ = wrapper.reset(seed=73013)
    assert tensor.shape == (3 * V2.dim,)
    payloads = _payloads()
    for payload in payloads[1:]:
        raw, reward, terminated, truncated, info = env.step(0)
        assert info == {"base_reward": 1.0} and reward == 1.0 and not terminated and not truncated
        assert [(e["entity_id"], e["class_name"]) for e in raw["world_entities"]] == [(e["entity_id"], e["class_name"]) for e in payload["entities"]]
        assert all(e["occluded"] is False and e["timestamp_ns"] == raw["timestamp_ns"] for e in raw["world_entities"])
        tensor, *_ = wrapper.step(0)
        assert tensor.shape == (3 * V2.dim,) and np.all(np.isfinite(tensor))
    classes = {e["class_name"] for p in payloads for e in p["entities"]}
    assert classes == {"enemy_normal", "gem_blue", "weapon_projectile", "weapon_zone", "weapon_orbit", "weapon_aura"}


def test_pie_fixture_matches_llt_raw_producer_fields():
    """PIE と LLT の fixture が共通の deploy_raw producer 項目を返す。

    reset の deploy_raw 全体と obs schema hash、各 step の共通項目を比べます。
    敵やジェムはレベル設定で数が変わるため、step の entities は比較しません。
    """
    pie = json.loads(FIXTURE.read_text(encoding="utf-8"))
    llt = json.loads(LLT_FIXTURE.read_text(encoding="utf-8"))
    assert pie["obs_schema_hash"] == llt["obs_schema_hash"]

    pie_reset = next(r for r in pie["responses"] if r["endpoint"] == "/reset")
    llt_reset = next(r for r in llt["responses"] if r["endpoint"] == "/reset")
    assert pie_reset["deploy_raw"] == llt_reset["deploy_raw"]

    pie_steps = [r for r in pie["responses"] if r["endpoint"] == "/step"]
    llt_steps = [r for r in llt["responses"] if r["endpoint"] == "/step"]
    assert [r["step"] for r in pie_steps] == [r["step"] for r in llt_steps]
    fields = ("elapsed_s", "camera", "player", "weapon_slots", "passive_slots", "duration_mult")
    for pie_response, llt_response in zip(pie_steps, llt_steps):
        pie_raw = pie_response["info"]["deploy_raw"]
        llt_raw = llt_response["info"]["deploy_raw"]
        assert {key: pie_raw[key] for key in fields} == {key: llt_raw[key] for key in fields}


def test_slot_type_ids_map_to_common_vocabulary_names():
    """C++ enum 値の type_id が Common 語彙の名前に、0 が空き枠（None）になる。

    fixture の初期武器（Knife, SantaWater, KingBible, Garlic, FireWand, Peachone）と並びが一致します。
    """
    raw = deploy_raw_to_raw(_payloads()[1], (1920, 1080), (0.0, 0.0))
    names = [slot["type_name"] for slot in raw["inventory"]["weapon_slots"]]
    assert names == json.loads(FIXTURE.read_text(encoding="utf-8"))["initial_weapons"]
    assert all(slot == {"index": i, "type_name": None, "level": None} for i, slot in enumerate(raw["inventory"]["passive_slots"]))


def test_movement_direction_comes_from_player_world_delta():
    """移動方向は前フレームとの自機位置の差を単位ベクトルにしたもので、reset 直後は (0, 0)。

    fixture の step 60 → 61 は x 方向にだけ動いている（-x）ので (-1, 0) になります。
    """
    env = DeployRawEnv(FakeSurvivorsEnv())
    raw, _ = env.reset()
    assert raw["temporal"]["movement_direction"] == (0.0, 0.0)
    env.step(0)
    raw, *_ = env.step(0)
    assert raw["temporal"]["movement_direction"] == pytest.approx((-1.0, 0.0))


def test_env_rejects_disabled_or_missing_deploy_raw():
    """/params の失敗や deploy_raw が無い応答は flat obs に戻さず例外にする。

    reset・step の両方の経路で同じく拒否します。
    """
    with pytest.raises(ContractError, match="params"):
        DeployRawEnv(FakeSurvivorsEnv(params_ok=False)).reset()
    with pytest.raises(ContractError, match="reset response"):
        DeployRawEnv(FakeSurvivorsEnv(drop_deploy_raw=True)).reset()
    env = DeployRawEnv(FakeSurvivorsEnv())
    env.reset()
    env.env.drop = True
    with pytest.raises(ContractError, match="step info"):
        env.step(0)


def _set(path, value):
    """payload の入れ子の位置 path を value に書き換える mutation を作る。

    value が _DELETE ならキーを消し、path の最後が新しいキーなら未知キーの追加になります。
    """
    def apply(payload):
        """payload をその場で書き換える。

        deepcopy した payload に対して呼び出します。
        """
        target = payload
        for key in path[:-1]:
            target = target[key]
        if value is _DELETE:
            del target[path[-1]]
        else:
            target[path[-1]] = value
    return apply


_DELETE = object()


@pytest.mark.parametrize("mutation", [
    _set(("extra",), 1),
    _set(("duration_mult",), _DELETE),
    _set(("schema_version",), "survivors_deploy_raw.v0"),
    _set(("elapsed_s",), math.nan),
    _set(("elapsed_s",), "1.0"),
    _set(("elapsed_s",), -1.0),
    _set(("camera", "zoom"), 1.0),
    _set(("camera", "cull_margin"), _DELETE),
    _set(("camera", "half_width"), 0),
    _set(("camera", "center_x"), math.inf),
    _set(("player", "hp_ratio"), 1.5),
    _set(("player", "level"), True),
    _set(("player", "world_y"), None),
    _set(("duration_mult",), 0),
    _set(("weapon_slots",), []),
    _set(("weapon_slots", 0, "index"), 1),
    _set(("weapon_slots", 0, "extra"), 0),
    _set(("weapon_slots", 0, "level"), 0),
    _set(("passive_slots", 0, "level"), 1),
    _set(("passive_slots", 0, "type_id"), -1),
    _set(("entities", 0, "extra"), 0),
    _set(("entities", 0, "warning"), _DELETE),
    _set(("entities", 0, "world_x"), math.nan),
    _set(("entities", 0, "radius_world"), "8"),
    _set(("entities", 0, "class_name"), "hazard"),
    _set(("entities", 0, "entity_id"), -1),
    _set(("entities", 0, "entity_id"), True),
    _set(("entities", 0, "slot"), "1"),
    _set(("entities", 0, "ttl_true_s"), math.inf),
    _set(("entities", 0, "warning"), 1),
    _set(("entities",), {}),
])
def test_deploy_raw_parser_is_fail_closed(mutation):
    """deploy_raw の未知キー・欠損キー・非数・型違い・範囲外を全階層で拒否する。

    トップ・camera・player・スロット・entity の各階層に同じ規則が効いていることを確かめます。
    """
    payload = deepcopy(_payloads()[1])
    deploy_raw_to_raw(payload, (1920, 1080), (0.0, 0.0))
    mutation(payload)
    with pytest.raises(ContractError):
        deploy_raw_to_raw(payload, (1920, 1080), (0.0, 0.0))

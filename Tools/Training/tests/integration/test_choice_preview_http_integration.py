"""level-up preview のHTTP facadeとtyped clientの結合契約を検証する。

初心者向け: fake endpointでもpending ID・候補集合・schemaがずれた応答を必ず拒否する。
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]


def _load_survivors_module(monkeypatch):
    """重量runtimeをfakeへ置換してSurvivors clientだけを読み込む。

    初心者向け: UE5を起動できない環境でもwire validationを同じmethod経由で試す。
    """

    gym = types.ModuleType("gymnasium")
    gym.Env = object
    gym.spaces = types.SimpleNamespace(Discrete=lambda *_: None, Box=lambda **_: None)
    monkeypatch.setitem(sys.modules, "gymnasium", gym)

    numpy = types.ModuleType("numpy")
    numpy.ndarray = list
    numpy.float32 = float
    numpy.inf = float("inf")
    numpy.array = lambda value, dtype=None: list(value)
    monkeypatch.setitem(sys.modules, "numpy", numpy)

    base = types.ModuleType("base.base_ue5_env")

    class FakeBase:
        """test transportへPOSTを転送する最小base。

        初心者向け: retry等の既存基盤ではなくpreview response契約だけにテストを集中する。
        """

        def _post_json(self, endpoint, payload, timeout=10, retries=2):
            return self.transport.post(endpoint, payload)

    base.BaseUE5Env = FakeBase
    monkeypatch.setitem(sys.modules, "base.base_ue5_env", base)

    sb3 = types.ModuleType("stable_baselines3")
    sb3_common = types.ModuleType("stable_baselines3.common")
    sb3_monitor = types.ModuleType("stable_baselines3.common.monitor")

    class Monitor:
        """wrapped envを保持するだけのtest monitor。

        初心者向け: 本物のSB3を導入せずforward methodの動作だけを確認する。
        """

        def __init__(self, env):
            self.env = env

    sb3_monitor.Monitor = Monitor
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", sb3_common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.monitor", sb3_monitor)

    monkeypatch.syspath_prepend(str(ROOT / "Tools/Training"))
    sys.modules.pop("games.survivors.survivors_env", None)
    sys.modules.pop("games.survivors.choice_preview", None)
    return importlib.import_module("games.survivors.survivors_env")


class FakePreviewTransport:
    """pending choiceをproduction適用した結果のように返すfake。

    初心者向け: responseを差し替えてclientのfail-closed検証も同じ入口から試せる。
    """

    def __init__(self) -> None:
        self.response = {
            "decision_id": "level-up-1-1-2",
            "obs_schema_hash": "schema",
            "base_obs": [0.0, 0.5, 1.0],
            "previews": [
                {
                    "choice_id": "choice-0",
                    "projected_obs": [0.25, 0.5, 1.0],
                    "changed_segments": ["weapon_slots"],
                },
                {
                    "choice_id": "choice-1",
                    "projected_obs": [0.0, 0.75, 1.0],
                    "changed_segments": ["passive_slots"],
                },
            ],
        }

    def post(self, endpoint, payload):
        """preview endpointへの正しいdecision requestだけを受け付ける。

        初心者向け: clientがchoice内容やobs patchを送っていないこともここで確認する。
        """

        assert endpoint == "/preview_level_up"
        assert payload == {"decision_id": "level-up-1-1-2"}
        return self.response


def _make_env(module):
    """初期化済みclient相当の最小stateを作る。

    初心者向け: schemaから得たdim/name/hashをtyped parserへ渡す実経路を再現する。
    """

    env = module.SurvivorsEnv.__new__(module.SurvivorsEnv)
    env._expected_schema_hash = "schema"
    env._obs_schema = [
        {"name": "weapon_slots", "dim": 1},
        {"name": "passive_slots", "dim": 1},
        {"name": "other", "dim": 1},
    ]
    env.observation_space = types.SimpleNamespace(shape=(3,))
    env.transport = FakePreviewTransport()
    return env


def test_preview_facade_returns_typed_exact_observations(monkeypatch) -> None:
    """正常応答をimmutable typed contractへ変換する。

    初心者向け: 候補順ではなくchoice ID集合で結び、各raw obsをそのまま保持する。
    """

    module = _load_survivors_module(monkeypatch)
    env = _make_env(module)
    preview = module.SurvivorsMonitor(env).preview_level_up(
        "level-up-1-1-2", {"choice-0", "choice-1"}
    )

    assert isinstance(preview, module.SurvivorsLevelUpPreview)
    assert preview.base_obs == (0.0, 0.5, 1.0)
    assert preview.by_choice_id["choice-0"].projected_obs == (0.25, 0.5, 1.0)
    assert preview.by_choice_id["choice-1"].changed_segments == ("passive_slots",)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("obs_schema_hash", "stale", "obs_schema_hash"),
        ("decision_id", "level-up-old", "decision_id"),
        ("previews", [], "choice_id set"),
    ],
)
def test_preview_facade_rejects_binding_mismatch(
    monkeypatch, field: str, value, message: str
) -> None:
    """schema・decision・choice集合の片側ずれを拒否する。

    初心者向け: 正しい数値に見えても別decisionの候補ならpolicyへ渡してはいけない。
    """

    module = _load_survivors_module(monkeypatch)
    env = _make_env(module)
    env.transport.response[field] = value

    with pytest.raises(ValueError, match=message):
        env.preview_level_up("level-up-1-1-2", {"choice-0", "choice-1"})


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_preview_facade_rejects_non_finite_observation(
    monkeypatch, bad_value: float
) -> None:
    """base/projectedの非finite値を全経路で拒否する。

    初心者向け: NaNやInfinityがscorerへ入り学習結果を壊す前に通信境界で止める。
    """

    module = _load_survivors_module(monkeypatch)
    env = _make_env(module)
    env.transport.response["previews"][0]["projected_obs"][0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        env.preview_level_up("level-up-1-1-2", {"choice-0", "choice-1"})


@pytest.mark.parametrize(
    "bad_schema",
    [
        [
            {"name": "weapon_slots", "dim": 1},
            {"name": "weapon_slots", "dim": 1},
            {"name": "other", "dim": 1},
        ],
        [
            {"name": "weapon_slots", "dim": 1},
            {"name": "passive_slots", "dim": 1},
        ],
        [
            {"name": "weapon_slots", "dim": True},
            {"name": "passive_slots", "dim": 1},
            {"name": "other", "dim": 1},
        ],
    ],
)
def test_preview_facade_rejects_invalid_live_schema(
    monkeypatch, bad_schema: list[dict]
) -> None:
    """重複名・合計dim不一致・bool dimのlive schemaを拒否する。

    初心者向け: hashだけ一致してもsegment境界が曖昧ならchanged_segmentsを安全に解釈できない。
    """

    module = _load_survivors_module(monkeypatch)
    env = _make_env(module)
    env._obs_schema = bad_schema

    with pytest.raises(ValueError, match="obs_schema"):
        env.preview_level_up("level-up-1-1-2", {"choice-0", "choice-1"})


def test_preview_facade_rejects_unknown_wire_field(monkeypatch) -> None:
    """version外のresponse fieldを黙認しない。

    初心者向け: serverとclientの契約versionがずれた応答を既知形式としてscorerへ渡さない。
    """

    module = _load_survivors_module(monkeypatch)
    env = _make_env(module)
    env.transport.response["projection_version"] = 2

    with pytest.raises(ValueError, match="unknown"):
        env.preview_level_up("level-up-1-1-2", {"choice-0", "choice-1"})

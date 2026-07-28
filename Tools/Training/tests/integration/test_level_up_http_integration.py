"""level-up HTTP の pending・ack・再送を fake transport で検証する。

初心者向け: タイムアウト後に同じ選択を再送しても、適用が一度だけになる契約を確認する。
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def _load_survivors_module(monkeypatch):
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
        def _post_json(self, endpoint, payload, timeout=10, retries=2):
            for attempt in range(retries + 1):
                try:
                    return self.transport.post(endpoint, payload)
                except TimeoutError:
                    if attempt >= retries:
                        raise
            raise AssertionError("unreachable")

    base.BaseUE5Env = FakeBase
    monkeypatch.setitem(sys.modules, "base.base_ue5_env", base)

    sb3 = types.ModuleType("stable_baselines3")
    sb3_common = types.ModuleType("stable_baselines3.common")
    sb3_monitor = types.ModuleType("stable_baselines3.common.monitor")

    class Monitor:
        def __init__(self, env):
            self.env = env

    sb3_monitor.Monitor = Monitor
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", sb3_common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.monitor", sb3_monitor)

    monkeypatch.syspath_prepend(str(ROOT / "Tools/Training"))
    sys.modules.pop("games.survivors.survivors_env", None)
    return importlib.import_module("games.survivors.survivors_env")


class FakeLevelUpHttpTransport:
    """UE5 endpoint と同じ JSON 契約を持つ stateful fake。"""

    def __init__(self):
        self.applies = 0
        self.cached: dict | None = None
        self.timeout_delivered = False
        self.item_selection_mode = "external"

    def step(self) -> dict:
        return {
            "obs": [float(self.applies)],
            "reward": 0.0,
            "done": False,
            "truncated": False,
            "info": {
                "level_up_pending": self.applies == 0,
                "level_up_choices": (
                    [{"choice_id": "choice-0", "type": "weapon_new"}]
                    if self.applies == 0
                    else []
                ),
                "level_up_decision_id": (
                    "level-up-1-1-2" if self.applies == 0 else ""
                ),
            },
        }

    def post(self, endpoint, payload, **_kwargs) -> dict:
        if endpoint == "/params":
            assert payload == {"item_selection_mode": "auto"}
            assert self.cached is not None
            self.item_selection_mode = "auto"
            return {"status": "ok"}

        assert endpoint == "/level_up_choice"
        assert payload == {
            "decision_id": "level-up-1-1-2",
            "choice_id": "choice-0",
        }
        if self.cached is None:
            assert self.item_selection_mode == "external"
            self.applies += 1
            self.cached = {
                "status": "applied",
                "decision_id": payload["decision_id"],
                "choice_id": payload["choice_id"],
                "obs": [1.0],
                "obs_schema_hash": "schema",
                "info": {
                    "level_up_pending": False,
                    "level_up_choices": [],
                },
            }
        if not self.timeout_delivered:
            self.timeout_delivered = True
            raise TimeoutError("response lost after apply")
        return self.cached


def test_monitor_choice_ack_and_duplicate_response_are_idempotent(monkeypatch) -> None:
    module = _load_survivors_module(monkeypatch)
    transport = FakeLevelUpHttpTransport()
    env = module.SurvivorsEnv.__new__(module.SurvivorsEnv)
    env._expected_schema_hash = "schema"
    env._prev_obs = [0.0]
    env.transport = transport
    monitor = module.SurvivorsMonitor(env)

    pending = transport.step()
    assert pending["info"]["level_up_pending"] is True
    assert pending["info"]["level_up_choices"][0]["choice_id"] == "choice-0"

    first = monitor.choose_level_up("level-up-1-1-2", "choice-0")
    transport.post("/params", {"item_selection_mode": "auto"})
    duplicate = monitor.choose_level_up("level-up-1-1-2", "choice-0")

    assert first == duplicate
    assert first[0] == [1.0]
    assert first[1]["level_up_pending"] is False
    assert transport.applies == 1

    after = transport.step()
    assert after["obs"] == [1.0]
    assert after["info"]["level_up_pending"] is False
    assert after["info"]["level_up_choices"] == []

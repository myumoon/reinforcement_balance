"""ParamApplier のユニットテスト。"""
import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.param_applier import ParamApplier


class TestKeyMapping:
    def test_all_keys_mapped(self):
        applier = ParamApplier()
        keys = {"min_enemies", "max_enemies", "speed_mult", "spawn_rate_mult",
                "max_enemy_type_id", "enemy_hp_scale", "enemy_damage_scale", "time_scaling"}
        for k in keys:
            assert k in applier._KEY_MAP, f"{k} not in _KEY_MAP"

    def test_key_translation(self):
        applier = ParamApplier()
        assert applier._KEY_MAP["min_enemies"] == "MinActiveEnemies"
        assert applier._KEY_MAP["max_enemies"] == "MaxActiveEnemies"
        assert applier._KEY_MAP["speed_mult"] == "EnemySpeedMult"
        assert applier._KEY_MAP["spawn_rate_mult"] == "SpawnRateMult"
        assert applier._KEY_MAP["max_enemy_type_id"] == "MaxEnemyTypeId"
        assert applier._KEY_MAP["enemy_hp_scale"] == "EnemyHPScale"
        assert applier._KEY_MAP["enemy_damage_scale"] == "EnemyDamageScale"
        assert applier._KEY_MAP["time_scaling"] == "TimeScalingEnabled"

    def test_apply_returns_false_when_no_env(self):
        applier = ParamApplier(raw_env=None)
        result = applier.apply({"min_enemies": 4, "max_enemies": 6})
        assert result is False

    def test_apply_with_raw_env(self):
        class _MockEnv:
            def __init__(self):
                self.last_params = {}
            def set_params(self, **kwargs) -> bool:
                self.last_params = kwargs
                return True
        env = _MockEnv()
        applier = ParamApplier(raw_env=env)
        result = applier.apply({"min_enemies": 4, "max_enemies": 6, "speed_mult": 1.0,
                                 "spawn_rate_mult": 1.0, "max_enemy_type_id": 1,
                                 "enemy_hp_scale": 1.0, "enemy_damage_scale": 1.0,
                                 "time_scaling": False})
        assert result is True
        assert env.last_params["MinActiveEnemies"] == 4
        assert env.last_params["MaxActiveEnemies"] == 6
        assert env.last_params["TimeScalingEnabled"] is False

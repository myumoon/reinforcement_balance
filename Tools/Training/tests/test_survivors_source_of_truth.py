from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games.survivors.survivors_source_of_truth import build_survivors_source_of_truth
from games.survivors.survivors_eureka_config import SurvivorsEurekaConfig


def test_survivors_source_of_truth_extracts_cpp_constants():
    repo_root = Path(__file__).resolve().parents[3]
    sot = build_survivors_source_of_truth(repo_root)

    assert sot["reward_constants"]["AliveReward"] == 0.001
    assert sot["reward_constants"]["ItemReward"] == 1.0
    assert sot["reward_constants"]["KillReward"] == 2.0
    assert sot["player_constants"]["MaxPlayerHP"] == 70.0
    assert sot["observation_constants"]["EnemyDensityDirCount"] == 16
    assert sot["gem_xp_values"] == [2.0, 9.0, 10.0]
    assert len(sot["garlic_table"]) == 8
    assert any(enemy["name"] == "GiantBat" for enemy in sot["enemy_types"])
    assert next(enemy for enemy in sot["enemy_types"] if enemy["name"] == "Bat")["speed"] == 70.0
    assert next(enemy for enemy in sot["enemy_types"] if enemy["name"] == "Werewolf")["xp_drop"] == 9.0
    assert sot["directional_density"]["axis_mapping"]["+X"] == 8
    assert sot["directional_density"]["axis_mapping"]["+Y"] == 12
    assert sot["directional_density"]["axis_mapping"]["-Y"] == 4
    assert "atan2" in sot["directional_density"]["formula"]


def test_survivors_eureka_config_uses_source_of_truth_for_metric():
    config = SurvivorsEurekaConfig()
    config._source_of_truth = {
        "reward_constants": {"AliveReward": 0.5, "ItemReward": 1.0, "KillReward": 2.0},
        "player_constants": {"MaxPlayerHP": 70.0},
    }

    assert config.compute_primary_metric([11.0], [2]) == 10.0


def test_survivors_eureka_config_has_no_cpp_constant_assignments():
    config_path = Path(__file__).resolve().parents[1] / "games" / "survivors" / "survivors_eureka_config.py"
    text = config_path.read_text(encoding="utf-8")

    forbidden = [
        "_ALIVE_REWARD",
        "_ITEM_REWARD",
        "_KILL_REWARD",
        "_MAX_PLAYER_HP",
        "_ITEM_XP",
        "_XP_BASE",
        "_ENEMY_DPS",
        "_MIN_AURA_RADIUS",
    ]
    for name in forbidden:
        assert name not in text

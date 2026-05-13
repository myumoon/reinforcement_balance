from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games.survivors.survivors_source_of_truth import build_survivors_source_of_truth


def test_survivors_source_of_truth_extracts_cpp_constants():
    repo_root = Path(__file__).resolve().parents[3]
    sot = build_survivors_source_of_truth(repo_root)

    assert sot["reward_constants"]["AliveReward"] == 0.001
    assert sot["reward_constants"]["ItemReward"] == 1.0
    assert sot["reward_constants"]["KillReward"] == 2.0
    assert sot["player_constants"]["MaxPlayerHP"] == 70.0
    assert sot["observation_constants"]["EnemyDensityDirCount"] == 16

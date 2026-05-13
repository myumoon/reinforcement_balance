from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games.survivors.survivors_reward_validator import validate_survivors_reward_code


def _source_of_truth():
    return {
        "reward_constants": {
            "ItemReward": 1.0,
            "KillReward": 2.0,
        },
        "player_constants": {
            "MaxPlayerHP": 70.0,
        },
    }


def test_validator_rejects_stale_item_reward_and_hp_constants():
    code = """
if base_reward >= 2.9:
    shaped += 0.1
hp_abs = obs[12] * 100.0
"""

    findings = validate_survivors_reward_code(code, _source_of_truth())

    assert any(f["code"] == "ITEM_REWARD_MISMATCH" for f in findings)
    assert any(f["code"] == "MAX_HP_MISMATCH" for f in findings)


def test_validator_allows_current_item_reward_threshold():
    code = """
if base_reward >= 1.0:
    shaped += 0.1
"""

    findings = validate_survivors_reward_code(code, _source_of_truth())

    assert findings == []

"""実 UE5 PIE の HTTP 応答から deploy_raw fixture を取得する。

LLT fixture と同じ seed・初期武器・行動列を使い、指定したファイルへ reset と選択した step の応答を保存します。
UE5 Editor の Survivors HTTP server が起動している状態で実行してください。
"""

import argparse
import json
from pathlib import Path

from games.survivors.survivors_env import SurvivorsEnv

SEED = 73013
CAPTURE_STEPS = {60, 61, 62, 63, 180, 181, 360}
WEAPON_IDS = (4, 9, 7, 1, 8, 13)
WEAPON_NAMES = ["Knife", "SantaWater", "KingBible", "Garlic", "FireWand", "Peachone"]


def main() -> None:
    """同じ初期条件で PIE を動かし、reset と指定 step の raw 応答を保存する。

    action は step k ごとに (k - 1) % 9 とします。episode が途中で終わった場合は fixture を出力しません。
    """
    parser = argparse.ArgumentParser(description="Capture the Survivors deploy_raw fixture from a live UE5 PIE session.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON fixture path.")
    parser.add_argument("--port", type=int, default=8767, help="Survivors HTTP port (default: 8767).")
    args = parser.parse_args()

    slots = [{"weapon_id": weapon_id, "level": 4} for weapon_id in WEAPON_IDS]
    env = SurvivorsEnv(port=args.port, frame_skip=1)
    assert env.set_params(deploy_raw=True, initial_weapon_slots=slots), "/params failed"
    env.reset(seed=SEED)
    reset_response = env.last_reset_response
    assert "deploy_raw" in reset_response, "reset response has no deploy_raw"
    responses = [{"endpoint": "/reset", "step": 0, "deploy_raw": reset_response["deploy_raw"]}]

    for step in range(1, 361):
        _, _, done, truncated, info = env.step((step - 1) % 9)
        assert "deploy_raw" in info, f"step {step}: no deploy_raw"
        if step in CAPTURE_STEPS:
            responses.append({"endpoint": "/step", "step": step, "info": {"deploy_raw": info["deploy_raw"]}})
        if done or truncated:
            raise SystemExit(f"episode ended at step {step}")

    env.close()
    args.output.write_text(json.dumps({
        "description": (
            "deploy_raw captured from a live UE5 Editor PIE session via SurvivorsHttpEnvService "
            "(/params deploy_raw:true + initial_weapon_slots, /reset seed, /step with steps=1). "
            "Same seed/loadout/action rule as the LLT fixture deploy_raw_llt_v1.json."
        ),
        "seed": SEED,
        "action_rule": "action of step k (1-based) = (k - 1) % 9",
        "initial_weapons": WEAPON_NAMES,
        "obs_schema_hash": reset_response["obs_schema_hash"],
        "responses": responses,
    }, indent=2), encoding="utf-8")
    print("wrote", args.output, "steps", [response["step"] for response in responses])


if __name__ == "__main__":
    main()

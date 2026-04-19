"""SB3 の訓練済みモデルを ONNX 形式にエクスポートする。

使い方:
  python export_onnx.py --model models/balance_model --output ../../ReinBalance/Content/Models/balance_model.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import onnx
from stable_baselines3 import PPO


def export(model_path: Path, output_path: Path) -> None:
    model = PPO.load(str(model_path))
    policy = model.policy
    policy.eval()

    obs_dim = model.observation_space.shape[0]
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        policy,
        dummy_obs,
        str(output_path),
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )

    onnx.checker.check_model(str(output_path))
    print(f"[INFO] ONNX model saved to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("models/balance_model"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("../../ReinBalance/Content/Models/balance_model.onnx"),
    )
    args = p.parse_args()
    export(args.model, args.output)


if __name__ == "__main__":
    main()

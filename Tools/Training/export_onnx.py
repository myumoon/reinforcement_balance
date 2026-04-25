"""SB3 の訓練済みモデルを ONNX 形式にエクスポートする。

使い方:
  python export_onnx.py --model models/balance_model --output models/balance_model.onnx
  python export_onnx.py --model models/coin_model    --output models/coin_model.onnx
  # 出力ファイルを UE5 の Content Browser にドラッグ&ドロップでインポートする
"""

import argparse
from pathlib import Path

import torch
import onnx
from stable_baselines3 import PPO


class _DeterministicActor(torch.nn.Module):
    """PPO ポリシーから決定論的アクションのみを抽出するラッパー。

    SB3 の ActorCriticPolicy.forward() は (actions, values, log_prob) を返すが、
    UE5 NNE 推論では値関数・log_prob は不要なので除外し、入出力を 1 テンソルに絞る。
    - 連続行動: deterministic=True で平均アクション (Gaussian の mean) を返す
    - 離散行動: argmax インデックスを float32 にキャストして返す（NNE は float32 のみ受け付けるため）
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.policy._predict(obs, deterministic=True)
        return action.float()  # 離散 (int64) も float32 に統一


def export(model_path: Path, output_path: Path) -> None:
    model = PPO.load(str(model_path))
    policy = model.policy
    policy.eval()

    obs_dim = model.observation_space.shape[0]
    dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)

    actor = _DeterministicActor(policy)
    actor.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        actor,
        dummy_obs,
        str(output_path),
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )

    onnx.checker.check_model(str(output_path))

    # 入出力テンソル情報を表示
    m = onnx.load(str(output_path))
    inputs = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    outputs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
    print(f"[INFO] ONNX model saved to {output_path}")
    print(f"[INFO]   inputs:  {inputs}")
    print(f"[INFO]   outputs: {outputs}")
    print("[INFO] 次のステップ: UE5 Content Browser にこの .onnx をドラッグ&ドロップしてインポートしてください。")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("models/balance_model"),
                   help="SB3 モデルパス (.zip 拡張子は省略可)")
    p.add_argument("--output", type=Path, default=Path("models/balance_model.onnx"),
                   help="ONNX 出力先")
    args = p.parse_args()
    export(args.model, args.output)


if __name__ == "__main__":
    main()

"""Packaged Survivors Deployable Student Policy の synthetic dataset 評価 CLI。
正式 package と combat dataset を再検証し、actor KL/value Huber を JSON report として保存します。
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch as th
from games.survivors.combat_distillation_dataset import CombatDistillationDataset
from games.survivors.deployable_policy_package import load_deployable_policy_package
from games.survivors.deployable_policy_trainer import DeployableCombatPolicy, sequence_distillation_loss
def _parser() -> argparse.ArgumentParser:
    """artifact path だけを要求する軽量 evaluation CLI parser を作る。
    --help は package/model を開かないため、引数なし discovery でも exit 0 になります。
    """
    parser = argparse.ArgumentParser(description="Evaluate a deployable Survivors combat policy.")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser
def evaluate_package(package_dir: Path, dataset_path: Path) -> dict[str, float | int | str]:
    """formal package を復元して dataset 全体の distillation 指標を返す。
    schema hash を package/dataset 間で照合し、別 DeployObs 世代の評価を拒否します。
    """
    manifest, payload = load_deployable_policy_package(package_dir)
    dataset = CombatDistillationDataset.load(dataset_path)
    if dataset.deploy_schema_hash != manifest["deploy_schema_hash"]:
        raise ValueError("evaluation deploy schema hash mismatch")
    model = DeployableCombatPolicy(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    with th.no_grad():
        observations = th.from_numpy(np.array(dataset.observations, copy=True))
        resets = th.from_numpy(np.array(dataset.episode_reset_mask, copy=True)).bool()
        logits, values = model.forward_sequence(observations, resets)
        losses = sequence_distillation_loss(logits, values, dataset)
    return {
        "schema_version": "survivors.deployable_policy_eval.v1", "sequences": len(dataset.episode_ids),
        **{name: float(value) for name, value in losses.items()},
    }
def main(argv: list[str] | None = None) -> int:
    """CLI 入力を評価して report を canonical な JSON object として保存する。
    artifact 読込・binding・出力 error は stderr と exit 2 へまとめます。
    """
    args = _parser().parse_args(argv)
    try:
        report = evaluate_package(args.package_dir, args.dataset)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"deployable policy evaluation failed: {exc}", file=sys.stderr)
        return 2
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

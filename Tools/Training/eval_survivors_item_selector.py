"""Survivors ItemSelector 閉ループ評価 report を保存する CLI。

この入口は checkpoint binding を確認して evaluator を組み立てる。UE5 固有の live feature 抽出は
呼出し側 adapter が ``item_decision_features`` を step info に入れて接続する。
"""

from __future__ import annotations

import argparse
import json
import sys
import importlib
from pathlib import Path
from typing import Any, Callable

from games.survivors.item_selection_strategy import CardHeuristicV1Strategy, ItemSelectionStrategy, RandomCardStrategy
from games.survivors.item_selector_eval import EvaluationCell, ItemSelectorClosedLoopEvaluator, PairedItemSelectorEvaluator
from games.survivors.item_selector_model import ItemSelector
from games.survivors.item_selector_trainer import ItemSelectorTrainer


class ItemSelectorEvaluationCliError(ValueError):
    """閉ループ評価 CLI の入力・checkpoint・report 保存エラーを表す。

    途中 report を成功結果として出力しないよう、main で終了コード 2 にまとめる。
    """


def load_item_selector(
    checkpoint: Path,
    *,
    context_dim: int,
    candidate_dim: int,
    nmax: int,
    target_capability_hash: str,
    device: str,
) -> ItemSelectionStrategy:
    """target capability と feature shape を照合して selector checkpoint を復元する。

    学習時と異なる Nmax や feature schema の checkpoint を、推論で偶然ロードすることを防ぐ。
    """
    model = ItemSelector(context_dim=context_dim, candidate_dim=candidate_dim)
    trainer = ItemSelectorTrainer(
        model,
        target_capability_hash=target_capability_hash,
        nmax=nmax,
        device=device,
    )
    trainer.load_checkpoint(checkpoint)
    return ItemSelectionStrategy(model, device=device)


def run_closed_loop_evaluation(
    environment: Any,
    *,
    selector: ItemSelectionStrategy,
    movement_policy: Callable[[Any], Any],
    episode_count: int,
    seed_start: int,
    max_movement_steps: int,
) -> dict[str, Any]:
    """injected environment で閉ループ評価を実行して JSON 向け report を返す。

    UE5 接続の生成を関数外へ出すことで、同じ制御を mock integration test と実機 adapter で共有する。
    """
    report = ItemSelectorClosedLoopEvaluator(
        strategy=selector,
        movement_policy=movement_policy,
        max_movement_steps=max_movement_steps,
    ).evaluate(environment, episode_count=episode_count, seed_start=seed_start)
    return {
        "summary": dict(report.summary),
        "episodes": [
            {
                "seed": episode.seed,
                "total_reward": episode.total_reward,
                "movement_steps": episode.movement_steps,
                "terminated": episode.terminated,
                "truncated": episode.truncated,
                "selections": [
                    {"decision_id": choice.decision_id, "choice_id": choice.choice_id}
                    for choice in episode.selections
                ],
            }
            for episode in report.episodes
        ],
    }


def _parser() -> argparse.ArgumentParser:
    """checkpoint と評価境界を明示する CLI 引数を定義する。

    live UE5 adapter はプロジェクト固有の movement/feature producer を必要とするため、この CLI は checkpoint 検証までを提供する。
    """
    parser = argparse.ArgumentParser(description="Run a paired Survivors ItemSelector closed-loop evaluation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--context-dim", type=int, required=True)
    parser.add_argument("--candidate-dim", type=int, required=True)
    parser.add_argument("--nmax", type=int, required=True)
    parser.add_argument("--target-capability-hash", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-factory", required=True, help="module:callable accepting EvaluationCell")
    parser.add_argument("--movement-policy", required=True, help="module:callable accepting an observation")
    parser.add_argument("--manifest", type=Path, required=True, help="frozen seed/scenario manifest JSON")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--required-episodes", type=int)
    parser.add_argument("--max-movement-steps", type=int, default=100000)
    return parser


def _load_callable(spec: str) -> Callable[..., Any]:
    """明示 adapter を import する。CLI 自身は UE5 接続を暗黙生成しない。"""
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ItemSelectorEvaluationCliError("adapter must be module:callable")
    value = getattr(importlib.import_module(module), name)
    if not callable(value):
        raise ItemSelectorEvaluationCliError("adapter must be callable")
    return value


def main(argv: list[str] | None = None) -> int:
    """checkpoint を含む三戦略を、明示 adapter と frozen manifest で実行する。"""
    args = _parser().parse_args(argv)
    try:
        selector = load_item_selector(
            args.checkpoint,
            context_dim=args.context_dim,
            candidate_dim=args.candidate_dim,
            nmax=args.nmax,
            target_capability_hash=args.target_capability_hash,
            device=args.device,
        )
        raw_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        cells = tuple(EvaluationCell(**cell) for cell in raw_manifest["cells"])
        manifest = PairedItemSelectorEvaluator.freeze_manifest(cells)
        # A provided manifest must already be frozen; accepting altered seed matrices would leak evaluation data.
        if raw_manifest.get("manifest_hash") != manifest["manifest_hash"]:
            raise ItemSelectorEvaluationCliError("manifest hash mismatch")
        previous = json.loads(args.resume.read_text(encoding="utf-8")) if args.resume else None
        evaluator = PairedItemSelectorEvaluator(
            environment_factory=_load_callable(args.environment_factory),
            movement_policy=_load_callable(args.movement_policy),
            max_movement_steps=args.max_movement_steps,
        )
        payload = evaluator.evaluate(
            strategies={"random": RandomCardStrategy(), "card_heuristic_v1": CardHeuristicV1Strategy(), "item_selector": selector},
            manifest=manifest,
            existing_result=previous,
            required_episodes=args.required_episodes,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"item selector evaluation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

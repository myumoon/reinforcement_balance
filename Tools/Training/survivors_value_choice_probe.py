"""Saved source と level-up preview から value choice ranking v3 を生成する CLI。

preview/context/source の phase・step・decision・obs/schema hash を相互検証し、正式 row または
明示的な zero-state smoke row だけを検証済み JSONL へ追記する。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from games.survivors.choice_preview import parse_level_up_preview
from games.survivors.recurrent_policy_session import (
    CriticContext,
    load_critic_context,
    observation_hash,
)
from games.survivors.value_choice_schema import (
    append_value_choice_ranking,
    build_value_choice_ranking,
)
from games.survivors.value_scorer import ValueScorer


class ValueChoiceProbeError(ValueError):
    """probe 引数・preview・context binding が不正な場合の例外。

    argparse の process exit と runtime validation を main の終了コード 3 へ統一する。
    """


class _ProbeArgumentParser(argparse.ArgumentParser):
    """必須引数欠落を probe 契約エラーへ変換する parser。

    argparse が直接 exit せず、テストと自動化が invalid input を同じコードで扱えるようにする。
    """

    def error(self, message: str) -> None:
        """CLI 構文エラーを ValueChoiceProbeError として送出する。

        artifact gate 不通過と同じ stderr 経路を使い、部分 JSONL を生成しない。
        """
        raise ValueChoiceProbeError(f"invalid probe arguments: {message}")


def _parser() -> argparse.ArgumentParser:
    """value choice probe の versioned CLI 引数を定義する。

    manifest・preview・context・output を全て明示必須にし、cwd からの自動探索を禁止する。
    """
    parser = _ProbeArgumentParser(
        description="Score Survivors level-up candidate values."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preview-json", type=Path, required=True)
    parser.add_argument("--context-npz", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--zero-state-smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser


def _read_preview(path: Path) -> Mapping[str, Any]:
    """environment_step 付き preview JSON object を strict に読む。

    HTTP payload の四 field に step binding を加え、未知 field と JSON 構文不正を拒否する。
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueChoiceProbeError("preview JSON could not be read") from exc
    if not isinstance(payload, Mapping):
        raise ValueChoiceProbeError("preview JSON must contain an object")
    expected = {
        "environment_step",
        "decision_id",
        "obs_schema_hash",
        "base_obs",
        "previews",
    }
    if set(payload) != expected:
        raise ValueChoiceProbeError("preview JSON fields mismatch")
    if (
        type(payload["environment_step"]) is not int
        or payload["environment_step"] < 0
    ):
        raise ValueChoiceProbeError(
            "preview environment_step must be non-negative"
        )
    return payload


def _parse_bound_preview(
    payload: Mapping[str, Any],
    scorer: ValueScorer,
) -> tuple[Any, list[str], np.ndarray]:
    """preview を source obs schema と自身の decision/choice set へ束縛する。

    parse 後に context と照合するため、candidate 順と raw base/candidate batch を保持する。
    """
    descriptor_schema = scorer.source.descriptor["observation_schema"]
    expected_schema_hash = (
        descriptor_schema["reported_hash"] or descriptor_schema["sha256"]
    )
    if payload["obs_schema_hash"] != expected_schema_hash:
        raise ValueChoiceProbeError(
            "preview observation schema hash does not match source"
        )
    previews = payload["previews"]
    if not isinstance(previews, list):
        raise ValueChoiceProbeError("preview candidates must be an array")
    choice_ids: list[str] = []
    for candidate in previews:
        if not isinstance(candidate, Mapping):
            raise ValueChoiceProbeError("preview candidate must be an object")
        choice_id = candidate.get("choice_id")
        if not isinstance(choice_id, str):
            raise ValueChoiceProbeError("preview choice_id must be a string")
        choice_ids.append(choice_id)
    segment_names = [
        segment["name"] for segment in descriptor_schema["ordered_segments"]
    ]
    parsed = parse_level_up_preview(
        {
            key: payload[key]
            for key in (
                "decision_id",
                "obs_schema_hash",
                "base_obs",
                "previews",
            )
        },
        expected_decision_id=payload["decision_id"],
        expected_schema_hash=expected_schema_hash,
        expected_choice_ids=choice_ids,
        obs_dim=scorer.source.observation_dim,
        schema_segment_names=segment_names,
    )
    batch = np.asarray(
        [
            parsed.by_choice_id[choice_id].projected_obs
            for choice_id in choice_ids
        ],
        dtype=np.float32,
    )
    return parsed, choice_ids, batch


def _bind_context(
    payload: Mapping[str, Any],
    base_obs: np.ndarray,
    context: CriticContext,
) -> None:
    """context の phase・step・decision・pending obs hash を preview へ照合する。

    各 binding sibling を個別比較し、別 decision の正しい形の state も formal score に使わない。
    """
    context.validate_integrity()
    if context.phase != "pending_pre_commit":
        raise ValueChoiceProbeError("context phase mismatch")
    if context.environment_step != payload["environment_step"]:
        raise ValueChoiceProbeError("context environment step mismatch")
    if context.decision_id != payload["decision_id"]:
        raise ValueChoiceProbeError("context decision id mismatch")
    if context.pending_obs_hash != observation_hash(base_obs):
        raise ValueChoiceProbeError("context pending observation hash mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    """candidate score と ranking append を実行し success=0 / invalid=3 を返す。

    zero-state flag は入力 state を無視して schema shape の零 state を作り、row の label-ready を false に固定する。
    """
    try:
        args = _parser().parse_args(argv)
        scorer = ValueScorer.load(args.manifest, device=args.device)
        payload = _read_preview(args.preview_json)
        parsed, choice_ids, raw_batch = _parse_bound_preview(payload, scorer)
        loaded_context = load_critic_context(args.context_npz)
        _bind_context(
            payload,
            np.asarray(parsed.base_obs, dtype=np.float32),
            loaded_context,
        )
        if args.zero_state_smoke:
            context = CriticContext.zero_state(
                environment_step=loaded_context.environment_step,
                decision_id=loaded_context.decision_id,
                pending_obs=np.asarray(parsed.base_obs, dtype=np.float32),
                episode_start=loaded_context.episode_start,
                policy_state_schema=dict(scorer.source.policy_state_schema),
            )
        else:
            if loaded_context.context_mode == "zero_state_smoke":
                raise ValueChoiceProbeError(
                    "zero-state context requires --zero-state-smoke"
                )
            context = loaded_context
        values = scorer.score(
            raw_batch,
            context,
            choice_ids=choice_ids,
            allow_zero_state_smoke=args.zero_state_smoke,
        )
        descriptor = scorer.source.descriptor
        row = build_value_choice_ranking(
            source_identity_sha256=descriptor["identity_sha256"],
            manifest_sha256=scorer.source.manifest_sha256,
            model_sha256=descriptor["artifacts"]["model"]["sha256"],
            vecnormalize_sha256=descriptor["artifacts"]["vecnormalize"][
                "sha256"
            ],
            observation_schema_sha256=descriptor["observation_schema"][
                "sha256"
            ],
            policy_state_schema_sha256=scorer.source.policy_state_schema[
                "policy_state_schema_hash"
            ],
            context_sha256=context.context_sha256,
            context_mode=context.context_mode,
            environment_step=context.environment_step,
            decision_id=context.decision_id,
            pending_obs_sha256=context.pending_obs_hash,
            values=values,
            zero_state_smoke=args.zero_state_smoke,
        )
        append_value_choice_ranking(args.output_jsonl, row)
        print(
            json.dumps(
                {
                    "schema_version": row["schema_version"],
                    "decision_id": context.decision_id,
                    "tie": row["tie"],
                    "ready_for_training_label": False,
                    "output_jsonl": str(args.output_jsonl),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError) as exc:
        print(f"[ERROR] value choice probe failed: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


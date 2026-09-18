"""02-04/05-01 parity subprocess worker (Deployment side)。

Training 側 subprocess が export した ItemSelector package を、Deployment の
ONNX adapter (OnnxItemSelector) と ItemSession だけで読み込み、同じ shared
fixture から UiIntentV1 を再構成する。games.* を import しないのはこの
subprocess の契約であり、self-check で確認する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reinbalance_survivors_contracts.item_decision import ItemDecisionFeatures

from survivors.perception_snapshot import (
    UI_PRESENTATION_SCHEMA_HASH,
    NormalizedRoi,
    UiCandidateTargetV1,
    UiPresentationSnapshotV1,
)
from survivors.runtime.item_selector_runtime import OnnxItemSelector
from survivors.runtime.item_session import ItemSession


def _ui_presentation(raw: dict) -> UiPresentationSnapshotV1:
    """shared fixture の wire 表現から UiPresentationSnapshotV1 を組み立てる。"""
    return UiPresentationSnapshotV1(
        schema_hash=UI_PRESENTATION_SCHEMA_HASH,
        snapshot_id=raw["snapshot_id"],
        frame_id=raw["frame_id"],
        parser_artifact_hash=raw["parser_artifact_hash"],
        screen_state=raw["screen_state"],
        candidate_set_hash=raw["candidate_set_hash"],
        inventory_hash=raw["inventory_hash"],
        source_content_hash=raw["source_content_hash"],
        ui_state_key=raw["ui_state_key"],
        candidates=tuple(
            UiCandidateTargetV1(
                choice_id=candidate["choice_id"],
                choice_index=candidate["choice_index"],
                semantic_kind=candidate["semantic_kind"],
                roi=NormalizedRoi(*candidate["roi"]),
                validity=candidate["validity"],
                confidence=candidate["confidence"],
            )
            for candidate in raw["candidates"]
        ),
        buttons=tuple(),
    )


def main() -> int:
    """Training 側が export した package を ONNX adapter で読み、choose_card の canonical bytes を書き出す。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--intent-out", required=True)
    parser.add_argument("--result-out", required=True)
    args = parser.parse_args()

    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    item_context = ItemDecisionFeatures.from_wire(case["item_context"])
    ui_presentation = _ui_presentation(case["ui_presentation"])

    artifact = OnnxItemSelector.load(Path(args.package))
    session = ItemSession(artifact)
    outcome = session.decide(
        item_context,
        ui_presentation,
        decision_policy_id=case["hash_kwargs"]["decision_policy_id"],
        decision_rule_id=case["hash_kwargs"]["decision_rule_id"],
        decision_config_hash=case["hash_kwargs"]["decision_config_hash"],
    )
    if outcome.intent is None:
        raise SystemExit(f"parity fixture must pass the confidence gate: {outcome.reason}")

    Path(args.intent_out).write_bytes(outcome.intent.canonical_bytes() + b"\n")
    forbidden = sorted(name for name in sys.modules if name == "games" or name.startswith("games."))
    Path(args.result_out).write_text(json.dumps({"forbidden_imports": forbidden}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""02-04/05-01 parity subprocess worker (Training side)。

Deployment 側テストプロセスから subprocess として起動される。games.* を import
できるのはこの subprocess だけであり、呼び出し元 (Deployment) プロセスへは
一切 import を波及させない。checkpoint を新規作成して ItemSelector package を
export し、同じ package から生成した UiIntentV1 の canonical bytes を出力する。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes
from reinbalance_survivors_contracts.item_decision import ItemDecisionFeatures

from export_survivors_item_selector import export_item_selector_artifact
from games.survivors.item_selector_artifact import ItemSelectorArtifact
from games.survivors.item_selector_dataset import SplitManifest
from games.survivors.item_selector_model import ItemSelector
from games.survivors.item_selector_session import ItemSelectorArtifactSession
from games.survivors.item_selector_trainer import ItemSelectorTrainer

_CAPABILITY = "1" * 64
_SOURCE = "2" * 64
_TEACHER = "3" * 64
_CONTEXT_DIM = 24
_CANDIDATE_DIM = 9
_NMAX = 2
_FEATURE_SCHEMA = "context_only_v1"


def _dataset_row(decision_id: str, split: str, episode_id: str, target: list[float]) -> dict:
    """dims=24/9 の合成学習行を、export_fixture 精度と同じ直接エンコード済み形式で作る。"""
    return {
        "decision_id": decision_id,
        "episode_id": episode_id,
        "split": split,
        "label_action_kind": "choose_card",
        "context_features": [0.1 * (index % 7) for index in range(_CONTEXT_DIM)],
        "candidate_features": [
            [0.05 * (index % 5) for index in range(_CANDIDATE_DIM)],
            [0.05 * ((index + 1) % 5) for index in range(_CANDIDATE_DIM)],
        ],
        "candidate_mask": [True, True],
        "candidate_item_ids": ["wand", "knife"],
        "candidate_kinds": ["weapon", "weapon"],
        "teacher_soft_target": target,
        "teacher_scores": target,
        "kind": "weapon",
        "time_bucket": "0-5m",
        "slot_occupancy_bucket": "open",
    }


def _build_package(output_dir: Path) -> Path:
    """dims=24/9 の較正済み ItemSelector package を新規 checkpoint から export する。

    checkpoint は未学習の乱数初期化重みをそのまま保存するだけであり、dataset の行内容
    自体は package の推論結果に影響しない（parity test が検証したいのは
    Training/Deployment 双方の adapter が同じ package から同じ bytes を出せることであり、
    モデル品質ではない）。
    """
    raw = [{"decision_id": f"parity-d-{index}", "episode_id": f"parity-e-{index}"} for index in range(6)]
    split_manifest = SplitManifest.freeze(raw, seed="parity-runtime-test")
    rows = [
        _dataset_row(
            item["decision_id"],
            split_manifest.split_for(item),
            item["episode_id"],
            [0.9, 0.1] if index % 2 == 0 else [0.1, 0.9],
        )
        for index, item in enumerate(raw)
    ]
    dataset = output_dir / "dataset"
    (dataset / "shards").mkdir(parents=True)
    split_manifest.commit(dataset / "split_manifest.json")
    development = [row for row in rows if row["split"] != "test"]
    test = [row for row in rows if row["split"] == "test"]
    (dataset / "shards" / "development.jsonl").write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in development)
    )
    (dataset / "shards" / "test.jsonl").write_bytes(
        b"".join(canonical_json_bytes(row) + b"\n" for row in test)
    )
    dataset_identity = canonical_hash({"rows": rows, "split": split_manifest.manifest_id})
    (dataset / "manifest.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "survivors.item_selector_dataset.v1",
                "dataset_identity": dataset_identity,
                "split_manifest": "split_manifest.json",
                "shards": [{"logical_id": "shards/development.jsonl"}],
                "test_shards": [{"logical_id": "shards/test.jsonl"}],
            }
        )
    )

    checkpoint = output_dir / "checkpoint.pt"
    model = ItemSelector(context_dim=_CONTEXT_DIM, candidate_dim=_CANDIDATE_DIM)
    ItemSelectorTrainer(model, target_capability_hash=_CAPABILITY, nmax=_NMAX).save_checkpoint(
        checkpoint,
        optimizer=None,
        epoch=1,
        best_val_ndcg=0.5,
    )
    vocabulary = output_dir / "vocabulary.json"
    vocabulary.write_bytes(canonical_json_bytes(["knife", "wand"]))
    package_dir = output_dir / "package"
    export_item_selector_artifact(
        checkpoint=checkpoint,
        dataset_dir=dataset,
        output_dir=package_dir,
        target_capability_hash=_CAPABILITY,
        nmax=_NMAX,
        context_dim=_CONTEXT_DIM,
        candidate_dim=_CANDIDATE_DIM,
        feature_schema=_FEATURE_SCHEMA,
        vocabulary_json=vocabulary,
        temperature=1.0,
        confidence_threshold=0.0,
        student_output_temperature=1.0,
        dataset_identity=dataset_identity,
        source_descriptor_identity=_SOURCE,
        teacher_verdict_identity=_TEACHER,
        run_id="parity-runtime-test",
        device="cpu",
    )
    return package_dir


def main() -> int:
    """shared fixture を読み、package を build して choose_card の canonical bytes を書き出す。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--intent-out", required=True)
    parser.add_argument("--result-out", required=True)
    args = parser.parse_args()

    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    package_dir = _build_package(work_dir)

    item_context = ItemDecisionFeatures.from_wire(case["item_context"])
    session = ItemSelectorArtifactSession(ItemSelectorArtifact.load(package_dir))
    outcome = session.decide(item_context, **case["hash_kwargs"])
    if outcome.intent is None:
        raise SystemExit(f"parity fixture must pass the confidence gate: {outcome.reason}")

    Path(args.intent_out).write_bytes(outcome.intent.canonical_bytes() + b"\n")
    forbidden = sorted(
        name for name in sys.modules if name == "survivors" or name.startswith("survivors.")
    )
    Path(args.result_out).write_text(
        json.dumps({"package_dir": str(package_dir), "forbidden_imports": forbidden}),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

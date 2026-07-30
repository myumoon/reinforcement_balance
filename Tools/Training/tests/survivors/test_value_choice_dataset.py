"""Survivors choice trace shard の原子性・再開・厳格 schema を検証する。

二 shard の round-trip と破損 fixture を使い、record dedup、dtype、hash、quarantine、
commit 時だけの manifest 集計を永続化境界で確認する。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from reinbalance_survivors_contracts.canonical_json import canonical_hash

from games.survivors.value_choice_dataset import (
    DatasetError,
    DatasetWriter,
    read_dataset,
)

SOURCE = "a" * 64


def _record(
    decision_id: str,
    *,
    selected_choice_id: str = "choice-a",
) -> tuple[dict, dict[str, np.ndarray]]:
    """有効な choice trace row と array 群を返す。

    teacher label と behavior selection を別 object に置き、LSTM の非自明 shape も
    round-trip 対象へ含める。
    """

    metadata = {
        "episode_logical_id": "episode-7",
        "decision_id": decision_id,
        "environment_step": 11,
        "candidate_choice_ids": ["choice-a", "choice-b"],
        "behavior": {
            "policy": "epsilon_source_scorer",
            "epsilon": 0.2,
            "selected_choice_id": selected_choice_id,
            "propensity": 0.9 if selected_choice_id == "choice-a" else 0.1,
        },
        "teacher_label": {
            "policy": "source_scorer",
            "best_choice_id": "choice-a",
            "ordered_choice_ids": ["choice-a", "choice-b"],
            "normalized_returns": [2.0, 1.0],
        },
        "replay_events": [
            {"kind": "reset", "seed": 7},
            {"kind": "external_decision", "decision_id": decision_id},
        ],
        "artifact_identity": {
            "source_identity_sha256": SOURCE,
            "fidelity_verdict_sha256": "b" * 64,
        },
    }
    arrays = {
        "pending_obs": np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        "candidate_obs": np.asarray(
            [[0.25, 0.5, 1.0], [0.0, 0.75, 1.0]],
            dtype=np.float64,
        ),
        "pi_h": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "pi_c": np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 1,
        "vf_h": np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 2,
        "vf_c": np.arange(24, dtype=np.float32).reshape(2, 3, 4) + 3,
        "movement_actions": np.asarray([1, 8, 3], dtype=np.int64),
    }
    return metadata, arrays


def test_two_shards_round_trip_preserves_rows_arrays_hashes_and_order(
    tmp_path: Path,
) -> None:
    """二 shard の row order と全 ndarray を書込み順のまま復元する。

    float/int は固定 dtype へ正規化し、LSTM shape と各 array の canonical hash が
    read-back 後にも一致することを確認する。
    """

    writer = DatasetWriter(
        tmp_path / "dataset",
        dataset_id="survivors-choice-test",
        source_identity_sha256=SOURCE,
    )
    expected_ids = []
    expected_arrays = []
    for shard_index in range(2):
        writer.start_shard(f"shard-{shard_index:05d}")
        metadata, arrays = _record(f"decision-{shard_index}")
        expected_ids.append(writer.append(metadata, arrays))
        expected_arrays.append(arrays)
        writer.commit_shard()

    snapshot = read_dataset(tmp_path / "dataset")
    assert [row["record_id"] for row in snapshot.rows] == expected_ids
    assert snapshot.manifest["record_count"] == 2
    for row_index, (actual, expected) in enumerate(
        zip(snapshot.arrays, expected_arrays)
    ):
        for name, original in expected.items():
            restored = actual[name]
            expected_dtype = np.int32 if name == "movement_actions" else np.float32
            normalized = np.asarray(original, dtype=expected_dtype)
            assert restored.dtype == expected_dtype
            assert restored.shape == normalized.shape
            np.testing.assert_array_equal(restored, normalized)
            assert canonical_hash(
                {
                    "dtype": str(restored.dtype),
                    "shape": list(restored.shape),
                    "values": restored.tolist(),
                }
            ) == snapshot.rows[row_index]["arrays"][name]["sha256"]


def test_retry_duplicate_is_deduplicated_and_histogram_changes_only_on_commit(
    tmp_path: Path,
) -> None:
    """同じ source/episode/decision の retry を一行へ畳み込む。

    append 中は manifest の count/histogram を変更せず、shard commit 後にだけ一回
    反映されることを確認する。
    """

    root = tmp_path / "dataset"
    writer = DatasetWriter(
        root,
        dataset_id="survivors-choice-test",
        source_identity_sha256=SOURCE,
    )
    writer.start_shard("shard-00000")
    metadata, arrays = _record("decision-retry")
    first = writer.append(metadata, arrays)
    second = writer.append(metadata, arrays)
    manifest_before = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert first == second
    assert manifest_before["record_count"] == 0
    assert manifest_before["histogram"] == {
        "behavior_policy": {},
        "candidate_count": {},
        "selected_choice_id": {},
        "teacher_choice_id": {},
    }

    writer.commit_shard()
    manifest_after = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_after["record_count"] == 1
    assert manifest_after["histogram"]["candidate_count"] == {"2": 1}
    assert manifest_after["histogram"]["selected_choice_id"] == {"choice-a": 1}


def test_conflicting_duplicate_is_quarantined_without_manifest_mutation(
    tmp_path: Path,
) -> None:
    """同じ canonical record ID に異なる内容を再送した shard を隔離する。

    正常な同一内容 retry の dedup と対称に、array 内容だけを変えた衝突は DatasetError とし、
    active shard、record count、histogram のいずれも publish しない。
    """

    root = tmp_path / "dataset"
    writer = DatasetWriter(
        root,
        dataset_id="survivors-choice-test",
        source_identity_sha256=SOURCE,
    )
    writer.start_shard("shard-00000")
    metadata, arrays = _record("decision-conflict")
    writer.append(metadata, arrays)
    conflicting_arrays = {
        name: value.copy()
        for name, value in arrays.items()
    }
    conflicting_arrays["pending_obs"][0] = 0.75

    with pytest.raises(DatasetError, match="conflicting content"):
        writer.append(metadata, conflicting_arrays)

    snapshot = read_dataset(root)
    assert snapshot.manifest["record_count"] == 0
    assert snapshot.manifest["histogram"] == {
        "behavior_policy": {},
        "candidate_count": {},
        "selected_choice_id": {},
        "teacher_choice_id": {},
    }
    assert writer.active_shard_id is None
    assert list((root / "quarantine").iterdir())


def test_source_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    """既存 dataset と異なる source identity の writer を拒否する。

    shard 単位の見かけ上の整合性より dataset 全体の teacher provenance を優先し、
    異なる source の row が一つの manifest に混ざらないようにする。
    """

    root = tmp_path / "dataset"
    DatasetWriter(
        root,
        dataset_id="survivors-choice-test",
        source_identity_sha256=SOURCE,
    )
    with pytest.raises(DatasetError, match="source"):
        DatasetWriter(
            root,
            dataset_id="survivors-choice-test",
            source_identity_sha256="f" * 64,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nan", "finite"),
        ("unknown_choice", "choice"),
    ],
)
def test_invalid_rows_are_aborted_to_quarantine(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """NaN と未知 selected choice を publish 前に quarantine する。

    JSON と choice binding の sibling validation を同じ append gate に適用し、失敗 shard を
    再利用可能な active transaction として残さない。
    """

    root = tmp_path / mutation
    writer = DatasetWriter(
        root,
        dataset_id="survivors-choice-test",
        source_identity_sha256=SOURCE,
    )
    writer.start_shard("shard-00000")
    metadata, arrays = _record("decision-invalid")
    if mutation == "nan":
        arrays["pending_obs"][0] = np.nan
    else:
        metadata["behavior"]["selected_choice_id"] = "choice-unknown"

    with pytest.raises(DatasetError, match=message):
        writer.append(metadata, arrays)
    assert list((root / "quarantine").iterdir())
    assert read_dataset(root).manifest["record_count"] == 0


@pytest.mark.parametrize("corruption", ["interrupted", "partial_npz", "count_mismatch"])
def test_recovery_quarantines_incomplete_or_mismatched_shards(
    tmp_path: Path,
    corruption: str,
) -> None:
    """process interruption、partial NPZ、JSONL/NPZ count mismatch を隔離する。

    commit marker のない staging/orphan shard と内部 count が一致しない shard を
    manifest へ推測追加せず、次の writer 起動時に quarantine へ移す。
    """

    root = tmp_path / corruption
    writer = DatasetWriter(
        root,
        dataset_id="survivors-choice-test",
        source_identity_sha256=SOURCE,
    )
    if corruption == "interrupted":
        writer.start_shard("shard-interrupted")
        metadata, arrays = _record("decision-interrupted")
        writer.append(metadata, arrays)
    else:
        orphan = root / "shards" / f"shard-{corruption}"
        orphan.mkdir(parents=True)
        (orphan / "rows.jsonl").write_text(
            json.dumps({"record_id": "bad"}) + "\n",
            encoding="utf-8",
        )
        if corruption == "partial_npz":
            (orphan / "arrays.npz").write_bytes(b"PK\x03\x04partial")
        else:
            with (orphan / "arrays.npz").open("wb") as stream:
                np.savez(stream, row_count=np.asarray(0, dtype=np.int32))

    recovered = DatasetWriter(
        root,
        dataset_id="survivors-choice-test",
        source_identity_sha256=SOURCE,
    )
    assert recovered.manifest["record_count"] == 0
    assert list((root / "quarantine").iterdir())

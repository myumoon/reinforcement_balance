"""Value choice probe CLI の cross-binding と zero-state 出力を検証する。

実 source fixture と NPZ context を使い、必須引数・preview binding・JSONL publish の入口を
end-to-end で確認する。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.recurrent_policy_session import write_critic_context
from games.survivors.value_choice_schema import validate_value_choice_ranking
from games.survivors.value_scorer import ValueScorer
from survivors_value_choice_probe import main
from value_scorer_fixtures import build_saved_value_source


def _probe_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """probe 用 manifest・preview JSON・captured context NPZ を作る。

    preview base hash と decision/step を session capture から得た値へ正確に揃える。
    """
    manifest_path, _, _ = build_saved_value_source(
        tmp_path,
        recurrent=True,
    )
    scorer = ValueScorer.load(manifest_path)
    pending = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    context = scorer.new_session().begin_level_up(
        environment_step=9,
        decision_id="decision-9",
        pending_obs=pending,
        episode_start=False,
    )
    context_path = tmp_path / "context.npz"
    write_critic_context(context_path, context)
    schema_hash = scorer.source.descriptor["observation_schema"]["sha256"]
    preview_path = tmp_path / "preview.json"
    preview_path.write_text(
        json.dumps(
            {
                "environment_step": 9,
                "decision_id": "decision-9",
                "obs_schema_hash": schema_hash,
                "base_obs": pending.tolist(),
                "previews": [
                    {
                        "choice_id": "choice-z",
                        "projected_obs": [0.25, 0.5, 1.0],
                        "changed_segments": ["observation"],
                    },
                    {
                        "choice_id": "choice-a",
                        "projected_obs": [0.0, 0.75, 1.0],
                        "changed_segments": ["observation"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, preview_path, context_path


def test_zero_state_smoke_forces_false_and_publishes_valid_row(
    tmp_path: Path,
) -> None:
    """zero-state smoke が label-ready false の ranking だけを append する。

    flag の有無で source binding は緩めず、context mode だけを明示的な診断値へ置き換える。
    """
    manifest_path, preview_path, context_path = _probe_inputs(tmp_path)
    output = tmp_path / "ranking.jsonl"

    assert main(
        [
            "--manifest",
            str(manifest_path),
            "--preview-json",
            str(preview_path),
            "--context-npz",
            str(context_path),
            "--output-jsonl",
            str(output),
            "--zero-state-smoke",
        ]
    ) == 0
    row = json.loads(output.read_text(encoding="utf-8"))
    validate_value_choice_ranking(row)
    assert row["context"]["mode"] == "zero_state_smoke"
    assert row["ready_for_training_label"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("environment_step", 10),
        ("decision_id", "other-decision"),
        ("base_obs", [9.0, 0.5, 1.0]),
    ],
)
def test_probe_rejects_all_context_binding_mismatch_siblings(
    tmp_path: Path,
    field: str,
    value,
) -> None:
    """preview の step・decision・base obs mismatch を全て publish 前に拒否する。

    一つの binding だけ正しい row を formal artifact として残さない。
    """
    manifest_path, preview_path, context_path = _probe_inputs(tmp_path)
    payload = json.loads(preview_path.read_text(encoding="utf-8"))
    payload[field] = value
    preview_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "ranking.jsonl"

    assert main(
        [
            "--manifest",
            str(manifest_path),
            "--preview-json",
            str(preview_path),
            "--context-npz",
            str(context_path),
            "--output-jsonl",
            str(output),
        ]
    ) == 3
    assert not output.exists()


def test_probe_requires_all_four_artifact_arguments() -> None:
    """manifest・preview・context・output の省略を invalid とする。

    自動 path 探索で別 run の artifact を混ぜず、argparse exit も終了コード 3 に統一する。
    """
    assert main([]) == 3


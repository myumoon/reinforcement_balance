"""Level-up 中の recurrent state 遷移と exactly-once commit を検証する。

pending・preview・retry を非 movement observation として凍結し、selected post-choice
observation だけが actor/critic state を一度進めることを確認する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.recurrent_policy_session import RecurrentSessionError
from games.survivors.value_scorer import ValueScorer
from value_scorer_fixtures import build_saved_value_source


def test_pending_preview_retry_freeze_then_selected_commits_once(
    tmp_path: Path,
) -> None:
    """pending と百回 preview で state が変わらないことを確認する。

    selected post-choice observation の commit 後だけ state hash が変わり、二重 commit は拒否する。
    """
    manifest_path, _, _ = build_saved_value_source(tmp_path, recurrent=True)
    scorer = ValueScorer.load(manifest_path)
    session = scorer.new_session()
    pending = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    candidates = np.asarray(
        [[0.25, 0.5, 1.0], [0.0, 0.75, 1.0]],
        dtype=np.float32,
    )
    context = session.begin_level_up(
        environment_step=11,
        decision_id="decision-11",
        pending_obs=pending,
        episode_start=False,
    )
    frozen_hash = session.state_hash

    for _ in range(100):
        scorer.score(candidates, context)
        session.validate_pending_retry(
            environment_step=11,
            decision_id="decision-11",
            pending_obs=pending,
        )
        assert session.state_hash == frozen_hash

    commit = session.commit_selected(
        environment_step=11,
        decision_id="decision-11",
        selected_choice_id="choice-a",
        selected_post_obs=candidates[0],
    )
    assert commit.commit_count == 1
    assert commit.state_before_hash == frozen_hash
    assert commit.state_after_hash == session.state_hash
    assert commit.state_after_hash != frozen_hash
    assert session.finalize_level_up().commit_count == 1

    with pytest.raises(RecurrentSessionError, match="already|duplicate"):
        session.commit_selected(
            environment_step=11,
            decision_id="decision-11",
            selected_choice_id="choice-a",
            selected_post_obs=candidates[0],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("environment_step", 12, "step"),
        ("decision_id", "other", "decision"),
        ("pending_obs", np.asarray([9.0, 0.5, 1.0], dtype=np.float32), "obs"),
    ],
)
def test_session_rejects_context_and_retry_binding_mismatches(
    tmp_path: Path,
    field: str,
    value,
    message: str,
) -> None:
    """step・decision・pending obs の全 binding sibling を拒否する。

    retry payload が一項目でも変わった場合は新しい policy timestep と推測しない。
    """
    manifest_path, _, _ = build_saved_value_source(tmp_path, recurrent=True)
    scorer = ValueScorer.load(manifest_path)
    session = scorer.new_session()
    pending = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    session.begin_level_up(
        environment_step=11,
        decision_id="decision-11",
        pending_obs=pending,
        episode_start=False,
    )
    arguments = {
        "environment_step": 11,
        "decision_id": "decision-11",
        "pending_obs": pending,
    }
    arguments[field] = value

    with pytest.raises(RecurrentSessionError, match=message):
        session.validate_pending_retry(**arguments)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"environment_step": 12}, "step"),
        ({"decision_id": "other"}, "decision"),
        ({"pending_obs_hash": "0" * 64}, "obs"),
        ({"phase": "selected_post_commit"}, "phase"),
    ],
)
def test_session_rejects_all_formal_context_binding_siblings(
    tmp_path: Path,
    updates: dict,
    message: str,
) -> None:
    """context phase・step・decision・obs hash の不一致を全て拒否する。

    seal が正しく再計算された別 context でも、現在 pending session との binding は緩めない。
    """
    manifest_path, _, _ = build_saved_value_source(tmp_path, recurrent=True)
    scorer = ValueScorer.load(manifest_path)
    session = scorer.new_session()
    context = session.begin_level_up(
        environment_step=11,
        decision_id="decision-11",
        pending_obs=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        episode_start=False,
    )

    with pytest.raises(RecurrentSessionError, match=message):
        session.validate_context(context.with_updates(**updates))


def test_pending_session_rejects_all_external_movement_advances(
    tmp_path: Path,
) -> None:
    """pending/base obs と retry obs を movement policy へ渡せないことを確認する。

    decision が開いている間の public advance を一律拒否し、retry 回数で state を進めない。
    """
    manifest_path, _, _ = build_saved_value_source(tmp_path, recurrent=True)
    scorer = ValueScorer.load(manifest_path)
    session = scorer.new_session()
    pending = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    session.begin_level_up(
        environment_step=11,
        decision_id="decision-11",
        pending_obs=pending,
        episode_start=False,
    )
    frozen_hash = session.state_hash

    with pytest.raises(RecurrentSessionError, match="pending"):
        session.advance_movement(pending, episode_start=False)
    assert session.state_hash == frozen_hash


def test_finalize_rejects_zero_commit_and_reference_transition_matches(
    tmp_path: Path,
) -> None:
    """commit 0 回の session を閉じず、正常 commit を reference rollout と比較する。

    pending を policy に入力しないため、同じ initial state から selected obs を一回進めた結果と一致する。
    """
    manifest_path, _, _ = build_saved_value_source(tmp_path, recurrent=True)
    scorer = ValueScorer.load(manifest_path)
    selected = np.asarray([0.25, 0.5, 1.0], dtype=np.float32)
    pending = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)

    formal = scorer.new_session()
    formal.begin_level_up(
        environment_step=3,
        decision_id="decision-3",
        pending_obs=pending,
        episode_start=False,
    )
    with pytest.raises(RecurrentSessionError, match="exactly once|not committed"):
        formal.finalize_level_up()
    formal_commit = formal.commit_selected(
        environment_step=3,
        decision_id="decision-3",
        selected_choice_id="choice-a",
        selected_post_obs=selected,
    )

    reference = scorer.new_session()
    reference_step = reference.advance_movement(selected, episode_start=False)
    assert formal_commit.movement_action == reference_step.movement_action
    assert formal.state_hash == reference.state_hash

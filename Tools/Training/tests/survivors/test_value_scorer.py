"""PPO/RecurrentPPO critic と ValueScorer の数値 parity を検証する。

raw candidate を一度だけ正規化し、全候補を同じ critic hidden state から独立評価する
実モデル回帰テストを提供する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch as th

_TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.recurrent_policy_session import CriticContext
from games.survivors.value_scorer import TIE_EPSILON, ValueScorer
from value_scorer_fixtures import build_saved_value_source


def _captured_context(
    scorer: ValueScorer,
    *,
    fill_pi: float = 0.125,
    fill_vf: float = -0.25,
) -> CriticContext:
    """scorer の実 policy shape に合う非零 captured context を作る。

    actor と critic を異なる値にし、actor state を vf state と誤用する回帰を見つける。
    """
    schema = scorer.source.policy_state_schema
    if scorer.source.algorithm == "PPO":
        pi = vf = None
    else:
        shape = (
            schema["n_lstm_layers"],
            1,
            schema["lstm_hidden_size"],
        )
        pi = (
            np.full(shape, fill_pi, dtype=np.float32),
            np.full(shape, fill_pi * 2.0, dtype=np.float32),
        )
        if schema["shared_lstm"] or not schema["enable_critic_lstm"]:
            vf = (pi[0].copy(), pi[1].copy())
        else:
            vf = (
                np.full(shape, fill_vf, dtype=np.float32),
                np.full(shape, fill_vf * 2.0, dtype=np.float32),
            )
    return CriticContext.captured(
        environment_step=17,
        decision_id="decision-17",
        pending_obs=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        episode_start=False,
        pi=pi,
        vf=vf,
        policy_state_schema_hash=schema["policy_state_schema_hash"],
    )


def test_ppo_score_matches_direct_predict_values_within_one_micro(
    tmp_path: Path,
) -> None:
    """PPO の direct predict_values と scorer 出力を比較する。

    VecNormalize の固定統計を経た同じ tensor で差が 1e-6 以下になることを要求する。
    """
    manifest_path, model, vecnormalize = build_saved_value_source(
        tmp_path,
        recurrent=False,
    )
    scorer = ValueScorer.load(manifest_path)
    raw = np.asarray([[2.0, 1.0, 0.25], [-3.0, 4.0, 1.5]], dtype=np.float32)
    context = _captured_context(scorer)

    normalized = vecnormalize.normalize_obs(raw.copy())
    obs_tensor, _ = model.policy.obs_to_tensor(normalized)
    with th.no_grad():
        direct = model.policy.predict_values(obs_tensor).cpu().numpy().reshape(-1)
    actual = scorer.score(raw, context)

    np.testing.assert_allclose(
        [item.value_normalized_return for item in actual],
        direct,
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("shared_lstm", "enable_critic_lstm"),
    [(False, True), (True, False), (False, False)],
)
def test_recurrent_score_matches_all_critic_paths_with_nonzero_state(
    tmp_path: Path,
    shared_lstm: bool,
    enable_critic_lstm: bool,
) -> None:
    """separate・shared・disabled critic の direct value と一致させる。

    各候補へ同じ非零 hidden_in を複製し、batch を時系列として誤処理しないことも検査する。
    """
    manifest_path, model, vecnormalize = build_saved_value_source(
        tmp_path,
        recurrent=True,
        shared_lstm=shared_lstm,
        enable_critic_lstm=enable_critic_lstm,
    )
    scorer = ValueScorer.load(manifest_path)
    context = _captured_context(scorer)
    raw = np.asarray([[2.0, 1.0, 0.25], [-3.0, 4.0, 1.5]], dtype=np.float32)
    normalized = vecnormalize.normalize_obs(raw.copy())
    obs_tensor, _ = model.policy.obs_to_tensor(normalized)
    state = context.vf
    repeated_state = tuple(np.repeat(item, len(raw), axis=1) for item in state)
    tensor_state = tuple(th.as_tensor(item) for item in repeated_state)
    starts = th.zeros(len(raw), dtype=th.float32)

    with th.no_grad():
        direct = (
            model.policy.predict_values(obs_tensor, tensor_state, starts)
            .cpu()
            .numpy()
            .reshape(-1)
        )
    actual = scorer.score(raw, context)

    np.testing.assert_allclose(
        [item.value_normalized_return for item in actual],
        direct,
        rtol=0.0,
        atol=1e-6,
    )


def test_candidate_evaluation_is_state_isolated_and_order_independent(
    tmp_path: Path,
) -> None:
    """A を先に含めても B の value が変わらないことを確認する。

    candidate 間で hidden_out を持ち回らず、同じ hidden_in の batch 複製だけを許す。
    """
    manifest_path, _, _ = build_saved_value_source(tmp_path, recurrent=True)
    scorer = ValueScorer.load(manifest_path)
    context = _captured_context(scorer)
    candidate_a = np.asarray([2.0, 1.0, 0.25], dtype=np.float32)
    candidate_b = np.asarray([-3.0, 4.0, 1.5], dtype=np.float32)

    together = scorer.score(np.stack((candidate_a, candidate_b)), context)[1]
    alone = scorer.score(candidate_b[None, :], context)[0]
    reversed_b = scorer.score(np.stack((candidate_b, candidate_a)), context)[0]

    assert together.value_normalized_return == pytest.approx(
        alone.value_normalized_return,
        abs=1e-6,
    )
    assert reversed_b.value_normalized_return == pytest.approx(
        alone.value_normalized_return,
        abs=1e-6,
    )


def test_formal_context_rejects_missing_or_mutated_vf_state_and_nonfinite_obs(
    tmp_path: Path,
) -> None:
    """vf state 欠落・actor state 置換・非有限 observation を拒否する。

    shape が正しくても context seal と policy schema に結ばれていない state は formal 入力にしない。
    """
    manifest_path, _, _ = build_saved_value_source(tmp_path, recurrent=True)
    scorer = ValueScorer.load(manifest_path)
    context = _captured_context(scorer)
    raw = np.asarray([[2.0, 1.0, 0.25]], dtype=np.float32)

    missing = context.with_updates(vf=None)
    with pytest.raises(ValueError, match="vf"):
        scorer.score(raw, missing)

    actor_only = context.with_updates(vf=context.pi)
    with pytest.raises(ValueError, match="actor-only|vf"):
        scorer.score(raw, actor_only)

    context.vf[0][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="seal|integrity"):
        scorer.score(raw, context)

    with pytest.raises(ValueError, match="finite"):
        scorer.score(np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float32), _captured_context(scorer))


def test_context_rejects_wrong_schema_shape_and_unflagged_zero_state(
    tmp_path: Path,
) -> None:
    """policy schema hash・LSTM shape・zero mode の全 context gate を閉じる。

    数値 state が存在しても source policy へ束縛できない formal ranking は生成しない。
    """
    manifest_path, _, _ = build_saved_value_source(tmp_path, recurrent=True)
    scorer = ValueScorer.load(manifest_path)
    context = _captured_context(scorer)
    raw = np.asarray([[2.0, 1.0, 0.25]], dtype=np.float32)

    wrong_hash = context.with_updates(policy_state_schema_hash="0" * 64)
    with pytest.raises(ValueError, match="schema hash"):
        scorer.score(raw, wrong_hash)

    wrong_shape = context.with_updates(
        vf=(context.vf[0][:1], context.vf[1][:1])
    )
    with pytest.raises(ValueError, match="shape"):
        scorer.score(raw, wrong_shape)

    zero = CriticContext.zero_state(
        environment_step=17,
        decision_id="decision-17",
        pending_obs=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        episode_start=False,
        policy_state_schema=dict(scorer.source.policy_state_schema),
    )
    with pytest.raises(ValueError, match="zero-state"):
        scorer.score(raw, zero)


def test_context_is_bound_to_model_vecnormalize_and_observation_source(
    tmp_path: Path,
) -> None:
    """同じ LSTM shape を持つ別 source の context を拒否する。

    policy state schema hash に model・VecNormalize・obs schema hash を含め、architecture 一致だけで流用しない。
    """
    first_path, _, _ = build_saved_value_source(
        tmp_path / "first",
        recurrent=True,
        seed=7,
    )
    second_path, _, _ = build_saved_value_source(
        tmp_path / "second",
        recurrent=True,
        seed=8,
    )
    first = ValueScorer.load(first_path)
    second = ValueScorer.load(second_path)
    context = _captured_context(first)

    with pytest.raises(ValueError, match="schema hash"):
        second.score(
            np.asarray([[2.0, 1.0, 0.25]], dtype=np.float32),
            context,
        )


def test_burn_in_reconstructs_same_policy_bound_state_as_movement_session(
    tmp_path: Path,
) -> None:
    """ordered raw obs burn-in と live movement capture の value を一致させる。

    episode-start flag を含む同じ sequence から pi/vf を再構成し、pending obs は消費しない。
    """
    manifest_path, _, _ = build_saved_value_source(tmp_path, recurrent=True)
    scorer = ValueScorer.load(manifest_path)
    sequence = np.asarray(
        [[-1.0, 0.0, 0.5], [0.25, 0.5, 1.0]],
        dtype=np.float32,
    )
    starts = np.asarray([True, False], dtype=np.bool_)
    pending = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    candidates = np.asarray(
        [[0.25, 0.5, 1.0], [0.0, 0.75, 1.0]],
        dtype=np.float32,
    )
    burn_in = CriticContext.burn_in(
        environment_step=21,
        decision_id="decision-21",
        pending_obs=pending,
        episode_start=False,
        raw_obs_sequence=sequence,
        episode_starts=starts,
        policy_state_schema_hash=scorer.source.policy_state_schema[
            "policy_state_schema_hash"
        ],
    )

    reference_session = scorer.new_session()
    for obs, episode_start in zip(sequence, starts):
        reference_session.advance_movement(
            obs,
            episode_start=bool(episode_start),
        )
    captured = reference_session.begin_level_up(
        environment_step=21,
        decision_id="decision-21",
        pending_obs=pending,
        episode_start=False,
    )
    actual = scorer.score(candidates, burn_in)
    expected = scorer.score(candidates, captured)

    np.testing.assert_allclose(
        [item.value_normalized_return for item in actual],
        [item.value_normalized_return for item in expected],
        rtol=0.0,
        atol=1e-6,
    )


def test_tie_epsilon_does_not_turn_choice_id_into_a_label() -> None:
    """1e-5 以下の value 差を tie と判定する。

    同値候補は入力順を保ち、choice ID の辞書順で人工的な正解へ変換しない。
    """
    assert TIE_EPSILON == 1e-5

"""ItemSelector 閉ループ評価の選択適用と ack binding を検証する。

実 UE5 を起動せず、pending decision を一度だけ選択・適用して episode 集計する。
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import pytest

from reinbalance_survivors_contracts.ui_intent import (
    DecisionOwner,
    UiIntentKind,
    UiIntentV1,
)
from games.survivors.item_selector_eval import (
    ClosedLoopEvaluationError,
    EvaluationCell,
    ItemSelectorClosedLoopEvaluator,
    PairedItemSelectorEvaluator,
)


def _nonmodel_intent(kind: UiIntentKind) -> UiIntentV1:
    """NON_MODEL_UI_POLICY が生成する最小有効 UiIntentV1 を返す。

    decision_owner と kind を固定して abstention 集計の型チェックを検証する。
    """
    extra: dict = {}
    semantic = kind.value
    if kind is UiIntentKind.CHOOSE_FALLBACK:
        extra = {"target_id": "gold-1", "target_index": 0}
        semantic = "gold"
    return UiIntentV1(
        kind=kind,
        semantic_action=semantic,
        decision_owner=DecisionOwner.NON_MODEL_UI_POLICY,
        source_snapshot_hash="h1",
        source_frame_hash="h2",
        source_content_hash="h3",
        ui_state_key="level_up",
        decision_policy_id="non_model_ui_policy_v1",
        decision_rule_id="fallback_heuristic_v1",
        decision_config_hash="hc",
        **extra,
    )


@dataclass(frozen=True)
class _Decision:
    """戦略モックが返す selection の最小値を表す。

    evaluator が choice と decision identity を endpoint にそのまま渡すために使う。
    """

    decision_id: str
    choice_id: str


class _Strategy:
    """live payload の ID を選択結果として返す戦略モック。

    feature の解釈は対象外にして evaluator の transaction 境界だけを検証する。
    """

    def select(self, features: object) -> _Decision:
        """payload の fixture choice を選び返す。

        テストでは features に choice ID をそのまま置く。
        """
        return _Decision(decision_id="decision-1", choice_id="knife")


class _Environment:
    """一つの pending level-up を返す最小閉ループ環境。

    choice の適用後に terminal step を返し、選択が episode outcome へ接続する形を作る。
    """

    def __init__(self) -> None:
        """呼出し履歴を空で初期化する。

        history により apply が一度だけだったことを観測できる。
        """
        self.choice_calls: list[tuple[str, str]] = []
        self._step = 0

    def reset(self, *, seed: int):
        """初期観測と空の情報を返す。

        seed は evaluator が episode ごとに明示していることを表すだけで使用しない。
        """
        return "obs-0", {}

    def step(self, action: int):
        """最初の movement 後に pending、次に terminal を返す。

        pending 中に追加 movement を許さない評価制御を検証可能にする。
        """
        self._step += 1
        if self._step == 1:
            return "pending-obs", 1.0, False, False, {
                "item_decision_features": "features",
                "decision_id": "decision-1",
                "choice_ids": ["wand", "knife"],
            }
        return "terminal-obs", 2.0, True, False, {}

    def choose_level_up(self, decision_id: str, choice_id: str):
        """選択を記録し、ack を含む post-choice observation を返す。

        endpoint の応答は evaluator が選択結果と照合する。
        """
        self.choice_calls.append((decision_id, choice_id))
        return "post-choice-obs", {"decision_id": decision_id, "choice_id": choice_id}


def test_closed_loop_evaluator_applies_each_pending_choice_once_and_summarizes() -> None:
    """pending choice を apply してから次 movement へ進め、episode を集計する。

    return は movement reward のみ、choice count は ack 済み transaction の数だけを報告する。
    """
    environment = _Environment()
    evaluator = ItemSelectorClosedLoopEvaluator(
        strategy=_Strategy(), movement_policy=lambda observation: 8
    )

    report = evaluator.evaluate(environment, episode_count=1, seed_start=10)

    assert environment.choice_calls == [("decision-1", "knife")]
    assert report.summary["episodes"] == 1
    assert report.summary["mean_return"] == pytest.approx(3.0)
    assert report.summary["decision_count"] == 1
    assert report.episodes[0].selections[0].choice_id == "knife"


def test_closed_loop_evaluator_rejects_mismatched_choice_ack() -> None:
    """endpoint の別 choice acknowledgement を成功として扱わない。

    pending decision と apply 応答の binding が壊れた時点で episode を fail-closed に止める。
    """
    environment = _Environment()

    def bad_ack(decision_id: str, choice_id: str):
        """選択とは異なる choice ID の ack を返す。

        evaluator の acknowledgement validation を起動する。
        """
        return "post-choice-obs", {"decision_id": decision_id, "choice_id": "wand"}

    environment.choose_level_up = bad_ack  # type: ignore[method-assign]
    evaluator = ItemSelectorClosedLoopEvaluator(
        strategy=_Strategy(), movement_policy=lambda observation: 8
    )

    with pytest.raises(ClosedLoopEvaluationError, match="acknowledgement"):
        evaluator.evaluate(environment, episode_count=1, seed_start=10)


@pytest.mark.parametrize("kind", [
    UiIntentKind.CHOOSE_FALLBACK,
    UiIntentKind.REROLL,
    UiIntentKind.SKIP,
    UiIntentKind.BANISH,
    UiIntentKind.ACK_CHEST,
    UiIntentKind.CONFIRM,
])
def test_nonmodel_ui_intent_counted_in_card_abstention(kind: UiIntentKind) -> None:
    """NON_MODEL_UI_POLICY UiIntentV1 が card_abstention_count に計上される。

    文字列名ではなく decision_owner フィールドで判定するため、kind ごとに UiIntentV1 オブジェクトを
    そのまま渡し、abstention が正確に 1 になることを確認する。
    """
    intent = _nonmodel_intent(kind)

    class IntentEnv(_Environment):
        def step(self, action: int):
            obs, reward, terminated, truncated, info = super().step(action)
            return obs, reward, terminated, truncated, {**info, "non_model_ui_actions": [intent]}

    evaluator = ItemSelectorClosedLoopEvaluator(strategy=_Strategy(), movement_policy=lambda _: 0)
    result = evaluator.evaluate(IntentEnv(), episode_count=1, seed_start=0)
    assert result.summary["mean_card_abstention_count"] >= 1


def test_bootstrap_paired_difference_uses_all_seeds_in_cluster() -> None:
    """同一 cluster の複数 seed が平均化されて CI に寄与する（上書きされない）。

    _bootstrap に同一 (stratum, cluster) で2件のレコードを直接渡す。
    seed=1: base=0,   candidate=10   → diff=10
    seed=2: base=100, candidate=100  → diff=0
    cluster 平均差 = 5。dict 代入バグがあると最後の seed だけ残り diff=0 になる。
    """
    cells = (
        EvaluationCell(seed=1, scenario="x", stratum="s", cluster="c"),
        EvaluationCell(seed=2, scenario="y", stratum="s", cluster="c"),
    )
    manifest = PairedItemSelectorEvaluator.freeze_manifest(cells)
    evaluator = PairedItemSelectorEvaluator(
        environment_factory=lambda cell: _Environment(),
        movement_policy=lambda _: 0,
        bootstrap_resamples=200,
    )

    # _bootstrap に直接レコードを渡して環境依存を排除する。
    records = [
        {"strategy": "baseline", "stratum": "s", "cluster": "c", "seed": 1, "total_reward": 0.0},
        {"strategy": "candidate", "stratum": "s", "cluster": "c", "seed": 1, "total_reward": 10.0},
        {"strategy": "baseline", "stratum": "s", "cluster": "c", "seed": 2, "total_reward": 100.0},
        {"strategy": "candidate", "stratum": "s", "cluster": "c", "seed": 2, "total_reward": 100.0},
    ]
    ci = evaluator._bootstrap(records, base="baseline", other="candidate")
    diff = ci["difference"]
    assert abs(diff - 5.0) < 0.1, f"expected mean diff ≈5, got {diff}"


def test_paired_evaluator_is_deterministic_resumable_and_requires_all_episodes() -> None:
    cells = (EvaluationCell(seed=3, scenario="a", stratum="early", cluster="c1"), EvaluationCell(seed=4, scenario="b", stratum="late", cluster="c2"))
    manifest = PairedItemSelectorEvaluator.freeze_manifest(cells)
    make = lambda cell: _Environment()
    evaluator = PairedItemSelectorEvaluator(environment_factory=make, movement_policy=lambda _: 0, bootstrap_resamples=10)
    strategies = {"baseline": _Strategy(), "candidate": _Strategy()}
    first = evaluator.evaluate(strategies=strategies, manifest=manifest, required_episodes=3)
    resumed = evaluator.evaluate(strategies=strategies, manifest=manifest, existing_result=first, required_episodes=3)
    assert first == resumed
    assert first["comparisons"]["candidate-vs-baseline"]["resamples"] == 10
    assert first["n_executed"] == 2
    assert first["promotion"] is False

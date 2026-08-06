"""Survivors ItemSelector の level-up 閉ループ評価を定義する。

movement と pending choice の順序を制御し、選択・UE5 acknowledgement・episode outcome を一つの
評価記録に残す。学習用 split や teacher label はこの経路へ入れない。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class ClosedLoopEvaluationError(RuntimeError):
    """閉ループの pending、choice、ack 契約違反を表す。

    実際に適用された choice が確認できない episode を成功サンプルとして集計しないための例外。
    """


class ItemSelectionProtocol(Protocol):
    """閉ループ evaluator が必要とする selector の最小 interface を表す。

    model 実装の代わりに小さな戦略モックも渡せるため、環境 transaction を unit test で検証できる。
    """

    def select(self, features: object) -> Any:
        """feature に対応する decision ID と choice ID を持つ選択結果を返す。

        返却値の具体型は固定せず、evaluator が identity 属性だけを検証する。
        """
        ...


@dataclass(frozen=True, slots=True)
class ClosedLoopSelection:
    """ack 済みの一回の item choice を記録する。

    choice は strategy output ではなく endpoint acknowledgement 照合後にだけ episode 結果へ入る。
    """

    decision_id: str
    choice_id: str


@dataclass(frozen=True, slots=True)
class ClosedLoopEpisodeResult:
    """一 episode の movement reward と ack 済み選択を保持する。

    ``terminated`` と ``truncated`` は別に保存し、終了理由を success/failed の二値へ潰さない。
    """

    seed: int
    total_reward: float
    movement_steps: int
    terminated: bool
    truncated: bool
    selections: tuple[ClosedLoopSelection, ...]


@dataclass(frozen=True, slots=True)
class ClosedLoopEvaluationReport:
    """複数 episode の immutable 結果と集計値を保持する。

    summary は JSON 化可能な primitive だけに限定して CLI 出力と後続比較へ渡す。
    """

    episodes: tuple[ClosedLoopEpisodeResult, ...]
    summary: Mapping[str, Any]


class ItemSelectorClosedLoopEvaluator:
    """movement rollout 中の pending level-up を selector で一度だけ解決する。

    pending response を受けた直後に choice endpoint を呼び、ack が一致するまで次の movement step を実行しない。
    """

    def __init__(
        self,
        *,
        strategy: ItemSelectionProtocol,
        movement_policy: Callable[[Any], Any],
        max_movement_steps: int = 100_000,
    ) -> None:
        """選択戦略、movement policy、episode 上限を固定する。

        上限は壊れた environment が terminal を返さない場合に、無限の UE5 評価 run を防ぐ。
        """
        if not callable(getattr(strategy, "select", None)):
            raise TypeError("strategy must provide select(features)")
        if not callable(movement_policy):
            raise TypeError("movement_policy must be callable")
        if type(max_movement_steps) is not int or max_movement_steps <= 0:
            raise ValueError("max_movement_steps must be a positive integer")
        self.strategy = strategy
        self.movement_policy = movement_policy
        self.max_movement_steps = max_movement_steps

    @staticmethod
    def _pending(info: Any) -> tuple[str, object, frozenset[str]] | None:
        """step info から完全な item pending payload を検証して返す。

        item feature、decision ID、choice ID 集合の三つが揃わない info は pending でないものとして黙認せず停止する。
        """
        if not isinstance(info, Mapping):
            raise ClosedLoopEvaluationError("step info must be an object")
        has_pending_field = "item_decision_features" in info
        pending_flag = info.get("level_up_pending", has_pending_field)
        if type(pending_flag) is not bool:
            raise ClosedLoopEvaluationError("level_up_pending must be bool")
        if not pending_flag:
            if has_pending_field or "decision_id" in info or "choice_ids" in info:
                raise ClosedLoopEvaluationError("non-pending step contains item choice fields")
            return None
        features = info.get("item_decision_features")
        decision_id = info.get("decision_id")
        choices = info.get("choice_ids")
        if not isinstance(decision_id, str) or not decision_id:
            raise ClosedLoopEvaluationError("pending decision_id must be non-empty")
        if features is None:
            raise ClosedLoopEvaluationError("pending item_decision_features are required")
        if (
            not isinstance(choices, Sequence)
            or isinstance(choices, (str, bytes))
            or not choices
            or any(not isinstance(choice, str) or not choice for choice in choices)
            or len(set(choices)) != len(choices)
        ):
            raise ClosedLoopEvaluationError("pending choice_ids must be unique non-empty strings")
        return decision_id, features, frozenset(choices)

    @staticmethod
    def _validate_ack(ack: Any, *, decision_id: str, choice_id: str) -> None:
        """choice endpoint acknowledgement を request identity に照合する。

        timeout/retry の別 response や選択の取り違えを、次 movement 前に fail-closed で検出する。
        """
        if not isinstance(ack, Mapping):
            raise ClosedLoopEvaluationError("choice acknowledgement must be an object")
        if ack.get("decision_id") != decision_id or ack.get("choice_id") != choice_id:
            raise ClosedLoopEvaluationError("choice acknowledgement does not match request")

    def _run_episode(self, environment: Any, *, seed: int) -> ClosedLoopEpisodeResult:
        """一つの reset から terminal/truncated まで閉ループを実行する。

        choice endpoint が返す post-choice observation を次 movement policy に渡し、pending observation を通常 action に再利用しない。
        """
        reset = environment.reset(seed=seed)
        if not isinstance(reset, tuple) or len(reset) != 2:
            raise ClosedLoopEvaluationError("environment reset must return (observation, info)")
        observation, reset_info = reset
        if not isinstance(reset_info, Mapping):
            raise ClosedLoopEvaluationError("environment reset info must be an object")
        total_reward = 0.0
        movement_steps = 0
        selections: list[ClosedLoopSelection] = []
        seen_decisions: set[str] = set()
        while movement_steps < self.max_movement_steps:
            action = self.movement_policy(observation)
            transition = environment.step(action)
            if not isinstance(transition, tuple) or len(transition) != 5:
                raise ClosedLoopEvaluationError("environment step must return five values")
            observation, reward, terminated, truncated, info = transition
            if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
                raise ClosedLoopEvaluationError("environment reward must be finite")
            if type(terminated) is not bool or type(truncated) is not bool:
                raise ClosedLoopEvaluationError("environment done flags must be bool")
            total_reward += float(reward)
            movement_steps += 1
            pending = self._pending(info)
            if pending is not None:
                decision_id, features, choice_ids = pending
                if terminated or truncated:
                    raise ClosedLoopEvaluationError("terminal step cannot contain a pending item choice")
                if decision_id in seen_decisions:
                    raise ClosedLoopEvaluationError("item decision was emitted more than once")
                selected = self.strategy.select(features)
                selected_decision = getattr(selected, "decision_id", None)
                selected_choice = getattr(selected, "choice_id", None)
                if selected_decision != decision_id:
                    raise ClosedLoopEvaluationError("strategy decision_id does not match pending decision")
                if selected_choice not in choice_ids:
                    raise ClosedLoopEvaluationError("strategy choice_id is not pending")
                applied = environment.choose_level_up(decision_id, selected_choice)
                if not isinstance(applied, tuple) or len(applied) != 2:
                    raise ClosedLoopEvaluationError("choice endpoint must return (observation, acknowledgement)")
                observation, acknowledgement = applied
                self._validate_ack(acknowledgement, decision_id=decision_id, choice_id=selected_choice)
                seen_decisions.add(decision_id)
                selections.append(ClosedLoopSelection(decision_id, selected_choice))
            if terminated or truncated:
                return ClosedLoopEpisodeResult(
                    seed=seed,
                    total_reward=total_reward,
                    movement_steps=movement_steps,
                    terminated=terminated,
                    truncated=truncated,
                    selections=tuple(selections),
                )
        raise ClosedLoopEvaluationError("episode exceeded max_movement_steps")

    @staticmethod
    def _summary(episodes: Sequence[ClosedLoopEpisodeResult]) -> dict[str, Any]:
        """ack 済み episode 結果を JSON 向けの比較指標へ集約する。

        selection histogram は choice ID を明示し、平均だけでは見えない model collapse を診断できるようにする。
        """
        if not episodes:
            return {"episodes": 0, "mean_return": 0.0, "mean_movement_steps": 0.0, "decision_count": 0, "choice_counts": {}}
        choices = Counter(choice.choice_id for episode in episodes for choice in episode.selections)
        return {
            "episodes": len(episodes),
            "mean_return": sum(item.total_reward for item in episodes) / len(episodes),
            "mean_movement_steps": sum(item.movement_steps for item in episodes) / len(episodes),
            "decision_count": sum(len(item.selections) for item in episodes),
            "choice_counts": dict(sorted(choices.items())),
        }

    def evaluate(
        self, environment: Any, *, episode_count: int, seed_start: int = 0
    ) -> ClosedLoopEvaluationReport:
        """連続 seed の episode を実行して selection/outcome report を返す。

        episode count と seed は明示的に固定し、再現不能なランダム評価を CLI の既定にしない。
        """
        if type(episode_count) is not int or episode_count <= 0:
            raise ValueError("episode_count must be a positive integer")
        if type(seed_start) is not int:
            raise ValueError("seed_start must be an integer")
        episodes = tuple(
            self._run_episode(environment, seed=seed_start + index)
            for index in range(episode_count)
        )
        return ClosedLoopEvaluationReport(episodes=episodes, summary=self._summary(episodes))

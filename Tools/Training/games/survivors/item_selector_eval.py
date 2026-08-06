"""Survivors item selector の再開可能な paired closed-loop evaluator。"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from reinbalance_survivors_contracts.item_decision import ItemDecisionFeatures
from reinbalance_survivors_contracts.ui_intent import DecisionOwner, UiIntentKind, UiIntentV1

from games.survivors.item_selection_strategy import CardTargetRef, card_target_from_intent


class ClosedLoopEvaluationError(RuntimeError): pass

class ItemSelectionProtocol(Protocol):
    def select(self, features: object) -> Any: ...

@dataclass(frozen=True, slots=True)
class ClosedLoopSelection:
    decision_id: str
    choice_id: str

@dataclass(frozen=True, slots=True)
class ClosedLoopEpisodeResult:
    seed: int
    total_reward: float
    movement_steps: int
    terminated: bool
    truncated: bool
    selections: tuple[ClosedLoopSelection, ...]
    scenario: str = "default"
    survival: float = 0.0
    level: int = 0
    gems: int = 0
    kills: int = 0
    choices: int = 0
    evolution_count: int = 0
    union_count: int = 0
    fallback_count: int = 0
    card_abstention_count: int = 0
    latency_ms: float = 0.0

@dataclass(frozen=True, slots=True)
class ClosedLoopEvaluationReport:
    episodes: tuple[ClosedLoopEpisodeResult, ...]
    summary: Mapping[str, Any]

@dataclass(frozen=True, slots=True)
class EvaluationCell:
    seed: int
    scenario: str = "default"
    stratum: str = "default"
    cluster: str = "default"

def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()

class ItemSelectorClosedLoopEvaluator:
    def __init__(self, *, strategy: ItemSelectionProtocol, movement_policy: Callable[[Any], Any], max_movement_steps: int = 100_000) -> None:
        if not callable(getattr(strategy, "select", None)) or not callable(movement_policy): raise TypeError("strategy.select and movement_policy must be callable")
        if type(max_movement_steps) is not int or max_movement_steps <= 0: raise ValueError("max_movement_steps must be positive")
        self.strategy, self.movement_policy, self.max_movement_steps = strategy, movement_policy, max_movement_steps

    @staticmethod
    def _pending(info: Any) -> tuple[str, object, frozenset[str]] | None:
        if not isinstance(info, Mapping): raise ClosedLoopEvaluationError("step info must be an object")
        pending = info.get("level_up_pending", "item_decision_features" in info)
        if type(pending) is not bool: raise ClosedLoopEvaluationError("level_up_pending must be bool")
        if not pending:
            if any(k in info for k in ("item_decision_features", "decision_id", "choice_ids")): raise ClosedLoopEvaluationError("non-pending step contains item choice fields")
            return None
        did, features, choices = info.get("decision_id"), info.get("item_decision_features"), info.get("choice_ids")
        if not isinstance(did, str) or not did or features is None or not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)) or not choices or any(not isinstance(x,str) or not x for x in choices) or len(set(choices)) != len(choices): raise ClosedLoopEvaluationError("invalid pending item choice")
        return did, features, frozenset(choices)

    @staticmethod
    def _metric(info: Mapping[str, Any], names: tuple[str, ...], default: Any) -> Any:
        for name in names:
            if name in info: return info[name]
        return default

    @staticmethod
    def _validate_ack(ack: Any, *, decision_id: str, choice_id: str) -> None:
        if not isinstance(ack, Mapping) or ack.get("decision_id") != decision_id or ack.get("choice_id") != choice_id: raise ClosedLoopEvaluationError("choice acknowledgement does not match request")

    def _choice(self, features: object, selected: Any, decision_id: str) -> tuple[str, int]:
        # UiIntent is the production boundary. Legacy minimal mocks remain accepted for transaction tests only.
        if isinstance(selected, UiIntentV1):
            if selected.kind is not UiIntentKind.CHOOSE_CARD or selected.decision_owner is not DecisionOwner.ITEM_SELECTOR_SESSION: raise ClosedLoopEvaluationError("strategy must return item-selector choose_card intent")
            try: target = card_target_from_intent(features, selected)
            except (TypeError, ValueError) as exc: raise ClosedLoopEvaluationError(str(exc)) from exc
            if target.decision_id != decision_id: raise ClosedLoopEvaluationError("strategy decision_id does not match pending decision")
            return target.card_id, target.candidate_index
        if getattr(selected, "decision_id", None) != decision_id: raise ClosedLoopEvaluationError("strategy decision_id does not match pending decision")
        return getattr(selected, "choice_id", None), -1

    def _run_episode(self, environment: Any, *, seed: int, scenario: str = "default") -> ClosedLoopEpisodeResult:
        try: reset = environment.reset(seed=seed, options={"scenario": scenario})
        except TypeError: reset = environment.reset(seed=seed)
        if not isinstance(reset, tuple) or len(reset) != 2 or not isinstance(reset[1], Mapping): raise ClosedLoopEvaluationError("environment reset must return (observation, info)")
        observation, latest = reset; reward = 0.0; steps = 0; selections: list[ClosedLoopSelection] = []; latency = 0.; seen: set[str] = set(); abstentions = 0; fallbacks = 0
        while steps < self.max_movement_steps:
            observation, value, terminated, truncated, info = environment.step(self.movement_policy(observation))
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)) or type(terminated) is not bool or type(truncated) is not bool: raise ClosedLoopEvaluationError("invalid environment transition")
            reward += float(value); steps += 1; latest = info
            fallbacks += int(self._metric(info, ("fallback_count", "fallback"), 0))
            nonmodel_step = self._metric(info, ("non_model_ui_actions", "card_abstentions"), 0)
            if isinstance(nonmodel_step, Sequence) and not isinstance(nonmodel_step, (str, bytes)):
                for x in nonmodel_step:
                    if isinstance(x, UiIntentV1):
                        if x.decision_owner is DecisionOwner.NON_MODEL_UI_POLICY:
                            abstentions += 1
                    elif isinstance(x, str) and x in {"choose_fallback", "reroll", "skip", "banish", "ack_chest", "confirm"}:
                        abstentions += 1
            elif isinstance(nonmodel_step, int):
                abstentions += nonmodel_step
            pending = self._pending(info)
            if pending:
                did, features, allowed = pending
                if terminated or truncated or did in seen: raise ClosedLoopEvaluationError("invalid repeated or terminal item decision")
                began = time.perf_counter(); selected = self.strategy.select(features); latency += (time.perf_counter()-began)*1000
                choice, _ = self._choice(features, selected, did)
                if choice not in allowed: raise ClosedLoopEvaluationError("strategy choice_id is not pending")
                observation, ack = environment.choose_level_up(did, choice); self._validate_ack(ack, decision_id=did, choice_id=choice)
                seen.add(did); selections.append(ClosedLoopSelection(did, choice))
                if isinstance(ack, Mapping):
                    latest = ack
                    fallbacks += int(self._metric(ack, ("fallback_count", "fallback"), 0))
                    nonmodel_ack = self._metric(ack, ("non_model_ui_actions", "card_abstentions"), 0)
                    if isinstance(nonmodel_ack, Sequence) and not isinstance(nonmodel_ack, (str, bytes)):
                        for x in nonmodel_ack:
                            if isinstance(x, UiIntentV1):
                                if x.decision_owner is DecisionOwner.NON_MODEL_UI_POLICY:
                                    abstentions += 1
                            elif isinstance(x, str) and x in {"choose_fallback", "reroll", "skip", "banish", "ack_chest", "confirm"}:
                                abstentions += 1
                    elif isinstance(nonmodel_ack, int):
                        abstentions += nonmodel_ack
            if terminated or truncated:
                return ClosedLoopEpisodeResult(seed, reward, steps, terminated, truncated, tuple(selections), scenario, float(self._metric(latest,("survival","survival_time","elapsed_time"),steps)), int(self._metric(latest,("level",),0)), int(self._metric(latest,("gems","gems_collected"),0)), int(self._metric(latest,("kills",),0)), len(selections), int(self._metric(latest,("evolution_count","evolutions"),0)), int(self._metric(latest,("union_count","unions"),0)), fallbacks, abstentions, latency)
        raise ClosedLoopEvaluationError("episode exceeded max_movement_steps")

    @staticmethod
    def _summary(episodes: Sequence[ClosedLoopEpisodeResult]) -> dict[str, Any]:
        n=len(episodes)
        if not n: return {"episodes":0,"mean_return":0.0,"mean_movement_steps":0.0,"decision_count":0,"choice_counts":{}}
        choices=Counter(x.choice_id for e in episodes for x in e.selections)
        means={name: sum(getattr(e,name) for e in episodes)/n for name in ("total_reward","movement_steps","survival","level","gems","kills","choices","evolution_count","union_count","fallback_count","card_abstention_count","latency_ms")}
        return {"episodes":n,"mean_return":means.pop("total_reward"),"mean_movement_steps":means.pop("movement_steps"),"decision_count":sum(len(e.selections) for e in episodes),"choice_counts":dict(sorted(choices.items())), **{"mean_"+k:v for k,v in means.items()}}

    def evaluate(self, environment: Any, *, episode_count: int, seed_start: int=0) -> ClosedLoopEvaluationReport:
        if type(episode_count) is not int or episode_count<=0 or type(seed_start) is not int: raise ValueError("episode_count must be positive and seed_start must be int")
        episodes=tuple(self._run_episode(environment, seed=seed_start+i) for i in range(episode_count)); return ClosedLoopEvaluationReport(episodes,self._summary(episodes))


class PairedItemSelectorEvaluator:
    """freeze 済み matrix を strategy ごとに同一 cell で実行・resume する evaluator。"""
    def __init__(self, *, environment_factory: Callable[[EvaluationCell], Any], movement_policy: Callable[[Any], Any], max_movement_steps: int=100_000, bootstrap_resamples: int=10_000) -> None:
        self.environment_factory, self.movement_policy, self.max_movement_steps, self.bootstrap_resamples = environment_factory, movement_policy, max_movement_steps, bootstrap_resamples

    @staticmethod
    def freeze_manifest(cells: Sequence[EvaluationCell]) -> dict[str, Any]:
        if not cells or len({(x.seed,x.scenario) for x in cells}) != len(cells): raise ValueError("manifest cells must be non-empty and unique")
        payload={"cells":[asdict(x) for x in cells]}; return {**payload,"manifest_hash":canonical_hash(payload)}

    def evaluate(self, *, strategies: Mapping[str, ItemSelectionProtocol], manifest: Mapping[str, Any], existing_result: Mapping[str, Any] | None=None, required_episodes: int | None=None) -> dict[str, Any]:
        cells=tuple(EvaluationCell(**x) for x in manifest.get("cells", [])); frozen=self.freeze_manifest(cells)
        if manifest.get("manifest_hash") != frozen["manifest_hash"]: raise ClosedLoopEvaluationError("manifest hash mismatch")
        names=tuple(strategies)
        if not names: raise ValueError("at least one strategy is required")
        previous: dict[tuple[str,int,str],dict[str,Any]]={}
        if existing_result is not None:
            body={k:v for k,v in existing_result.items() if k!="result_hash"}
            if existing_result.get("result_hash") != canonical_hash(body) or existing_result.get("manifest_hash") != frozen["manifest_hash"]: raise ClosedLoopEvaluationError("existing result hash or manifest does not match")
            previous={(r["strategy"],r["seed"],r["scenario"]):r for r in existing_result.get("episodes",[])}
        records=[]
        for name, strategy in strategies.items():
            for cell in cells:
                key=(name,cell.seed,cell.scenario)
                if key in previous: records.append(previous[key]); continue
                episode=ItemSelectorClosedLoopEvaluator(strategy=strategy,movement_policy=self.movement_policy,max_movement_steps=self.max_movement_steps)._run_episode(self.environment_factory(cell),seed=cell.seed,scenario=cell.scenario)
                records.append({"strategy":name,"stratum":cell.stratum,"cluster":cell.cluster,**asdict(episode),"selections":[asdict(x) for x in episode.selections]})
        required=required_episodes if required_episodes is not None else len(cells)
        executed=min(sum(1 for x in records if x["strategy"]==name) for name in names)
        comparisons={f"{name}-vs-{names[0]}":self._bootstrap(records,names[0],name) for name in names[1:]}
        body={"manifest_hash":frozen["manifest_hash"],"strategies":list(names),"episodes":records,"comparisons":comparisons,"n_required":required,"n_executed":executed,"promotion":executed>=required}
        return {**body,"result_hash":canonical_hash(body)}

    def _bootstrap(self, records: Sequence[Mapping[str,Any]], base: str, other: str) -> dict[str, Any]:
        # resample clusters inside each stratum, always retaining paired strategy records.
        # M3 fix: accumulate into lists — dict assignment loses multiple seeds in the same cluster.
        paired: dict[str, dict[str, dict[str, list[float]]]] = {}
        for row in records:
            if row["strategy"] in (base, other):
                paired.setdefault(str(row["stratum"]), {}).setdefault(str(row["cluster"]), {}).setdefault(str(row["strategy"]), []).append(float(row["total_reward"]))
        # Per-cluster difference: mean(other) − mean(base) across all seeds in the cluster.
        usable = {
            s: [
                sum(v[other]) / len(v[other]) - sum(v[base]) / len(v[base])
                for v in clusters.values()
                if base in v and other in v
            ]
            for s, clusters in paired.items()
        }
        usable = {s: v for s, v in usable.items() if v}
        observed=sum(sum(v) for v in usable.values())/sum(len(v) for v in usable.values()) if usable else 0.0
        rng=random.Random(0); samples=[]
        for _ in range(self.bootstrap_resamples):
            vals=[rng.choice(values) for values in usable.values() for _ in range(len(values))]; samples.append(sum(vals)/len(vals) if vals else 0.0)
        samples.sort(); lo=samples[int(.025*(len(samples)-1))] if samples else 0.; hi=samples[int(.975*(len(samples)-1))] if samples else 0.
        return {"metric":"total_reward","difference":observed,"ci_95":[lo,hi],"resamples":self.bootstrap_resamples,"stratified_by":"stratum","clustered_by":"cluster"}

"""AgentRuntime の性能・30分 soak を検証する (05-01 PR3 Task4)。

やさしい説明: combat/selector/combined の latency 分布と、長時間 synthetic schedule
での memory growth・episode reset・NaN/範囲外 action の有無をまとめて確認します。
GPU 版 combat 推論は別 PR で追加予定のため、ここでの計測はすべて CPU 推論です
（詳細は docs/deployment/agent_runtime.md の Formality 節を参照）。
"""

from __future__ import annotations

import gc
import json
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Final

import numpy as np

from reinbalance_survivors_contracts.canonical_json import sha256_hex
from reinbalance_survivors_contracts.item_decision import CandidateFeatures, ItemDecisionFeatures
from reinbalance_survivors_contracts.target_action import ActionSemantics

from survivors.perception_snapshot import (
    UI_PRESENTATION_SCHEMA_HASH,
    NormalizedRoi,
    UiCandidateTargetV1,
    UiPresentationSnapshotV1,
)
from survivors.runtime import artifact_bundle as ab
from survivors.runtime.agent_runtime import AgentRuntime, decisions_to_jsonl
from survivors.runtime.artifact_bundle import RuntimeBundle
from survivors.runtime.combat_session import CombatSession
from survivors.runtime.decision_scheduler import DecisionScheduler
from survivors.runtime.item_selector_runtime import OnnxItemSelector
from survivors.runtime.item_session import ItemSession

from . import _runtime_fixtures as fx
from .test_agent_runtime import _TickingClock, _fresh_now, _snap

TICK_NS: Final[int] = DecisionScheduler().interval_ns

# GPU 版 combat 推論は別 PR (05-01 option C) で追加予定。ここでは CPU 推論のみ計測する。
THRESHOLD_COMBAT_P95_MS: Final[float] = 8.0
THRESHOLD_SELECTOR_P95_MS: Final[float] = 5.0
THRESHOLD_COMBINED_P99_MS: Final[float] = 20.0

_NMAX: Final[int] = 2
_CONTEXT_DIM: Final[int] = 24
_CANDIDATE_DIM: Final[int] = 9
_FEATURE_SCHEMA: Final[str] = "context_only_v1"
_HIDDEN_DIM: Final[int] = 256
_SEED: Final[int] = 1234


def _hash(label: str) -> str:
    """ラベルから決定的な 64 桁 hex hash を作る。

    やさしい説明: source_snapshot_hash 等の埋め合わせ用に決定的な文字列が必要なだけです。
    """
    return sha256_hex(label.encode("utf-8"))


def _build_combat_policy() -> ab.CombatPolicy:
    """hidden_dim=256 の現実的な combat policy を組み立てる。

    やさしい説明: golden fixture と同じ組み立て方だが、latency 計測が意味を持つ
    大きさの hidden_dim を使います。
    """
    model, config = fx.build_combat_model(hidden_dim=_HIDDEN_DIM, seed=_SEED)
    runtime_model = ab.CombatGruPolicy(**config)
    runtime_model.load_state_dict(model.state_dict(), strict=True)
    runtime_model.eval()
    return ab.CombatPolicy(model=runtime_model, **config)


def _build_item_selector(package_dir: Path) -> OnnxItemSelector:
    """nmax=2/context_dim=24/candidate_dim=9 の実 ItemSelector package を書き出して読む。

    やさしい説明: parity テストで確認済みの実 combo をそのまま再利用します。
    """
    fx.write_item_selector_package(
        package_dir,
        target_capability_hash=_hash("performance-target"),
        nmax=_NMAX,
        context_dim=_CONTEXT_DIM,
        candidate_dim=_CANDIDATE_DIM,
        feature_schema=_FEATURE_SCHEMA,
    )
    return OnnxItemSelector.load(package_dir)


def _build_runtime(
    package_dir: Path, *, clock_ns: Callable[[], int] = time.perf_counter_ns
) -> AgentRuntime:
    """combat + item selector を両方持つ現実的な AgentRuntime を組み立てる。

    やさしい説明: 決定的 JSONL テストが combat/UI 双方の経路を再現できるようにします。
    clock_ns を差し替えると inference timeout 判定も壁時計に依存しなくなります
    （デフォルトは AgentRuntime 自身の既定値 time.perf_counter_ns のままです）。
    """
    combat_policy = _build_combat_policy()
    selector = _build_item_selector(package_dir)
    bundle = RuntimeBundle.from_golden_fixture(combat_policy, item_selector=selector)
    return AgentRuntime(bundle, clock_ns=clock_ns)


def _item_context() -> ItemDecisionFeatures:
    """2 候補 (wand/knife) の level-up 選択 fixture を返す。

    やさしい説明: test_ui_policy_runtime_parity.py の _build_case と同じ形を使います。
    """
    candidates = [
        CandidateFeatures(
            kind="item_card", item_id="wand", new_level=2, owned=True, is_new=False,
            is_evolve=False, is_union=False, has_prerequisite=False, slot_capacity=1,
        ),
        CandidateFeatures(
            kind="item_card", item_id="knife", new_level=1, owned=False, is_new=True,
            is_evolve=False, is_union=False, has_prerequisite=False, slot_capacity=1,
        ),
    ]
    return ItemDecisionFeatures(
        decision_id=_hash("decision"),
        feature_schema=_FEATURE_SCHEMA,
        elapsed_time=125.0,
        level=6,
        hp_ratio=0.7,
        xp_ratio=0.4,
        weapon_slots=(1, 0, 0, 0, 0, 0),
        passive_slots=(0, 0, 0, 0, 0, 0),
        empty_slot_count=11,
        evolution_readiness=0.3,
        choice_count=2,
        card_mask=(True, True),
        fallback_kind="chicken",
        ui_state_validity=0.95,
        ui_state_age=0.2,
        candidates=candidates,
        max_item_cards=_NMAX,
    )


def _ui_presentation(item_context: ItemDecisionFeatures) -> UiPresentationSnapshotV1:
    """item_context と一致する 2 候補の UiPresentationSnapshotV1 を返す。

    やさしい説明: ItemSession の winner 解決に必要な choice_index/validity/semantic_kind を揃えます。
    """
    return UiPresentationSnapshotV1(
        schema_hash=UI_PRESENTATION_SCHEMA_HASH,
        snapshot_id=item_context.decision_id,
        frame_id=_hash("frame"),
        parser_artifact_hash=_hash("parser_artifact"),
        screen_state="level_up_items",
        candidate_set_hash=_hash("candidate_set"),
        inventory_hash=_hash("inventory"),
        source_content_hash=_hash("content"),
        ui_state_key=_hash("ui_state_key"),
        candidates=(
            UiCandidateTargetV1(
                choice_id="wand", choice_index=0, semantic_kind="item_card",
                roi=NormalizedRoi(0.1, 0.1, 0.3, 0.3), validity=True, confidence=0.9,
            ),
            UiCandidateTargetV1(
                choice_id="knife", choice_index=1, semantic_kind="item_card",
                roi=NormalizedRoi(0.4, 0.1, 0.6, 0.3), validity=True, confidence=0.85,
            ),
        ),
        buttons=(),
    )


def _measure_latencies_ns(
    call: Callable[[int], None], *, iterations: int, warmup: int
) -> list[int]:
    """call(i) を iterations 回計測し、warmup 回を除いた ns 単位 latency 一覧を返す。

    やさしい説明: 初回の cold start ノイズを外し、安定した percentile を得ます。
    """
    latencies: list[int] = []
    for i in range(iterations + warmup):
        start = time.perf_counter_ns()
        call(i)
        elapsed = time.perf_counter_ns() - start
        if i >= warmup:
            latencies.append(elapsed)
    return latencies


def _percentile_ms(latencies_ns: list[int], percentile: float) -> float:
    """ns 単位の latency 一覧から ms 単位の percentile を返す。

    やさしい説明: perception_benchmark.py と同じ np.percentile ベースの計算に揃えます。
    """
    return float(np.percentile(latencies_ns, percentile)) / 1e6


_VOLATILE_DECISION_FIELDS: Final[tuple[str, ...]] = (
    "decision_id",
    "inference_started_ns",
    "inference_finished_ns",
)


def _strip_volatile_fields(jsonl: bytes) -> list[dict]:
    """decisions_to_jsonl の出力から呼び出しごとに変わる trace field を取り除く。

    やさしい説明: decision_id は uuid4、inference_*_ns は壁時計なので、
    再現性比較の対象は決定内容 (kind/action_index/confidence/reason 等) に絞ります。
    """
    rows = [json.loads(line) for line in jsonl.splitlines() if line]
    for row in rows:
        for field in _VOLATILE_DECISION_FIELDS:
            row.pop(field, None)
    return rows


class TestDeterministicDecisionLog:
    """[Task4-1] recorded snapshot sequence から決定的な decisions JSONL を作れることを検証する。

    やさしい説明: 同じ録画済み sequence を 2 回流し、decision の中身 (trace id を除く) が
    完全一致することを確かめます。
    """

    def test_recorded_sequence_produces_reproducible_jsonl(self, tmp_path: Path) -> None:
        """test_recorded_sequence_produces_reproducible_jsonl の契約を検証する。

        やさしい説明: combat / chest / level-up / death を含む sequence を 2 回再生して比較します。
        decide() は 1 回につき clock_ns() を 2 回消費する (started_ns / inference_finished_ns)
        ため、8 step 分の _TickingClock には 16 個の値を渡し、両 run で同一シーケンスを
        与えて inference timeout 判定が壁時計のばらつきで揺れないようにします。
        """

        def run(run_index: int) -> bytes:
            runtime = _build_runtime(
                tmp_path / f"package-{run_index}",
                clock_ns=_TickingClock(*range(0, 16_000, 1_000)),
            )
            states = [
                "gameplay", "gameplay", "chest", "gameplay",
                "level_up_items", "gameplay", "death", "gameplay",
            ]
            base_ts = 1_000_000_000
            decisions = []
            for step, state in enumerate(states):
                snap = _snap(state, ts=base_ts + step * TICK_NS)
                decision = runtime.decide(snap, now_ns=_fresh_now(snap), episode_start=(step == 0))
                decisions.append(decision)
            return decisions_to_jsonl(decisions)

        first = run(0)
        second = run(1)
        assert len(first.splitlines()) == 8
        assert _strip_volatile_fields(first) == _strip_volatile_fields(second)


class TestPerformanceBudgets:
    """[Task4-2] combat/selector/combined latency percentile を予算内で確認する。

    やさしい説明: GPU 対応が無い現時点では、これらはすべて CPU 推論の計測値です。
    """

    def test_combat_inference_p95_within_cpu_budget(self) -> None:
        """test_combat_inference_p95_within_cpu_budget の契約を検証する。

        やさしい説明: 孤立した CombatSession.decide だけを繰り返し計測します。
        """
        combat_policy = _build_combat_policy()
        session = CombatSession(combat_policy, action_semantics=ActionSemantics.default_v1().actions)
        rng = np.random.default_rng(_SEED)
        obs_dim = session.observation_dim

        def call(i: int) -> None:
            obs = rng.standard_normal(obs_dim).astype(np.float32)
            session.decide(obs, episode_start=(i == 0))

        latencies_ns = _measure_latencies_ns(call, iterations=200, warmup=20)
        p95_ms = _percentile_ms(latencies_ns, 95)
        assert p95_ms <= THRESHOLD_COMBAT_P95_MS, (
            f"combat inference p95={p95_ms:.3f}ms exceeds CPU budget "
            f"{THRESHOLD_COMBAT_P95_MS}ms (GPU support tracked separately)"
        )

    def test_item_selector_inference_p95_within_budget(self, tmp_path: Path) -> None:
        """test_item_selector_inference_p95_within_budget の契約を検証する。

        やさしい説明: 孤立した ItemSession.decide だけを繰り返し計測します。
        """
        selector = _build_item_selector(tmp_path / "package")
        session = ItemSession(selector)
        item_context = _item_context()
        ui_presentation = _ui_presentation(item_context)

        def call(_: int) -> None:
            session.decide(
                item_context,
                ui_presentation,
                decision_policy_id="performance-policy-v1",
                decision_rule_id="performance-rule-v1",
                decision_config_hash=_hash("decision_config"),
            )

        latencies_ns = _measure_latencies_ns(call, iterations=200, warmup=20)
        p95_ms = _percentile_ms(latencies_ns, 95)
        assert p95_ms <= THRESHOLD_SELECTOR_P95_MS, (
            f"item selector inference p95={p95_ms:.3f}ms exceeds budget {THRESHOLD_SELECTOR_P95_MS}ms"
        )

    def test_combined_decision_p99_within_budget(self, tmp_path: Path) -> None:
        """test_combined_decision_p99_within_budget の契約を検証する。

        やさしい説明: full AgentRuntime.decide の cadence 込み latency を計測します。
        snapshot 組み立ては計測区間の外で行い、decide 自体の latency だけを見ます。
        """
        runtime = _build_runtime(tmp_path / "package")
        warmup = 20
        iterations = 200
        base_ts = 1_000_000_000
        base_now = base_ts + 5_000_000
        latencies_ns: list[int] = []
        for i in range(warmup + iterations):
            snap = _snap("gameplay", ts=base_ts + i * TICK_NS)
            now_ns = base_now + i * TICK_NS
            start = time.perf_counter_ns()
            runtime.decide(snap, now_ns=now_ns, episode_start=(i == 0))
            elapsed = time.perf_counter_ns() - start
            if i >= warmup:
                latencies_ns.append(elapsed)

        p99_ms = _percentile_ms(latencies_ns, 99)
        assert p99_ms <= THRESHOLD_COMBINED_P99_MS, (
            f"combined decide() p99={p99_ms:.3f}ms exceeds budget {THRESHOLD_COMBINED_P99_MS}ms"
        )


class TestThirtyMinuteSoak:
    """[Task4-3] 30 分相当の synthetic schedule で memory growth / state reset / エラー 0 を確認する。

    やさしい説明: 15Hz cadence で 27,000 tick (=30分) を壁時計を待たずに進め、
    メモリ増加・recurrent state reset・NaN/範囲外 action が無いことを確認します。
    """

    def test_thirty_minute_synthetic_schedule_is_stable(self) -> None:
        """test_thirty_minute_synthetic_schedule_is_stable の契約を検証する。

        やさしい説明: 約1分ごとに death を挟んで episode reset を繰り返し検証します。
        """
        combat_policy = _build_combat_policy()
        bundle = RuntimeBundle.from_golden_fixture(combat_policy)
        runtime = AgentRuntime(bundle)

        total_ticks = 30 * 60 * 15  # 30分 x 15Hz cadence
        boundary_every = 900  # 約1分ごとに death -> episode reset を挟む
        sample_every = 3_000

        base_ts = 1_000_000_000
        base_now = base_ts + 5_000_000
        exception_count = 0
        memory_samples: list[int] = []

        # ponytail: tracemalloc は Python object しか追跡できず、torch/ONNX Runtime の
        # native allocation は見えない (hidden=256 GRU を毎 tick 回しても数十 KB しか
        # 見えないのはこのため)。native 側の growth まで見たくなったら psutil の RSS
        # 計測に置き換える。
        tracemalloc.start()
        try:
            for tick in range(total_ticks):
                is_boundary = tick > 0 and tick % boundary_every == 0
                state = "death" if is_boundary else "gameplay"
                snap = _snap(state, ts=base_ts + tick * TICK_NS)
                now_ns = base_now + tick * TICK_NS
                try:
                    decision = runtime.decide(snap, now_ns=now_ns, episode_start=(tick == 0))
                except Exception:
                    exception_count += 1
                    continue

                # AgentDecision.__post_init__ が action_index/confidence の不正値で
                # 例外を送出する (上の except で exception_count に計上済み) ため、
                # ここへ到達した decision は既に valid であることが構造上保証されている。
                if decision.kind == "move":
                    assert 0 <= decision.action_index < ab.REQUIRED_ACTION_DIM
                    assert np.isfinite(decision.confidence)

                if is_boundary:
                    assert decision.kind == "no_op"
                    assert runtime.combat_session.lstm_state_copy() is None

                if tick % sample_every == 0:
                    gc.collect()
                    current, _ = tracemalloc.get_traced_memory()
                    memory_samples.append(current)
        finally:
            tracemalloc.stop()

        assert exception_count == 0, f"{exception_count} decide() calls raised unexpectedly"
        # 最初の数サンプルは warm-up ノイズなので除外し、以降の増加だけを見る。
        stable_samples = memory_samples[3:]
        growth = max(stable_samples) - min(stable_samples)
        assert growth < 10 * 1024 * 1024, f"traced memory grew by {growth} bytes across the soak"

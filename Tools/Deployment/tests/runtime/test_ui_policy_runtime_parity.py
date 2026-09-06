"""UI policy: runtime (05-01) と Training evaluator (02-04) の byte-identical parity を検証する。

decide_non_model_ui_intent() が共通 policy から同じ canonical wire を生成することを確認する。
ItemSelector (choose_card) は item_selector_session が生成し、05-03 は intent を生成しない。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)
from reinbalance_survivors_contracts.ui_intent import (
    ContractValidationError,
    DecisionOwner,
    UiIntentKind,
    UiIntentV1,
    allowed_semantic_actions,
)
from reinbalance_survivors_contracts.ui_policy import (
    ButtonOption,
    ButtonSemantic,
    FallbackTarget,
    NonModelUiPolicyConfigV1,
    ScreenState,
    UiPolicyInputV1,
    decide_non_model_ui_intent,
)


def _policy_input(
    screen_state: ScreenState,
    *,
    hp: float = 0.8,
    fallback_targets: list[dict] | None = None,
    button: dict | None = None,
) -> UiPolicyInputV1:
    """テスト用 UiPolicyInputV1 を返す。"""
    targets = [FallbackTarget.from_wire(t) for t in (fallback_targets or [])]
    btn = ButtonOption.from_wire(button) if button else None
    return UiPolicyInputV1(
        source_snapshot_hash="s" * 64,
        source_frame_hash="f" * 64,
        source_content_hash="c" * 64,
        ui_state_key="k" * 64,
        screen_state=screen_state,
        hp_fraction=hp,
        fallback_targets=tuple(targets),
        button=btn,
    )


class TestUiPolicyRuleCoverage:
    """全 reachable semantic が UiIntentV1 で表現できることを確認する。"""

    def test_choose_card_semantic_set(self):
        """choose_card の semantic_action は "choose_card" だけ。target の意味は target_id/semantic_kind で表す。"""
        allowed = allowed_semantic_actions(UiIntentKind.CHOOSE_CARD)
        assert "choose_card" in allowed

    def test_choose_fallback_semantics_covered(self):
        allowed = allowed_semantic_actions(UiIntentKind.CHOOSE_FALLBACK)
        assert "chicken" in allowed or "gold" in allowed

    def test_no_op_semantic_is_no_op(self):
        assert allowed_semantic_actions(UiIntentKind.NO_OP) == frozenset({"no_op"})

    def test_stop_semantic_is_stop(self):
        assert allowed_semantic_actions(UiIntentKind.STOP) == frozenset({"stop"})

    def test_choose_card_owner_is_item_selector(self):
        """choose_card の decision_owner は item_selector_session でなければならない。"""
        with pytest.raises(ContractValidationError, match="owned by|owner"):
            UiIntentV1(
                kind=UiIntentKind.CHOOSE_CARD,
                semantic_action="choose_card",
                decision_owner=DecisionOwner.NON_MODEL_UI_POLICY,  # wrong: must be ITEM_SELECTOR_SESSION
                source_snapshot_hash="s" * 64,
                source_frame_hash="f" * 64,
                source_content_hash="c" * 64,
                ui_state_key="k" * 64,
                target_index=0,
                candidate_set_hash="c" * 64,
                inventory_hash="i" * 64,
                decision_policy_id="p",
                decision_rule_id="r",
                decision_config_hash="d" * 64,
            )

    def test_non_model_policy_cannot_produce_choose_card(self):
        """non-model policy は choose_card を返せない (choose_fallback だけ)。"""
        inp = _policy_input(
            ScreenState.FALLBACK,
            fallback_targets=[
                {"target_id": "t1", "target_index": 0, "semantic": "chicken", "valid": True}
            ],
        )
        config = NonModelUiPolicyConfigV1.load_default()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.kind != UiIntentKind.CHOOSE_CARD


class TestNonModelPolicyParity:
    """02-04 Training evaluator と同じ policy ロジックが 05-01 で動く。"""

    def _config(self) -> NonModelUiPolicyConfigV1:
        return NonModelUiPolicyConfigV1.load_default()

    def test_ack_chest_on_chest_state(self):
        inp = _policy_input(ScreenState.CHEST)
        config = self._config()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.kind == UiIntentKind.ACK_CHEST
        assert intent.decision_owner == DecisionOwner.NON_MODEL_UI_POLICY
        assert intent.effect_owner == "ui_state_machine_v1"

    def test_confirm_on_confirm_state(self):
        inp = _policy_input(ScreenState.CONFIRM)
        config = self._config()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.kind == UiIntentKind.CONFIRM

    def test_choose_fallback_chicken_when_hp_low(self):
        """HP が低いとき chicken fallback を選ぶ。"""
        inp = _policy_input(
            ScreenState.FALLBACK,
            hp=0.10,
            fallback_targets=[
                {"target_id": "c1", "target_index": 0, "semantic": "chicken", "valid": True},
                {"target_id": "g1", "target_index": 1, "semantic": "gold", "valid": True},
            ],
        )
        config = self._config()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.kind == UiIntentKind.CHOOSE_FALLBACK
        assert intent.semantic_action == "chicken"

    def test_choose_fallback_gold_when_hp_high(self):
        """HP が十分高いとき gold fallback を選ぶ。"""
        inp = _policy_input(
            ScreenState.FALLBACK,
            hp=0.99,
            fallback_targets=[
                {"target_id": "c1", "target_index": 0, "semantic": "chicken", "valid": True},
                {"target_id": "g1", "target_index": 1, "semantic": "gold", "valid": True},
            ],
        )
        config = self._config()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.kind == UiIntentKind.CHOOSE_FALLBACK
        assert intent.semantic_action == "gold"

    def test_stop_on_empty_fallback(self):
        """fallback 候補が空 → stop。"""
        inp = _policy_input(ScreenState.FALLBACK, fallback_targets=[])
        config = self._config()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.kind == UiIntentKind.STOP

    def test_source_binding_copied_to_intent(self):
        """intent.source_snapshot_hash が policy_input.source_snapshot_hash を継承する。"""
        inp = _policy_input(ScreenState.CHEST)
        config = self._config()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.source_snapshot_hash == inp.source_snapshot_hash
        assert intent.source_frame_hash == inp.source_frame_hash
        assert intent.source_content_hash == inp.source_content_hash
        assert intent.ui_state_key == inp.ui_state_key

    def test_intent_canonical_bytes_are_stable(self):
        """同一入力から常に同じ canonical wire bytes が生成される。"""
        inp = _policy_input(ScreenState.CHEST)
        config = self._config()
        i1 = decide_non_model_ui_intent(inp, config)
        i2 = decide_non_model_ui_intent(inp, config)
        assert i1 is not None and i2 is not None
        assert canonical_json_bytes(i1.to_wire()) == canonical_json_bytes(i2.to_wire())

    def test_gameplay_returns_none(self):
        """gameplay 画面に対して policy は None を返す (combat session に委ねる)。"""
        inp = _policy_input(ScreenState.GAMEPLAY)
        config = self._config()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is None


class TestIntentOwnershipRules:
    """UiIntentV1 の owner / effect_owner ルールを確認する。"""

    def test_effect_owner_is_always_ui_state_machine(self):
        """05-03 だけが effect を持つ — effect_owner は固定値。"""
        inp = _policy_input(ScreenState.CHEST)
        config = NonModelUiPolicyConfigV1.load_default()
        intent = decide_non_model_ui_intent(inp, config)
        assert intent is not None
        assert intent.effect_owner == "ui_state_machine_v1"

    def test_ui_state_machine_cannot_generate_intent_in_decision_policy(self):
        """05-03 は intent を生成しない — decision_policy_id に ui_state_machine を含む intent は拒否。"""
        with pytest.raises(ContractValidationError, match="05-03"):
            UiIntentV1(
                kind=UiIntentKind.ACK_CHEST,
                semantic_action="ack_chest",
                decision_owner=DecisionOwner.NON_MODEL_UI_POLICY,
                source_snapshot_hash="s" * 64,
                source_frame_hash="f" * 64,
                source_content_hash="c" * 64,
                ui_state_key="k" * 64,
                decision_policy_id="ui_state_machine_v2",  # 禁止
                decision_rule_id="r",
                decision_config_hash="d" * 64,
            )

    def test_runtime_safety_owns_no_op_and_stop(self):
        for kind, semantic in [(UiIntentKind.NO_OP, "no_op"), (UiIntentKind.STOP, "stop")]:
            intent = UiIntentV1(
                kind=kind,
                semantic_action=semantic,
                decision_owner=DecisionOwner.RUNTIME_SAFETY,
                source_snapshot_hash="s" * 64,
                source_frame_hash="f" * 64,
                source_content_hash="c" * 64,
                ui_state_key="k" * 64,
            )
            assert intent.decision_owner == DecisionOwner.RUNTIME_SAFETY


# ---------------------------------------------------------------------------
# 共有 fixture と別 subprocess/cwd での byte-identical parity
# ---------------------------------------------------------------------------

# repository root は tests/runtime から 4 階層上。
_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE_ROOT = _REPO_ROOT / "Tools" / "Common" / "tests" / "fixtures"
_UI_POLICY_CASES = _FIXTURE_ROOT / "ui_policy_cases_v1.json"
_UI_INTENTS = _FIXTURE_ROOT / "ui_intents_v1.json"

# Training / Deployment いずれの package も import せず、共有契約だけで
# canonical UiIntent JSONL を生成する driver。cwd だけを変えて 2 回実行する。
_PARITY_DRIVER = r'''
import json
import sys
from pathlib import Path

from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes
from reinbalance_survivors_contracts.ui_policy import (
    NonModelUiPolicyConfigV1,
    UiPolicyInputV1,
    decide_non_model_ui_intent,
)

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
config = NonModelUiPolicyConfigV1.load_default()
lines = []
for case in data["cases"]:
    policy_input = UiPolicyInputV1.from_wire(case["input"])
    intent = decide_non_model_ui_intent(policy_input, config)
    wire = intent.to_wire() if intent is not None else None
    lines.append(canonical_json_bytes({"case": case["name"], "intent": wire}))

leaked = sorted(
    name for name in sys.modules
    if name == "games" or name.startswith("games.")
    or name == "survivors" or name.startswith("survivors.")
)
if leaked:
    sys.stderr.write("forbidden Training/Deployment imports: " + ",".join(leaked))
    raise SystemExit(2)

sys.stdout.buffer.write(b"\n".join(lines))
'''


def _load_cases() -> dict:
    """共有 ui_policy_cases_v1 fixture を読む。"""
    return json.loads(_UI_POLICY_CASES.read_text(encoding="utf-8"))


def _run_driver(driver_path: Path, cwd: Path) -> bytes:
    """driver を指定 cwd の別 process で実行し stdout bytes を返す。"""
    result = subprocess.run(
        [sys.executable, str(driver_path), str(_UI_POLICY_CASES)],
        cwd=str(cwd),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    return result.stdout


@pytest.fixture(scope="module")
def parity_driver(tmp_path_factory) -> Path:
    """共有契約だけを使う parity driver script を書き出す。"""
    path = tmp_path_factory.mktemp("parity") / "emit_ui_intents.py"
    path.write_text(_PARITY_DRIVER, encoding="utf-8")
    return path


class TestSharedFixtureParity:
    """[指摘11] 共有 fixture の全 case で期待 intent と一致する。"""

    def test_fixture_exists(self):
        assert _UI_POLICY_CASES.is_file()
        assert _UI_INTENTS.is_file()

    def test_every_case_matches_expected_intent(self):
        data = _load_cases()
        config = NonModelUiPolicyConfigV1.load_default()
        for case in data["cases"]:
            policy_input = UiPolicyInputV1.from_wire(case["input"])
            intent = decide_non_model_ui_intent(policy_input, config)
            actual = intent.to_wire() if intent is not None else None
            assert actual == case["expected"], case["name"]

    def test_installed_config_hash_matches_fixture(self):
        data = _load_cases()
        assert NonModelUiPolicyConfigV1.load_default().config_hash == data["config_hash"]

    def test_canonical_jsonl_sha256_matches_fixture(self):
        """05-01 が生成する JSONL が 02-04 と同じ sha256 になる。"""
        data = _load_cases()
        config = NonModelUiPolicyConfigV1.load_default()
        lines = []
        for case in data["cases"]:
            policy_input = UiPolicyInputV1.from_wire(case["input"])
            intent = decide_non_model_ui_intent(policy_input, config)
            wire = intent.to_wire() if intent is not None else None
            lines.append(canonical_json_bytes({"case": case["name"], "intent": wire}))
        assert sha256_hex(b"\n".join(lines)) == data["expected_jsonl_sha256"]


class TestCrossProcessParity:
    """[指摘11] 別 subprocess / 別 cwd から byte-identical JSONL を得る。"""

    def test_training_and_deployment_cwd_produce_identical_bytes(self, parity_driver):
        training_cwd = _REPO_ROOT / "Tools" / "Training"
        deployment_cwd = _REPO_ROOT / "Tools" / "Deployment"
        assert training_cwd.is_dir() and deployment_cwd.is_dir()
        training_out = _run_driver(parity_driver, training_cwd)
        deployment_out = _run_driver(parity_driver, deployment_cwd)
        assert training_out == deployment_out
        assert training_out

    def test_cross_process_output_matches_fixture_sha256(self, parity_driver):
        data = _load_cases()
        out = _run_driver(parity_driver, _REPO_ROOT / "Tools" / "Deployment")
        assert sha256_hex(out) == data["expected_jsonl_sha256"]

    def test_repo_root_cwd_produces_identical_bytes(self, parity_driver):
        """cwd 依存の相対 import が混ざっていないこと。"""
        from_root = _run_driver(parity_driver, _REPO_ROOT)
        from_deployment = _run_driver(parity_driver, _REPO_ROOT / "Tools" / "Deployment")
        assert from_root == from_deployment

    def test_driver_does_not_import_training_or_deployment(self, parity_driver):
        """driver 内の leak 検査が実際に有効であること (成功終了で確認)。"""
        result = subprocess.run(
            [sys.executable, str(parity_driver), str(_UI_POLICY_CASES)],
            cwd=str(_REPO_ROOT / "Tools" / "Training"),
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0
        assert b"forbidden" not in result.stderr


class TestUiIntentGoldenWire:
    """[指摘11] 共有 ui_intents_v1 fixture の required / forbidden field を検証する。"""

    def _fixture(self) -> dict:
        return json.loads(_UI_INTENTS.read_text(encoding="utf-8"))

    def test_valid_intents_round_trip(self):
        for case in self._fixture()["valid"]:
            intent = UiIntentV1.from_wire(case["wire"])
            assert intent.to_wire() == case["wire"], case["name"]

    def test_invalid_intents_are_rejected(self):
        for case in self._fixture()["invalid"]:
            with pytest.raises(ContractValidationError):
                UiIntentV1.from_wire(case["wire"])

    def test_valid_intent_bytes_are_stable(self):
        for case in self._fixture()["valid"]:
            intent = UiIntentV1.from_wire(case["wire"])
            assert intent.canonical_bytes() == intent.canonical_bytes()
            assert len(intent.intent_hash()) == 64


class TestRuntimeInferencePerformance:
    """[指摘11] combat 推論の p95 / p99 latency が decision 予算内に収まる。"""

    def test_combat_decide_latency_percentiles(self, golden_combat_policy):
        import statistics
        import time

        import numpy as np

        from survivors.runtime.combat_session import CombatSession
        from survivors.runtime.decision_scheduler import DEFAULT_INFERENCE_TIMEOUT_NS

        session = CombatSession(golden_combat_policy)
        session.reset_episode()
        rng = np.random.default_rng(3)
        observation_dim = golden_combat_policy.observation_dim

        # 最初の数回は lazy init を含むため warm-up として捨てる。
        for _ in range(20):
            session.decide(rng.normal(size=observation_dim).astype("float32"))

        samples: list[int] = []
        for _ in range(300):
            observation = rng.normal(size=observation_dim).astype("float32")
            started = time.perf_counter_ns()
            session.decide(observation)
            samples.append(time.perf_counter_ns() - started)

        samples.sort()
        p95 = samples[int(len(samples) * 0.95) - 1]
        p99 = samples[int(len(samples) * 0.99) - 1]
        # CPU 実行での 1 tick 予算 (scheduler の inference timeout) を超えないこと。
        assert p95 <= DEFAULT_INFERENCE_TIMEOUT_NS, f"p95={p95 / 1e6:.2f} ms"
        assert p99 <= DEFAULT_INFERENCE_TIMEOUT_NS, f"p99={p99 / 1e6:.2f} ms"
        assert statistics.median(samples) > 0


class TestSustainedRunStability:
    """[指摘11] 長時間 run で state 形状が保たれ NaN / 範囲外 action が出ない。"""

    def test_long_run_keeps_state_shape_and_finite_values(self, golden_combat_policy):
        import numpy as np

        from survivors.runtime.combat_session import CombatSession

        session = CombatSession(golden_combat_policy)
        session.reset_episode()
        rng = np.random.default_rng(23)
        expected_shape = golden_combat_policy.lstm_state_shape
        observation_dim = golden_combat_policy.observation_dim

        # 15 Hz × 30 分 = 27000 tick 相当。CI 時間の都合で代表 3000 tick を回す。
        for index in range(3000):
            observation = rng.normal(size=observation_dim).astype("float32")
            decision = session.decide(observation, episode_start=(index % 900 == 0))
            assert 0 <= decision.action_index < golden_combat_policy.action_dim
            assert np.isfinite(decision.confidence)
            state = session.lstm_state_copy()
            assert state is not None
            # state が積み上がらず常に同じ形状であること (memory 増加なし)。
            assert state[0].shape == expected_shape
            assert state[1].shape == expected_shape

        final = session.lstm_state_copy()
        assert final is not None
        assert np.all(np.isfinite(final[0]))
        assert np.all(np.isfinite(final[1]))

"""Survivors choice collector の transaction、policy、resume を検証する。

fake choice API と実 recurrent scorer を組み合わせ、pending/preview retry が state を
進めず、apply 後の observation だけが exactly once commit されることを確認する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reinbalance_survivors_contracts.fidelity_verdict import (
    FidelityVerdict,
    GATING_KEYS,
)

from games.survivors.choice_preview import (
    SurvivorsChoicePreview,
    SurvivorsLevelUpPreview,
)
from games.survivors.value_choice_collector import (
    ChoiceTraceCollector,
    CollectorError,
    epsilon_source_scorer_propensities,
)
from games.survivors.value_choice_dataset import DatasetWriter, read_dataset
from games.survivors.value_scorer import ValueScorer
from value_scorer_fixtures import build_saved_value_source


def _fidelity() -> tuple[FidelityVerdict, dict[str, str]]:
    """current-hash integration verdict fixture を作る。

    production collector と同じ共有 validator を通せる最小の visibility metric と
    13 producer hash を用意する。
    """

    from reinbalance_survivors_contracts.fidelity_verdict import FidelityMetric

    hashes = {key: "a" * 64 for key in GATING_KEYS}
    verdict = FidelityVerdict(
        "integration",
        {
            "target_profile_hash": "1" * 64,
            "target_build_attestation_hash": "2" * 64,
            "report_scope": "exact_target",
            "producer_allowlist_version": "fidelity_producer_paths.v1",
            "producer_manifest_hash": "3" * 64,
            "resolved_producers": {key: [] for key in GATING_KEYS},
        },
        (
            FidelityMetric(
                "deploy_obs_visibility",
                0.01,
                "normalized_error",
                True,
                None,
                True,
            ),
        ),
        (),
        {
            "git_commit": "fixture",
            "workspace_dirty_summary": "",
            "audit_tool_version": "fixture",
            "dependency_versions": {},
            "operator": "pytest",
            "timestamp": "2026-07-31T00:00:00Z",
        },
        hashes,
    )
    return verdict, hashes


class FakeChoiceEnv:
    """preview/apply の応答消失を注入できる idempotent fake API。

    同じ decision/choice の再送だけを許可し、server-side apply count と request event を
    collector の retry 契約から観測できるようにする。
    """

    def __init__(self) -> None:
        """timeout fault と server-side cached ack の初期状態を作る。

        schema hash は source fixture 構築後に実 descriptor の値へ置換し、collector の
        schema binding を迂回しない。
        """

        self.preview_calls: list[str] = []
        self.choice_calls: list[tuple[str, str]] = []
        self.applies = 0
        self._preview_timeout = True
        self._choice_timeout = True
        self._ack: tuple[np.ndarray, dict] | None = None
        self.schema_hash = "schema"

    def preview_level_up(
        self,
        decision_id: str,
        expected_choice_ids: list[str],
    ) -> SurvivorsLevelUpPreview:
        """同じ decision への最初の preview 応答だけを失う。

        retry 時にも候補 observation は変えず、no-commit preview の呼出し回数だけを
        記録する。
        """

        self.preview_calls.append(decision_id)
        if self._preview_timeout:
            self._preview_timeout = False
            raise TimeoutError("preview response lost")
        assert expected_choice_ids == ["choice-a", "choice-b"]
        return SurvivorsLevelUpPreview(
            decision_id,
            self.schema_hash,
            (0.0, 0.5, 1.0),
            {
                "choice-a": SurvivorsChoicePreview(
                    "choice-a",
                    (0.25, 0.5, 1.0),
                    ("weapon_slots",),
                ),
                "choice-b": SurvivorsChoicePreview(
                    "choice-b",
                    (0.0, 0.75, 1.0),
                    ("passive_slots",),
                ),
            },
        )

    def choose_level_up(
        self,
        decision_id: str,
        choice_id: str,
    ) -> tuple[np.ndarray, dict]:
        """最初の apply 後だけ ack timeout を発生させる。

        duplicate request は cache 済み ack を返し、production mutation が一度だけに
        なる UE5 endpoint 契約を模倣する。
        """

        self.choice_calls.append((decision_id, choice_id))
        if self._ack is None:
            self.applies += 1
            selected = (
                np.asarray([0.25, 0.5, 1.0], dtype=np.float32)
                if choice_id == "choice-a"
                else np.asarray([0.0, 0.75, 1.0], dtype=np.float32)
            )
            self._ack = (
                selected,
                {"level_up_pending": False, "level_up_choices": []},
            )
        if self._choice_timeout:
            self._choice_timeout = False
            raise TimeoutError("choice ack lost")
        return self._ack


class FakeEpisodeEnv(FakeChoiceEnv):
    """pending 中の ``step`` を即座に失敗させる episode fake。

    最初の movement 後に external decision を発行し、choice ack 後の次 movement で
    terminal にすることで collector の制御順序を観測する。
    """

    def __init__(self) -> None:
        """HTTP retry state に episode step/pending state を追加する。

        server-side apply が完了した時点で pending を解除し、ack response の消失とは
        production mutation state を分離する。
        """

        super().__init__()
        self.pending = False
        self.step_calls = 0

    def reset(self, *, seed: int):
        """固定初期 observation と空 info を返す。

        reset seed は replay event 側で検証し、fake dynamics 自体は seed に依存させない。
        """

        assert seed == 13
        return np.asarray([0.0, 0.5, 1.0], dtype=np.float32), {}

    def step(self, action: int):
        """pending 中の movement を拒否し、二 tick の episode を返す。

        action 値は policy 出力として受理し、step call の順序だけを choice transaction と
        比較する。
        """

        assert isinstance(action, int)
        if self.pending:
            raise AssertionError("/step while pending")
        self.step_calls += 1
        if self.step_calls == 1:
            self.pending = True
            return (
                np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
                0.0,
                False,
                False,
                {
                    "level_up_pending": True,
                    "level_up_decision_id": "decision-episode",
                    "level_up_choices": [
                        {"choice_id": "choice-a"},
                        {"choice_id": "choice-b"},
                    ],
                },
            )
        return (
            np.asarray([0.25, 0.5, 1.0], dtype=np.float32),
            0.0,
            True,
            False,
            {"level_up_pending": False, "level_up_choices": []},
        )

    def choose_level_up(
        self,
        decision_id: str,
        choice_id: str,
    ) -> tuple[np.ndarray, dict]:
        """idempotent apply 後に pending を解除する。

        response timeout が起きても server mutation は完了済みなので、次 retry まで
        ``step`` を許可しない collector 側順序と独立に状態を更新する。
        """

        try:
            return super().choose_level_up(decision_id, choice_id)
        finally:
            if self._ack is not None:
                self.pending = False


def _collector(tmp_path: Path, *, epsilon: float = 0.2):
    """実 recurrent source と writer を formal collector へ組み立てる。

    source/fidelity identity を dataset と collector の両方へ渡し、fixture でも正式 gate を
    迂回しない。
    """

    manifest_path, _, _ = build_saved_value_source(
        tmp_path / "source",
        recurrent=True,
    )
    scorer = ValueScorer.load(manifest_path)
    source_id = scorer.source.descriptor["identity_sha256"]
    writer = DatasetWriter(
        tmp_path / "dataset",
        dataset_id="survivors-choice-test",
        source_identity_sha256=source_id,
    )
    writer.start_shard("shard-00000")
    verdict, hashes = _fidelity()
    session = scorer.new_session()
    env = FakeChoiceEnv()
    env.schema_hash = (
        scorer.source.descriptor["observation_schema"]["reported_hash"]
        or scorer.source.descriptor["observation_schema"]["sha256"]
    )
    collector = ChoiceTraceCollector(
        env=env,
        scorer=scorer,
        session=session,
        writer=writer,
        source_identity_sha256=source_id,
        fidelity_verdict=verdict,
        current_gating_producer_hashes=hashes,
        epsilon=epsilon,
        seed=13,
        journal_path=tmp_path / "collector-journal.json",
    )
    return collector, env, scorer, session, writer


def test_epsilon_source_scorer_propensity_depends_on_candidate_count() -> None:
    """epsilon-greedy の全 candidate propensity を正確に計算する。

    teacher best には exploit と uniform explore の双方を加え、それ以外には
    epsilon / candidate_count だけを割り当てる。
    """

    assert epsilon_source_scorer_propensities(
        ["a", "b"],
        "a",
        epsilon=0.2,
    ) == {"a": 0.9, "b": 0.1}
    assert epsilon_source_scorer_propensities(
        ["a", "b", "c", "d"],
        "c",
        epsilon=0.2,
    ) == {"a": 0.05, "b": 0.05, "c": 0.8500000000000001, "d": 0.05}


def test_retry_transaction_commits_once_and_matches_reference_state(
    tmp_path: Path,
) -> None:
    """HTTP/preview retry 後も一 record・一 recurrent commit に限定する。

    selected post-choice observation を直接一回進めた reference rollout と、次 movement
    action および actor/critic LSTM state が一致することを確認する。
    """

    collector, env, scorer, session, writer = _collector(tmp_path)
    pending = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    result = collector.collect_decision(
        episode_logical_id="episode-13",
        environment_step=7,
        decision_id="decision-7",
        pending_obs=pending,
        episode_start=False,
        choice_ids=["choice-a", "choice-b"],
    )
    writer.commit_shard()

    selected_obs = np.asarray(
        (
            [0.25, 0.5, 1.0]
            if result.selected_choice_id == "choice-a"
            else [0.0, 0.75, 1.0]
        ),
        dtype=np.float32,
    )
    reference = scorer.new_session()
    reference_step = reference.advance_movement(
        selected_obs,
        episode_start=False,
    )

    assert result.commit.commit_count == 1
    assert result.commit.movement_action == reference_step.movement_action
    assert session.state_hash == reference.state_hash
    for actual, expected in zip(session.states[0], reference.states[0]):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(session.states[1], reference.states[1]):
        np.testing.assert_array_equal(actual, expected)
    assert env.applies == 1
    assert env.preview_calls == ["decision-7", "decision-7"]
    assert env.choice_calls == [
        ("decision-7", result.selected_choice_id),
        ("decision-7", result.selected_choice_id),
    ]
    snapshot = read_dataset(tmp_path / "dataset")
    assert snapshot.manifest["record_count"] == 1
    row = snapshot.rows[0]
    assert row["behavior"]["selected_choice_id"] == result.selected_choice_id
    assert row["teacher_label"]["best_choice_id"] in {"choice-a", "choice-b"}
    assert row["record_id"] == result.record_id


def test_process_resume_reuses_decision_and_deduplicates_record(
    tmp_path: Path,
) -> None:
    """同一 decision の process resume を deterministic record ID へ収束させる。

    journal に保存した behavior selection を再利用し、retry duplicate を writer が
    二行目へ増やさないことを確認する。
    """

    collector, env, _, _, writer = _collector(tmp_path)
    pending = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    first = collector.collect_decision(
        episode_logical_id="episode-13",
        environment_step=7,
        decision_id="decision-resume",
        pending_obs=pending,
        episode_start=False,
        choice_ids=["choice-a", "choice-b"],
    )
    second = collector.collect_decision(
        episode_logical_id="episode-13",
        environment_step=7,
        decision_id="decision-resume",
        pending_obs=pending,
        episode_start=False,
        choice_ids=["choice-a", "choice-b"],
    )
    writer.commit_shard()

    assert first.record_id == second.record_id
    assert first.selected_choice_id == second.selected_choice_id
    assert env.applies == 1
    assert read_dataset(tmp_path / "dataset").manifest["record_count"] == 1


def test_new_process_recovers_uncommitted_record_with_same_server_apply(
    tmp_path: Path,
) -> None:
    """append 後・shard commit 前の process interruption を再収集する。

    中断 staging は quarantine されても journal の同じ selection/decision と server cached
    ack を使い、production apply count=1 のまま新 shard へ一 record を確定する。
    """

    collector, env, scorer, _, _ = _collector(tmp_path)
    pending = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    first = collector.collect_decision(
        episode_logical_id="episode-13",
        environment_step=7,
        decision_id="decision-process-resume",
        pending_obs=pending,
        episode_start=False,
        choice_ids=["choice-a", "choice-b"],
    )

    resumed_writer = DatasetWriter(
        tmp_path / "dataset",
        dataset_id="survivors-choice-test",
        source_identity_sha256=collector.source_identity_sha256,
    )
    verdict, hashes = _fidelity()
    resumed = ChoiceTraceCollector(
        env=env,
        scorer=scorer,
        session=scorer.new_session(),
        writer=resumed_writer,
        source_identity_sha256=collector.source_identity_sha256,
        fidelity_verdict=verdict,
        current_gating_producer_hashes=hashes,
        epsilon=collector.epsilon,
        seed=13,
        journal_path=tmp_path / "collector-journal.json",
    ).collect_decision(
        episode_logical_id="episode-13",
        environment_step=7,
        decision_id="decision-process-resume",
        pending_obs=pending,
        episode_start=False,
        choice_ids=["choice-a", "choice-b"],
    )
    resumed_writer.commit_shard()

    assert resumed.record_id == first.record_id
    assert resumed.selected_choice_id == first.selected_choice_id
    assert env.applies == 1
    snapshot = read_dataset(tmp_path / "dataset")
    assert snapshot.manifest["record_count"] == 1
    assert any(path.is_dir() for path in (tmp_path / "dataset/quarantine").iterdir())


def test_episode_never_steps_while_choice_is_pending_and_saves_replay(
    tmp_path: Path,
) -> None:
    """pending fixture で choice commit 前の追加 ``/step`` を禁止する。

    reset、全 movement action、external decision、全 retry request/ack、artifact identity が
    dataset replay events に ordered 保存されることも同じ episode で確認する。
    """

    collector, _, scorer, _, writer = _collector(tmp_path)
    env = FakeEpisodeEnv()
    env.schema_hash = (
        scorer.source.descriptor["observation_schema"]["reported_hash"]
        or scorer.source.descriptor["observation_schema"]["sha256"]
    )
    collector.env = env

    assert collector.collect_episode(
        seed=13,
        episode_logical_id="episode-13",
    ) == 1
    writer.commit_shard()

    assert env.step_calls == 2
    assert env.applies == 1
    row = read_dataset(tmp_path / "dataset").rows[0]
    kinds = [event["kind"] for event in row["replay_events"]]
    assert kinds[0:2] == ["artifact_identity", "reset"]
    assert kinds.count("action") == 1
    assert kinds.count("external_decision") == 1
    assert kinds.count("choice_preview_request") == 2
    assert kinds.count("choice_preview_ack") == 1
    assert kinds.count("choice_request") == 2
    assert kinds.count("choice_ack") == 1


def test_formal_collector_rejects_missing_baseline_stale_and_blocked_fidelity(
    tmp_path: Path,
) -> None:
    """formal collector の全 fidelity failure sibling を起動前に拒否する。

    missing、baseline、current hash 差、blocking verdict を warning 継続せず、dataset
    append や UE5 choice request より前に停止する。
    """

    manifest_path, _, _ = build_saved_value_source(
        tmp_path / "source",
        recurrent=True,
    )
    scorer = ValueScorer.load(manifest_path)
    source_id = scorer.source.descriptor["identity_sha256"]
    writer = DatasetWriter(
        tmp_path / "dataset",
        dataset_id="survivors-choice-test",
        source_identity_sha256=source_id,
    )
    verdict, hashes = _fidelity()

    cases = [
        (None, hashes),
        (verdict.to_wire() | {"verdict_stage": "baseline"}, hashes),
        (verdict, hashes | {"logic_private": "f" * 64}),
        (
            FidelityVerdict(
                verdict.verdict_stage,
                verdict.subject,
                verdict.metrics,
                (__import__(
                    "reinbalance_survivors_contracts.fidelity_verdict",
                    fromlist=["BlockingReason"],
                ).BlockingReason("manual", "blocked"),),
                verdict.provenance,
                verdict.gating_producer_hashes,
            ),
            hashes,
        ),
    ]
    for candidate, current in cases:
        with pytest.raises((CollectorError, ValueError)):
            ChoiceTraceCollector(
                env=FakeChoiceEnv(),
                scorer=scorer,
                session=scorer.new_session(),
                writer=writer,
                source_identity_sha256=source_id,
                fidelity_verdict=candidate,
                current_gating_producer_hashes=current,
            )


def test_collection_cli_exposes_seed_shard_manifest_ports_and_epsilon() -> None:
    """collection CLI の必須 provenance と collection range 引数を固定する。

    seed range、episode count、shard size、source manifest、UE5 ports、epsilon が parser
    から欠落して orchestration が暗黙既定へ戻らないことを確認する。
    """

    from collect_survivors_value_choices import _parser

    args = _parser().parse_args(
        [
            "--manifest",
            "source.json",
            "--fidelity-verdict",
            "fidelity.json",
            "--current-producer-hashes",
            "current.json",
            "--artifact-store",
            "/artifact/store",
            "--seed-start",
            "10",
            "--seed-end",
            "19",
            "--episode-count",
            "8",
            "--shard-size",
            "4",
            "--ue5-ports",
            "8767",
            "8777",
            "--epsilon",
            "0.25",
        ]
    )

    assert (args.seed_start, args.seed_end, args.episode_count) == (10, 19, 8)
    assert args.shard_size == 4
    assert args.manifest == Path("source.json")
    assert args.ue5_ports == [8767, 8777]
    assert args.epsilon == 0.25

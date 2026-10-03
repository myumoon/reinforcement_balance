"""正式蒸留収集（教師 flat obs + 同 step の deploy_raw → DeployObs v2）を fake teacher と golden fixture で検証する。

実 UE5・実教師 model は使わず、03-07 の deploy_raw fixture を返す fake env と、flat obs から決定的に
logits を作る fake teacher で、同じ step の記録・LSTM 境界リセット・padding/burn-in/split・
fail-closed の開始条件・artifact store 保存を確かめます。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from games.survivors import distillation_collector as collector
from games.survivors.combat_distillation_dataset import CombatDistillationDataset
from games.survivors.deploy_obs_wrapper import DeployObsWrapper
from games.survivors.deploy_raw_env import DeployRawEnv
from reinbalance_survivors_contracts.artifact_dag import validate_artifact_dag
from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.fidelity_verdict import (
    GATING_KEYS, BlockingReason, FidelityMetric, FidelityVerdict,
)

FIXTURE = Path(__file__).parent / "fixtures" / "deploy_raw_llt_v1.json"
V2 = DeployObsSchema.default_v2()
ACTION_DIM = 9
STEPS_PER_EPISODE = 7  # fixture は reset + step 7 応答。最後の step で終了するので記録は 7 step


def _payloads():
    """fixture の deploy_raw を応答順（reset → step ...）に並べて返す。

    /reset はトップレベル、/step は info の下にあるので、それぞれから取り出します。
    """
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [r["deploy_raw"] if r["endpoint"] == "/reset" else r["info"]["deploy_raw"] for r in data["responses"]]


class FakeSurvivorsEnv:
    """fixture の応答を返す SurvivorsEnv 代役。flat obs は応答番号そのもの。

    flat obs と deploy_raw は同じ応答から返すので、教師入力と DeployObs の step 対応を検証できます。
    最後の応答で terminated=True にし、episode 境界を作ります。
    """

    def __init__(self, drop_at=None, flat_override=None, params_ok=True):
        """deploy_raw を落とす応答番号・flat obs の差し替え・/params の成否を設定する。

        drop_at=k は k 番目の応答から deploy_raw を抜き、無効のままの UE5 を表します。
        """
        self.payloads, self.drop_at, self.flat_override, self.params_ok = _payloads(), drop_at, flat_override, params_ok
        self.cursor, self.params_calls, self.reset_seeds, self.closed = 0, [], [], False
        self.last_reset_response = None

    def _flat(self):
        """現在の応答番号を 1 要素の flat obs にする。

        差し替え指定（壊れた flat obs の再現用）があればそれをそのまま返します。
        """
        if self.flat_override is not None:
            return self.flat_override
        return np.array([float(self.cursor)], np.float32)

    def set_params(self, **kwargs):
        """/params 呼び出しを記録し、設定した成否を返す。

        SurvivorsEnv.set_params と同じく失敗は例外ではなく False で表します。
        """
        self.params_calls.append(kwargs)
        return self.params_ok

    def reset(self, *, seed=None, options=None):
        """応答 0 を last_reset_response に置き、flat obs と空 info を返す。

        渡された seed は記録し、episode ごとの seed 割り当てを検証できるようにします。
        """
        self.cursor = 0
        self.reset_seeds.append(seed)
        self.last_reset_response = {"obs": [0.0]}
        if self.drop_at != 0:
            self.last_reset_response["deploy_raw"] = self.payloads[0]
        return self._flat(), {}

    def step(self, action):
        """次の応答の deploy_raw を info に入れ、最後の応答で terminated を返す。

        flat obs と deploy_raw は同じ応答番号から作るので、同 step の対応が保たれます。
        """
        self.cursor += 1
        info = {"base_reward": 1.0}
        if self.drop_at != self.cursor:
            info["deploy_raw"] = self.payloads[self.cursor]
        return self._flat(), 1.0, self.cursor == len(self.payloads) - 1, False, info

    def close(self):
        """close されたことを記録する。

        正式経路が収集後に env を必ず閉じることの確認に使います。
        """
        self.closed = True


class FakeTeacher:
    """flat obs から決定的な出力を作る教師。呼び出し履歴を保持する。

    logits = arange(9)*0.1 + obs[0]、value = obs[0]、行動 = obs[0] mod 9、状態は呼び出し回数を数える int です。
    """

    action_dim = ACTION_DIM
    identity_sha256 = "d" * 64

    def __init__(self, logits_shape=(ACTION_DIM,)):
        """logits の形と呼び出し履歴を用意する。

        logits_shape を action_dim と違う値にすると、壊れた教師を再現できます。
        """
        self.logits_shape, self.calls = logits_shape, []

    def initial_state(self):
        """episode 開始時の状態 0 を返す。

        act のたびに 1 増えるので、境界で 0 に戻ったかを数値で確認できます。
        """
        return 0

    def act(self, obs, state, episode_start):
        """呼び出しを記録し、flat obs から決定的な (行動, logits, value, 次状態) を返す。

        記録は (flat obs の先頭値, 渡された状態, episode_start) の組です。
        """
        self.calls.append((float(obs[0]), state, episode_start))
        logits = (np.arange(ACTION_DIM, dtype=np.float32) * 0.1 + obs[0])[: self.logits_shape[0]]
        return int(obs[0]) % ACTION_DIM, logits, float(obs[0]), state + 1


def _expected_tensors():
    """同じ fixture を DeployRawEnv → DeployObsWrapper.release() に直接通した v2 tensor 列を返す。

    収集ループを通した記録と比較するための基準値です（行動は fake teacher と同じ規則）。
    """
    wrapper = DeployObsWrapper.release(DeployRawEnv(FakeSurvivorsEnv()), V2)
    tensor, _ = wrapper.reset(seed=0)
    out = [tensor]
    for t in range(STEPS_PER_EPISODE - 1):
        tensor, *_ = wrapper.step(t % ACTION_DIM)
        out.append(tensor)
    return np.stack(out)


def _collect(**kwargs):
    """fake env / fake teacher で収集し、(結果, env, teacher) を返す。

    既定は 2 episode・長さ 8・burn-in 2・validation なし・seed 100 で、kwargs で上書きできます。
    """
    env, teacher = kwargs.pop("env", FakeSurvivorsEnv()), kwargs.pop("teacher", FakeTeacher())
    options = {"episodes": 2, "sequence_length": 8, "burn_in": 2, "validation_every": 0, "seed": 100, **kwargs}
    return collector.collect_sequences(env, teacher, V2, **options), env, teacher


def test_teacher_outputs_and_deploy_obs_are_recorded_from_the_same_step():
    """各添字の DeployObs v2 は同じ fixture を release wrapper に通した tensor と一致し、教師出力も同じ step 由来。

    flat obs = 応答番号なので、logits[t] = arange*0.1 + t、value[t] = t、行動[t] = t のとき同じ応答から記録されています。
    """
    result, env, _ = _collect(episodes=1)
    dataset, actions = result.datasets["train"], result.teacher_actions["train"]
    expected = _expected_tensors()
    count = STEPS_PER_EPISODE
    np.testing.assert_array_equal(dataset.observations[0, :count], expected)
    for t in range(count):
        np.testing.assert_allclose(dataset.action_logits[0, t], np.arange(ACTION_DIM) * 0.1 + t, rtol=0, atol=1e-6)
        assert dataset.teacher_values[0, t] == t and actions[0, t] == t
    assert dataset.deploy_schema_hash == V2.schema_hash and dataset.teacher_actions is None
    assert env.params_calls == [{"deploy_raw": True}] and result.environment_steps == count


def test_teacher_state_resets_with_episode_start_at_every_episode_boundary():
    """episode ごとに教師状態を初期値へ戻し、最初の step だけ episode_start=True を渡す。

    2 episode 目の先頭でも状態は 0、以降は 1 ずつ進みます。reset seed は episode ごとに seed+episode です。
    """
    _, env, teacher = _collect(episodes=2)
    assert len(teacher.calls) == 2 * STEPS_PER_EPISODE
    for episode in range(2):
        calls = teacher.calls[episode * STEPS_PER_EPISODE:(episode + 1) * STEPS_PER_EPISODE]
        assert [c[1] for c in calls] == list(range(STEPS_PER_EPISODE))
        assert [c[2] for c in calls] == [True] + [False] * (STEPS_PER_EPISODE - 1)
        assert [c[0] for c in calls] == [float(t) for t in range(STEPS_PER_EPISODE)]
    assert env.reset_seeds == [100, 101]


def test_chunks_padding_burn_in_and_episode_level_split():
    """episode を sequence_length ごとの行に分け、padding・burn-in・split を episode 単位で割り当てる。

    5 episode・長さ 4・burn-in 2・validation_every 5 → train 4 episode（8 行）、validation 1 episode（2 行）。
    """
    result, _, _ = _collect(episodes=5, sequence_length=4, burn_in=2, validation_every=5)
    train, validation = result.datasets["train"], result.datasets["validation"]
    assert train.splits == ("train",) * 8 and validation.splits == ("validation",) * 2
    assert validation.episode_ids == ("episode-00004-chunk-0000", "episode-00004-chunk-0001")
    assert train.episode_ids[:2] == ("episode-00000-chunk-0000", "episode-00000-chunk-0001")
    np.testing.assert_array_equal(train.valid_mask[:2], [[True] * 4, [True, True, True, False]])
    np.testing.assert_array_equal(train.burn_in_mask[:2], [[True, True, False, False]] * 2)
    np.testing.assert_array_equal(train.episode_reset_mask[:, 0], True)
    assert not train.episode_reset_mask[:, 1:].any()
    assert np.all(train.action_logits[1, 3] == 0) and train.teacher_values[1, 3] == 0
    assert result.teacher_actions["train"][1, 3] == -1 and result.environment_steps == 5 * STEPS_PER_EPISODE
    train.assert_release_training_ready(V2)
    with pytest.raises(ValueError, match="split leakage"):
        validation.assert_release_training_ready(V2)


@pytest.mark.parametrize("drop_at", [0, 3])
def test_missing_deploy_raw_fails_instead_of_skipping(drop_at):
    """reset / step の応答に deploy_raw が無ければ、その step を飛ばさず収集を失敗させる。

    drop_at=0 は reset 応答、3 は途中の step 応答から deploy_raw を抜きます。
    """
    with pytest.raises(ValueError, match="deploy_raw"):
        _collect(env=FakeSurvivorsEnv(drop_at=drop_at))


def test_deploy_raw_opt_in_failure_stops_collection():
    """/params で deploy_raw を有効化できなければ収集を始めない。

    flat obs だけの応答へ黙って戻らないことを確かめます。
    """
    with pytest.raises(ValueError, match="deploy_raw"):
        _collect(env=FakeSurvivorsEnv(params_ok=False))


@pytest.mark.parametrize("flat", [np.array([np.nan], np.float32), np.zeros((1, 1), np.float32)])
def test_non_finite_or_malformed_flat_obs_is_rejected(flat):
    """flat obs が非数・多次元なら教師へ渡さず拒否する。

    UE5 応答の壊れた観測で教師出力を作らないための検査です。
    """
    with pytest.raises(collector.CollectionError, match="flat observation"):
        _collect(env=FakeSurvivorsEnv(flat_override=flat))


def test_malformed_teacher_logits_are_rejected():
    """教師の logits が action_dim と合わなければ記録しない。

    形の違う logits を dataset に入れず、収集を止めます。
    """
    with pytest.raises(collector.CollectionError, match="logits"):
        _collect(teacher=FakeTeacher(logits_shape=(ACTION_DIM - 1,)))


def test_oracle_values_mixed_into_deploy_obs_are_rejected(monkeypatch):
    """unobservable segment（enemy_hp）に oracle 値が入った観測が混ざれば dataset を作らない。

    release wrapper の出力を差し替えて privileged 値を注入し、release 観測検査で止まることを確かめます。
    """
    original = DeployObsWrapper.observation
    offset, _ = V2.layout["enemy_hp"]

    def leaky(self, raw):
        """release の出力に enemy_hp の oracle 値（valid・age 0）を書き足して返す。

        privileged 値が DeployObs に漏れた状況の再現です。
        """
        tensor = np.array(original(self, raw), copy=True)
        tensor[offset], tensor[V2.dim + offset], tensor[2 * V2.dim + offset] = 0.5, 1.0, 0.0
        return tensor

    monkeypatch.setattr(DeployObsWrapper, "observation", leaky)
    with pytest.raises(ValueError, match="oracle provenance"):
        _collect(episodes=1)


def _verdict(*, blocked=False, stage="integration"):
    """current-hash integration verdict と、その current hash を返す。

    blocked=True で blocking 行付き、stage で baseline などを作れます。
    """
    hashes = {key: "a" * 64 for key in GATING_KEYS}
    if stage == "baseline":
        hashes["deploy_obs_schema"] = hashes["deploy_release_adapter"] = "absent"
    rows = (BlockingReason("terminal", "not approved"),) if blocked else ()
    if stage == "baseline":
        rows = tuple(BlockingReason(c, "baseline") for c in ("action", "offer", "terminal"))
    verdict = FidelityVerdict(
        stage,
        {
            "target_profile_hash": "1" * 64, "target_build_attestation_hash": "2" * 64,
            "report_scope": "exact_target", "producer_allowlist_version": "fidelity_producer_paths.v1",
            "producer_manifest_hash": "3" * 64, "resolved_producers": {key: [] for key in GATING_KEYS},
        },
        (FidelityMetric("deploy_obs_visibility", 0.01, "normalized_error", True, None, True),),
        rows,
        {
            "git_commit": "fixture", "workspace_dirty_summary": "", "audit_tool_version": "fixture",
            "dependency_versions": {}, "operator": "pytest", "timestamp": "2026-10-03T00:00:00Z",
        },
        hashes,
    )
    return verdict, hashes


def _formal_inputs(tmp_path, verdict):
    """教師 descriptor と verdict JSON を tmp に書いてパスを返す。

    教師ロードは fake に差し替えるので、descriptor の中身は任意の JSON で構いません。
    """
    descriptor = tmp_path / "teacher_descriptor.json"
    descriptor.write_bytes(canonical_json_bytes({"identity_sha256": "d" * 64}))
    verdict_path = tmp_path / "verdict.json"
    verdict_path.write_bytes(canonical_json_bytes(verdict.to_wire()))
    return descriptor, verdict_path


def _run(tmp_path, *, verdict_path, hashes, descriptor, teacher_factory=None, env=None):
    """run_formal_collection を fake で呼び、(descriptor, 作られた env の一覧) を返す。

    5 episode・長さ 4・burn-in 2・validation_every 5・seed 7 の固定設定で収集します。
    """
    created = []

    def env_factory():
        """fake env を作って記録する。

        呼ばれた回数で「開始条件を通る前に接続していない」ことを確かめます。
        """
        created.append(env or FakeSurvivorsEnv())
        return created[-1]

    descriptor_node = collector.run_formal_collection(
        source_descriptor=descriptor, fidelity_verdict=verdict_path, current_gating_producer_hashes=hashes,
        env_factory=env_factory, artifact_store=tmp_path / "store", output=tmp_path / "out",
        episodes=5, sequence_length=4, burn_in=2, validation_every=5, seed=7,
        teacher_factory=teacher_factory or (lambda path: FakeTeacher()),
    )
    return descriptor_node, created


def test_formal_collection_saves_dataset_and_parent_identities_to_artifact_store(tmp_path):
    """正式経路は split dataset を保存し、教師 descriptor と verdict を親にした descriptor を store に登録する。

    store の files から dataset を復元でき、DAG 検証・identity_metadata（schema hash 等）・env の close を確認します。
    """
    verdict, hashes = _verdict()
    descriptor, verdict_path = _formal_inputs(tmp_path, verdict)
    node, created = _run(tmp_path, verdict_path=verdict_path, hashes=hashes, descriptor=descriptor)
    assert len(created) == 1 and created[0].closed
    assert node.node_kind == "combat_distillation_dataset"
    meta = node.identity_metadata
    assert meta["deploy_schema_hash"] == V2.schema_hash and meta["deploy_raw_schema_version"] == "survivors_deploy_raw.v1"
    assert meta["teacher_identity_sha256"] == "d" * 64 and meta["fidelity_verdict_identity_hash"] == verdict.identity_hash
    assert meta["sequences"] == {"train": 8, "validation": 2} and meta["environment_steps"] == 35
    store = ArtifactStore(tmp_path / "store")
    parents = []
    for parent in node.parents:
        ref = store.resolve(f"{node.logical_id}/descriptors/{parent.identity_hash}.json")
        parents.append(ArtifactDescriptor.from_wire(json.loads(store.object_path(ref.store_uri).read_bytes())))
    assert {p.identity_metadata["source_role"] for p in parents} == {"teacher_value_source", "integration_fidelity_verdict"}
    assert validate_artifact_dag([node, *parents]).node_count == 3
    stored = store.resolve(f"{node.logical_id}/descriptors/{node.identity_hash}.json")
    assert ArtifactDescriptor.from_wire(json.loads(store.object_path(stored.store_uri).read_bytes())) == node
    restored = tmp_path / "restored"
    restored.mkdir()
    for name in ("data.npz", "manifest.json"):
        ref = store.resolve(f"{node.logical_id}/train/{name}")
        assert ref in node.files
        (restored / name).write_bytes(store.object_path(ref.store_uri).read_bytes())
    CombatDistillationDataset.load(restored).assert_release_training_ready(V2)


def test_formal_gate_checks_descriptor_before_verdict_and_never_connects(tmp_path):
    """教師 descriptor の検証が最初。失敗すると verdict を読まず、env にも接続しない。

    verdict パスは存在しないものを渡し、descriptor の失敗が先に出ることで順序を確かめます。
    """
    _, hashes = _verdict()

    def broken_teacher(path):
        """descriptor 検証に失敗する教師ロードを再現する。

        load_value_source が ValueSourceLoadError（ValueError）を出す場合と同じ形です。
        """
        raise ValueError("descriptor invalid")

    with pytest.raises(ValueError, match="descriptor invalid"):
        _run(tmp_path, verdict_path=tmp_path / "missing.json", hashes=hashes,
             descriptor=tmp_path / "d.json", teacher_factory=broken_teacher)
    assert not (tmp_path / "out").exists() and not (tmp_path / "store").exists()


@pytest.mark.parametrize("case", ["missing", "stale_hash", "blocked", "baseline"])
def test_formal_gate_rejects_invalid_fidelity_verdict_before_collection(tmp_path, case):
    """verdict 欠損・producer hash 不一致・blocking あり・baseline のどれでも env に接続せず止まる。

    どの場合も dataset の出力 directory は作られません。
    """
    verdict, hashes = _verdict(blocked=case == "blocked", stage="baseline" if case == "baseline" else "integration")
    descriptor, verdict_path = _formal_inputs(tmp_path, verdict)
    if case == "missing":
        verdict_path.unlink()
    if case == "stale_hash":
        hashes = {**hashes, GATING_KEYS[0]: "b" * 64}
    created = []
    with pytest.raises(ValueError):
        collector.run_formal_collection(
            source_descriptor=descriptor, fidelity_verdict=verdict_path, current_gating_producer_hashes=hashes,
            env_factory=lambda: created.append(1), artifact_store=tmp_path / "store", output=tmp_path / "out",
            episodes=1, sequence_length=8, burn_in=2, teacher_factory=lambda path: FakeTeacher(),
        )
    assert created == [] and not (tmp_path / "out").exists()


def test_save_rechecks_fidelity_verdict(tmp_path):
    """保存関数を直接呼んでも verdict を再検証し、stale な verdict では何も保存しない。

    正式経路のモジュール入口が複数あっても同じ開始条件で止まることを確かめます。
    """
    verdict, hashes = _verdict()
    descriptor, _ = _formal_inputs(tmp_path, verdict)
    result, _, _ = _collect(episodes=1)
    with pytest.raises(ValueError, match="gating producer hashes differ"):
        collector.save_dataset_artifact(
            result, schema=V2, output=tmp_path / "out", artifact_store=tmp_path / "store", dataset_id="ds",
            teacher_identity_sha256="d" * 64, teacher_descriptor_path=descriptor, verdict=verdict,
            current_gating_producer_hashes={**hashes, GATING_KEYS[0]: "b" * 64}, collection_config={},
        )
    assert not (tmp_path / "out").exists()


def test_value_source_teacher_emits_logits_value_and_resets_lstm_state(tmp_path):
    """ValueSourceTeacher は tiny RecurrentPPO fixture から logits・value・次状態を返し、行動は logits の argmax。

    episode_start=True なら途中の状態を渡しても初期状態と同じ出力になります（LSTM 境界リセット）。
    """
    from games.survivors.value_source_loader import load_value_source
    from value_scorer_fixtures import build_saved_value_source

    manifest_path, _, _ = build_saved_value_source(tmp_path, recurrent=True)
    teacher = collector.ValueSourceTeacher(load_value_source(manifest_path))
    obs = np.array([1.0, -2.0, 0.5], np.float32)
    action, logits, value, state = teacher.act(obs, teacher.initial_state(), True)
    assert logits.shape == (teacher.action_dim,) and np.isfinite(value) and action == int(np.argmax(logits))
    _, _, _, state = teacher.act(obs + 1, state, False)
    again = teacher.act(obs, state, True)
    assert again[0] == action and np.allclose(again[1], logits) and np.isclose(again[2], value)

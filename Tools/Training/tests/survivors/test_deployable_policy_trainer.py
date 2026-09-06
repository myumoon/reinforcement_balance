"""Deployable combat policy の loss・curriculum・resume・formal gate を検証する。
UE5 を使わず、固定 sequence と最小 state holder で step-0 sealing と完全再開を確認する。
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np
import pytest
import torch as th
import torch.nn.functional as F
from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.fidelity_verdict import FidelityMetric, FidelityVerdict, GATING_KEYS
from reinbalance_survivors_contracts.perception_error import PerceptionErrorProfile
from games.survivors.combat_distillation_dataset import CombatDistillationDataset
from games.survivors.deployable_policy_trainer import (
    CurriculumConfig, DeployableCombatPolicy, DeployablePolicyTrainer,
    FormalDependencies, sequence_distillation_loss,
)
from reinbalance_survivors_contracts.perception_profile import (
    CALIBRATION_ARTIFACT_SCHEMA_VERSION,
    FittedPerceptionErrorProfile,
    _FORMAL_FACTORY_TOKEN,
)
SCHEMA = DeployObsSchema.default_v1()
def _dataset() -> CombatDistillationDataset:
    """burn-in 一枠と padding 一枠を持つ一 episode dataset を返す。
    release observation は全 field の canonical missing 表現で leakage を含めない。
    """
    observations = np.zeros((1, 4, SCHEMA.dim * 3), dtype=np.float32)
    neutral = np.concatenate([
        np.full(field.size, field.neutral, dtype=np.float32) for field in SCHEMA.fields
    ])
    observations[:, :, :SCHEMA.dim] = neutral
    observations[:, :, SCHEMA.dim * 2 :] = 1.0
    return CombatDistillationDataset(
        observations, np.array([[[9., -9.], [1., 0.], [0., 1.], [0., 0.]]], np.float32),
        np.array([[99., 1., 2., 0.]], np.float32),
        np.array([[1, 1, 1, 0]], np.bool_), np.array([[1, 0, 0, 0]], np.bool_),
        np.array([[1, 0, 0, 0]], np.bool_), ("ep",), ("train",), SCHEMA.schema_hash, 1,
        ("hud_inventory", "screen_world_observed", "temporal_inferred", "constant"),
    )
def test_sequence_loss_is_actor_kl_plus_value_huber_with_masks() -> None:
    """burn-in/padding を除いた位置だけで KL と Huber を合成する。
    除外位置へ巨大値を置いても手計算した二つの有効 timestep と一致することを確認する。
    """
    data = _dataset()
    student_logits = th.tensor([[[1e4, -1e4], [0., 0.], [1., -1.], [-1e4, 1e4]]])
    student_values = th.tensor([[1e4, 0., 4., -1e4]])
    losses = sequence_distillation_loss(student_logits, student_values, data)
    mask = th.tensor([False, True, True, False])
    expected_kl = F.kl_div(
        F.log_softmax(student_logits.reshape(-1, 2)[mask], dim=-1),
        F.softmax(th.tensor(data.action_logits).reshape(-1, 2)[mask], dim=-1),
        reduction="batchmean",
    )
    expected_value = F.huber_loss(
        student_values.reshape(-1)[mask], th.tensor(data.teacher_values).reshape(-1)[mask],
        delta=1.0,
    )
    th.testing.assert_close(losses["actor_kl"], expected_kl)
    th.testing.assert_close(losses["value_huber"], expected_value)
    th.testing.assert_close(losses["total"], expected_kl + expected_value)
def test_curriculum_has_four_fixed_stages_and_dagger_boundaries() -> None:
    """clean から full corruption へ四段階で進み、DAgger は固定境界だけで追加する。
    境界外・重複 shard を拒否し、resume state が stage と shard identity を保持する。
    """
    config = CurriculumConfig(stage_start_updates=(0, 2, 4, 6), dagger_add_updates=(4, 6))
    trainer = DeployablePolicyTrainer(
        DeployableCombatPolicy(SCHEMA.dim * 3, 2, hidden_dim=4), curriculum_config=config,
    )
    assert [stage.name for stage in trainer.curriculum.stages] == ["clean", "light", "measured", "full"]
    assert [stage.corruption_scale for stage in trainer.curriculum.stages] == [0., 1 / 3, 2 / 3, 1.]
    trainer.curriculum.advance(4)
    trainer.curriculum.add_dagger_shard("dagger-1", at_update=4)
    with pytest.raises(ValueError, match="boundary"):
        trainer.curriculum.add_dagger_shard("bad", at_update=5)
    trainer.curriculum.advance(6)
    with pytest.raises(ValueError, match="duplicate"):
        trainer.curriculum.add_dagger_shard("dagger-1", at_update=6)
class _ErrorState:
    """PerceptionErrorWrapper と同じ state API を持つ最小 fake。
    checkpoint が wrapper RNG sibling を列順に復元したことを観測する。
    """
    def __init__(self, value: int) -> None:
        """単一整数 state を初期値として保持する。
        実 wrapper の複雑な履歴は既存 test に任せ、trainer の配線だけを検証する。
        """
        self.value = value
    def get_corruption_state(self) -> dict:
        """serializable な fake RNG state を返す。
        checkpoint payload へ mutable object を直接共有しない形にする。
        """
        return {"value": self.value}
    def set_corruption_state(self, state: dict) -> None:
        """checkpoint の fake RNG state を復元する。
        呼出し結果は value から直接確認できる。
        """
        self.value = state["value"]
def _vecnormalize(mean: float) -> SimpleNamespace:
    """SB3 VecNormalize の保存対象属性を持つ軽量 fake を返す。
    obs/return running statistics と returns を別々に保持する。
    """
    rms = lambda value: SimpleNamespace(mean=np.array([value]), var=np.array([2.]), count=3.)
    return SimpleNamespace(
        obs_rms=rms(mean), ret_rms=rms(mean + 1), returns=np.array([mean + 2]),
        clip_obs=10., clip_reward=10., gamma=.99, epsilon=1e-8, norm_obs=True, norm_reward=True,
    )
def test_checkpoint_resumes_model_vecnormalize_error_curriculum_and_dagger(tmp_path: Path) -> None:
    """五種類の mutable training state が同じ checkpoint から復元される。
    保存後に全 state を変更し、load が model だけの部分 resume にならないことを確認する。
    """
    model = DeployableCombatPolicy(SCHEMA.dim * 3, 2, hidden_dim=4)
    trainer = DeployablePolicyTrainer(model, curriculum_config=CurriculumConfig((0, 1, 2, 3), (2, 3)))
    trainer.curriculum.advance(2)
    trainer.curriculum.add_dagger_shard("s1", at_update=2)
    trainer.training_steps = 2
    vec, error = _vecnormalize(5.), _ErrorState(7)
    checkpoint = tmp_path / "student.pt"
    expected_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
    trainer.save_checkpoint(checkpoint, vec_normalize=vec, error_wrappers=[error])
    for parameter in model.parameters():
        parameter.data.zero_()
    vec.obs_rms.mean[:] = -1
    error.value = -1
    trainer.curriculum.reset()
    trainer.load_checkpoint(checkpoint, vec_normalize=vec, error_wrappers=[error])
    for key, value in model.state_dict().items():
        th.testing.assert_close(value, expected_model[key])
    assert vec.obs_rms.mean.tolist() == [5.]
    assert error.value == 7
    assert trainer.curriculum.stage_index == 2
    assert trainer.curriculum.dagger_shards == ("s1",)
def test_step_zero_rejects_missing_formal_dependencies_and_dataset_leakage() -> None:
    """formal dependency と release dataset gate が optimizer mutation より先に動く。
    失敗後も training_steps=0 のため、一件処理済みとして resume されないことを確認する。
    """
    trainer = DeployablePolicyTrainer(
        DeployableCombatPolicy(SCHEMA.dim * 3, 2, hidden_dim=4), formal_mode=True,
        formal_dependencies=None,
    )
    with pytest.raises(ValueError, match="formal dependencies"):
        trainer.train_step(_dataset())
    assert trainer.training_steps == 0
    development = DeployablePolicyTrainer(DeployableCombatPolicy(SCHEMA.dim * 3, 2, hidden_dim=4))
    leaked = CombatDistillationDataset(**{**_dataset().__dict__, "splits": ("test",)})
    with pytest.raises(ValueError, match="split leakage"):
        development.train_step(leaked)
    assert development.training_steps == 0
def _formal_profile(session_id: str = "cal-1") -> FittedPerceptionErrorProfile:
    """テスト専用 development_only=False フィクスチャ。

    wire ローダー経由の synthetic formal profile 作成は禁止されたため、
    テストコードのみが参照できる _FORMAL_FACTORY_TOKEN を直接渡す。
    production code は絶対に使用しないこと。
    """
    cal_hash = canonical_hash({"synthetic_session_id": session_id})
    base = PerceptionErrorProfile(calibration_session_ids=[session_id])
    dummy_fit_hash = "a" * 64
    return FittedPerceptionErrorProfile(
        **base.to_wire(),
        calibration_session_hashes={session_id: cal_hash},
        field_sample_counts={"hp_ratio": 2},
        fit_code_hash=dummy_fit_hash,
        development_only=False,
        _factory_token=_FORMAL_FACTORY_TOKEN,
    )


def test_formal_dependency_object_rejects_bootstrap_profile_source() -> None:
    """fixture/bootstrap profile は formal dependency object 自体で拒否される。
    profile 内容を measured と推測せず、source kind の明示を必須にする。
    """
    with pytest.raises(ValueError, match="measured"):
        FormalDependencies(
            fidelity_verdict={}, current_gating_producer_hashes={}, perception_profile=None,
            required_perception_profile_hash="0" * 64, profile_source="bootstrap",
        )
    # development_only=True のプロファイルは validate() で production ガードに拒否される。
    dev_only_profile = PerceptionErrorProfile()
    bootstrap = FormalDependencies(
        fidelity_verdict={}, current_gating_producer_hashes={}, perception_profile=dev_only_profile,
        required_perception_profile_hash=dev_only_profile.profile_hash,
    )
    with pytest.raises(ValueError, match="production"):
        bootstrap.validate()
def test_formal_dependencies_reject_stale_fidelity_and_profile() -> None:
    """current producer hash 差と frozen measured profile hash 差を別々に拒否する。
    両依存が揃った正常 fixture だけが正式 identities を返すことも同時に確認する。
    """
    digits = "abcdef0123456789"
    hashes = {name: digits[index] * 64 for index, name in enumerate(GATING_KEYS)}
    verdict = FidelityVerdict(
        "integration",
        {
            "target_profile_hash": "1" * 64, "target_build_attestation_hash": "2" * 64,
            "report_scope": "exact_target", "producer_allowlist_version": "fidelity_producer_paths.v1",
            "producer_manifest_hash": "3" * 64,
            "resolved_producers": {name: [{"path": name, "sha256": digest}] for name, digest in hashes.items()},
        },
        (FidelityMetric("deploy_obs_visibility", 1., "ratio", True, None, True),), (),
        {
            "git_commit": "abc", "workspace_dirty_summary": "clean", "audit_tool_version": "test",
            "dependency_versions": {}, "operator": "pytest", "timestamp": "2026-08-09T00:00:00Z",
        }, hashes,
    )
    profile = _formal_profile("cal-1")
    dependencies = FormalDependencies(verdict, hashes, profile, profile.profile_hash)
    assert dependencies.validate() == {
        "fidelity_verdict": verdict.identity_hash, "perception_profile": profile.profile_hash,
    }
    stale_hashes = dict(hashes)
    stale_hashes["logic_public"] = "f" * 64
    with pytest.raises(ValueError, match="hashes differ"):
        FormalDependencies(verdict, stale_hashes, profile, profile.profile_hash).validate()
    with pytest.raises(ValueError, match="profile is stale"):
        FormalDependencies(verdict, hashes, profile, "0" * 64).validate()


def _make_formal_deps() -> FormalDependencies:
    """テスト用の valid FormalDependencies を返す。
    verify_current_fidelity が通る verdict と development_only=False profile を
    一か所で構築し、複数 test から参照します。
    """
    digits = "abcdef0123456789"
    hashes = {name: digits[index % len(digits)] * 64 for index, name in enumerate(GATING_KEYS)}
    verdict = FidelityVerdict(
        "integration",
        {
            "target_profile_hash": "1" * 64, "target_build_attestation_hash": "2" * 64,
            "report_scope": "exact_target", "producer_allowlist_version": "fidelity_producer_paths.v1",
            "producer_manifest_hash": "3" * 64,
            "resolved_producers": {name: [{"path": name, "sha256": digest}] for name, digest in hashes.items()},
        },
        (FidelityMetric("deploy_obs_visibility", 1., "ratio", True, None, True),), (),
        {
            "git_commit": "abc", "workspace_dirty_summary": "clean", "audit_tool_version": "test",
            "dependency_versions": {}, "operator": "pytest", "timestamp": "2026-08-09T00:00:00Z",
        }, hashes,
    )
    profile = _formal_profile("cal-1")
    return FormalDependencies(verdict, hashes, profile, profile.profile_hash)


def _publish_calibration_store(
    tmp_path: Path,
    store_dir: str,
    run_key: str,
    profile: object,
) -> tuple[object, str, str]:
    """producer実関数でcalibration commitを1件publishする。

    正規 producer の store と、攻撃者が自作する store の両方を同じ関数で作れるようにし、
    「攻撃者も producer と同じ形の store/commit を用意できる」ことをテストで再現します。
    戻り値は (store, commit logical ID, calibration descriptor identity hash)。
    """
    from benchmark_survivors_perception import (
        FormalBenchmarkRequest,
        _commit_calibration_package,
        calibration_commit_logical_id,
    )
    from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor
    from reinbalance_survivors_contracts.artifact_store import ArtifactStore
    from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes

    store = ArtifactStore(tmp_path / store_dir)
    capture_ref = store.put_bytes(
        logical_id="perception/capture/manifest.json",
        data=canonical_json_bytes({"sessions": ["cal-1"], "store": store_dir}),
        media_type="application/json",
    )
    capture_descriptor = ArtifactDescriptor(
        logical_id="perception/capture/dataset",
        node_kind="source_descriptor",
        producer_id="capture-fixture",
        producer_version="v1",
        identity_metadata={"manifest_logical_id": capture_ref.logical_id},
        files=(capture_ref,),
    )
    request = FormalBenchmarkRequest(
        store=store,
        capture_store_root=tmp_path / f"{store_dir}-captures",
        dependency_descriptors={"capture_dataset": capture_descriptor},
    )
    descriptors, _staged, raw_ref, artifact_ref = _commit_calibration_package(
        request,
        run_key,
        profile,
        {"capture_dataset_hash": capture_ref.sha256},
        request.calibration_logical_id(run_key),
        request.calibration_provenance_logical_id(run_key),
    )
    assert artifact_ref.logical_id.endswith("/profile.artifact.json")
    assert json.loads(store.object_path(raw_ref.store_uri).read_bytes()) == profile.to_wire()
    assert json.loads(store.object_path(artifact_ref.store_uri).read_bytes()) == (
        profile.to_artifact_wire()
    )
    return store, calibration_commit_logical_id(run_key), descriptors[1].identity_hash


def _formal_deps_payload(
    dependencies: FormalDependencies, store: object, commit_logical_id: str
) -> dict:
    """store形式のformal_dependencies payloadを組み立てる。

    期待 calibration descriptor hash はこの JSON には入りません（別チャネル管理）。
    """
    return {
        "fidelity_verdict": dependencies.fidelity_verdict.to_wire(),
        "perception_profile_store_root": str(store.root),
        "perception_calibration_commit_logical_id": commit_logical_id,
        "required_perception_profile_hash": dependencies.required_perception_profile_hash,
        "current_gating_producer_hashes": dict(
            dependencies.current_gating_producer_hashes
        ),
        "profile_source": dependencies.profile_source,
    }


def _producer_formal_deps_file(
    tmp_path: Path,
) -> tuple[Path, str, str, FormalDependencies]:
    """producerのcalibration commitを使うformal dependencies fixtureを作る。

    Training 側 loader は自由な logical ID ではなく producer が freeze した
    calibration commit のみを入口にするため、fixture も本番と同じ
    `_commit_calibration_package()` で store を作ります。
    戻り値は (JSONパス, commit logical ID, 別チャネルの期待 descriptor hash, 期待値)。
    """
    dependencies = _make_formal_deps()
    store, commit_logical_id, descriptor_hash = _publish_calibration_store(
        tmp_path, "artifact-store", "producer-integration", dependencies.perception_profile
    )
    path = tmp_path / "formal-deps.json"
    path.write_text(
        json.dumps(_formal_deps_payload(dependencies, store, commit_logical_id)),
        encoding="utf-8",
    )
    return path, commit_logical_id, descriptor_hash, dependencies


def test_load_formal_deps_reads_producer_artifact_envelope(tmp_path: Path) -> None:
    """producerのcalibration commitを経由したときだけformal profileをロードできる。"""
    from train_survivors_deployable_policy import _load_formal_deps

    path, commit_logical_id, descriptor_hash, expected = _producer_formal_deps_file(tmp_path)
    loaded = _load_formal_deps(
        path, required_calibration_descriptor_hash=descriptor_hash
    )

    assert loaded is not None
    assert commit_logical_id.startswith("perception/calibration_commit/")
    assert loaded.perception_profile.to_artifact_wire() == (
        expected.perception_profile.to_artifact_wire()
    )
    assert loaded.required_calibration_descriptor_hash == descriptor_hash
    # descriptor 束縛まで含めて step-0 gate を通過する。
    loaded.validate()

    stored = json.loads(path.read_text(encoding="utf-8"))
    # 別チャネルの期待 identity が違えば fail-closed になる。
    with pytest.raises(ValueError, match="does not match the expected"):
        _load_formal_deps(path, required_calibration_descriptor_hash="d" * 64)

    # 自由な logical ID（raw profile alias）は入口として受け付けない。
    payload = dict(stored)
    payload["perception_calibration_commit_logical_id"] = (
        "perception/calibration/producer-integration/profile.json"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration commit schema is not supported"):
        _load_formal_deps(path, required_calibration_descriptor_hash=descriptor_hash)


def test_load_formal_deps_requires_out_of_band_descriptor_hash(tmp_path: Path) -> None:
    """期待 descriptor hash をJSON内に書いても信頼根にはならない。

    store形式は別チャネルの期待値が無ければ必ず失敗し、JSONへ期待値キーを
    追加した場合も unknown key として拒否されます。
    """
    from train_survivors_deployable_policy import _load_formal_deps

    path, _commit, descriptor_hash, _expected = _producer_formal_deps_file(tmp_path)

    # 別チャネルの期待値なし = 正規 store でもロード不可。
    with pytest.raises(ValueError, match="out-of-band expected calibration descriptor hash"):
        _load_formal_deps(path)

    # JSON 側へ期待値を書き足しても schema で弾かれる（自己申告を受け付けない）。
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["required_calibration_descriptor_hash"] = descriptor_hash
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown or missing keys"):
        _load_formal_deps(path, required_calibration_descriptor_hash=descriptor_hash)


def test_load_formal_deps_rejects_fully_self_consistent_attacker_input(
    tmp_path: Path,
) -> None:
    """store・commit・descriptorを攻撃者が全て自作しても formal 化できない。

    攻撃者は producer と同じ関数で自分の store に calibration commit を publish でき、
    formal_dependencies.json も自由に書けます。それでも別チャネルの期待
    descriptor hash（正規 calibration 実行の値）と一致しないため拒否されます。
    """
    from train_survivors_deployable_policy import _load_formal_deps

    _path, _commit, trusted_hash, dependencies = _producer_formal_deps_file(tmp_path)

    # 攻撃者: 自分の store / commit / descriptor を自己整合的に用意する。
    forged_store, forged_commit, forged_hash = _publish_calibration_store(
        tmp_path, "attacker-store", "attacker-run", dependencies.perception_profile
    )
    assert forged_hash != trusted_hash
    forged_path = tmp_path / "attacker-deps.json"
    forged_path.write_text(
        json.dumps(_formal_deps_payload(dependencies, forged_store, forged_commit)),
        encoding="utf-8",
    )

    # 攻撃者が自分の hash を渡せる経路は存在しない（CLI 側は正規値で固定）。
    with pytest.raises(ValueError, match="does not match the expected"):
        _load_formal_deps(
            forged_path, required_calibration_descriptor_hash=trusted_hash
        )


def _forged_promoted_calibration_store(tmp_path: Path) -> tuple[Any, str, str]:
    """公開fitのdevelopment artifactを改ざんし攻撃者自身のcommitごとpublishする。

    fit_error_profile() の development fixture を development_only=False に書き換え、
    攻撃者が source/profile descriptor と calibration commit まで自己整合的に作ります。
    レビューで再現された攻撃入力をそのまま Training loader へ与えるための fixture です。
    戻り値は (store, commit logical ID, 攻撃者側 descriptor identity hash)。
    """
    from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor
    from reinbalance_survivors_contracts.artifact_store import ArtifactStore
    from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes
    from reinbalance_survivors_contracts.perception_profile import CalibrationResidual
    from survivors.perception_error_fit import fit_error_profile

    residuals = [
        CalibrationResidual(f"s{index}", "f0", "hp_ratio", 0.01, 1.0, 0)
        for index in range(2)
    ]
    wire = fit_error_profile(residuals, ["s0", "s1"], []).to_artifact_wire()
    assert wire["development_only"] is True
    wire["development_only"] = False
    store = ArtifactStore(tmp_path / "forged-store")
    refs = {
        name: store.put_bytes(
            logical_id=f"perception/package/calibration/forged/{name}",
            data=payload,
            media_type="application/json",
        )
        for name, payload in (
            ("profile.json", canonical_json_bytes(wire["profile"])),
            ("profile.artifact.json", canonical_json_bytes(wire)),
            (
                "provenance.json",
                canonical_json_bytes({
                    "schema_version": "perception_calibration_package.v1",
                    "profile_artifact": wire,
                    "subject_hashes": {},
                }),
            ),
        )
    }
    source = ArtifactDescriptor(
        logical_id="perception/capture/source",
        node_kind="source_descriptor",
        producer_id="perception_error_fit",
        producer_version="v2",
        identity_metadata={"split_manifest_hash": "e" * 64},
        files=(refs["profile.json"],),
    )
    node = ArtifactDescriptor(
        logical_id="perception/calibration/forged",
        node_kind="perception_calibration_profile",
        producer_id="perception_error_fit",
        producer_version="v2",
        identity_metadata={
            "profile_hash": wire["profile_hash"],
            "fit_code_hash": wire["fit_code_hash"],
            "subject_hashes": {},
        },
        parents=(source.node_ref(),),
        files=tuple(refs.values()),
    )
    commit_logical_id = "perception/calibration_commit/forged"
    store.put_bytes(
        logical_id=commit_logical_id,
        data=canonical_json_bytes({
            "schema_version": "perception_calibration_commit.v1",
            "run_key": "forged",
            "profile_descriptor_hash": node.identity_hash,
            "refs": [
                store.put_bytes(
                    logical_id=(
                        f"perception/package/descriptors/{descriptor.identity_hash}.json"
                    ),
                    data=canonical_json_bytes(descriptor.to_wire()),
                    media_type="application/json",
                ).to_wire()
                for descriptor in (source, node)
            ],
        }),
        media_type="application/json",
    )
    return store, commit_logical_id, node.identity_hash


def test_load_formal_deps_rejects_promoted_development_fixture_store(
    tmp_path: Path,
) -> None:
    """改ざんdevelopment fixture＋攻撃者commitを指すJSONは formal 化されない。

    攻撃者は store_root / commit logical ID / store 内 descriptor を全て自作できますが、
    期待 descriptor identity だけは JSON に書けないため、この経路は塞がれています。
    """
    from train_survivors_deployable_policy import _load_formal_deps

    _path, _commit, trusted_hash, dependencies = _producer_formal_deps_file(tmp_path)
    forged_store, forged_commit, forged_hash = _forged_promoted_calibration_store(tmp_path)
    assert forged_hash != trusted_hash
    forged_path = tmp_path / "forged-deps.json"
    forged_path.write_text(
        json.dumps(_formal_deps_payload(dependencies, forged_store, forged_commit)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="out-of-band expected calibration descriptor hash"):
        _load_formal_deps(forged_path)
    with pytest.raises(ValueError, match="does not match the expected"):
        _load_formal_deps(
            forged_path, required_calibration_descriptor_hash=trusted_hash
        )
    # 攻撃者hashをJSONへ書いても schema が受け付けない。
    payload = json.loads(forged_path.read_text(encoding="utf-8"))
    payload["required_calibration_descriptor_hash"] = forged_hash
    forged_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown or missing keys"):
        _load_formal_deps(
            forged_path, required_calibration_descriptor_hash=trusted_hash
        )


def test_cli_requires_out_of_band_calibration_hash_for_store_deps(
    tmp_path: Path, capsys: Any
) -> None:
    """CLIエントリポイントも別チャネルの期待hashなしでは formal 訓練を始めない。

    攻撃者は formal_dependencies.json を丸ごと差し替えられますが、
    --required-calibration-descriptor-hash を省略した時点で exit 2 になります。
    """
    from train_survivors_deployable_policy import main

    path, _commit, _hash, _expected = _producer_formal_deps_file(tmp_path)
    dataset_dir = tmp_path / "dataset"
    _dataset().save(dataset_dir)
    argv = [
        "--dataset", str(dataset_dir),
        "--output-dir", str(tmp_path / "out"),
        "--formal-deps", str(path),
    ]

    assert main(argv) == 2
    assert "out-of-band expected calibration descriptor hash" in capsys.readouterr().err


def test_cli_rejects_attacker_store_when_trusted_hash_differs(
    tmp_path: Path, capsys: Any
) -> None:
    """攻撃者storeを指すJSONでも、CLIの信頼済みhashと違えば exit 2 になる。"""
    from train_survivors_deployable_policy import main

    _path, _commit, trusted_hash, dependencies = _producer_formal_deps_file(tmp_path)
    forged_store, forged_commit, forged_hash = _publish_calibration_store(
        tmp_path, "attacker-store", "attacker-run", dependencies.perception_profile
    )
    assert forged_hash != trusted_hash
    forged_path = tmp_path / "attacker-deps.json"
    forged_path.write_text(
        json.dumps(_formal_deps_payload(dependencies, forged_store, forged_commit)),
        encoding="utf-8",
    )
    dataset_dir = tmp_path / "dataset"
    _dataset().save(dataset_dir)

    exit_code = main(
        [
            "--dataset", str(dataset_dir),
            "--output-dir", str(tmp_path / "out"),
            "--formal-deps", str(forged_path),
            "--required-calibration-descriptor-hash", trusted_hash,
        ]
    )
    assert exit_code == 2
    assert "does not match the expected" in capsys.readouterr().err


def test_store_formal_deps_load_from_training_cwd_without_pythonpath(
    tmp_path: Path,
) -> None:
    """documented cwdからrepo-root package名に依存せずstore形式をロードできる。"""
    path, _commit, descriptor_hash, _expected = _producer_formal_deps_file(tmp_path)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                "from train_survivors_deployable_policy import _load_formal_deps; "
                "assert not _load_formal_deps("
                "Path(sys.argv[1]), "
                "required_calibration_descriptor_hash=sys.argv[2])"
                ".perception_profile.development_only"
            ),
            str(path),
            descriptor_hash,
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_curriculum_corruption_scale_applied_in_train_step() -> None:
    """train_step が curriculum stage の corruption_scale を corrupt_fn へ渡す。
    clean stage では scale=0.0 で corrupt_fn が呼ばれず、full stage では scale=1.0 で呼ばれる。
    """
    recorded_scales: list[float] = []

    def track_corrupt(obs: np.ndarray, scale: float) -> np.ndarray:
        recorded_scales.append(scale)
        return obs

    config = CurriculumConfig(stage_start_updates=(0, 1, 2, 3), dagger_add_updates=(1, 2))
    model = DeployableCombatPolicy(SCHEMA.dim * 3, 2, hidden_dim=4)
    trainer = DeployablePolicyTrainer(model, curriculum_config=config, corrupt_fn=track_corrupt)

    # 4 steps: clean(0.0) → light(1/3) → measured(2/3) → full(1.0)
    # clean step では scale=0.0 なので corrupt_fn は呼ばれない
    for _ in range(4):
        trainer.train_step(_dataset())

    assert len(recorded_scales) == 3  # clean 以外の 3 stage で呼ばれる
    assert recorded_scales[0] == pytest.approx(1 / 3)   # light
    assert recorded_scales[1] == pytest.approx(2 / 3)   # measured
    assert recorded_scales[2] == pytest.approx(1.0)     # full


def test_dagger_datasets_mixed_in_compute_loss() -> None:
    """compute_loss が dagger_datasets をバッチへ連結して loss を変化させる。
    main dataset のみと DAgger 混合で total loss が変化することを確認する。
    """
    model = DeployableCombatPolicy(SCHEMA.dim * 3, 2, hidden_dim=4)
    trainer = DeployablePolicyTrainer(model)
    dataset = _dataset()
    # DAgger shard: teacher logits を main dataset と意図的に逆転させて loss を変化させる
    dagger_logits = np.array([[[[-0.9, 0.9], [-0.9, 0.9], [-0.9, 0.9], [0., 0.]]]], np.float32).squeeze(0)
    dagger = CombatDistillationDataset(
        np.array(dataset.observations), dagger_logits, np.array(dataset.teacher_values),
        np.array(dataset.valid_mask), np.array(dataset.burn_in_mask),
        np.array(dataset.episode_reset_mask), ("ep-d",), ("train",),
        SCHEMA.schema_hash, 1,
        ("hud_inventory", "screen_world_observed", "temporal_inferred", "constant"),
    )
    loss_no_dagger = trainer.compute_loss(dataset)
    loss_with_dagger = trainer.compute_loss(dataset, dagger_datasets=[dagger])
    assert not th.isclose(loss_no_dagger["total"], loss_with_dagger["total"])


def test_formal_resume_requires_current_dependencies(tmp_path: Path) -> None:
    """formal trainer が formal_dependencies=None で formal checkpoint を load_checkpoint すると
    validate_formal_dependencies が ValueError を送出して resume を拒否する。
    """
    formal_deps = _make_formal_deps()
    model = DeployableCombatPolicy(SCHEMA.dim * 3, 2, hidden_dim=4)
    saver = DeployablePolicyTrainer(model, formal_mode=True, formal_dependencies=formal_deps)
    # train_step を経由せず直接 save して formal checkpoint を作成する
    saver.training_steps = 0
    checkpoint = tmp_path / "formal.pt"
    saver.save_checkpoint(checkpoint)
    # formal_dependencies=None の formal trainer は resume 時に ValueError
    resuming = DeployablePolicyTrainer(
        DeployableCombatPolicy(SCHEMA.dim * 3, 2, hidden_dim=4),
        formal_mode=True, formal_dependencies=None,
    )
    with pytest.raises(ValueError):
        resuming.load_checkpoint(checkpoint)


def test_dagger_release_gate_rejects_unobservable_and_teacher_actions() -> None:
    """compute_loss が DAgger shard に対しても release gate を適用する。
    unobservable source class や teacher_actions を含む DAgger dataset は ValueError になる。
    """
    model = DeployableCombatPolicy(SCHEMA.dim * 3, 2, hidden_dim=4)
    trainer = DeployablePolicyTrainer(model)
    dataset = _dataset()
    # unobservable source class を含む DAgger dataset
    with pytest.raises(ValueError, match="unobservable"):
        bad_dagger = CombatDistillationDataset(
            np.array(dataset.observations), np.array(dataset.action_logits),
            np.array(dataset.teacher_values), np.array(dataset.valid_mask),
            np.array(dataset.burn_in_mask), np.array(dataset.episode_reset_mask),
            ("ep-bad",), ("train",),
            SCHEMA.schema_hash, 1,
            # unobservable が含まれるため release gate で拒否される
            ("unobservable",),
        )
        trainer.compute_loss(dataset, dagger_datasets=[bad_dagger])


def test_formal_resume_does_not_mutate_state_on_failure(tmp_path: Path) -> None:
    """load_checkpoint が formal gate で失敗した場合、model/optimizer/training_steps を変更しない。
    失敗後のパラメータが元の初期値と一致することを確認する。
    """
    formal_deps = _make_formal_deps()
    model = DeployableCombatPolicy(SCHEMA.dim * 3, 2, hidden_dim=4)
    saver = DeployablePolicyTrainer(model, formal_mode=True, formal_dependencies=formal_deps)
    saver.training_steps = 0
    checkpoint = tmp_path / "formal.pt"
    saver.save_checkpoint(checkpoint)

    # 元の model パラメータを記録
    fresh_model = DeployableCombatPolicy(SCHEMA.dim * 3, 2, hidden_dim=4)
    original_params = {k: v.detach().clone() for k, v in fresh_model.state_dict().items()}
    resuming = DeployablePolicyTrainer(
        fresh_model, formal_mode=True, formal_dependencies=None,
    )
    assert resuming.training_steps == 0

    with pytest.raises(ValueError):
        resuming.load_checkpoint(checkpoint)

    # 失敗後も training_steps が 0 のまま（state 変更なし）
    assert resuming.training_steps == 0
    for key, original in original_params.items():
        th.testing.assert_close(fresh_model.state_dict()[key], original)


def test_load_formal_deps_cli_reads_all_required_fields(tmp_path: Path) -> None:
    """CLI _load_formal_deps が current_gating_producer_hashes を含む valid JSON から
    FormalDependencies を構築し、キー欠落は ValueError で拒否する。
    """
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from train_survivors_deployable_policy import _load_formal_deps

    formal_deps = _make_formal_deps()
    digits = "abcdef0123456789"
    hashes = {name: digits[index % len(digits)] * 64 for index, name in enumerate(GATING_KEYS)}
    verdict = FidelityVerdict(
        "integration",
        {
            "target_profile_hash": "1" * 64, "target_build_attestation_hash": "2" * 64,
            "report_scope": "exact_target", "producer_allowlist_version": "fidelity_producer_paths.v1",
            "producer_manifest_hash": "3" * 64,
            "resolved_producers": {name: [{"path": name, "sha256": digest}] for name, digest in hashes.items()},
        },
        (FidelityMetric("deploy_obs_visibility", 1., "ratio", True, None, True),), (),
        {
            "git_commit": "abc", "workspace_dirty_summary": "clean", "audit_tool_version": "test",
            "dependency_versions": {}, "operator": "pytest", "timestamp": "2026-08-09T00:00:00Z",
        }, hashes,
    )
    # CLI テストでは development_only=True のプロファイルを使う。
    # formal profile (development_only=False) は wire 経由でのロードが禁止されており、
    # ArtifactStore 検証経路のみで取得可能。validate() のテストは別テストで行う。
    cal_hash = canonical_hash({"synthetic_session_id": "cal-1"})
    profile = FittedPerceptionErrorProfile(
        **PerceptionErrorProfile(calibration_session_ids=["cal-1"]).to_wire(),
        calibration_session_hashes={"cal-1": cal_hash},
        field_sample_counts={"hp_ratio": 2},
        fit_code_hash="a" * 64,
        development_only=True,
    )
    valid_data = {
        "fidelity_verdict": verdict.to_wire(),
        "perception_profile": profile.to_artifact_wire(),
        "required_perception_profile_hash": profile.profile_hash,
        "current_gating_producer_hashes": hashes,
        "profile_source": "measured",
    }
    deps_path = tmp_path / "formal_deps.json"
    deps_path.write_text(json.dumps(valid_data), encoding="utf-8")
    result = _load_formal_deps(deps_path)
    assert isinstance(result, FormalDependencies)
    # current_gating_producer_hashes が欠落すると ValueError
    for missing_key in ("current_gating_producer_hashes", "profile_source"):
        incomplete = {k: v for k, v in valid_data.items() if k != missing_key}
        incomplete_path = tmp_path / f"missing_{missing_key}.json"
        incomplete_path.write_text(json.dumps(incomplete), encoding="utf-8")
        with pytest.raises(ValueError):
            _load_formal_deps(incomplete_path)
    # 未知 field を含む JSON も ValueError
    extra_json = {**valid_data, "extra_unexpected_field": "value"}
    extra_path = tmp_path / "extra_field.json"
    extra_path.write_text(json.dumps(extra_json), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        _load_formal_deps(extra_path)

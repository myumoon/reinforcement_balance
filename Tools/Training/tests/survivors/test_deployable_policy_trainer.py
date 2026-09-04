"""Deployable combat policy の loss・curriculum・resume・formal gate を検証する。
UE5 を使わず、固定 sequence と最小 state holder で step-0 sealing と完全再開を確認する。
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
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
from survivors.perception_error_fit import (
    FittedPerceptionErrorProfile, _FORMAL_FACTORY_TOKEN, _current_fit_code_hash,
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
    """development_only=False の最小 FittedPerceptionErrorProfile。

    FormalDependencies.validate() の development_only 検証を通過するための
    最小フィクスチャ。formal runner pipeline と同じ factory token を使う。
    """
    cal_hash = canonical_hash({"synthetic_session_id": session_id})
    return FittedPerceptionErrorProfile(
        calibration_session_ids=[session_id],
        final_e2e_session_ids=[],
        calibration_session_hashes={session_id: cal_hash},
        field_sample_counts={"hp_ratio": 2},
        fit_code_hash=_current_fit_code_hash(),
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
    profile = PerceptionErrorProfile(calibration_session_ids=["cal-1"])
    valid_data = {
        "fidelity_verdict": verdict.to_wire(),
        "perception_profile": profile.to_wire(),
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

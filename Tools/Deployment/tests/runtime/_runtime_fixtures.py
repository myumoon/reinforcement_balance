"""runtime test 共通 fixture: golden combat package と formal artifact descriptor 生成。

正式 Artifact がなくても loader / session / scheduler の全テストを実行できるよう、
実物と同じ形の SB3 RecurrentPPO package・VecNormalize・artifact descriptor を
一時ディレクトリ上に組み立てる。生成物はすべて development 用途であり、
formal parent には使えない。
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    ArtifactRef,
    artifact_uri,
)
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes, sha256_hex
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.target_action import ActionSemantics, TargetProfileRef

from survivors.runtime.artifact_bundle import (
    COMBAT_PACKAGE_FILES,
    COMBAT_PACKAGE_SCHEMA_VERSION,
    RecurrentCombatPolicy,
)

# deploy schema の values/validity/age 3 平面ぶんが combat observation 次元になる。
DEPLOY_OBS_DIM: int = 3 * DeployObsSchema.default_v1().dim
DEFAULT_ACTION_DIM = 9
DEFAULT_HIDDEN = 8
DEFAULT_LAYERS = 1

# 04-10 final verdict が subject に持つ exact hash field 集合 (artifact_dag と同一契約)。
PERCEPTION_SUBJECT_FIELDS = (
    "parser_artifact_hash", "detector_artifact_hash", "model_hash", "build_hash",
    "assembler_schema_hash", "ui_presentation_schema_hash",
    "ui_presentation_golden_fixture_hash", "config_hash", "capture_dataset_hash",
    "calibration_profile_hash", "threshold_hash", "atlas_vocabulary_hash",
    "assembler_impl_hash", "roi_resolver_input_hash", "benchmark_fit_code_hash",
    "lineage_seal_hash",
)


def _hash_of(label: str) -> str:
    """ラベルから決定的な擬似 SHA-256 を作る。

    テスト間で安定した identity が必要な箇所に使う。
    """
    return sha256_hex(label.encode("utf-8"))


def _tiny_vec_env(observation_dim: int, action_dim: int):
    """RecurrentPPO 構築用の最小 vectorized env を返す。

    学習はせず、policy の observation/action space を確定するためだけに使う。
    """
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class _Tiny(gym.Env):
        """観測を単調に変化させるだけの決定的ダミー環境。"""

        def __init__(self) -> None:
            self.observation_space = spaces.Box(
                -10.0, 10.0, (observation_dim,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(action_dim)
            self._step = 0

        def reset(self, *, seed: int | None = None, options: Any = None):
            self._step = 0
            return np.zeros(observation_dim, dtype=np.float32), {}

        def step(self, action):
            self._step += 1
            obs = np.full(observation_dim, self._step * 0.01, dtype=np.float32)
            return obs, 0.0, self._step >= 8, False, {}

    return VecNormalize(DummyVecEnv([_Tiny]), norm_obs=True, norm_reward=True)


def build_golden_combat_policy(
    *,
    observation_dim: int = DEPLOY_OBS_DIM,
    action_dim: int = DEFAULT_ACTION_DIM,
    hidden_size: int = DEFAULT_HIDDEN,
    n_lstm_layers: int = DEFAULT_LAYERS,
    seed: int = 0,
) -> RecurrentCombatPolicy:
    """golden fixture 用の RecurrentPPO + VecNormalize ペアを構築する。

    重みは初期値のままでよい。action/state parity と gate の検証が目的であり、
    性能は問わない。
    """
    from sb3_contrib import RecurrentPPO

    venv = _tiny_vec_env(observation_dim, action_dim)
    # 統計を非自明にして、正規化が実際に効いていることを検出できるようにする。
    venv.reset()
    for _ in range(6):
        venv.step(np.array([0]))
    model = RecurrentPPO(
        "MlpLstmPolicy",
        venv,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_kwargs=dict(lstm_hidden_size=hidden_size, n_lstm_layers=n_lstm_layers),
        device="cpu",
        seed=seed,
        verbose=0,
    )
    model.policy.set_training_mode(False)
    venv.training = False
    venv.norm_reward = False
    return RecurrentCombatPolicy(
        model=model,
        vecnormalize=venv,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_lstm_layers=n_lstm_layers,
        lstm_hidden_size=hidden_size,
    )


def write_combat_package(
    package_dir: Path,
    policy: RecurrentCombatPolicy,
    *,
    target_profile: TargetProfileRef,
    target_capability_hash: str,
    choice_capability_hash: str,
    action_semantics: ActionSemantics | None = None,
    development_only: bool = False,
    formal_student_eligible: bool = True,
    decision_hz: int = 15,
    deploy_schema_hash: str | None = None,
    action_semantics_hash: str | None = None,
) -> dict[str, Any]:
    """combat package directory を実ファイルとして書き出し manifest を返す。

    policy.zip / vecnormalize.pkl / manifest.json の 3 点セットを、実際の
    content hash を計算したうえで生成する。異常系テストは戻り値の manifest を
    書き換えて再保存する。
    """
    root = Path(package_dir)
    root.mkdir(parents=True, exist_ok=True)
    semantics = action_semantics or ActionSemantics.default_v1()

    policy.model.save(root / "policy.zip")
    with (root / "vecnormalize.pkl").open("wb") as handle:
        pickle.dump(policy.vecnormalize, handle)

    manifest = {
        "schema_version": COMBAT_PACKAGE_SCHEMA_VERSION,
        "policy_sha256": sha256_hex((root / "policy.zip").read_bytes()),
        "vecnormalize_sha256": sha256_hex((root / "vecnormalize.pkl").read_bytes()),
        "deploy_schema_hash": deploy_schema_hash or DeployObsSchema.default_v1().schema_hash,
        "action_semantics_hash": action_semantics_hash or semantics.semantics_hash,
        "target_profile_ref_hash": target_profile.ref_hash,
        "target_capability_hash": target_capability_hash,
        "model_config": {
            "observation_dim": policy.observation_dim,
            "action_dim": policy.action_dim,
            "n_lstm_layers": policy.n_lstm_layers,
            "lstm_hidden_size": policy.lstm_hidden_size,
        },
        "runtime_profile": {
            "os_build": "Windows 11 26200",
            "gpu_name": "development-cpu",
            "driver_version": "0.0.0",
            "cuda_version": "none",
            "capture_backend": "dxgi",
            "decision_hz": decision_hz,
            "key_lease_duration_ms": 120,
            "choice_capability_hash": choice_capability_hash,
        },
        "development_only": development_only,
        "formal_student_eligible": formal_student_eligible,
        "formal_dependency_identities": {
            "fidelity_verdict": _hash_of("fidelity_verdict"),
            "perception_profile": _hash_of("perception_profile"),
        },
        "files": list(COMBAT_PACKAGE_FILES),
    }
    rewrite_combat_manifest(root, manifest)
    return manifest


def rewrite_combat_manifest(package_dir: Path, manifest: dict[str, Any]) -> None:
    """manifest.json を canonical JSON で書き直す。

    異常系テストが manifest を一部だけ書き換えて保存し直すために使う。
    """
    (Path(package_dir) / "manifest.json").write_bytes(canonical_json_bytes(manifest))


def perception_subject_hashes() -> dict[str, str]:
    """04-10 final verdict の subject hash 一式を返す。"""
    return {field: _hash_of(field) for field in PERCEPTION_SUBJECT_FIELDS}


def store_file_ref(store: ArtifactStore, logical_id: str, payload: bytes) -> ArtifactRef:
    """artifact store へ bytes を登録し ArtifactRef を返す。"""
    return store.put_bytes(
        logical_id=logical_id, data=payload, media_type="application/octet-stream"
    )


def build_formal_descriptors(
    *,
    store: ArtifactStore,
    combat_manifest: dict[str, Any],
    item_selector_manifest: dict[str, Any],
    target_capability_hash: str,
    choice_capability_hash: str,
    action_semantics: ActionSemantics | None = None,
    decision_hz: int = 15,
    verdict_passed: bool = True,
    verdict_development_only: bool = False,
    runtime_subject_hashes: dict[str, str] | None = None,
) -> list[ArtifactDescriptor]:
    """formal runtime DAG を満たす descriptor 一式を作る。

    runtime_bundle と 3 つの immutable parent (combat/item selector release、
    perception final verdict) を、artifact store 上の実 object と紐付けて返す。
    """
    semantics = action_semantics or ActionSemantics.default_v1()
    subject_hashes = perception_subject_hashes()
    runtime_subjects = runtime_subject_hashes or dict(subject_hashes)

    # DAG validator は全 ancestor descriptor を要求するため、root から順に構築する。
    source = ArtifactDescriptor(
        logical_id="phase5-source",
        node_kind="source_descriptor",
        producer_id="phase5",
        producer_version="1.0.0",
    )
    calibration = ArtifactDescriptor(
        logical_id="d04-perception-calibration",
        node_kind="perception_calibration_profile",
        producer_id="perception-calibration",
        producer_version="1.0.0",
        parents=(source.node_ref(),),
    )
    verdict = ArtifactDescriptor(
        logical_id="d04-perception-final",
        node_kind="perception_final_verdict",
        producer_id="perception-benchmark",
        producer_version="1.0.0",
        identity_metadata={
            "passed": verdict_passed,
            "development_only": verdict_development_only,
            "subject_hashes": subject_hashes,
        },
        parents=(calibration.node_ref(),),
        files=(store_file_ref(store, "d04-perception-final/verdict.json", b"verdict"),),
    )
    teacher_verdict = ArtifactDescriptor(
        logical_id="d01-teacher-verdict",
        node_kind="teacher_validation_verdict",
        producer_id="teacher",
        producer_version="1.0.0",
        parents=(source.node_ref(),),
    )
    dataset = ArtifactDescriptor(
        logical_id="d02-choice-dataset",
        node_kind="choice_dataset_release",
        producer_id="choice-trace",
        producer_version="1.0.0",
        parents=(teacher_verdict.node_ref(),),
    )
    combat_release = ArtifactDescriptor(
        logical_id="d03-deploy-student-release",
        node_kind="combat_student_release",
        producer_id="deployable-policy-trainer",
        producer_version="1.0.0",
        identity_metadata={"package_manifest_hash": canonical_hash(combat_manifest)},
        parents=(dataset.node_ref(),),
        files=(store_file_ref(store, "d03-deploy-student-release/policy.zip", b"policy"),),
    )
    selector_release = ArtifactDescriptor(
        logical_id="d02-item-selector-release",
        node_kind="item_selector_release",
        producer_id="item-selector-trainer",
        producer_version="1.0.0",
        identity_metadata={"package_manifest_hash": canonical_hash(item_selector_manifest)},
        parents=(dataset.node_ref(),),
        files=(store_file_ref(store, "d02-item-selector-release/model.onnx", b"onnx"),),
    )
    runtime = ArtifactDescriptor(
        logical_id="d05-runtime-bundle",
        node_kind="runtime_bundle",
        producer_id="recurrent-agent-runtime",
        producer_version="1.0.0",
        identity_metadata={
            "perception_subject_hashes": runtime_subjects,
            "target_capability_hash": target_capability_hash,
            "choice_capability_hash": choice_capability_hash,
            "action_semantics_hash": semantics.semantics_hash,
            "decision_hz": decision_hz,
        },
        parents=(
            combat_release.node_ref(),
            selector_release.node_ref(),
            verdict.node_ref(),
        ),
        files=(store_file_ref(store, "d05-runtime-bundle/startup.json", b"startup"),),
    )
    return [
        source, calibration, verdict, teacher_verdict, dataset,
        combat_release, selector_release, runtime,
    ]


def default_target_profile() -> TargetProfileRef:
    """テスト用の固定 target hardware profile を返す。"""
    return TargetProfileRef(
        build_id="vs-1.11.106",
        canonical_save_hash=_hash_of("canonical-save"),
        hardware_profile_id="dev-win64",
    )


def write_item_selector_package(
    package_dir: Path,
    *,
    target_capability_hash: str,
    nmax: int = 3,
    context_dim: int = 4,
    candidate_dim: int = 3,
    feature_schema: str = "context_only_v1",
    temperature: float = 1.0,
    student_output_temperature: float = 1.0,
    confidence_threshold: float = 0.0,
    item_vocabulary: list[str] | None = None,
) -> dict[str, Any]:
    """共有契約を満たす ItemSelector package を実ファイルとして書き出す。

    model.pt (TorchScript) / model.onnx / ui_policy_config.json / manifest.json を
    生成し、manifest の全 hash を実 byte から計算する。正式 artifact がなくても
    ONNX adapter と confidence gate を end-to-end で検証できる。
    """
    import torch as th

    from reinbalance_survivors_contracts.item_selector_package import (
        expected_onnx_tensor_manifest,
        installed_policy_impl_hash,
        policy_schema_hash,
    )
    from reinbalance_survivors_contracts.ui_policy import NonModelUiPolicyConfigV1

    root = Path(package_dir)
    root.mkdir(parents=True, exist_ok=True)
    vocabulary = sorted(item_vocabulary or ["knife", "wand", "whip"])

    class _TinySelector(th.nn.Module):
        """context と candidate の内積で候補を採点する最小 selector。"""

        def __init__(self) -> None:
            super().__init__()
            self.context_proj = th.nn.Linear(context_dim, 8)
            self.candidate_proj = th.nn.Linear(candidate_dim, 8)

        def forward(
            self,
            context_features: th.Tensor,
            candidate_features: th.Tensor,
            candidate_mask: th.Tensor,
        ) -> th.Tensor:
            """masked 候補を大きな負値へ潰した logits を返す。"""
            context = self.context_proj(context_features).unsqueeze(1)
            candidates = self.candidate_proj(candidate_features)
            logits = (context * candidates).sum(-1)
            return th.where(candidate_mask, logits, th.full_like(logits, -1.0e4))

    th.manual_seed(0)
    model = _TinySelector().eval()
    scripted = th.jit.script(model)
    scripted.save(str(root / "model.pt"))

    example = (
        th.zeros(1, context_dim, dtype=th.float32),
        th.zeros(1, nmax, candidate_dim, dtype=th.float32),
        th.ones(1, nmax, dtype=th.bool),
    )
    th.onnx.export(
        model,
        example,
        str(root / "model.onnx"),
        input_names=["context_features", "candidate_features", "candidate_mask"],
        output_names=["logits"],
        dynamic_axes={
            "context_features": {0: "batch"},
            "candidate_features": {0: "batch", 1: "candidates"},
            "candidate_mask": {0: "batch", 1: "candidates"},
            "logits": {0: "batch", 1: "candidates"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    installed = NonModelUiPolicyConfigV1.load_default()
    (root / "ui_policy_config.json").write_bytes(canonical_json_bytes(installed.to_wire()))

    file_hashes = {
        name: sha256_hex((root / name).read_bytes())
        for name in ("model.pt", "model.onnx", "ui_policy_config.json")
    }
    core: dict[str, Any] = {
        "schema_version": "survivors.item_selector_artifact.v1",
        "target_capability_hash": target_capability_hash,
        "nmax": nmax,
        "context_dim": context_dim,
        "candidate_dim": candidate_dim,
        "feature_schema": feature_schema,
        "vocabulary_hash": canonical_hash(vocabulary),
        "item_vocabulary": vocabulary,
        "temperature": temperature,
        "confidence_threshold": confidence_threshold,
        "student_output_temperature": student_output_temperature,
        "dataset_identity": _hash_of("item-selector-dataset"),
        "model_state_hash": file_hashes["model.pt"],
        "onnx_model_hash": file_hashes["model.onnx"],
        "files": file_hashes,
        "policy_schema_hash": policy_schema_hash(installed),
        "policy_config_hash": installed.config_hash,
        "policy_impl_hash": installed_policy_impl_hash(),
        "lineage": {
            "source_descriptor_identity": _hash_of("source-descriptor"),
            "teacher_verdict_identity": _hash_of("teacher-verdict"),
            "trace_dataset_identity": _hash_of("trace-dataset"),
            "model_training_run_id": "item-selector-fixture-run",
        },
        "dependency_versions": {
            "torch": "2.11.0", "onnx": "1.16.0",
            "onnxruntime": "1.18.1", "numpy": "1.26.4",
        },
    }
    inputs, outputs = expected_onnx_tensor_manifest(core)
    core["onnx_input_tensors"] = inputs
    core["onnx_output_tensors"] = outputs
    manifest = dict(core)
    manifest["artifact_identity"] = canonical_hash(core)
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest

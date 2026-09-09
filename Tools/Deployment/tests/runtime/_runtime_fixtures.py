"""runtime test 用の成果物ビルダー（combat package / ItemSelector package / trust registry）。

03-05 と 02-03 の wire 契約を、Tools/Training を一切 import せずに再現する。依存方向の
規則（docs/project_structure.md）は test にも等しく適用されるため、fixture は契約の
「形」だけを自前で組み立てる。

やさしい説明:
    テストで使う「本物そっくりの成果物」を作る道具箱です。学習側のコードを呼び出して
    作ると、実装が同じかどうかを確かめたことになりません。そこで、仕様書どおりの形を
    テスト側で独立に組み立て、それを実行側のローダーが読めるかを確認します。
    combat モデルも、学習側と同じ部品名（recurrent / actor / value）でここに書き起こし
    ます。これにより「別々に書いた同じ構造」が噛み合うことを検証できます。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch as th
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor, ArtifactRef
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.target_action import ActionSemantics, TargetProfileRef

from survivors.runtime.artifact_bundle import (
    COMBAT_PACKAGE_FILES,
    COMBAT_PACKAGE_SCHEMA_VERSION,
    REQUIRED_ACTION_DIM,
    REQUIRED_DECISION_HZ,
    TRUST_REGISTRY_SCHEMA_VERSION,
    TRUST_REGISTRY_SIGNATURE_SUFFIX,
    HostRuntimeProfile,
)

# combat model の既定次元。observation_dim は deploy schema の value/validity/age 三面。
DEFAULT_HIDDEN_DIM = 8
DEFAULT_OBSERVATION_DIM = 3 * DeployObsSchema.default_v1().dim

# 04-10 final verdict が subject として固定する exact hash field 群
# （artifact_dag._FORMAL_PERCEPTION_SUBJECT_HASH_FIELDS と同一契約）。
PERCEPTION_SUBJECT_FIELDS = (
    "parser_artifact_hash",
    "detector_artifact_hash",
    "model_hash",
    "build_hash",
    "assembler_schema_hash",
    "ui_presentation_schema_hash",
    "ui_presentation_golden_fixture_hash",
    "config_hash",
    "capture_dataset_hash",
    "calibration_profile_hash",
    "threshold_hash",
    "atlas_vocabulary_hash",
    "assembler_impl_hash",
    "roi_resolver_input_hash",
    "benchmark_fit_code_hash",
    "lineage_seal_hash",
)

# テスト用の固定 Ed25519 署名鍵。決定的にするため生成せず固定 seed から作る。
TEST_SIGNING_KEY_SEED = bytes(range(32))
# 「別チャネルの発行者ではない鍵」を表す第二の鍵。署名鍵取り違えの検証に使う。
UNTRUSTED_SIGNING_KEY_SEED = bytes(range(100, 132))


def hash_of(label: str) -> str:
    """label から決定的な SHA-256 hex を作る。

    やさしい説明: テストのなかで「毎回同じだが項目ごとに違う指紋」が欲しいときに使う
    小道具です。実物のハッシュではなく、識別できれば十分な場面で使います。
    """
    return sha256_hex(label.encode("utf-8"))


class TrainingShapedCombatPolicy(th.nn.Module):
    """Training の `DeployableCombatPolicy` と同じ submodule 名・形を持つ fixture model。

    やさしい説明: 学習側のモデルを import せずに、同じ部品構成（GRUCell の
    `recurrent`、線形層の `actor` と `value`）をテスト側で書き起こしたものです。
    ここで作った重みが実行側のモデルへ `strict=True` で読み込めることが、契約が
    一致している何よりの証拠になります。VecNormalize は 03-05 契約に存在しないため
    ここにもありません。
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        """GRUCell と 2 つの head を、学習側と同じ順序・同じ名前で作る。"""
        super().__init__()
        self.recurrent = th.nn.GRUCell(observation_dim, hidden_dim)
        self.actor = th.nn.Linear(hidden_dim, action_dim)
        self.value = th.nn.Linear(hidden_dim, 1)


def build_combat_model(
    *,
    observation_dim: int = DEFAULT_OBSERVATION_DIM,
    action_dim: int = REQUIRED_ACTION_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    seed: int = 0,
) -> tuple[TrainingShapedCombatPolicy, dict[str, int]]:
    """決定的な重みを持つ fixture combat model と、その model_config を返す。

    やさしい説明: 毎回同じ重みになるよう乱数の種を固定します。こうしておくと、
    「保存して読み直しても出力が同じ」というテストが安定します。
    """
    th.manual_seed(seed)
    model = TrainingShapedCombatPolicy(observation_dim, action_dim, hidden_dim).eval()
    model_config = {
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "hidden_dim": hidden_dim,
    }
    return model, model_config


def write_combat_package(
    package_dir: Path,
    model: TrainingShapedCombatPolicy,
    model_config: dict[str, int],
    *,
    deploy_schema_hash: str | None = None,
    development_only: bool = False,
    formal_student_eligible: bool = True,
) -> dict[str, Any]:
    """03-05 `package_deployable_policy` と同一形状の package を書き出す。

    やさしい説明: `manifest.json` と `model.pt` の 2 ファイルだけを持つ箱を作ります。
    `model.pt` の中身は `{model_config, model_state_dict}` の 2 キーで、manifest には
    ちょうど 9 個のキーを書きます。これは学習側が実際に出力する形そのままです。
    """
    root = Path(package_dir)
    root.mkdir(parents=True, exist_ok=True)

    th.save(
        {"model_config": dict(model_config), "model_state_dict": model.state_dict()},
        root / "model.pt",
    )
    schema = DeployObsSchema.default_v1()
    manifest = {
        "schema_version": COMBAT_PACKAGE_SCHEMA_VERSION,
        "checkpoint_sha256": hash_of("deployable-checkpoint"),
        "model_sha256": sha256_hex((root / "model.pt").read_bytes()),
        "deploy_schema_hash": deploy_schema_hash or schema.schema_hash,
        "model_config": dict(model_config),
        "development_only": development_only,
        "formal_student_eligible": formal_student_eligible,
        "formal_dependency_identities": {
            "fidelity_verdict": hash_of("fidelity-verdict"),
            "perception_profile": hash_of("perception-profile"),
        },
        "files": list(COMBAT_PACKAGE_FILES),
    }
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest


def rewrite_combat_manifest(package_dir: Path, manifest: dict[str, Any]) -> None:
    """既存 combat package の manifest だけを差し替える。

    やさしい説明: 異常系テストで「説明書だけ書き換えられた箱」を作るための道具です。
    """
    (Path(package_dir) / "manifest.json").write_bytes(canonical_json_bytes(manifest))


def default_target_profile() -> TargetProfileRef:
    """テスト用の固定 target identity 参照を返す。"""
    return TargetProfileRef(
        build_id="vs-1.11.106",
        canonical_save_hash=hash_of("canonical-save"),
        hardware_profile_id="dev-win64",
    )


def default_host_profile(**overrides: Any) -> HostRuntimeProfile:
    """テスト用の実 hardware / capture 値を返す。override で 1 項目だけ変えられる。

    やさしい説明: 「このPCで動かしています」という実際の値です。テストでは、この
    どれか 1 つをわざと変えて、起動が止まることを確かめます。
    """
    values: dict[str, Any] = {
        "os_build": "Windows 11 Pro 10.0.26200",
        "gpu_name": "NVIDIA GeForce RTX 4070",
        "driver_version": "561.09",
        "cuda_version": "12.6",
        "capture_backend": "dxcam",
        "decision_hz": REQUIRED_DECISION_HZ,
        "key_lease_duration_ms": 250,
    }
    values.update(overrides)
    return HostRuntimeProfile(**values)


def perception_subject_hashes() -> dict[str, str]:
    """04-10 final verdict の subject hash 一式を返す。"""
    return {name: hash_of(name) for name in PERCEPTION_SUBJECT_FIELDS}


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
    decision_hz: int = REQUIRED_DECISION_HZ,
    verdict_passed: bool = True,
    verdict_development_only: bool = False,
    runtime_subject_hashes: dict[str, str] | None = None,
) -> list[ArtifactDescriptor]:
    """formal runtime DAG を満たす descriptor 一式を作る。

    やさしい説明: 成果物どうしの「親子関係」を、根っこから順に組み立てます。DAG の
    検証器は途中の先祖もすべて要求するため、source から runtime_bundle まで全部
    そろえます。
    """
    semantics = action_semantics or ActionSemantics.default_v1()
    subject_hashes = perception_subject_hashes()
    runtime_subjects = runtime_subject_hashes or dict(subject_hashes)

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
        files=(store_file_ref(store, "d03-deploy-student-release/model.pt", b"model"),),
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
        source,
        calibration,
        verdict,
        teacher_verdict,
        dataset,
        combat_release,
        selector_release,
        runtime,
    ]


def signing_key(seed: bytes = TEST_SIGNING_KEY_SEED) -> Ed25519PrivateKey:
    """決定的な test 用 Ed25519 秘密鍵を返す。

    やさしい説明: 「正規リリース一覧」に署名する発行者役の鍵です。テストの中だけで
    使う鍵であり、本番用の固定鍵とは別物です。
    """
    return Ed25519PrivateKey.from_private_bytes(seed)


def public_key_hex(seed: bytes = TEST_SIGNING_KEY_SEED) -> str:
    """署名鍵に対応する Ed25519 公開鍵の hex を返す。"""
    from cryptography.hazmat.primitives import serialization

    raw = signing_key(seed).public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return raw.hex()


def release_entry(
    *,
    descriptors: list[ArtifactDescriptor],
    combat_manifest: dict[str, Any],
    item_selector_manifest: dict[str, Any],
    target_profile: TargetProfileRef,
    host_profile: HostRuntimeProfile,
    target_capability_hash: str,
    choice_capability_hash: str,
    action_semantics: ActionSemantics | None = None,
) -> dict[str, Any]:
    """descriptor 一式から trusted release registry の 1 entry を組み立てる。

    やさしい説明: 「このバンドルを本番で動かしてよい」という許可証の中身を作ります。
    実際の運用では別チャネルで配布されるものですが、テストではここで組み立てます。
    """
    semantics = action_semantics or ActionSemantics.default_v1()
    by_kind = {descriptor.node_kind: descriptor for descriptor in descriptors}
    return {
        "runtime_bundle_identity_hash": by_kind["runtime_bundle"].identity_hash,
        "combat_student_release_identity_hash": by_kind["combat_student_release"].identity_hash,
        "item_selector_release_identity_hash": by_kind["item_selector_release"].identity_hash,
        "perception_final_verdict_identity_hash": by_kind[
            "perception_final_verdict"
        ].identity_hash,
        "combat_package_manifest_hash": canonical_hash(combat_manifest),
        "item_selector_manifest_hash": canonical_hash(item_selector_manifest),
        "deploy_schema_hash": DeployObsSchema.default_v1().schema_hash,
        "action_semantics_hash": semantics.semantics_hash,
        "target_profile_ref_hash": target_profile.ref_hash,
        "target_capability_hash": target_capability_hash,
        "choice_capability_hash": choice_capability_hash,
        "host_profile": host_profile.to_wire(),
    }


def write_trust_registry(
    registry_path: Path,
    entries: list[dict[str, Any]],
    *,
    registry_id: str = "survivors-release-registry-test",
    schema_version: str = TRUST_REGISTRY_SCHEMA_VERSION,
    seed: bytes = TEST_SIGNING_KEY_SEED,
    tamper: bool = False,
) -> Path:
    """canonical JSON の registry と detached Ed25519 署名を書き出す。

    やさしい説明: 一覧ファイルと、その署名ファイル（`.sig`）を並べて置きます。署名は
    ファイルの中身そのものに対して行うので、あとから 1 文字でも書き換えると検証に
    失敗します。`tamper=True` にすると、署名したあとで中身を書き換えた状態を作れます。
    """
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": schema_version,
        "registry_id": registry_id,
        "releases": entries,
    }
    payload = canonical_json_bytes(document)
    signature = signing_key(seed).sign(payload)
    path.write_bytes(payload)
    path.with_name(path.name + TRUST_REGISTRY_SIGNATURE_SUFFIX).write_text(
        signature.hex(), encoding="utf-8"
    )
    if tamper:
        # 署名後に registry_id だけを差し替える（署名は古い内容のまま）。
        document["registry_id"] = registry_id + "-tampered"
        path.write_bytes(canonical_json_bytes(document))
    return path


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

    やさしい説明: `model.pt`（TorchScript）/ `model.onnx` / `ui_policy_config.json` /
    `manifest.json` を作り、manifest のハッシュはすべて実際のファイルの中身から
    計算します。正式な成果物が無くても、ONNX アダプタを端から端まで検証できます。
    """
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
        "dataset_identity": hash_of("item-selector-dataset"),
        "model_state_hash": file_hashes["model.pt"],
        "onnx_model_hash": file_hashes["model.onnx"],
        "files": file_hashes,
        "policy_schema_hash": policy_schema_hash(installed),
        "policy_config_hash": installed.config_hash,
        "policy_impl_hash": installed_policy_impl_hash(),
        "lineage": {
            "source_descriptor_identity": hash_of("source-descriptor"),
            "teacher_verdict_identity": hash_of("teacher-verdict"),
            "trace_dataset_identity": hash_of("trace-dataset"),
            "model_training_run_id": "item-selector-fixture-run",
        },
        "dependency_versions": {
            "torch": "2.11.0",
            "onnx": "1.16.0",
            "onnxruntime": "1.18.1",
            "numpy": "1.26.4",
        },
    }
    inputs, outputs = expected_onnx_tensor_manifest(core)
    core["onnx_input_tensors"] = inputs
    core["onnx_output_tensors"] = outputs
    manifest = dict(core)
    manifest["artifact_identity"] = canonical_hash(core)
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return manifest

"""Runtime artifact bundle: recurrent combat package + ItemSelector の検証済みロード。

SB3 RecurrentPPO policy と deploy VecNormalize、ONNX ItemSelector を、artifact store
上の immutable parent と exact subject hash を検証したうえでロードする単一の入口。
golden fixture (development_only=True, live_eligible=False) も同じ型で扱えるが、
`assert_live_eligible()` が正式起動を fail-closed で拒否する。OS input には触れない。

検証順序が本 module の要点である。信頼された identity を先に確定し、file content
hash を照合してから初めて model file を読む。これにより package 差し替えによる
任意コード実行を、読み込み前の段階で遮断する。
"""
from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reinbalance_survivors_contracts.artifact_dag import (
    ArtifactDagValidationError,
    validate_formal_runtime_dag,
)
from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    ArtifactRef,
)
from reinbalance_survivors_contracts.artifact_store import (
    ArtifactStore,
    ArtifactStoreError,
)
from reinbalance_survivors_contracts.canonical_json import canonical_hash, sha256_hex
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.target_action import ActionSemantics, TargetProfileRef
from reinbalance_survivors_contracts.ui_policy import NonModelUiPolicyConfigV1

from .item_selector_runtime import ItemSelectorRuntimeError, OnnxItemSelector

# combat package は SB3 RecurrentPPO policy と deploy VecNormalize を同梱する。
COMBAT_PACKAGE_SCHEMA_VERSION = "survivors.recurrent_combat_package.v1"

COMBAT_POLICY_FILENAME = "policy.zip"
COMBAT_VECNORMALIZE_FILENAME = "vecnormalize.pkl"
COMBAT_MANIFEST_FILENAME = "manifest.json"
COMBAT_PACKAGE_FILES: tuple[str, ...] = (
    COMBAT_MANIFEST_FILENAME,
    COMBAT_POLICY_FILENAME,
    COMBAT_VECNORMALIZE_FILENAME,
)

_COMBAT_MANIFEST_KEYS = frozenset({
    "schema_version",
    "policy_sha256",
    "vecnormalize_sha256",
    "deploy_schema_hash",
    "action_semantics_hash",
    "target_profile_ref_hash",
    "target_capability_hash",
    "model_config",
    "runtime_profile",
    "development_only",
    "formal_student_eligible",
    "formal_dependency_identities",
    "files",
})
_MODEL_CONFIG_KEYS = frozenset({
    "observation_dim", "action_dim", "n_lstm_layers", "lstm_hidden_size",
})
# plan 05-01 タスク2: OS/GPU/driver/CUDA/capture backend、decision Hz、key lease、choice capability。
_RUNTIME_PROFILE_KEYS = frozenset({
    "os_build", "gpu_name", "driver_version", "cuda_version", "capture_backend",
    "decision_hz", "key_lease_duration_ms", "choice_capability_hash",
})
_FORMAL_DEPENDENCY_KEYS = frozenset({"fidelity_verdict", "perception_profile"})

# 9 direction/idle 固定。2-action model 等を move decision に使わせない。
REQUIRED_ACTION_DIM = 9
# combat tick は 15 Hz 固定 (plan 05-01 / 00-03 action contract)。
REQUIRED_DECISION_HZ = 15

BUNDLE_DEVELOPMENT_SENTINEL = "golden_fixture"
BUNDLE_FORMAL_SENTINEL = "formal"

_RUNTIME_NODE_KIND = "runtime_bundle"
_COMBAT_RELEASE_NODE_KIND = "combat_student_release"
_ITEM_SELECTOR_RELEASE_NODE_KIND = "item_selector_release"
_PERCEPTION_VERDICT_NODE_KIND = "perception_final_verdict"


class BundleLoadError(ValueError):
    """artifact bundle の schema / hash / capability 検証失敗。

    起動前に artifact の取り違えや改変を拒否するための fail-closed 例外。
    このエラーが出た bundle は live 起動も shadow 起動もしてはならない。
    """


def _is_sha256(value: Any) -> bool:
    """小文字 64 桁の SHA-256 identity を確認する。"""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _require_sha256(value: Any, label: str) -> str:
    """SHA-256 でなければ BundleLoadError を送出する。"""
    if not _is_sha256(value):
        raise BundleLoadError(f"{label} must be lowercase SHA-256")
    return value


def _positive_int(value: Any, label: str) -> int:
    """正の int だけを受理する。bool や float を混入させない。"""
    if type(value) is not int or value <= 0:
        raise BundleLoadError(f"{label} must be a positive int")
    return value


def _resolve_package_file(root: Path, name: str) -> Path:
    """package 内の exact file path を root escape / symlink を拒否して解決する。

    manifest が宣言した名前だけを、root 直下の実 file としてのみ受け付ける。
    `..` を含む相対 path や symlink は、別 artifact を読ませる差し替え経路になる。
    """
    if "/" in name or "\\" in name or name in ("", ".", ".."):
        raise BundleLoadError(f"package file name {name!r} must be a plain file name")
    candidate = root / name
    if candidate.is_symlink():
        raise BundleLoadError(f"package file {name!r} must not be a symlink")
    if not candidate.is_file():
        raise BundleLoadError(f"package file {name!r} is missing")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise BundleLoadError(f"cannot resolve package file {name!r}: {exc}") from exc
    if resolved.parent != resolved_root:
        raise BundleLoadError(f"package file {name!r} escapes the package root")
    return resolved


def _verified_bytes(root: Path, name: str, expected_sha256: str) -> bytes:
    """宣言 hash と一致する場合だけ file 内容を返す。

    model file を deserialize する前に必ずこの関数を通す。hash 不一致の file を
    torch / pickle に渡さないことが、任意コード実行を防ぐ唯一の境界である。
    """
    path = _resolve_package_file(root, name)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise BundleLoadError(f"cannot read package file {name!r}: {exc}") from exc
    actual = sha256_hex(payload)
    if actual != expected_sha256:
        raise BundleLoadError(
            f"package file {name!r} hash mismatch: {actual!r} != {expected_sha256!r}"
        )
    return payload


def _read_combat_manifest(root: Path) -> dict[str, Any]:
    """combat package manifest を読み、schema と必須 field を検証する。

    manifest は JSON だけを読む。pickle 経路を通さないため、この段階では
    任意コード実行が起こり得ない。
    """
    manifest_path = _resolve_package_file(root, COMBAT_MANIFEST_FILENAME)
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleLoadError(f"cannot read combat package manifest: {exc}") from exc
    if not isinstance(manifest, dict) or set(manifest) != _COMBAT_MANIFEST_KEYS:
        raise BundleLoadError("combat package manifest fields mismatch")
    if manifest.get("schema_version") != COMBAT_PACKAGE_SCHEMA_VERSION:
        raise BundleLoadError("unsupported combat package schema_version")
    if manifest.get("development_only") is not False:
        raise BundleLoadError("development_only combat package cannot start live bundle")
    if manifest.get("formal_student_eligible") is not True:
        raise BundleLoadError("formal_student_eligible=false combat package rejected")
    for name in (
        "policy_sha256", "vecnormalize_sha256", "deploy_schema_hash",
        "action_semantics_hash", "target_profile_ref_hash", "target_capability_hash",
    ):
        _require_sha256(manifest.get(name), name)

    identities = manifest.get("formal_dependency_identities")
    if not isinstance(identities, dict) or set(identities) != _FORMAL_DEPENDENCY_KEYS:
        raise BundleLoadError("combat package formal dependency identities are missing")
    for name, identity in identities.items():
        _require_sha256(identity, f"formal_dependency_identities.{name}")

    if manifest.get("files") != list(COMBAT_PACKAGE_FILES):
        raise BundleLoadError("combat package file list mismatch")
    if {entry.name for entry in root.iterdir()} != set(COMBAT_PACKAGE_FILES):
        raise BundleLoadError("combat package directory contents mismatch")

    model_config = manifest.get("model_config")
    if not isinstance(model_config, dict) or set(model_config) != _MODEL_CONFIG_KEYS:
        raise BundleLoadError("combat model_config fields mismatch")
    for key in sorted(_MODEL_CONFIG_KEYS):
        _positive_int(model_config.get(key), f"model_config.{key}")

    profile = manifest.get("runtime_profile")
    if not isinstance(profile, dict) or set(profile) != _RUNTIME_PROFILE_KEYS:
        raise BundleLoadError("combat package runtime_profile fields mismatch")
    for key in ("os_build", "gpu_name", "driver_version", "cuda_version", "capture_backend"):
        if not isinstance(profile.get(key), str) or not profile[key]:
            raise BundleLoadError(f"runtime_profile.{key} must be a non-empty string")
    _require_sha256(profile.get("choice_capability_hash"), "runtime_profile.choice_capability_hash")
    if profile.get("decision_hz") != REQUIRED_DECISION_HZ:
        raise BundleLoadError(f"runtime_profile.decision_hz must be {REQUIRED_DECISION_HZ}")
    _positive_int(profile.get("key_lease_duration_ms"), "runtime_profile.key_lease_duration_ms")
    return manifest


def _validate_action_binding(
    manifest: Mapping[str, Any], action_semantics: ActionSemantics
) -> None:
    """action_dim と ActionSemantics の順序・hash を bundle 側で検証する。

    action_dim が正整数であることだけを見ると、2-action の別 model でも move
    decision を返してしまう。9 direction/idle の exact ordering hash まで照合する。
    """
    if action_semantics.num_actions != REQUIRED_ACTION_DIM:
        raise BundleLoadError(
            f"action semantics must declare {REQUIRED_ACTION_DIM} actions, "
            f"got {action_semantics.num_actions}"
        )
    declared_dim = manifest["model_config"]["action_dim"]
    if declared_dim != REQUIRED_ACTION_DIM:
        raise BundleLoadError(
            f"combat model action_dim must be {REQUIRED_ACTION_DIM}, got {declared_dim}"
        )
    if manifest["action_semantics_hash"] != action_semantics.semantics_hash:
        raise BundleLoadError("combat package action_semantics_hash mismatch")


def _load_recurrent_policy(policy_bytes_path: Path) -> Any:
    """検証済み policy.zip から SB3 RecurrentPPO を CPU 推論モードでロードする。

    Deployment 側では policy architecture を再定義しない。SB3 が保存した
    recurrent actor をそのまま読むことで、saved actor との drift を作らない。
    """
    try:
        from sb3_contrib import RecurrentPPO  # noqa: PLC0415  # 重い依存を遅延 import
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise BundleLoadError(f"sb3-contrib is required for combat runtime: {exc}") from exc
    try:
        model = RecurrentPPO.load(policy_bytes_path, device="cpu")
    except Exception as exc:  # noqa: BLE001  # SB3 は多様な例外型を送出する
        raise BundleLoadError(f"cannot load recurrent combat policy: {exc}") from exc
    policy = model.policy
    if getattr(policy, "lstm_actor", None) is None:
        raise BundleLoadError("combat policy has no actor LSTM; wrong recurrent policy")
    policy.set_training_mode(False)
    return model


def _load_deploy_vecnormalize(payload: bytes, observation_dim: int) -> Any:
    """検証済み VecNormalize を deploy 設定 (training=False, norm_reward=False) で復元する。

    obs 統計だけを使うため gym env は要らない。訓練時統計を更新しない設定に
    固定し、正規化条件が訓練時と一致することを observation 次元で照合する。
    """
    try:
        from stable_baselines3.common.vec_env import VecNormalize  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise BundleLoadError(f"stable-baselines3 is required: {exc}") from exc
    try:
        vecnormalize = pickle.loads(payload)
    except Exception as exc:  # noqa: BLE001  # pickle は多様な例外型を送出する
        raise BundleLoadError(f"cannot load deploy VecNormalize: {exc}") from exc
    if not isinstance(vecnormalize, VecNormalize):
        raise BundleLoadError("deploy VecNormalize payload is not a VecNormalize")
    obs_rms = getattr(vecnormalize, "obs_rms", None)
    if obs_rms is None or not hasattr(obs_rms, "mean"):
        raise BundleLoadError("deploy VecNormalize has no observation statistics")
    mean = np.asarray(obs_rms.mean, dtype=np.float64)
    if mean.shape != (observation_dim,):
        raise BundleLoadError(
            f"deploy VecNormalize obs statistics shape {mean.shape} "
            f"does not match observation_dim {observation_dim}"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(np.asarray(obs_rms.var))):
        raise BundleLoadError("deploy VecNormalize statistics contain non-finite values")
    # deploy 時は統計を更新せず、reward 正規化も行わない。
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    if not getattr(vecnormalize, "norm_obs", False):
        raise BundleLoadError("deploy VecNormalize must normalize observations")
    return vecnormalize


def _assert_policy_matches_manifest(model: Any, manifest: Mapping[str, Any]) -> None:
    """ロード済み policy の次元が manifest 宣言と一致することを確認する。

    manifest だけを書き換えた package や、別 shape の policy を検出する。
    """
    config = manifest["model_config"]
    policy = model.policy
    lstm = policy.lstm_actor
    if int(lstm.num_layers) != config["n_lstm_layers"]:
        raise BundleLoadError("combat policy n_lstm_layers mismatch")
    if int(lstm.hidden_size) != config["lstm_hidden_size"]:
        raise BundleLoadError("combat policy lstm_hidden_size mismatch")

    action_space = getattr(model, "action_space", None)
    action_n = getattr(action_space, "n", None)
    if action_n is None or int(action_n) != config["action_dim"]:
        raise BundleLoadError("combat policy action space does not match action_dim")

    obs_space = getattr(model, "observation_space", None)
    obs_shape = getattr(obs_space, "shape", None)
    if obs_shape is None or tuple(obs_shape) != (config["observation_dim"],):
        raise BundleLoadError("combat policy observation space does not match observation_dim")


@dataclass(frozen=True)
class RecurrentCombatPolicy:
    """SB3 RecurrentPPO model と deploy VecNormalize の検証済みペア。

    combat session はこの型だけを受け取り、policy architecture を自前で
    持たない。observation_dim / action_dim / LSTM 形状も併せて公開する。
    """

    model: Any
    vecnormalize: Any
    observation_dim: int
    action_dim: int
    n_lstm_layers: int
    lstm_hidden_size: int

    @property
    def lstm_state_shape(self) -> tuple[int, int, int]:
        """actor LSTM state の shape `[n_layers, 1, hidden]` を返す。"""
        return (self.n_lstm_layers, 1, self.lstm_hidden_size)


def _load_combat_package(
    package_dir: Path, action_semantics: ActionSemantics
) -> tuple[RecurrentCombatPolicy, dict[str, Any]]:
    """combat package を hash 検証してから RecurrentPPO と VecNormalize をロードする。

    manifest 検証 → file hash 照合 → deserialize の順序を守る。順序を入れ替えると
    未検証 file を pickle に渡すことになり、fail-closed が成立しない。
    """
    root = Path(package_dir)
    if not root.is_dir():
        raise BundleLoadError(f"combat package directory not found: {root}")
    manifest = _read_combat_manifest(root)
    _validate_action_binding(manifest, action_semantics)

    policy_path = _resolve_package_file(root, COMBAT_POLICY_FILENAME)
    _verified_bytes(root, COMBAT_POLICY_FILENAME, manifest["policy_sha256"])
    vecnormalize_payload = _verified_bytes(
        root, COMBAT_VECNORMALIZE_FILENAME, manifest["vecnormalize_sha256"]
    )

    config = manifest["model_config"]
    model = _load_recurrent_policy(policy_path)
    _assert_policy_matches_manifest(model, manifest)
    vecnormalize = _load_deploy_vecnormalize(vecnormalize_payload, config["observation_dim"])

    policy = RecurrentCombatPolicy(
        model=model,
        vecnormalize=vecnormalize,
        observation_dim=config["observation_dim"],
        action_dim=config["action_dim"],
        n_lstm_layers=config["n_lstm_layers"],
        lstm_hidden_size=config["lstm_hidden_size"],
    )
    return policy, manifest


def _descriptors_by_kind(
    descriptors: Sequence[ArtifactDescriptor],
) -> dict[str, ArtifactDescriptor]:
    """node_kind をキーに descriptor を引ける dict を作る。

    同じ kind が複数ある formal bundle は曖昧なので拒否する。
    """
    result: dict[str, ArtifactDescriptor] = {}
    for descriptor in descriptors:
        if descriptor.node_kind in result:
            raise BundleLoadError(
                f"duplicate artifact descriptor for node kind {descriptor.node_kind!r}"
            )
        result[descriptor.node_kind] = descriptor
    return result


def _verify_store_restore(store: ArtifactStore, descriptors: Sequence[ArtifactDescriptor]) -> tuple[str, ...]:
    """全 descriptor の file を artifact store 上で restore 検証する。

    store 上に object が存在し、hash と size が descriptor 宣言と一致することを
    確認する。1 件でも欠損 / 破損があれば live 起動しない。
    """
    verified: list[str] = []
    for descriptor in descriptors:
        for entry in descriptor.files:
            ref = entry if isinstance(entry, ArtifactRef) else ArtifactRef.from_wire(entry)
            try:
                verification = store.verify(ref, expected_size_bytes=ref.size_bytes)
            except ArtifactStoreError as exc:
                raise BundleLoadError(
                    f"artifact store verification failed for {ref.logical_id!r}: {exc}"
                ) from exc
            if not verification.ok:
                raise BundleLoadError(
                    f"artifact store restore check failed for {ref.logical_id!r}: "
                    f"{verification.reason}"
                )
            verified.append(ref.logical_id)
    return tuple(sorted(verified))


def _require_metadata_hash(descriptor: ArtifactDescriptor, key: str) -> str:
    """descriptor.identity_metadata から SHA-256 を必須で取り出す。"""
    value = descriptor.identity_metadata.get(key)
    if not _is_sha256(value):
        raise BundleLoadError(
            f"{descriptor.node_kind}.identity_metadata.{key} must be lowercase SHA-256"
        )
    return str(value)


@dataclass(frozen=True)
class RuntimeBundle:
    """検証済み combat + ItemSelector artifact の runtime コンテナ。

    live-eligible かどうかを startup 時に確定し、golden fixture からの
    正式起動を fail-closed で拒否する。OS input には触れない。
    """

    development_only: bool
    live_eligible: bool
    combat_policy: RecurrentCombatPolicy
    deploy_schema: DeployObsSchema
    deploy_schema_hash: str
    action_semantics: ActionSemantics
    ui_policy_config: NonModelUiPolicyConfigV1
    startup_report: dict[str, Any]
    # ItemSelector は省略可 — combat-only golden fixture に None を許容
    _item_selector: Any = field(default=None, repr=False)

    @property
    def item_selector(self) -> Any:
        """OnnxItemSelector または None を返す。

        None の場合は item session を起動できないため、呼び出し側が guard する。
        """
        return self._item_selector

    @property
    def action_dim(self) -> int:
        """combat policy の action 空間サイズを返す。"""
        return self.combat_policy.action_dim

    def assert_live_eligible(self) -> None:
        """live 起動が許可されていない bundle に対して BundleLoadError を送出する。

        formal artifact が揃っていない場合は本番起動を防ぐ。
        """
        if self.development_only or not self.live_eligible:
            raise BundleLoadError(
                "bundle is development_only or not live_eligible; formal artifacts required"
            )

    @classmethod
    def from_golden_fixture(
        cls,
        combat_policy: RecurrentCombatPolicy,
        *,
        ui_policy_config: NonModelUiPolicyConfigV1 | None = None,
        item_selector: Any = None,
        deploy_schema: DeployObsSchema | None = None,
        action_semantics: ActionSemantics | None = None,
    ) -> "RuntimeBundle":
        """golden fixture 用の development_only=True, live_eligible=False bundle を返す。

        formal package を必要とせず、テストが全 loader / session / scheduler を実行できる。
        この bundle は assert_live_eligible() で必ず拒否される。
        """
        if not isinstance(combat_policy, RecurrentCombatPolicy):
            raise BundleLoadError("combat_policy must be RecurrentCombatPolicy")
        config = ui_policy_config or NonModelUiPolicyConfigV1.load_default()
        schema = deploy_schema or DeployObsSchema.default_v1()
        semantics = action_semantics or ActionSemantics.default_v1()
        return cls(
            development_only=True,
            live_eligible=False,
            combat_policy=combat_policy,
            deploy_schema=schema,
            deploy_schema_hash=schema.schema_hash,
            action_semantics=semantics,
            ui_policy_config=config,
            startup_report={
                "bundle_kind": BUNDLE_DEVELOPMENT_SENTINEL,
                "development_only": True,
                "live_eligible": False,
            },
            _item_selector=item_selector,
        )

    @classmethod
    def load(
        cls,
        combat_package_dir: Path,
        item_selector_dir: Path,
        *,
        artifact_store: ArtifactStore,
        descriptors: Sequence[ArtifactDescriptor],
        target_profile: TargetProfileRef,
        action_semantics: ActionSemantics | None = None,
    ) -> "RuntimeBundle":
        """formal package を全 gate 通過後にロードし live-capable bundle を返す。

        gate 順序は (1) artifact DAG の immutable parent と exact subject hash、
        (2) artifact store 上の restore 検証、(3) package manifest の identity 照合、
        (4) hardware / action / time / capability の一致。いずれかが欠けたら起動しない。
        perception verdict hash は呼び出し側の引数ではなく、検証済み descriptor から
        導出する。文字列の形だけを見て startup report へ複製することはしない。
        """
        semantics = action_semantics or ActionSemantics.default_v1()
        if not isinstance(artifact_store, ArtifactStore):
            raise BundleLoadError("artifact_store must be an ArtifactStore")
        if not isinstance(target_profile, TargetProfileRef):
            raise BundleLoadError("target_profile must be a TargetProfileRef")

        descriptor_list = list(descriptors)
        if not descriptor_list:
            raise BundleLoadError("formal bundle requires artifact descriptors")

        # (1) immutable parent と 04-10 exact subject hash を DAG validator で強制する。
        try:
            dag_report = validate_formal_runtime_dag(descriptor_list)
        except ArtifactDagValidationError as exc:
            raise BundleLoadError(f"formal runtime DAG rejected: {exc}") from exc

        by_kind = _descriptors_by_kind(descriptor_list)
        for required_kind in (
            _RUNTIME_NODE_KIND,
            _COMBAT_RELEASE_NODE_KIND,
            _ITEM_SELECTOR_RELEASE_NODE_KIND,
            _PERCEPTION_VERDICT_NODE_KIND,
        ):
            if required_kind not in by_kind:
                raise BundleLoadError(f"formal bundle missing descriptor: {required_kind}")

        runtime_descriptor = by_kind[_RUNTIME_NODE_KIND]
        combat_descriptor = by_kind[_COMBAT_RELEASE_NODE_KIND]
        selector_descriptor = by_kind[_ITEM_SELECTOR_RELEASE_NODE_KIND]
        verdict_descriptor = by_kind[_PERCEPTION_VERDICT_NODE_KIND]

        # (2) restore 検証済み artifact store の上でだけ live bundle を作る。
        verified_objects = _verify_store_restore(artifact_store, descriptor_list)

        # (3) package manifest を、信頼済み descriptor 側の identity と突き合わせる。
        combat_policy, combat_manifest = _load_combat_package(Path(combat_package_dir), semantics)
        expected_combat_manifest_hash = _require_metadata_hash(
            combat_descriptor, "package_manifest_hash"
        )
        if canonical_hash(combat_manifest) != expected_combat_manifest_hash:
            raise BundleLoadError(
                "combat package manifest hash does not match combat_student_release descriptor"
            )

        try:
            item_selector = OnnxItemSelector.load(Path(item_selector_dir))
        except ItemSelectorRuntimeError as exc:
            raise BundleLoadError(f"ItemSelector package rejected: {exc}") from exc
        expected_selector_manifest_hash = _require_metadata_hash(
            selector_descriptor, "package_manifest_hash"
        )
        if canonical_hash(item_selector.manifest) != expected_selector_manifest_hash:
            raise BundleLoadError(
                "ItemSelector manifest hash does not match item_selector_release descriptor"
            )

        # (4) hardware / action / time / capability の一致を bundle 全体で照合する。
        installed_config = NonModelUiPolicyConfigV1.load_default()
        if item_selector.ui_policy_config.to_wire() != installed_config.to_wire():
            raise BundleLoadError("ItemSelector UI policy config mismatch with installed policy")

        deploy_schema = DeployObsSchema.default_v1()
        if deploy_schema.schema_hash != combat_manifest["deploy_schema_hash"]:
            raise BundleLoadError(
                f"combat package deploy_schema_hash {combat_manifest['deploy_schema_hash']!r} "
                f"does not match installed default schema {deploy_schema.schema_hash!r}"
            )
        if combat_policy.observation_dim != 3 * deploy_schema.dim:
            raise BundleLoadError(
                "combat observation_dim does not match deploy schema value/validity/age planes"
            )

        if combat_manifest["target_profile_ref_hash"] != target_profile.ref_hash:
            raise BundleLoadError("combat package target hardware profile mismatch")
        runtime_target_hash = _require_metadata_hash(runtime_descriptor, "target_capability_hash")
        if combat_manifest["target_capability_hash"] != runtime_target_hash:
            raise BundleLoadError("combat package target_capability_hash mismatch")
        if item_selector.manifest.get("target_capability_hash") != runtime_target_hash:
            raise BundleLoadError("ItemSelector target_capability_hash mismatch")

        runtime_profile = combat_manifest["runtime_profile"]
        declared_capability = runtime_descriptor.identity_metadata.get("choice_capability_hash")
        if runtime_profile["choice_capability_hash"] != declared_capability:
            raise BundleLoadError("runtime choice capability hash mismatch")
        if runtime_descriptor.identity_metadata.get("action_semantics_hash") != (
            semantics.semantics_hash
        ):
            raise BundleLoadError("runtime descriptor action_semantics_hash mismatch")
        if runtime_descriptor.identity_metadata.get("decision_hz") != REQUIRED_DECISION_HZ:
            raise BundleLoadError("runtime descriptor decision_hz mismatch")

        perception_verdict_hash = verdict_descriptor.identity_hash
        subject_hashes = verdict_descriptor.identity_metadata.get("subject_hashes")
        if not isinstance(subject_hashes, Mapping) or not subject_hashes:
            raise BundleLoadError("perception_final_verdict subject_hashes are missing")

        startup_report = _build_startup_report(
            combat_manifest=combat_manifest,
            item_selector=item_selector,
            runtime_descriptor=runtime_descriptor,
            perception_verdict_hash=perception_verdict_hash,
            perception_subject_hashes=subject_hashes,
            target_profile=target_profile,
            action_semantics=semantics,
            deploy_schema=deploy_schema,
            combat_policy=combat_policy,
            verified_objects=verified_objects,
            dag_identity_hashes=dag_report.topological_identity_hashes,
        )
        return cls(
            development_only=False,
            live_eligible=True,
            combat_policy=combat_policy,
            deploy_schema=deploy_schema,
            deploy_schema_hash=combat_manifest["deploy_schema_hash"],
            action_semantics=semantics,
            ui_policy_config=installed_config,
            startup_report=startup_report,
            _item_selector=item_selector,
        )


def _build_startup_report(
    *,
    combat_manifest: Mapping[str, Any],
    item_selector: OnnxItemSelector,
    runtime_descriptor: ArtifactDescriptor,
    perception_verdict_hash: str,
    perception_subject_hashes: Mapping[str, Any],
    target_profile: TargetProfileRef,
    action_semantics: ActionSemantics,
    deploy_schema: DeployObsSchema,
    combat_policy: RecurrentCombatPolicy,
    verified_objects: Sequence[str],
    dag_identity_hashes: Sequence[str],
) -> dict[str, Any]:
    """起動時の dependency / device / profile / schema summary を生成する。

    artifact 取り違えに運用者が気づけるよう、主要 identity を一か所へまとめる。
    plan 05-01 が要求する OS/GPU/driver/CUDA/capture backend、action semantics、
    decision Hz、key lease duration、choice capability も含める。
    """
    profile = combat_manifest["runtime_profile"]
    return {
        "bundle_kind": BUNDLE_FORMAL_SENTINEL,
        "development_only": False,
        "live_eligible": True,
        "combat_deploy_schema_hash": combat_manifest["deploy_schema_hash"],
        "combat_policy_sha256": combat_manifest["policy_sha256"],
        "combat_vecnormalize_sha256": combat_manifest["vecnormalize_sha256"],
        "combat_formal_dependency_identities": dict(
            combat_manifest["formal_dependency_identities"]
        ),
        "combat_observation_dim": combat_policy.observation_dim,
        "combat_action_dim": combat_policy.action_dim,
        "combat_lstm_state_shape": list(combat_policy.lstm_state_shape),
        "deploy_schema_dim": deploy_schema.dim,
        "deploy_schema_version": deploy_schema.schema_version,
        "item_selector_artifact_identity": item_selector.manifest.get("artifact_identity"),
        "item_selector_nmax": item_selector.nmax,
        "item_selector_feature_schema": item_selector.feature_schema,
        "item_selector_confidence_threshold": item_selector.confidence_threshold,
        "item_selector_temperature": item_selector.temperature,
        "ui_policy_config_hash": item_selector.ui_policy_config.config_hash,
        "action_semantics_hash": action_semantics.semantics_hash,
        "action_semantics_actions": list(action_semantics.actions),
        "target_profile_ref_hash": target_profile.ref_hash,
        "target_capability_hash": combat_manifest["target_capability_hash"],
        "runtime_bundle_identity_hash": runtime_descriptor.identity_hash,
        "perception_verdict_hash": perception_verdict_hash,
        "perception_subject_hashes": dict(perception_subject_hashes),
        "os_build": profile["os_build"],
        "gpu_name": profile["gpu_name"],
        "driver_version": profile["driver_version"],
        "cuda_version": profile["cuda_version"],
        "capture_backend": profile["capture_backend"],
        "decision_hz": profile["decision_hz"],
        "key_lease_duration_ms": profile["key_lease_duration_ms"],
        "choice_capability_hash": profile["choice_capability_hash"],
        "artifact_store_verified_objects": list(verified_objects),
        "artifact_dag_identity_hashes": list(dag_identity_hashes),
    }

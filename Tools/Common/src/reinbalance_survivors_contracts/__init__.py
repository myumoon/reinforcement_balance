"""reinbalance-survivors-contracts。

Survivors の契約型・canonical JSON / hash、および Training と Deployment が共有する
pure な非モデル UI policy の唯一の source of truth。

依存方向（docs/project_structure.md 参照）:

    reinbalance_survivors_contracts  <-  Tools/Training
    reinbalance_survivors_contracts  <-  Tools/Deployment

本パッケージは SB3・PyTorch・UE5 HTTP client・OpenCV・DXcam・および Training /
Deployment の module を import してはならない。NumPy は遅延 array ヘルパーでのみ使う。
"""

from __future__ import annotations

from .canonical_json import (
    CanonicalJsonError,
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)
from .current_fidelity import (
    GENERATED_INPUT_DESCRIPTOR_SCHEMA_VERSION,
    CurrentFidelityContext,
    GeneratedInputSourceIdentity,
    resolve_current_gating_producer_hashes,
)
from .ubt_action_graph import (
    UBT_ACTION_GRAPH_SCHEMA_VERSION,
    UbtActionGraphAttestation,
)
from .artifact_dag import (
    ALLOWED_PARENT_KINDS,
    ArtifactDagReport,
    ArtifactDagValidationError,
    validate_artifact_dag,
)
from .artifact_identity import (
    ARTIFACT_DESCRIPTOR_SCHEMA_VERSION,
    ARTIFACT_NODE_KINDS,
    ARTIFACT_NODE_REF_SCHEMA_VERSION,
    ARTIFACT_REF_SCHEMA_VERSION,
    OBJECT_URI_PREFIX,
    RESTORE_TEST_VERDICT_SCHEMA_VERSION,
    VALIDATION_VERDICT_SCHEMA_VERSION,
    ArtifactDescriptor,
    ArtifactNodeRef,
    ArtifactRef,
    RestoreTestVerdict,
    ValidationVerdict,
    artifact_uri,
    is_sha256_hex,
    parse_artifact_uri,
)
from .deploy_obs import (
    DEPLOY_OBS_SCHEMA_VERSION,
    DeployObsField,
    DeployObsSchema,
    DeployObservation,
)
from .item_decision import (
    CANDIDATE_FEATURES_SCHEMA_VERSION,
    ITEM_DECISION_SCHEMA_VERSION,
    CandidateFeatures,
    ItemDecisionFeatures,
)
from .perception_error import (
    PERCEPTION_ERROR_SCHEMA_VERSION,
    PerceptionErrorProfile,
)
from .target_action import (
    ACTION_SEMANTICS_SCHEMA_VERSION,
    TARGET_PROFILE_SCHEMA_VERSION,
    ActionSemantics,
    TargetProfileRef,
)
from .ui_intent import (
    CHOOSE_FALLBACK_SEMANTICS,
    EFFECT_OWNER_UI_STATE_MACHINE,
    UI_INTENT_SCHEMA_VERSION,
    ContractValidationError,
    DecisionOwner,
    UiIntentKind,
    UiIntentV1,
    allowed_semantic_actions,
)
from .ui_policy import (
    NON_MODEL_UI_POLICY_CONFIG_SCHEMA_VERSION,
    UI_POLICY_INPUT_SCHEMA_VERSION,
    ButtonOption,
    ButtonSemantic,
    FallbackSemantic,
    FallbackTarget,
    NonModelUiPolicyConfigV1,
    ScreenState,
    UiPolicyInputV1,
    decide_non_model_ui_intent,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # canonical json（canonical JSON）
    "CanonicalJsonError",
    "canonical_json_bytes",
    "sha256_hex",
    "canonical_hash",
    "GENERATED_INPUT_DESCRIPTOR_SCHEMA_VERSION",
    "GeneratedInputSourceIdentity",
    "CurrentFidelityContext",
    "resolve_current_gating_producer_hashes",
    "UBT_ACTION_GRAPH_SCHEMA_VERSION",
    "UbtActionGraphAttestation",
    # artifact identity / DAG（Artifact 識別・DAG）
    "OBJECT_URI_PREFIX",
    "ARTIFACT_REF_SCHEMA_VERSION",
    "ARTIFACT_NODE_REF_SCHEMA_VERSION",
    "ARTIFACT_DESCRIPTOR_SCHEMA_VERSION",
    "VALIDATION_VERDICT_SCHEMA_VERSION",
    "RESTORE_TEST_VERDICT_SCHEMA_VERSION",
    "ARTIFACT_NODE_KINDS",
    "ArtifactRef",
    "ArtifactNodeRef",
    "ArtifactDescriptor",
    "ValidationVerdict",
    "RestoreTestVerdict",
    "artifact_uri",
    "parse_artifact_uri",
    "is_sha256_hex",
    "ALLOWED_PARENT_KINDS",
    "ArtifactDagValidationError",
    "ArtifactDagReport",
    "validate_artifact_dag",
    # deploy obs（DeployObs）
    "DEPLOY_OBS_SCHEMA_VERSION",
    "DeployObsField",
    "DeployObsSchema",
    "DeployObservation",
    # item decision（アイテム決定）
    "ITEM_DECISION_SCHEMA_VERSION",
    "CANDIDATE_FEATURES_SCHEMA_VERSION",
    "CandidateFeatures",
    "ItemDecisionFeatures",
    # perception error（perception 誤差）
    "PERCEPTION_ERROR_SCHEMA_VERSION",
    "PerceptionErrorProfile",
    # target / action（target・アクション）
    "TARGET_PROFILE_SCHEMA_VERSION",
    "ACTION_SEMANTICS_SCHEMA_VERSION",
    "TargetProfileRef",
    "ActionSemantics",
    # ui intent（UI intent）
    "UI_INTENT_SCHEMA_VERSION",
    "EFFECT_OWNER_UI_STATE_MACHINE",
    "CHOOSE_FALLBACK_SEMANTICS",
    "ContractValidationError",
    "UiIntentKind",
    "DecisionOwner",
    "UiIntentV1",
    "allowed_semantic_actions",
    # ui policy（UI policy）
    "UI_POLICY_INPUT_SCHEMA_VERSION",
    "NON_MODEL_UI_POLICY_CONFIG_SCHEMA_VERSION",
    "ScreenState",
    "FallbackSemantic",
    "ButtonSemantic",
    "FallbackTarget",
    "ButtonOption",
    "UiPolicyInputV1",
    "NonModelUiPolicyConfigV1",
    "decide_non_model_ui_intent",
    # helpers（ヘルパー）
    "contract_fingerprint",
]


def contract_fingerprint() -> dict[str, str]:
    """共有契約の安定した identity 文字列を返す。

    cross-cwd / cross-package の smoke テストで、Training と Deployment が *同一* の
    インストール済み module と同じ schema hash を解決することを検証するために使う。
    値は code identity のみに依存し、cwd・プラットフォーム・プロセスに依存してはならない。
    """
    return {
        "package_version": __version__,
        "ui_intent_v1_qualname": f"{UiIntentV1.__module__}.{UiIntentV1.__qualname__}",
        "non_model_ui_policy_config_hash": NonModelUiPolicyConfigV1.default_config().config_hash,
        "deploy_obs_v1_schema_hash": DeployObsSchema.default_v1().schema_hash,
        "action_semantics_v1_hash": ActionSemantics.default_v1().semantics_hash,
    }

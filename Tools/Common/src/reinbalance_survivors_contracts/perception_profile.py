"""CalibrationResidual と FittedPerceptionErrorProfile の共有契約。

Training と Deployment が共有する calibration 残差型と fitted profile 型を定義する。
formal token はモジュール内 sentinel で管理し、wire ローダーは development_only=False を
fail-closed にする。formal profile は producer が発行した calibration descriptor /
calibration commit を exact ref で辿る検証経路（from_store_artifact /
from_calibration_commit）でのみ取得でき、任意の store+ref の提示では取得できない。
"""

from __future__ import annotations

import json as _json
import math
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

from .artifact_dag import ArtifactDagValidationError, validate_artifact_dag
from .artifact_identity import ArtifactDescriptor, ArtifactRef
from .artifact_store import ArtifactStore, ArtifactStoreError
from .canonical_json import canonical_hash, sha256_hex
from .perception_error import ITEM_CATEGORY_SIZE, PerceptionErrorProfile

CALIBRATION_ARTIFACT_SCHEMA_VERSION: Final[str] = "perception_calibration_profile.v1"

# producer(benchmark_survivors_perception)が発行する calibration node の node_kind。
CALIBRATION_PROFILE_NODE_KIND: Final[str] = "perception_calibration_profile"

# producer が calibration package を freeze したときに store へ書く commit の schema。
CALIBRATION_COMMIT_SCHEMA_VERSION: Final[str] = "perception_calibration_commit.v1"

# calibration descriptor が参照する provenance ファイルの schema。
CALIBRATION_PROVENANCE_SCHEMA_VERSION: Final[str] = "perception_calibration_package.v1"

# calibration descriptor が必ず持つ 3 ファイルの basename。
_CALIBRATION_FILE_NAMES: Final[frozenset[str]] = frozenset(
    {"profile.json", "profile.artifact.json", "provenance.json"}
)

# producer が staging する descriptor object の logical id 接頭辞。
_DESCRIPTOR_LOGICAL_PREFIX: Final[str] = "perception/package/descriptors/"

# formal profile 生成を wire ローダーから分離するための sentinel。
# 公開 API に含めない。fit runner と store 検証経路のみが参照する。
_FORMAL_FACTORY_TOKEN = object()

_RESIDUAL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "hp_ratio", "xp_ratio", "timer_seconds", "inventory_hash", "coord_noise",
        "coord_quantization_px", "burst_enter", "burst_exit", "burst_dropout",
        "unknown_screen_collapse", "unknown_screen_collapse_duration",
        "item_category", "enemy_category",
    }
)


class HashMismatchError(ValueError):
    """artifact content hash が seal/verdict の exact hash と一致しない。"""


class FormalVerdictPromotionError(ValueError):
    """synthetic 公開コンストラクタまたは wire ローダーから formal flag を構築しようとした。"""


class InvalidResidualError(ValueError):
    """residual field/type/range が fit 契約外である。"""


def _require_sha256(value: object, label: str) -> str:
    if not (isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)):
        raise ValueError(f"{label} must be a 64-character lowercase SHA-256")
    return value  # type: ignore[return-value]


def _load_verified_json(store: ArtifactStore, ref: ArtifactRef) -> dict[str, Any]:
    """logical binding・store verify・実バイト列を全て確認して JSON object を読む。

    ref が store の logical index と exact 一致し、content hash / size / media type も
    一致した場合だけ中身を返します。どれか一つでもずれていれば読み込まずに失敗します。
    """
    if not isinstance(ref, ArtifactRef):
        raise TypeError("verified JSON read requires an ArtifactRef")
    try:
        resolved = store.resolve(ref.logical_id)
        verification = store.verify(ref)
        data_bytes = store.object_path(ref.store_uri).read_bytes()
    except (ArtifactStoreError, OSError) as exc:
        raise HashMismatchError(f"calibration artifact store read failed: {exc}") from exc
    if resolved != ref:
        raise HashMismatchError(
            f"calibration artifact ref {ref.logical_id!r} is not bound to its logical id"
        )
    if not verification.ok:
        raise HashMismatchError(
            f"calibration artifact store verification failed: {verification.reason}"
        )
    if (
        ref.media_type != "application/json"
        or len(data_bytes) != ref.size_bytes
        or sha256_hex(data_bytes) != ref.sha256
    ):
        raise HashMismatchError("calibration artifact content changed after verification")
    try:
        value: Any = _json.loads(data_bytes.decode("utf-8"))
    except (UnicodeDecodeError, _json.JSONDecodeError) as exc:
        raise ValueError(f"calibration artifact is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("calibration artifact must be a JSON object")
    return value


def _strict_number(value: object, label: str) -> float:
    import numpy as np
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(float(value)):
        raise InvalidResidualError(f"{label} must be a finite number (bool is forbidden)")
    return float(value)


@dataclass(frozen=True, slots=True)
class CalibrationResidual:
    """キャリブレーション残差の1標本。confidence と age_frames で重み付けする。"""

    session_id: str
    frame_id: str
    field: str
    residual: float
    confidence: float
    age_frames: int
    latency_frames: float = 0.0
    ground_truth_category: int | None = None
    predicted_category: int | None = None

    def __post_init__(self) -> None:
        if type(self.session_id) is not str or not self.session_id:
            raise InvalidResidualError("session_id must be a non-empty string")
        if type(self.frame_id) is not str or not self.frame_id:
            raise InvalidResidualError("frame_id must be a non-empty string")
        if self.field not in _RESIDUAL_FIELDS:
            raise InvalidResidualError(f"unsupported residual field {self.field!r}")
        object.__setattr__(self, "residual", _strict_number(self.residual, "residual"))
        if self.field in {
            "burst_enter", "burst_exit", "burst_dropout",
            "unknown_screen_collapse",
        } and not 0.0 <= self.residual <= 1.0:
            raise InvalidResidualError(f"{self.field} residual must be in [0, 1]")
        if self.field in {
            "coord_quantization_px", "unknown_screen_collapse_duration",
        } and self.residual < 0.0:
            raise InvalidResidualError(f"{self.field} residual must be non-negative")
        confidence = _strict_number(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise InvalidResidualError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        if type(self.age_frames) is not int or self.age_frames < 0:
            raise InvalidResidualError("age_frames must be a non-negative integer")
        latency = _strict_number(self.latency_frames, "latency_frames")
        if latency < 0.0:
            raise InvalidResidualError("latency_frames must be non-negative")
        object.__setattr__(self, "latency_frames", latency)
        category_pair = (self.ground_truth_category, self.predicted_category)
        if (category_pair[0] is None) != (category_pair[1] is None):
            raise InvalidResidualError("category ground truth/prediction must be provided together")
        if category_pair[0] is not None:
            if self.field not in {"item_category", "enemy_category"}:
                raise InvalidResidualError("category labels require a category residual field")
            if any(type(v) is not int or not 0 <= v < ITEM_CATEGORY_SIZE for v in category_pair):
                raise InvalidResidualError("category labels are outside the fixed vocabulary")
        elif self.field in {"item_category", "enemy_category"}:
            raise InvalidResidualError("category residual fields require category labels")


@dataclass(frozen=True)
class FittedPerceptionErrorProfile(PerceptionErrorProfile):
    """calibration artifact メタデータを保持する PerceptionErrorProfile のサブタイプ。

    formal profile は wire ローダーでは取得できない（development_only=False の wire は拒否）。
    fit runner（Deployment）または producer descriptor 束縛のストア検証経路のみが
    _FORMAL_FACTORY_TOKEN を渡せる。ストア経路で復元した profile は、どの
    calibration descriptor に束縛されたかを calibration_descriptor_hash で保持する。
    """

    calibration_session_hashes: Mapping[str, str] = field(default_factory=dict)
    field_sample_counts: Mapping[str, int] = field(default_factory=dict)
    fit_code_hash: str = ""
    development_only: bool = True
    # store 検証経路で束縛された calibration descriptor の identity hash（wire には含めない）。
    calibration_descriptor_hash: str = ""
    _factory_token: InitVar[object | None] = None

    def __post_init__(self, _factory_token: object | None) -> None:
        super().__post_init__()
        if type(self.calibration_descriptor_hash) is not str:
            raise ValueError("calibration_descriptor_hash must be a string")
        if self.calibration_descriptor_hash:
            _require_sha256(self.calibration_descriptor_hash, "calibration_descriptor_hash")
        hashes = dict(self.calibration_session_hashes)
        if set(hashes) != set(self.calibration_session_ids):
            raise ValueError("calibration_session_hashes must exactly match calibration ids")
        for session_id, content_hash in hashes.items():
            if type(session_id) is not str or not session_id:
                raise ValueError("calibration session hash key must be non-empty")
            _require_sha256(content_hash, f"calibration_session_hashes[{session_id!r}]")
        counts = dict(self.field_sample_counts)
        if not counts or not all(type(n) is str and type(c) is int and c > 0 for n, c in counts.items()):
            raise ValueError("field_sample_counts must contain positive integer counts")
        _require_sha256(self.fit_code_hash, "fit_code_hash")
        if type(self.development_only) is not bool:
            raise ValueError("development_only must be bool")
        if not self.development_only and _factory_token is not _FORMAL_FACTORY_TOKEN:
            raise FormalVerdictPromotionError(
                "formal fitted profile requires the verified formal factory"
            )
        object.__setattr__(self, "calibration_session_hashes", MappingProxyType(hashes))
        object.__setattr__(self, "field_sample_counts", MappingProxyType(counts))

    def to_artifact_wire(self) -> dict[str, Any]:
        return {
            "schema_version": CALIBRATION_ARTIFACT_SCHEMA_VERSION,
            "profile": self.to_wire(),
            "profile_hash": self.profile_hash,
            "calibration_session_hashes": dict(self.calibration_session_hashes),
            "field_sample_counts": dict(self.field_sample_counts),
            "fit_code_hash": self.fit_code_hash,
            "development_only": self.development_only,
        }

    @classmethod
    def from_artifact_wire(cls, data: Mapping[str, Any]) -> "FittedPerceptionErrorProfile":
        """wire から development_only=True の profile のみをロードする。

        development_only=False の wire は fail-closed として拒否する。
        formal profile はストア検証経路（from_store_artifact）のみで取得できる。
        """
        return cls._from_artifact_mapping(data, factory_token=None)

    @classmethod
    def _from_artifact_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        factory_token: object | None,
        calibration_descriptor_hash: str = "",
    ) -> "FittedPerceptionErrorProfile":
        """artifact envelopeを検証し、許可された経路だけformal tokenを渡す。

        wire に含まれない calibration_descriptor_hash は store 検証経路だけが渡し、
        「どの producer descriptor に束縛された profile か」を下流へ伝えます。
        """
        expected = {
            "schema_version", "profile", "profile_hash",
            "calibration_session_hashes", "field_sample_counts", "fit_code_hash",
            "development_only",
        }
        if not isinstance(data, Mapping) or set(data) != expected:
            raise ValueError("calibration artifact fields do not match schema")
        if data["schema_version"] != CALIBRATION_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported calibration artifact schema")
        if type(data["development_only"]) is not bool:
            raise ValueError("calibration artifact development_only must be bool")
        if data["development_only"] is False and factory_token is not _FORMAL_FACTORY_TOKEN:
            raise FormalVerdictPromotionError(
                "formal calibration profile cannot be loaded from raw wire; "
                "use the artifact store verification path"
            )
        profile = PerceptionErrorProfile.from_wire(data["profile"])
        if profile.profile_hash != data["profile_hash"]:
            raise HashMismatchError("calibration artifact profile hash mismatch")
        fitted = cls(
            **profile.to_wire(),
            calibration_session_hashes=data["calibration_session_hashes"],
            field_sample_counts=data["field_sample_counts"],
            fit_code_hash=data["fit_code_hash"],
            development_only=data["development_only"],
            calibration_descriptor_hash=calibration_descriptor_hash,
            _factory_token=factory_token,
        )
        if fitted.to_artifact_wire() != dict(data):
            raise HashMismatchError("calibration artifact failed canonical reconstruction")
        return fitted

    @classmethod
    def _from_verified_bytes(
        cls, data_bytes: bytes, expected_sha256: str
    ) -> "FittedPerceptionErrorProfile":
        """自己申告hash付きbytesを検査する。formal tokenは発行しない。"""
        actual = sha256_hex(data_bytes)
        if actual != expected_sha256:
            raise HashMismatchError(
                f"calibration artifact content hash mismatch: expected {expected_sha256!r}, got {actual!r}"
            )
        try:
            data: Any = _json.loads(data_bytes.decode("utf-8"))
        except (UnicodeDecodeError, _json.JSONDecodeError) as exc:
            raise ValueError(f"calibration artifact is not valid JSON: {exc}") from exc
        return cls._from_artifact_mapping(data, factory_token=None)

    @classmethod
    def from_store_artifact(
        cls,
        store: ArtifactStore,
        *,
        descriptors: Sequence[ArtifactDescriptor],
        expected_calibration_identity_hash: str,
    ) -> "FittedPerceptionErrorProfile":
        """producer descriptor chain に束縛された calibration profile だけを formal 化する。

        呼び出し元が用意した store/ref の組み合わせだけでは formal token を発行しません。
        producer が発行した `perception_calibration_profile` descriptor（親 DAG 込み）と、
        その descriptor identity を「別経路で保証した期待値」の両方を要求し、
        descriptor の file ref・raw profile・provenance の subject hashes まで
        全て store 上の実バイト列と exact 一致することを確認してから復元します。

        expected_calibration_identity_hash は descriptor の files（＝envelope の sha256）
        まで covered なので、development_only だけを書き換えた artifact は必ずここで落ちます。
        """
        if not isinstance(store, ArtifactStore):
            raise TypeError("formal calibration profile requires an ArtifactStore")
        nodes = tuple(descriptors) if descriptors is not None else ()
        if not nodes or not all(isinstance(node, ArtifactDescriptor) for node in nodes):
            raise TypeError(
                "formal calibration profile requires producer ArtifactDescriptor objects"
            )
        _require_sha256(
            expected_calibration_identity_hash, "expected_calibration_identity_hash"
        )
        try:
            validate_artifact_dag(nodes)
        except ArtifactDagValidationError as exc:
            raise FormalVerdictPromotionError(
                f"calibration descriptor DAG is not valid: {exc}"
            ) from exc
        matched = [
            node for node in nodes if node.node_kind == CALIBRATION_PROFILE_NODE_KIND
        ]
        if len(matched) != 1:
            raise FormalVerdictPromotionError(
                "calibration commit must contain exactly one "
                f"{CALIBRATION_PROFILE_NODE_KIND} descriptor"
            )
        node = matched[0]
        if node.identity_hash != expected_calibration_identity_hash:
            raise FormalVerdictPromotionError(
                "calibration descriptor identity does not match the expected "
                "calibration commit identity"
            )
        by_identity = {other.identity_hash: other for other in nodes}
        sources = [
            by_identity[parent.identity_hash]
            for parent in node.parents
            if by_identity[parent.identity_hash].node_kind == "source_descriptor"
        ]
        if len(sources) != 1 or len(node.parents) != 1:
            raise FormalVerdictPromotionError(
                "calibration descriptor must declare exactly one capture source parent"
            )
        if not sources[0].files:
            raise FormalVerdictPromotionError(
                "calibration capture source declares no manifest file"
            )
        # 親 capture source が参照する実ファイルも store 上に存在することを要求する。
        for parent_file in sources[0].files:
            if not store.verify(parent_file).ok:
                raise FormalVerdictPromotionError(
                    "calibration capture source file is missing from the artifact store"
                )
        files: dict[str, ArtifactRef] = {}
        for file_ref in node.files:
            name = file_ref.logical_id.rsplit("/", 1)[-1]
            if name in files:
                raise FormalVerdictPromotionError(
                    f"calibration descriptor declares duplicate {name!r} files"
                )
            files[name] = file_ref
        if set(files) != set(_CALIBRATION_FILE_NAMES):
            raise FormalVerdictPromotionError(
                "calibration descriptor must declare exactly "
                f"{sorted(_CALIBRATION_FILE_NAMES)}"
            )
        envelope = _load_verified_json(store, files["profile.artifact.json"])
        fitted = cls._from_artifact_mapping(
            envelope,
            factory_token=_FORMAL_FACTORY_TOKEN,
            calibration_descriptor_hash=node.identity_hash,
        )
        metadata = dict(node.identity_metadata)
        if (
            metadata.get("profile_hash") != fitted.profile_hash
            or metadata.get("fit_code_hash") != fitted.fit_code_hash
        ):
            raise HashMismatchError(
                "calibration descriptor identity metadata does not match the stored profile"
            )
        if _load_verified_json(store, files["profile.json"]) != fitted.to_wire():
            raise HashMismatchError(
                "calibration raw profile does not match the artifact envelope"
            )
        provenance = _load_verified_json(store, files["provenance.json"])
        if (
            set(provenance) != {"schema_version", "profile_artifact", "subject_hashes"}
            or provenance["schema_version"] != CALIBRATION_PROVENANCE_SCHEMA_VERSION
            or provenance["profile_artifact"] != fitted.to_artifact_wire()
            or provenance["subject_hashes"] != metadata.get("subject_hashes")
        ):
            raise HashMismatchError(
                "calibration provenance does not match the descriptor subject hashes"
            )
        return fitted

    @classmethod
    def from_calibration_commit(
        cls,
        store: ArtifactStore,
        *,
        commit_logical_id: str,
        expected_calibration_identity_hash: str,
    ) -> "FittedPerceptionErrorProfile":
        """producer の calibration commit だけを入口に formal profile を復元する。

        任意の root/logical ID を渡して profile を読む経路を塞ぐための境界です。
        store に置かれた `perception_calibration_commit.v1` を exact ref で開き、
        commit が記録した descriptor object を読み直して DAG を復元してから
        from_store_artifact へ委譲します。commit が申告する profile_descriptor_hash が
        呼び出し元の期待値と違えば、その時点で fail-closed になります。
        """
        if not isinstance(store, ArtifactStore):
            raise TypeError("formal calibration profile requires an ArtifactStore")
        if type(commit_logical_id) is not str or not commit_logical_id:
            raise ValueError("calibration commit logical id must be a non-empty string")
        _require_sha256(
            expected_calibration_identity_hash, "expected_calibration_identity_hash"
        )
        try:
            commit_ref = store.resolve(commit_logical_id)
        except (ArtifactStoreError, OSError) as exc:
            raise HashMismatchError(f"calibration commit store read failed: {exc}") from exc
        if commit_ref is None:
            raise FormalVerdictPromotionError(
                f"calibration commit not found in store: {commit_logical_id!r}"
            )
        payload = _load_verified_json(store, commit_ref)
        if (
            set(payload) != {"schema_version", "run_key", "profile_descriptor_hash", "refs"}
            or payload["schema_version"] != CALIBRATION_COMMIT_SCHEMA_VERSION
        ):
            raise FormalVerdictPromotionError("calibration commit schema is not supported")
        if payload["profile_descriptor_hash"] != expected_calibration_identity_hash:
            raise FormalVerdictPromotionError(
                "calibration commit profile descriptor hash does not match the expected identity"
            )
        refs_wire = payload["refs"]
        if not isinstance(refs_wire, list) or not refs_wire:
            raise FormalVerdictPromotionError("calibration commit refs must be a non-empty list")
        descriptors: list[ArtifactDescriptor] = []
        seen: set[str] = set()
        for wire in refs_wire:
            ref = ArtifactRef.from_wire(wire)
            if ref.logical_id in seen:
                raise FormalVerdictPromotionError(
                    "calibration commit contains duplicate logical ids"
                )
            seen.add(ref.logical_id)
            if not ref.logical_id.startswith(_DESCRIPTOR_LOGICAL_PREFIX):
                continue
            descriptor = ArtifactDescriptor.from_wire(_load_verified_json(store, ref))
            if (
                ref.logical_id
                != f"{_DESCRIPTOR_LOGICAL_PREFIX}{descriptor.identity_hash}.json"
            ):
                raise FormalVerdictPromotionError(
                    "calibration commit descriptor object is not content-addressed by identity"
                )
            descriptors.append(descriptor)
        if not descriptors:
            raise FormalVerdictPromotionError(
                "calibration commit does not stage any artifact descriptor"
            )
        return cls.from_store_artifact(
            store,
            descriptors=tuple(descriptors),
            expected_calibration_identity_hash=expected_calibration_identity_hash,
        )

    @property
    def artifact_hash(self) -> str:
        return canonical_hash(self.to_artifact_wire())

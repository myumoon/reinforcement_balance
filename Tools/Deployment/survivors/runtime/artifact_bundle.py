"""Runtime artifact bundle: 03-05 combat package と 02-03 ItemSelector package の検証済みロード。

事前固定の trust anchor（別チャネルで配布される Ed25519 署名付き trusted release
registry）に登録された exact descriptor identity と一致する場合にだけ live-capable
bundle を組み立てる。combat model は 03-05 が発行する `manifest.json` + `model.pt`
（`{model_config, model_state_dict}`）契約だけを読み、SB3 `.zip` や独立
`vecnormalize.pkl` は要求も推測もしない。

やさしい説明:
    「学習で作ったAIの入った箱（package）を、実行側で開けてよいか確かめてから開ける」
    ための入口です。確認は3種類あります。(1) 箱の説明書どおりの中身か（ハッシュ照合）、
    (2) その箱が“正規リリース一覧”に載っているか（trust anchor）、(3) いま動かそうと
    しているPCの実際のスペックが、そのリリースが想定したスペックと一致するか。
    どれか1つでも合わなければ起動しません。

    重要な設計方針として、呼び出し元がその場で自作した package / descriptor /
    ArtifactStore だけでは絶対に live 起動できません。別途配布された署名済み一覧に
    その identity が載っていることが必須です。さらに「誰の署名を信じるか」も呼び出し
    元は選べません。`RuntimeBundle.load()` は trust anchor を引数で受け取らず、常に
    `load_production_trust_anchor()` を通して source 固定鍵
    `PRODUCTION_TRUST_ANCHOR_PUBLIC_KEYS` でのみ registry を検証します。
    また pickle 等の任意コード実行を伴う読み込みは一切使わず、JSON と
    `torch.load(weights_only=True)` だけを使います。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch as th
import torch.nn as nn
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from reinbalance_survivors_contracts.artifact_dag import (
    ArtifactDagValidationError,
    validate_formal_runtime_dag,
)
from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    ArtifactRef,
    is_sha256_hex,
)
from reinbalance_survivors_contracts.artifact_store import ArtifactStore, ArtifactStoreError
from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.target_action import ActionSemantics, TargetProfileRef
from reinbalance_survivors_contracts.ui_policy import NonModelUiPolicyConfigV1

from .item_selector_runtime import ItemSelectorRuntimeError, OnnxItemSelector

# --- 03-05 deployable student package 契約 -----------------------------------
# 正本は Tools/Training/games/survivors/deployable_policy_package.py。
# Training を import せず、同じ wire 契約を Deployment 側で独立に再検証する。
COMBAT_PACKAGE_SCHEMA_VERSION = "survivors.deployable_policy_package.v1"
COMBAT_MANIFEST_FILENAME = "manifest.json"
COMBAT_MODEL_FILENAME = "model.pt"
COMBAT_PACKAGE_FILES: tuple[str, ...] = (COMBAT_MANIFEST_FILENAME, COMBAT_MODEL_FILENAME)

_COMBAT_MANIFEST_KEYS = frozenset({
    "schema_version",
    "checkpoint_sha256",
    "model_sha256",
    "deploy_schema_hash",
    "model_config",
    "development_only",
    "formal_student_eligible",
    "formal_dependency_identities",
    "files",
})
_COMBAT_MANIFEST_HASH_KEYS = ("checkpoint_sha256", "model_sha256", "deploy_schema_hash")
_COMBAT_MODEL_CONFIG_KEYS = frozenset({"observation_dim", "action_dim", "hidden_dim"})
_COMBAT_MODEL_PAYLOAD_KEYS = frozenset({"model_config", "model_state_dict"})
_FORMAL_DEPENDENCY_KEYS = frozenset({"fidelity_verdict", "perception_profile"})

# 9 direction/idle 固定。2-action の別 model を move decision に使わせない。
REQUIRED_ACTION_DIM = 9
# combat tick は 15 Hz 固定 (plan 05-01 / 00-03 action contract)。
REQUIRED_DECISION_HZ = 15

BUNDLE_DEVELOPMENT_SENTINEL = "golden_fixture"
BUNDLE_FORMAL_SENTINEL = "formal"

_RUNTIME_NODE_KIND = "runtime_bundle"
_COMBAT_RELEASE_NODE_KIND = "combat_student_release"
_ITEM_SELECTOR_RELEASE_NODE_KIND = "item_selector_release"
_PERCEPTION_VERDICT_NODE_KIND = "perception_final_verdict"

# --- trust anchor 契約 --------------------------------------------------------
TRUST_REGISTRY_SCHEMA_VERSION = "survivors.trusted_release_registry.v1"
# registry の所在は別チャネル（配備時の環境変数）で与える。repo 内の相対 path を
# 既定にしないことで、repo を書ける攻撃者が registry を差し替える経路を作らない。
TRUST_REGISTRY_PATH_ENV = "REINBALANCE_SURVIVORS_TRUST_REGISTRY"
TRUST_REGISTRY_SIGNATURE_SUFFIX = ".sig"

# 本番 release 署名鍵（Ed25519 公開鍵 32 byte の hex）。read-only な source 側に
# 固定する。まだ本番鍵を発行していないため空であり、この状態では
# `load_production_trust_anchor()` は必ず失敗する（fail-closed の既定）。
PRODUCTION_TRUST_ANCHOR_PUBLIC_KEYS: tuple[str, ...] = ()

_REGISTRY_DOCUMENT_KEYS = frozenset({"schema_version", "registry_id", "releases"})
_RELEASE_ENTRY_HASH_KEYS = (
    "runtime_bundle_identity_hash",
    "combat_student_release_identity_hash",
    "item_selector_release_identity_hash",
    "perception_final_verdict_identity_hash",
    "combat_package_manifest_hash",
    "item_selector_manifest_hash",
    "deploy_schema_hash",
    "action_semantics_hash",
    "target_profile_ref_hash",
    "target_capability_hash",
    "choice_capability_hash",
)
_RELEASE_ENTRY_KEYS = frozenset(_RELEASE_ENTRY_HASH_KEYS) | {"host_profile"}

# plan 05-01 タスク2 が startup report と一致を要求する実 hardware / target 値。
_HOST_PROFILE_STR_KEYS = (
    "os_build",
    "gpu_name",
    "driver_version",
    "cuda_version",
    "capture_backend",
)
_HOST_PROFILE_INT_KEYS = ("decision_hz", "key_lease_duration_ms")
_HOST_PROFILE_KEYS = frozenset(_HOST_PROFILE_STR_KEYS) | frozenset(_HOST_PROFILE_INT_KEYS)

__all__ = [
    "COMBAT_PACKAGE_SCHEMA_VERSION",
    "COMBAT_PACKAGE_FILES",
    "REQUIRED_ACTION_DIM",
    "REQUIRED_DECISION_HZ",
    "TRUST_REGISTRY_SCHEMA_VERSION",
    "TRUST_REGISTRY_PATH_ENV",
    "PRODUCTION_TRUST_ANCHOR_PUBLIC_KEYS",
    "BundleLoadError",
    "TrustAnchorError",
    "CombatGruPolicy",
    "CombatPolicy",
    "HostRuntimeProfile",
    "HostProfileMismatch",
    "TrustedRelease",
    "TrustAnchor",
    "RuntimeBundle",
    "load_production_trust_anchor",
]


class BundleLoadError(ValueError):
    """artifact bundle の schema / hash / trust / capability 検証失敗。

    やさしい説明: 「この成果物では起動してはいけない」と判断したときに投げる例外です。
    起動前に取り違えや改ざんを止めるためのもので、この例外が出た bundle は live 起動も
    shadow 起動もしてはいけません。
    """


class TrustAnchorError(BundleLoadError):
    """trusted release registry の署名・schema 検証失敗。

    やさしい説明: 「正規リリース一覧」そのものが本物だと確認できなかったときの例外です。
    一覧が信用できない以上、その中身も一切使いません。
    """


def _require_sha256(value: Any, label: str) -> str:
    """小文字 64 桁 SHA-256 でなければ BundleLoadError を送出する。

    やさしい説明: 指紋（ハッシュ）の書式チェックです。空文字や短縮値を identity の
    代用にさせないための入口検査で、`is_sha256_hex` の共有実装をそのまま使います。
    """
    if not is_sha256_hex(value):
        raise BundleLoadError(f"{label} must be a lowercase 64-hex sha256")
    return str(value)


def _positive_int(value: Any, label: str) -> int:
    """正の int だけを受理する。bool や float の混入を拒否する。

    やさしい説明: 次元や周波数に 0・負数・True/False・小数が紛れ込むのを防ぎます。
    Python では `True` が `int` の一種なので、型を厳密に見る必要があります。
    """
    if type(value) is not int or value <= 0:
        raise BundleLoadError(f"{label} must be a positive int")
    return value


def _resolve_package_file(root: Path, name: str) -> Path:
    """package 内の exact file path を root escape / symlink を拒否して解決する。

    やさしい説明: 「箱の中の、この名前のファイル」だけを受け付けます。`..` を含む
    相対パスや symlink を許すと、箱の外にある別のファイルを読ませられてしまうため、
    実体パスが箱の直下にあることまで確認します。
    """
    if "/" in name or "\\" in name or name in ("", ".", ".."):
        raise BundleLoadError(f"package file name {name!r} must be a plain file name")
    candidate = root / name
    try:
        if candidate.is_symlink():
            raise BundleLoadError(f"package file {name!r} must not be a symlink")
        if not candidate.is_file():
            raise BundleLoadError(f"package file {name!r} is missing")
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise BundleLoadError(f"cannot resolve package file {name!r}: {exc}") from exc
    if resolved.parent != resolved_root:
        raise BundleLoadError(f"package file {name!r} escapes the package root")
    return resolved


def _verified_bytes(root: Path, name: str, expected_sha256: str) -> tuple[Path, bytes]:
    """宣言 hash と一致する場合だけ file の実体 path と内容を返す。

    やさしい説明: モデルを読み込む前に必ず通す関門です。ファイルの中身から計算した
    指紋が説明書の指紋と一致しなければ、その時点で止めます。未検証のファイルを
    deserialize しないことが、差し替えによる任意コード実行を防ぐ唯一の境界です。
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
    return path, payload


# --- hardware / target profile ------------------------------------------------


@dataclass(frozen=True)
class HostProfileMismatch:
    """host profile 1 field 分の期待値と実測値の食い違い。

    やさしい説明: 「どの項目が」「何を期待していて」「実際は何だったか」を持つ小さな
    記録です。存在チェックだけでは分からない具体的な差分を、そのままエラー文へ出せます。
    """

    field_name: str
    expected: Any
    actual: Any

    def describe(self) -> str:
        """人が読める 1 行の差分表現を返す。"""
        return f"{self.field_name}: expected {self.expected!r}, got {self.actual!r}"


@dataclass(frozen=True)
class HostRuntimeProfile:
    """runtime 起動ホストの実 OS / GPU / driver / capture 値と時間契約。

    やさしい説明: `TargetProfileRef` は build や save の「識別子」しか持たないため、
    OS ビルド名や GPU 名といった生の値を比較できません。この型がその生値を運びます。
    起動時に実際のマシンから取得した値をここに入れ、信頼済みリリースが想定していた
    値と 1 項目ずつ突き合わせます。
    """

    os_build: str
    gpu_name: str
    driver_version: str
    cuda_version: str
    capture_backend: str
    decision_hz: int
    key_lease_duration_ms: int

    def __post_init__(self) -> None:
        """全 field の型と範囲を fail-closed で検証する。

        やさしい説明: 空文字や 0 を「未設定のまま通す」ことがないようにします。
        decision_hz は plan が 15 Hz に固定しているため、その値だけを許可します。
        """
        for name in _HOST_PROFILE_STR_KEYS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise BundleLoadError(f"host profile {name} must be a non-empty string")
        for name in _HOST_PROFILE_INT_KEYS:
            _positive_int(getattr(self, name), f"host profile {name}")
        if self.decision_hz != REQUIRED_DECISION_HZ:
            raise BundleLoadError(
                f"host profile decision_hz must be {REQUIRED_DECISION_HZ}, got {self.decision_hz}"
            )

    def to_wire(self) -> dict[str, Any]:
        """canonical JSON 化できる dict 表現を返す。"""
        return {name: getattr(self, name) for name in sorted(_HOST_PROFILE_KEYS)}

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "HostRuntimeProfile":
        """exact field 集合を要求して wire 表現から復元する。

        やさしい説明: 項目の過不足があれば読み込みません。知らない項目を黙って
        無視すると、比較したつもりの項目が実は比較されていない状態が起こります。
        """
        if not isinstance(data, Mapping) or set(data) != _HOST_PROFILE_KEYS:
            raise BundleLoadError("host profile fields mismatch")
        return cls(**{name: data[name] for name in _HOST_PROFILE_KEYS})

    def mismatches(self, expected: "HostRuntimeProfile") -> tuple[HostProfileMismatch, ...]:
        """期待 profile との差分を field 単位で列挙する。

        やさしい説明: 「値が入っているか」ではなく「同じ値か」を比べます。GPU を
        載せ替えた、driver を上げた、capture 方式を変えた、といった変化をここで
        検出し、どの項目が違うのかをそのまま報告できるようにします。
        """
        return tuple(
            HostProfileMismatch(
                field_name=name,
                expected=getattr(expected, name),
                actual=getattr(self, name),
            )
            for name in sorted(_HOST_PROFILE_KEYS)
            if getattr(self, name) != getattr(expected, name)
        )


# --- trusted release registry -------------------------------------------------


@dataclass(frozen=True)
class TrustedRelease:
    """信頼 root に登録された 1 リリース分の exact identity 集合と期待 host profile。

    やさしい説明: 「このバンドルだけを本番で起動してよい」という許可証にあたります。
    どの descriptor identity か、どの manifest 指紋か、どのハードウェアを想定して
    いるかが、すべて具体値で書かれています。実行時にはこの値と実物を突き合わせます。
    """

    runtime_bundle_identity_hash: str
    combat_student_release_identity_hash: str
    item_selector_release_identity_hash: str
    perception_final_verdict_identity_hash: str
    combat_package_manifest_hash: str
    item_selector_manifest_hash: str
    deploy_schema_hash: str
    action_semantics_hash: str
    target_profile_ref_hash: str
    target_capability_hash: str
    choice_capability_hash: str
    host_profile: HostRuntimeProfile

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "TrustedRelease":
        """registry の 1 entry を exact field 集合で検証して読む。

        やさしい説明: 許可証の項目がそろっているか、指紋がすべて正しい書式かを見ます。
        1 つでも欠けたり増えたりしていれば、その許可証は使いません。
        """
        if not isinstance(data, Mapping) or set(data) != _RELEASE_ENTRY_KEYS:
            raise TrustAnchorError("trusted release entry fields mismatch")
        values: dict[str, Any] = {
            name: _require_sha256(data[name], f"trusted release {name}")
            for name in _RELEASE_ENTRY_HASH_KEYS
        }
        values["host_profile"] = HostRuntimeProfile.from_wire(data["host_profile"])
        return cls(**values)


@dataclass(frozen=True)
class TrustAnchor:
    """署名検証済み trusted release registry。live 起動を許可する唯一の根拠。

    やさしい説明: 「別のルートで配られた正規リリース一覧」を読み込んだ結果です。
    この一覧は Ed25519 署名で守られており、署名鍵は runtime の source 側に固定されて
    います。呼び出し元がその場で作った package や descriptor をいくら渡しても、この
    一覧に identity が載っていなければ live 起動にはなりません。
    """

    registry_id: str
    signer_public_key: str
    source_path: str
    releases: Mapping[str, TrustedRelease]

    def __contains__(self, runtime_bundle_identity_hash: str) -> bool:
        """runtime bundle identity が登録済みかを返す。"""
        return runtime_bundle_identity_hash in self.releases

    def release_for(self, runtime_bundle_identity_hash: str) -> TrustedRelease:
        """exact identity 一致で TrustedRelease を引く。未登録なら拒否する。

        やさしい説明: 一覧に載っていない identity は、ここで必ず止まります。部分一致や
        「空でなければ通す」ような緩い判定は行いません。
        """
        _require_sha256(runtime_bundle_identity_hash, "runtime bundle identity hash")
        release = self.releases.get(runtime_bundle_identity_hash)
        if release is None:
            raise BundleLoadError(
                "runtime bundle identity "
                f"{runtime_bundle_identity_hash!r} is not registered in trusted release "
                f"registry {self.registry_id!r}; caller-produced artifacts cannot become "
                "live-eligible"
            )
        return release

    @classmethod
    def load(
        cls,
        registry_path: Path | str,
        *,
        verification_public_keys: Sequence[str],
        signature_path: Path | str | None = None,
    ) -> "TrustAnchor":
        """detached Ed25519 署名を検証してから registry JSON を読む。

        やさしい説明: 手順は「署名を確かめる → 中身を読む」の順で、逆にはしません。
        署名が合わない一覧は 1 byte も解釈しません。読み込みは JSON だけで、pickle
        などの任意コード実行につながる形式は一切使いません。

        `verification_public_keys` は Ed25519 公開鍵 32 byte の hex 文字列。本番経路は
        `load_production_trust_anchor()` が source 固定の鍵を渡します。
        """
        payload_path = Path(registry_path)
        sig_path = (
            Path(signature_path)
            if signature_path is not None
            else payload_path.with_name(payload_path.name + TRUST_REGISTRY_SIGNATURE_SUFFIX)
        )
        keys = tuple(verification_public_keys)
        if not keys:
            raise TrustAnchorError(
                "no trust anchor verification key was supplied; refusing to trust registry "
                f"{str(payload_path)!r}"
            )

        try:
            payload = payload_path.read_bytes()
            signature_hex = sig_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise TrustAnchorError(f"cannot read trusted release registry: {exc}") from exc

        try:
            signature = bytes.fromhex(signature_hex)
        except ValueError as exc:
            raise TrustAnchorError("trusted release registry signature is not hex") from exc

        signer = cls._verify_signature(payload, signature, keys)
        document = cls._decode_document(payload, payload_path)
        releases = cls._decode_releases(document["releases"])
        return cls(
            registry_id=str(document["registry_id"]),
            signer_public_key=signer,
            source_path=str(payload_path),
            releases=releases,
        )

    @staticmethod
    def _verify_signature(
        payload: bytes, signature: bytes, keys: Sequence[str]
    ) -> str:
        """固定鍵集合のいずれかで detached 署名を検証し、通った鍵を返す。

        やさしい説明: 配られた一覧が、こちらが知っている発行者の鍵で署名されている
        ことを確かめます。どの鍵でも検証できなければ、その一覧は使いません。
        """
        for key_hex in keys:
            try:
                key_bytes = bytes.fromhex(key_hex)
                public_key = Ed25519PublicKey.from_public_bytes(key_bytes)
            except ValueError:
                # 鍵の書式不正は「その鍵では検証できない」として次の鍵を試す。
                continue
            try:
                public_key.verify(signature, payload)
            except InvalidSignature:
                continue
            return key_hex
        raise TrustAnchorError(
            "trusted release registry signature does not verify against any pinned key"
        )

    @staticmethod
    def _decode_document(payload: bytes, payload_path: Path) -> dict[str, Any]:
        """署名対象 byte 列を JSON として読み、canonical 形と schema を固定する。

        やさしい説明: 署名した byte 列と、解釈した中身が 1 対 1 に対応することまで
        確認します。空白や重複キーで見た目を変えられると、「署名された内容」と
        「実際に使う内容」がずれる余地が生まれるため、正規形と完全一致を要求します。
        """
        try:
            document = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise TrustAnchorError(f"trusted release registry is not valid JSON: {exc}") from exc
        if not isinstance(document, dict) or set(document) != _REGISTRY_DOCUMENT_KEYS:
            raise TrustAnchorError("trusted release registry fields mismatch")
        if document.get("schema_version") != TRUST_REGISTRY_SCHEMA_VERSION:
            raise TrustAnchorError("unsupported trusted release registry schema_version")
        registry_id = document.get("registry_id")
        if not isinstance(registry_id, str) or not registry_id:
            raise TrustAnchorError("trusted release registry_id must be a non-empty string")
        try:
            canonical = canonical_json_bytes(document)
        except ValueError as exc:
            raise TrustAnchorError(
                f"trusted release registry is not canonical JSON compatible: {exc}"
            ) from exc
        if canonical != payload:
            raise TrustAnchorError(
                "trusted release registry bytes are not canonical JSON; the signed bytes "
                f"and the parsed document must correspond exactly ({str(payload_path)!r})"
            )
        return document

    @staticmethod
    def _decode_releases(entries: Any) -> dict[str, TrustedRelease]:
        """release 配列を identity 索引へ変換し、重複登録を拒否する。

        やさしい説明: 同じ identity が 2 回出てくると、どちらの許可証が効くのか
        曖昧になります。曖昧なまま起動しないよう、重複はエラーにします。
        """
        if not isinstance(entries, list) or not entries:
            raise TrustAnchorError("trusted release registry must list at least one release")
        releases: dict[str, TrustedRelease] = {}
        for entry in entries:
            release = TrustedRelease.from_wire(entry)
            identity = release.runtime_bundle_identity_hash
            if identity in releases:
                raise TrustAnchorError(
                    f"duplicate trusted release entry for runtime bundle {identity!r}"
                )
            releases[identity] = release
        return releases


def load_production_trust_anchor(registry_path: Path | str | None = None) -> TrustAnchor:
    """source 固定の本番鍵と、別チャネル指定の registry path で trust anchor を作る。

    やさしい説明: 本番起動用の入口です。署名鍵は書き換えにくい source 側の定数
    `PRODUCTION_TRUST_ANCHOR_PUBLIC_KEYS` に固定し、一覧ファイルの場所は配備時の
    環境変数で渡します。まだ本番鍵を発行していないため、既定ではこの関数は必ず
    失敗します（鍵が無いのに起動してしまうより、起動しないほうが安全なため）。
    """
    if not PRODUCTION_TRUST_ANCHOR_PUBLIC_KEYS:
        raise TrustAnchorError(
            "no production trust anchor public key is pinned in "
            "artifact_bundle.PRODUCTION_TRUST_ANCHOR_PUBLIC_KEYS; live startup is refused"
        )
    resolved = registry_path if registry_path is not None else os.environ.get(
        TRUST_REGISTRY_PATH_ENV
    )
    if not resolved:
        raise TrustAnchorError(
            f"trusted release registry path is not set; export {TRUST_REGISTRY_PATH_ENV}"
        )
    return TrustAnchor.load(
        resolved, verification_public_keys=PRODUCTION_TRUST_ANCHOR_PUBLIC_KEYS
    )


# --- combat policy ------------------------------------------------------------


class CombatGruPolicy(nn.Module):
    """03-05 `DeployableCombatPolicy` と同一 topology の推論専用 GRU policy。

    やさしい説明: 学習側で作られた小さなAIと「まったく同じ部品構成」を、実行側で
    独立に組み立てたものです。部品名（recurrent / actor / value）と形が完全に一致
    しないと重みの読み込みが失敗するので、モデルの取り違えをそこで検出できます。

    学習側は Tools/Training にありますが、依存方向の規則により import できません。
    そのため構造だけを写した実装をここに持ちます。VecNormalize は 03-05 契約に
    存在しないため、この経路にも一切登場しません。
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        """正の次元から GRUCell と actor / value head を組み立てる。

        やさしい説明: 観測を受け取る再帰層と、行動を選ぶ頭、状態価値を出す頭の 3 つを
        作ります。actor と value は同じ再帰状態を共有します。
        """
        super().__init__()
        self.observation_dim = _positive_int(observation_dim, "model_config.observation_dim")
        self.action_dim = _positive_int(action_dim, "model_config.action_dim")
        self.hidden_dim = _positive_int(hidden_dim, "model_config.hidden_dim")
        self.recurrent = nn.GRUCell(self.observation_dim, self.hidden_dim)
        self.actor = nn.Linear(self.hidden_dim, self.action_dim)
        self.value = nn.Linear(self.hidden_dim, 1)

    def initial_hidden_state(self, batch_size: int = 1) -> th.Tensor:
        """episode 先頭で使うゼロ hidden state を返す。

        やさしい説明: 新しいエピソードが始まったら記憶を白紙に戻すための初期値です。
        """
        return th.zeros(
            (_positive_int(batch_size, "batch_size"), self.hidden_dim), dtype=th.float32
        )

    def step(self, observation: th.Tensor, hidden: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """1 timestep 進めて action logits・state value・新 hidden state を返す。

        やさしい説明: 観測を 1 つ受け取り、次の行動の点数と、更新後の記憶を返します。
        combat session は毎 tick これを呼びます。形は [batch, dim] を想定します。
        """
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise BundleLoadError("combat observation shape mismatch")
        if hidden.ndim != 2 or hidden.shape != (observation.shape[0], self.hidden_dim):
            raise BundleLoadError("combat hidden state shape mismatch")
        new_hidden = self.recurrent(observation, hidden)
        return self.actor(new_hidden), self.value(new_hidden).squeeze(-1), new_hidden


@dataclass(frozen=True)
class CombatPolicy:
    """検証済み combat model と、その宣言次元をまとめた runtime コンテナ。

    やさしい説明: 「読み込みと検証が全部終わったAI」を持ち歩くための入れ物です。
    combat session はこの型だけを受け取り、package の検証をやり直しません。
    """

    model: CombatGruPolicy
    observation_dim: int
    action_dim: int
    hidden_dim: int

    @property
    def hidden_state_shape(self) -> tuple[int, int]:
        """GRU hidden state の shape `[1, hidden_dim]` を返す。"""
        return (1, self.hidden_dim)

    def initial_hidden_state(self, batch_size: int = 1) -> th.Tensor:
        """episode 先頭のゼロ hidden state を返す。"""
        return self.model.initial_hidden_state(batch_size)


def _read_combat_manifest(root: Path) -> dict[str, Any]:
    """03-05 combat package manifest を読み、9 key 契約と eligibility を検証する。

    やさしい説明: 箱の説明書だけを先に読みます。ここは JSON しか読まないので、この
    段階で任意コードが動く余地はありません。開発用（development_only）や正式非対象
    （formal_student_eligible=false）の成果物は、この時点で弾きます。
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
        raise BundleLoadError("development_only combat package cannot start a live bundle")
    if manifest.get("formal_student_eligible") is not True:
        raise BundleLoadError("formal_student_eligible=false combat package rejected")
    for name in _COMBAT_MANIFEST_HASH_KEYS:
        _require_sha256(manifest.get(name), f"combat manifest {name}")

    identities = manifest.get("formal_dependency_identities")
    if not isinstance(identities, dict) or set(identities) != _FORMAL_DEPENDENCY_KEYS:
        raise BundleLoadError("combat package formal dependency identities are missing")
    for name in sorted(_FORMAL_DEPENDENCY_KEYS):
        _require_sha256(identities.get(name), f"formal_dependency_identities.{name}")

    if manifest.get("files") != list(COMBAT_PACKAGE_FILES):
        raise BundleLoadError("combat package file list mismatch")
    try:
        actual_entries = {entry.name for entry in root.iterdir()}
    except OSError as exc:
        raise BundleLoadError(f"cannot list combat package directory: {exc}") from exc
    if actual_entries != set(COMBAT_PACKAGE_FILES):
        raise BundleLoadError("combat package directory contents mismatch")

    model_config = manifest.get("model_config")
    if not isinstance(model_config, dict) or set(model_config) != _COMBAT_MODEL_CONFIG_KEYS:
        raise BundleLoadError("combat model_config fields mismatch")
    for key in sorted(_COMBAT_MODEL_CONFIG_KEYS):
        _positive_int(model_config.get(key), f"model_config.{key}")
    return manifest


def _load_combat_package(package_dir: Path) -> tuple[CombatPolicy, dict[str, Any]]:
    """combat package を hash 検証してから `model.pt` の重みだけを読み込む。

    やさしい説明: 順番が重要です。説明書を読む → ファイルの指紋を照合する →
    そのあとで初めてモデルを読む、という順を守ります。読み込みは
    `weights_only=True` に固定しており、テンソルと素の値以外は復元しません。
    つまり model.pt に仕込まれた任意コードが動く経路がありません。
    """
    root = Path(package_dir)
    try:
        is_missing = not root.is_dir()
    except OSError as exc:
        raise BundleLoadError(f"cannot inspect combat package directory: {exc}") from exc
    if is_missing:
        raise BundleLoadError(f"combat package directory not found: {root}")
    manifest = _read_combat_manifest(root)

    model_path, _ = _verified_bytes(root, COMBAT_MODEL_FILENAME, manifest["model_sha256"])
    try:
        payload = th.load(model_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001  # torch は多様な例外型を送出する
        raise BundleLoadError(f"cannot load combat model.pt: {exc}") from exc

    if not isinstance(payload, dict) or set(payload) != _COMBAT_MODEL_PAYLOAD_KEYS:
        raise BundleLoadError("combat model.pt structure mismatch")
    model_config = payload["model_config"]
    if not isinstance(model_config, dict) or set(model_config) != _COMBAT_MODEL_CONFIG_KEYS:
        raise BundleLoadError("combat model.pt model_config fields mismatch")
    if model_config != manifest["model_config"]:
        raise BundleLoadError("combat model.pt model_config does not match the manifest")

    model = CombatGruPolicy(
        observation_dim=model_config["observation_dim"],
        action_dim=model_config["action_dim"],
        hidden_dim=model_config["hidden_dim"],
    )
    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise BundleLoadError("combat model.pt model_state_dict must be a mapping")
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, KeyError, TypeError) as exc:
        raise BundleLoadError(f"combat model state_dict incompatible: {exc}") from exc
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    policy = CombatPolicy(
        model=model,
        observation_dim=model_config["observation_dim"],
        action_dim=model_config["action_dim"],
        hidden_dim=model_config["hidden_dim"],
    )
    return policy, manifest


# --- descriptor / store helpers ----------------------------------------------


def _descriptors_by_kind(
    descriptors: Sequence[ArtifactDescriptor],
) -> dict[str, ArtifactDescriptor]:
    """node_kind をキーに descriptor を引ける dict を作る。

    やさしい説明: 同じ種類の成果物が 2 つあると、どちらを起動するのか決まりません。
    曖昧な bundle は拒否します。
    """
    result: dict[str, ArtifactDescriptor] = {}
    for descriptor in descriptors:
        if descriptor.node_kind in result:
            raise BundleLoadError(
                f"duplicate artifact descriptor for node kind {descriptor.node_kind!r}"
            )
        result[descriptor.node_kind] = descriptor
    return result


def _verify_store_restore(
    store: ArtifactStore, descriptors: Sequence[ArtifactDescriptor]
) -> tuple[str, ...]:
    """全 descriptor の file を artifact store 上で restore 検証する。

    やさしい説明: 成果物の実体が保管庫にちゃんと存在し、壊れていないかを確認します。
    1 件でも欠損・破損していれば live 起動しません。
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
    if not is_sha256_hex(value):
        raise BundleLoadError(
            f"{descriptor.node_kind}.identity_metadata.{key} must be a lowercase 64-hex sha256"
        )
    return str(value)


def _require_exact(actual: Any, expected: Any, label: str) -> None:
    """厳密一致でなければ、両方の値を含むエラーを送出する。

    やさしい説明: 「入っているか」ではなく「同じか」を見る比較の共通処理です。
    エラー文に期待値と実際の値の両方を出して、原因を追えるようにします。
    """
    if actual != expected:
        raise BundleLoadError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


# --- runtime bundle -----------------------------------------------------------


@dataclass(frozen=True)
class RuntimeBundle:
    """検証済み combat policy + ItemSelector artifact の runtime コンテナ。

    やさしい説明: 起動時に「本番として動かしてよいか」を確定させ、その結論を
    `live_eligible` に固定して持ち回ります。golden fixture から作った bundle は
    必ず `live_eligible=False` になり、`assert_live_eligible()` が本番起動を止めます。
    この型は OS 入力（キーやマウス）には一切触れません。
    """

    development_only: bool
    live_eligible: bool
    combat_policy: CombatPolicy
    deploy_schema: DeployObsSchema
    deploy_schema_hash: str
    action_semantics: ActionSemantics
    ui_policy_config: NonModelUiPolicyConfigV1
    startup_report: dict[str, Any]
    host_profile: HostRuntimeProfile | None = None
    # ItemSelector は combat-only golden fixture では省略できる。
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

        やさしい説明: 開発用の成果物で本番を起動しようとしたら、ここで止めます。
        """
        if self.development_only or not self.live_eligible:
            raise BundleLoadError(
                "bundle is development_only / not live_eligible; formal artifacts and a "
                "trusted release registry entry are required for live startup"
            )

    @classmethod
    def from_golden_fixture(
        cls,
        combat_policy: CombatPolicy,
        *,
        ui_policy_config: NonModelUiPolicyConfigV1 | None = None,
        item_selector: Any = None,
        deploy_schema: DeployObsSchema | None = None,
        action_semantics: ActionSemantics | None = None,
    ) -> "RuntimeBundle":
        """golden fixture 用の development_only=True / live_eligible=False bundle を返す。

        やさしい説明: 正式な成果物がなくても loader / session / scheduler のテストを
        回せるようにするための入口です。ここから作った bundle は本番起動できません。
        """
        if not isinstance(combat_policy, CombatPolicy):
            raise BundleLoadError("combat_policy must be a CombatPolicy")
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
            host_profile=None,
            _item_selector=item_selector,
        )

    @classmethod
    def load(
        cls,
        *,
        combat_package_dir: Path | str,
        item_selector_dir: Path | str,
        artifact_store: ArtifactStore,
        descriptors: Sequence[ArtifactDescriptor],
        target_profile: TargetProfileRef,
        host_profile: HostRuntimeProfile,
        trust_registry_path: Path | str | None = None,
        action_semantics: ActionSemantics | None = None,
    ) -> "RuntimeBundle":
        """live-capable bundle を、信頼 root への exact 一致を前提にロードする。

        やさしい説明: 起動可否を次の順で判定します。
        (0) 信頼 root（trust anchor）そのものを、source 固定の本番鍵で確立できるか、
        (1) 成果物の系譜（DAG）が正しいか、(2) その runtime bundle が別チャネル配布の
        署名済みリリース一覧に **載っているか**、(3) 保管庫に実体があるか、
        (4) 実際のハードウェアがリリース想定と一致するか、(5) 箱の中身の指紋が
        すべて一致するか。ここまで全部通って初めて `live_eligible=True` になります。

        重要: 呼び出し元は trust anchor そのものを渡せません。anchor は必ず
        `load_production_trust_anchor()` を通して作られ、署名検証には source 側に
        固定された `PRODUCTION_TRUST_ANCHOR_PUBLIC_KEYS` だけを使います。
        `trust_registry_path` で渡せるのは「一覧ファイルの置き場所」だけであり、
        誰の署名を信じるかは呼び出し元から一切変更できません。None のときは
        環境変数 `TRUST_REGISTRY_PATH_ENV` から解決します。

        やさしい説明（なぜこの形か）: 以前は呼び出し元が `TrustAnchor` を直接渡せた
        ため、自分で鍵を作って自分で「正規リリース一覧」を署名すれば、その場で作った
        成果物でも本番起動できてしまいました。許可証を自分で発行できてしまっては
        許可証の意味が無いので、発行者の鍵は source 側に固定してあります。

        本番鍵が未固定、registry path が未指定、署名が固定鍵で検証できない、のいずれ
        でも `TrustAnchorError`（`BundleLoadError` の派生）で停止します。
        """
        semantics = action_semantics or ActionSemantics.default_v1()
        # (0) 信頼 root の確立。caller 由来の入力を見る前に、source 固定鍵で署名検証
        # できた registry だけを anchor として採用する。失敗は TrustAnchorError
        # （BundleLoadError の派生）としてそのまま伝播させる。
        trust_anchor = load_production_trust_anchor(trust_registry_path)
        if not isinstance(artifact_store, ArtifactStore):
            raise BundleLoadError("artifact_store must be an ArtifactStore")
        if not isinstance(target_profile, TargetProfileRef):
            raise BundleLoadError("target_profile must be a TargetProfileRef")
        if not isinstance(host_profile, HostRuntimeProfile):
            raise BundleLoadError("host_profile must be a HostRuntimeProfile")

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

        # (2) 信頼 root への exact identity 一致。ここが live 起動の唯一の根拠。
        release = trust_anchor.release_for(runtime_descriptor.identity_hash)
        _require_exact(
            combat_descriptor.identity_hash,
            release.combat_student_release_identity_hash,
            "combat_student_release identity_hash",
        )
        _require_exact(
            selector_descriptor.identity_hash,
            release.item_selector_release_identity_hash,
            "item_selector_release identity_hash",
        )
        _require_exact(
            verdict_descriptor.identity_hash,
            release.perception_final_verdict_identity_hash,
            "perception_final_verdict identity_hash",
        )

        # (3) restore 検証済み artifact store の上でだけ live bundle を作る。
        verified_objects = _verify_store_restore(artifact_store, descriptor_list)

        # (4) 実 hardware / target profile 値を、信頼済みリリースの期待値と直接比較する。
        differences = host_profile.mismatches(release.host_profile)
        if differences:
            raise BundleLoadError(
                "runtime host profile does not match the trusted release target profile: "
                + "; ".join(item.describe() for item in differences)
            )
        _require_exact(
            target_profile.ref_hash, release.target_profile_ref_hash, "target_profile_ref_hash"
        )

        # (5) package 内容の指紋照合。ここから先で初めて実 file を読む。
        combat_policy, combat_manifest = _load_combat_package(Path(combat_package_dir))
        combat_manifest_hash = canonical_hash(combat_manifest)
        _require_exact(
            combat_manifest_hash,
            release.combat_package_manifest_hash,
            "combat package manifest hash (trusted release)",
        )
        _require_exact(
            combat_manifest_hash,
            _require_metadata_hash(combat_descriptor, "package_manifest_hash"),
            "combat package manifest hash (release descriptor)",
        )

        try:
            item_selector = OnnxItemSelector.load(Path(item_selector_dir))
        except ItemSelectorRuntimeError as exc:
            raise BundleLoadError(f"ItemSelector package rejected: {exc}") from exc
        selector_manifest_hash = canonical_hash(item_selector.manifest)
        _require_exact(
            selector_manifest_hash,
            release.item_selector_manifest_hash,
            "ItemSelector manifest hash (trusted release)",
        )
        _require_exact(
            selector_manifest_hash,
            _require_metadata_hash(selector_descriptor, "package_manifest_hash"),
            "ItemSelector manifest hash (release descriptor)",
        )

        # deploy schema / action semantics / capability の三方向照合。
        installed_config = NonModelUiPolicyConfigV1.load_default()
        if item_selector.ui_policy_config.to_wire() != installed_config.to_wire():
            raise BundleLoadError("ItemSelector UI policy does not match the installed policy")

        deploy_schema = DeployObsSchema.default_v1()
        _require_exact(
            combat_manifest["deploy_schema_hash"],
            deploy_schema.schema_hash,
            "combat package deploy_schema_hash (installed schema)",
        )
        _require_exact(
            combat_manifest["deploy_schema_hash"],
            release.deploy_schema_hash,
            "combat package deploy_schema_hash (trusted release)",
        )
        if combat_policy.observation_dim != 3 * deploy_schema.dim:
            raise BundleLoadError(
                "combat observation_dim does not match deploy schema value/validity/age "
                f"planes: expected {3 * deploy_schema.dim}, got {combat_policy.observation_dim}"
            )
        if semantics.num_actions != REQUIRED_ACTION_DIM:
            raise BundleLoadError(
                f"action semantics must declare {REQUIRED_ACTION_DIM} actions, "
                f"got {semantics.num_actions}"
            )
        _require_exact(combat_policy.action_dim, REQUIRED_ACTION_DIM, "combat model action_dim")
        _require_exact(
            semantics.semantics_hash, release.action_semantics_hash, "action_semantics_hash"
        )
        _require_exact(
            runtime_descriptor.identity_metadata.get("action_semantics_hash"),
            semantics.semantics_hash,
            "runtime descriptor action_semantics_hash",
        )
        _require_exact(
            runtime_descriptor.identity_metadata.get("decision_hz"),
            REQUIRED_DECISION_HZ,
            "runtime descriptor decision_hz",
        )

        runtime_target_hash = _require_metadata_hash(runtime_descriptor, "target_capability_hash")
        _require_exact(
            runtime_target_hash, release.target_capability_hash, "target_capability_hash"
        )
        _require_exact(
            item_selector.manifest.get("target_capability_hash"),
            runtime_target_hash,
            "ItemSelector target_capability_hash",
        )
        _require_exact(
            _require_metadata_hash(runtime_descriptor, "choice_capability_hash"),
            release.choice_capability_hash,
            "choice_capability_hash",
        )

        perception_subject_hashes = verdict_descriptor.identity_metadata.get("subject_hashes")
        if not isinstance(perception_subject_hashes, Mapping):
            raise BundleLoadError("perception_final_verdict subject_hashes are missing")

        startup_report = _build_startup_report(
            combat_manifest=combat_manifest,
            combat_policy=combat_policy,
            item_selector=item_selector,
            deploy_schema=deploy_schema,
            action_semantics=semantics,
            target_profile=target_profile,
            host_profile=host_profile,
            trust_anchor=trust_anchor,
            runtime_descriptor=runtime_descriptor,
            verdict_descriptor=verdict_descriptor,
            perception_subject_hashes=perception_subject_hashes,
            dag_identity_hashes=dag_report.topological_identity_hashes,
            verified_objects=verified_objects,
        )
        return cls(
            development_only=False,
            live_eligible=True,
            combat_policy=combat_policy,
            deploy_schema=deploy_schema,
            deploy_schema_hash=deploy_schema.schema_hash,
            action_semantics=semantics,
            ui_policy_config=item_selector.ui_policy_config,
            startup_report=startup_report,
            host_profile=host_profile,
            _item_selector=item_selector,
        )


def _build_startup_report(
    *,
    combat_manifest: Mapping[str, Any],
    combat_policy: CombatPolicy,
    item_selector: OnnxItemSelector,
    deploy_schema: DeployObsSchema,
    action_semantics: ActionSemantics,
    target_profile: TargetProfileRef,
    host_profile: HostRuntimeProfile,
    trust_anchor: TrustAnchor,
    runtime_descriptor: ArtifactDescriptor,
    verdict_descriptor: ArtifactDescriptor,
    perception_subject_hashes: Mapping[str, Any],
    dag_identity_hashes: Sequence[str],
    verified_objects: Sequence[str],
) -> dict[str, Any]:
    """起動時の dependency / hash / hardware summary を 1 か所にまとめる。

    やさしい説明: 「いま何で起動したのか」を後から人が確認できるようにする報告書です。
    plan 05-01 タスク2 が求める OS / GPU / driver / CUDA / capture backend の実値と、
    主要な identity をここに載せます。秘密情報は含めません。
    """
    return {
        "bundle_kind": BUNDLE_FORMAL_SENTINEL,
        "development_only": False,
        "live_eligible": True,
        "trust_registry_id": trust_anchor.registry_id,
        "trust_registry_signer_public_key": trust_anchor.signer_public_key,
        "runtime_bundle_identity_hash": runtime_descriptor.identity_hash,
        "combat_package_schema_version": combat_manifest["schema_version"],
        "combat_checkpoint_sha256": combat_manifest["checkpoint_sha256"],
        "combat_model_sha256": combat_manifest["model_sha256"],
        "combat_deploy_schema_hash": combat_manifest["deploy_schema_hash"],
        "combat_formal_dependency_identities": dict(
            combat_manifest["formal_dependency_identities"]
        ),
        "combat_observation_dim": combat_policy.observation_dim,
        "combat_action_dim": combat_policy.action_dim,
        "combat_hidden_dim": combat_policy.hidden_dim,
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
        "target_build_id": target_profile.build_id,
        "target_hardware_profile_id": target_profile.hardware_profile_id,
        "perception_verdict_hash": verdict_descriptor.identity_hash,
        "perception_subject_hashes": dict(perception_subject_hashes),
        "dag_identity_hashes": list(dag_identity_hashes),
        "verified_store_objects": list(verified_objects),
        **host_profile.to_wire(),
    }

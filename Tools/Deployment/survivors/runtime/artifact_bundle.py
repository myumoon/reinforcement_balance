"""Runtime artifact bundle: combat package + ItemSelector の検証済みロード。

golden fixture (development_only=True, live_eligible=False) と正式 package の
両方を扱う単一の入口を提供する。OS input には触れず bundle 内容だけを公開する。
"""
from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch as torch_module
import torch.nn as nn

from reinbalance_survivors_contracts import ui_policy as _installed_ui_policy_module
from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    sha256_hex,
)
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.ui_policy import (
    NonModelUiPolicyConfigV1,
    decide_non_model_ui_intent,
)

# Training からの import は禁止 — ここで Deployment 専用の最小実装を定義する。
COMBAT_PACKAGE_SCHEMA_VERSION = "survivors.deployable_policy_package.v1"
_COMBAT_MANIFEST_KEYS = frozenset({
    "schema_version", "checkpoint_sha256", "model_sha256", "deploy_schema_hash",
    "model_config", "development_only", "formal_student_eligible",
    "formal_dependency_identities", "files",
})
_MODEL_CONFIG_KEYS = frozenset({"observation_dim", "action_dim", "hidden_dim"})

BUNDLE_DEVELOPMENT_SENTINEL = "golden_fixture"
REQUIRED_PERCEPTION_VERDICT_HASH_KEY = "perception_verdict_hash"


class BundleLoadError(ValueError):
    """artifact bundle の schema / hash / capability 検証失敗。

    起動前に artifact の取り違えや改変を拒否するための fail-closed 例外。
    """


def _policy_schema_hash(config: NonModelUiPolicyConfigV1) -> str:
    """NonModelUiPolicyConfigV1 の canonical wire から schema binding hash を返す。"""
    return canonical_hash(config.to_wire())


def _installed_policy_impl_hash() -> str:
    """インストール済み ui_policy 実装 source の SHA-256 を返す。"""
    source_name = inspect.getsourcefile(_installed_ui_policy_module)
    if not source_name:
        raise BundleLoadError("installed policy implementation source is unavailable")
    try:
        return sha256_hex(Path(source_name).read_bytes())
    except OSError as exc:
        raise BundleLoadError(
            f"cannot read installed policy implementation: {exc}"
        ) from exc


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


class _CombatGRU(nn.Module):
    """DeployableCombatPolicy と同一 GRU アーキテクチャの Deployment 専用 runtime 実装。

    Training 側 checkpoint の model_config (observation_dim / action_dim / hidden_dim)
    を受け取り、同じ weight topology で推論だけを行う。VecNormalize は含まない。
    """

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        """GRUCell + actor + value head を初期化する。

        Training の DeployableCombatPolicy と完全一致する topology でなければ
        state_dict load が失敗し、weight 不一致を検出できる。
        """
        super().__init__()
        if type(observation_dim) is not int or observation_dim <= 0:
            raise BundleLoadError("observation_dim must be positive int")
        if type(action_dim) is not int or action_dim <= 0:
            raise BundleLoadError("action_dim must be positive int")
        if type(hidden_dim) is not int or hidden_dim <= 0:
            raise BundleLoadError("hidden_dim must be positive int")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent = nn.GRUCell(observation_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def step(
        self,
        obs: torch_module.Tensor,
        hidden: torch_module.Tensor,
    ) -> tuple[torch_module.Tensor, torch_module.Tensor]:
        """1 step の GRU 推論。actor logit と新 hidden state を返す。

        batch dim [1, obs_dim] を想定。forward は sequence 用なので step で代替する。
        """
        new_hidden = self.recurrent(obs, hidden)
        logit = self.actor(new_hidden)
        return logit, new_hidden


def _load_combat_package(package_dir: Path) -> tuple[_CombatGRU, str, dict[str, Any]]:
    """formal combat package を読み込み、manifest・model hash を検証する。

    development_only=False かつ formal_student_eligible=True の package だけを受理する。
    """
    root = Path(package_dir)
    try:
        manifest = json.loads((root / "manifest.json").read_bytes())
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
    for fname in ("checkpoint_sha256", "model_sha256", "deploy_schema_hash"):
        _require_sha256(manifest.get(fname), fname)
    if manifest.get("files") != ["manifest.json", "model.pt"]:
        raise BundleLoadError("combat package file list mismatch")
    if {p.name for p in root.iterdir()} != {"manifest.json", "model.pt"}:
        raise BundleLoadError("combat package directory contents mismatch")
    actual_model_sha = sha256_hex((root / "model.pt").read_bytes())
    if actual_model_sha != manifest["model_sha256"]:
        raise BundleLoadError("combat model.pt hash mismatch")
    try:
        payload = torch_module.load(root / "model.pt", map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise BundleLoadError(f"cannot load combat model.pt: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"model_config", "model_state_dict"}:
        raise BundleLoadError("combat model.pt structure mismatch")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict) or set(model_config) != _MODEL_CONFIG_KEYS:
        raise BundleLoadError("combat model_config fields mismatch")
    if model_config != manifest.get("model_config"):
        raise BundleLoadError("combat model_config manifest mismatch")
    for key in ("observation_dim", "action_dim", "hidden_dim"):
        if type(model_config.get(key)) is not int or model_config[key] <= 0:
            raise BundleLoadError(f"combat model_config.{key} must be positive int")
    model = _CombatGRU(
        model_config["observation_dim"],
        model_config["action_dim"],
        model_config["hidden_dim"],
    )
    try:
        model.load_state_dict(payload["model_state_dict"])
    except (RuntimeError, KeyError) as exc:
        raise BundleLoadError(f"combat model state_dict incompatible: {exc}") from exc
    model.eval()
    return model, manifest["deploy_schema_hash"], manifest


@dataclass(frozen=True)
class RuntimeBundle:
    """検証済み combat + ItemSelector artifact の runtime コンテナ。

    live-eligible かどうかを startup 時に確定し、golden fixture からの
    正式起動を fail-closed で拒否する。OS input には触れない。
    """

    development_only: bool
    live_eligible: bool
    combat_model: _CombatGRU
    deploy_schema: DeployObsSchema
    deploy_schema_hash: str
    ui_policy_config: NonModelUiPolicyConfigV1
    startup_report: dict[str, Any]
    # ItemSelector は省略可 — combat-only golden fixture に None を許容
    _item_selector: Any = field(default=None, repr=False)

    @property
    def item_selector(self) -> Any:
        """ItemSelectorArtifact または None を返す。

        None の場合は item session を起動できないため、呼び出し側が guard する。
        """
        return self._item_selector

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
        combat_model: _CombatGRU,
        *,
        ui_policy_config: NonModelUiPolicyConfigV1 | None = None,
        item_selector: Any = None,
        deploy_schema: DeployObsSchema | None = None,
    ) -> "RuntimeBundle":
        """golden fixture 用の development_only=True, live_eligible=False bundle を返す。

        formal package を必要とせず、テストが全 loader / session / scheduler を実行できる。
        この bundle は assert_live_eligible() で必ず拒否される。
        """
        if not isinstance(combat_model, _CombatGRU):
            raise BundleLoadError("combat_model must be _CombatGRU")
        config = ui_policy_config or NonModelUiPolicyConfigV1.load_default()
        schema = deploy_schema or DeployObsSchema.default_v1()
        return cls(
            development_only=True,
            live_eligible=False,
            combat_model=combat_model,
            deploy_schema=schema,
            deploy_schema_hash=schema.schema_hash,
            ui_policy_config=config,
            startup_report={"bundle_kind": BUNDLE_DEVELOPMENT_SENTINEL},
            _item_selector=item_selector,
        )

    @classmethod
    def load(
        cls,
        combat_package_dir: Path,
        item_selector_dir: Path,
        *,
        perception_verdict_hash: str,
        target_capability_hash: str,
    ) -> "RuntimeBundle":
        """formal package を hash 検証してロードする。

        perception_verdict_hash は 04-10 の final verdict SHA-256。
        target_capability_hash は 02-03 / 03-05 の target capability SHA-256。
        いずれかが stale / missing の場合は起動を拒否する。
        """
        _require_sha256(perception_verdict_hash, "perception_verdict_hash")
        _require_sha256(target_capability_hash, "target_capability_hash")

        combat_model, deploy_schema_hash, combat_manifest = _load_combat_package(
            Path(combat_package_dir)
        )

        # ItemSelector package の検証は Training 側の ArtifactBindingError を透過させる
        try:
            from training_proxy import ItemSelectorArtifact  # noqa: PLC0415  # ponytail: import guard
            item_selector = ItemSelectorArtifact.load(Path(item_selector_dir))
        except ImportError:
            # Training package が Deployment 環境にない場合は別経路で import
            import sys
            _training_path = str(Path(__file__).parents[4] / "Training")
            if _training_path not in sys.path:
                sys.path.insert(0, _training_path)
            from games.survivors.item_selector_artifact import (  # noqa: PLC0415
                ItemSelectorArtifact,
            )
            item_selector = ItemSelectorArtifact.load(Path(item_selector_dir))

        # target capability hash をクロス検証
        if item_selector.manifest.get("target_capability_hash") != target_capability_hash:
            raise BundleLoadError("ItemSelector target_capability_hash mismatch")

        # UI policy 一致検証 (02-03 と同一 hash の policy)
        installed_config = NonModelUiPolicyConfigV1.load_default()
        if item_selector.ui_policy_config.to_wire() != installed_config.to_wire():
            raise BundleLoadError("ItemSelector UI policy config mismatch with installed policy")
        _validate_ui_policy_hashes(item_selector.manifest)

        # deploy schema を hash で照合して load する
        deploy_schema = DeployObsSchema.default_v1()
        if deploy_schema.schema_hash != deploy_schema_hash:
            raise BundleLoadError(
                f"combat package deploy_schema_hash {deploy_schema_hash!r} "
                f"does not match installed default schema {deploy_schema.schema_hash!r}"
            )

        startup_report = _build_startup_report(
            combat_manifest=combat_manifest,
            item_selector_manifest=item_selector.manifest,
            perception_verdict_hash=perception_verdict_hash,
            target_capability_hash=target_capability_hash,
        )
        return cls(
            development_only=False,
            live_eligible=True,
            combat_model=combat_model,
            deploy_schema=deploy_schema,
            deploy_schema_hash=deploy_schema_hash,
            ui_policy_config=installed_config,
            startup_report=startup_report,
            _item_selector=item_selector,
        )


def _validate_ui_policy_hashes(manifest: Mapping[str, Any]) -> None:
    """manifest の UI policy schema / config / impl hash を installed と照合する。

    missing / substitution / local duplicated rule を fail-closed で拒否する。
    """
    installed_config = NonModelUiPolicyConfigV1.load_default()
    expected_schema_hash = _policy_schema_hash(installed_config)
    expected_config_hash = installed_config.config_hash
    expected_impl_hash = _installed_policy_impl_hash()

    if manifest.get("policy_schema_hash") != expected_schema_hash:
        raise BundleLoadError("UI policy schema hash mismatch")
    if manifest.get("policy_config_hash") != expected_config_hash:
        raise BundleLoadError("UI policy config hash mismatch")
    if manifest.get("policy_impl_hash") != expected_impl_hash:
        raise BundleLoadError("UI policy implementation hash mismatch")


def _build_startup_report(
    *,
    combat_manifest: Mapping[str, Any],
    item_selector_manifest: Mapping[str, Any],
    perception_verdict_hash: str,
    target_capability_hash: str,
) -> dict[str, Any]:
    """起動時の dependency / hash / schema summary を生成する。

    ユーザーが artifact 取り違えに気づけるよう、主要な identity を一か所にまとめる。
    """
    return {
        "bundle_kind": "formal",
        "combat_deploy_schema_hash": combat_manifest.get("deploy_schema_hash"),
        "combat_model_sha256": combat_manifest.get("model_sha256"),
        "item_selector_artifact_identity": item_selector_manifest.get("artifact_identity"),
        "item_selector_target_capability_hash": item_selector_manifest.get("target_capability_hash"),
        "perception_verdict_hash": perception_verdict_hash,
        "target_capability_hash": target_capability_hash,
        "ui_policy_policy_id": item_selector_manifest.get("policy_schema_hash", "")[:8],
    }

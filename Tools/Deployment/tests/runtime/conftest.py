"""runtime test 用の共有 pytest fixture。

ItemSelector runtime 単体テスト用の最小 package fixture と、artifact bundle 用の
formal 成果物一式（combat package・trust registry 込み）の両方を提供する。

やさしい説明:
    ItemSelector 側は Torch や署名基盤を使わず、小さな埋め込み ONNX と共有 UI policy
    だけでロード・推論・改ざん拒否を試せる箱を各テストへ作る。artifact bundle 側は
    ONNX export と package 書き出しが 1 件あたり数秒かかるため、session scope で 1 度
    だけ組み立てて全テストで使い回す。異常系テストは、この一式を tmp_path へコピー
    してから壊して使う。元の一式を直接壊すと、あとに続くテストまで巻き添えで失敗
    するため。
"""

from __future__ import annotations

import base64
import inspect
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)
from reinbalance_survivors_contracts import ui_policy as installed_ui_policy
from reinbalance_survivors_contracts.ui_policy import NonModelUiPolicyConfigV1
from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor
from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.target_action import TargetProfileRef

from survivors.runtime.artifact_bundle import HostRuntimeProfile, TrustAnchor

from . import _runtime_fixtures as fx


_TINY_SELECTOR_ONNX = base64.b64decode(
    "CAg6wwMKOxILcmVkdWNlX2F4ZXMiCENvbnN0YW50KiIKBXZhbHVlKhYIARAHOgr/"
    "//////////8BQgRheGVzoAEECkUKEmNhbmRpZGF0ZV9mZWF0dXJlcwoLcmVkdWNl"
    "X2F4ZXMSBnNjb3JlcyIJUmVkdWNlU3VtKg8KCGtlZXBkaW1zGACgAQIKNhIMbWFz"
    "a2VkX3ZhbHVlIghDb25zdGFudCocCgV2YWx1ZSoQEAEiBABAHMZCBm1hc2tlZKAB"
    "BAo1Cg5jYW5kaWRhdGVfbWFzawoGc2NvcmVzCgxtYXNrZWRfdmFsdWUSBmxvZ2l0"
    "cyIFV2hlcmUSEnRpbnlfaXRlbV9zZWxlY3RvclonChBjb250ZXh0X2ZlYXR1cmVz"
    "EhMKEQgBEg0KBxIFYmF0Y2gKAggEWjcKEmNhbmRpZGF0ZV9mZWF0dXJlcxIhCh8I"
    "ARIbCgcSBWJhdGNoCgwSCmNhbmRpZGF0ZXMKAggDWi8KDmNhbmRpZGF0ZV9tYXNr"
    "Eh0KGwgJEhcKBxIFYmF0Y2gKDBIKY2FuZGlkYXRlc2InCgZsb2dpdHMSHQobCAES"
    "FwoHEgViYXRjaAoMEgpjYW5kaWRhdGVzQgQKABAR"
)

_UNMASKED_SELECTOR_ONNX = base64.b64decode(
    "CAg61AIKOxILcmVkdWNlX2F4ZXMiCENvbnN0YW50KiIKBXZhbHVlKhYIARAHOgr/"
    "//////////8BQgRheGVzoAEECkUKEmNhbmRpZGF0ZV9mZWF0dXJlcwoLcmVkdWNl"
    "X2F4ZXMSBmxvZ2l0cyIJUmVkdWNlU3VtKg8KCGtlZXBkaW1zGACgAQISEnRpbnlf"
    "aXRlbV9zZWxlY3RvclonChBjb250ZXh0X2ZlYXR1cmVzEhMKEQgBEg0KBxIFYmF0"
    "Y2gKAggEWjcKEmNhbmRpZGF0ZV9mZWF0dXJlcxIhCh8IARIbCgcSBWJhdGNoCgwS"
    "CmNhbmRpZGF0ZXMKAggDWi8KDmNhbmRpZGF0ZV9tYXNrEh0KGwgJEhcKBxIFYmF0"
    "Y2gKDBIKY2FuZGlkYXRlc2InCgZsb2dpdHMSHQobCAESFwoHEgViYXRjaAoMEgpj"
    "YW5kaWRhdGVzQgQKABAR"
)

# 候補数軸を静的 4 に固定した ONNX（M2 static-onnx-axes-pass-loader 回帰用）。
# やさしい説明: カード枚数を動かせないモデルを load 時に拒否できるかを試す不正な箱。
_STATIC_CANDIDATES_SELECTOR_ONNX = base64.b64decode(
    "CAg6pQMKOxILcmVkdWNlX2F4ZXMiCENvbnN0YW50KiIKBXZhbHVlKhYIARAHOgr/////"
    "//////8BQgRheGVzoAEECkUKEmNhbmRpZGF0ZV9mZWF0dXJlcwoLcmVkdWNlX2F4ZXMS"
    "BnNjb3JlcyIJUmVkdWNlU3VtKg8KCGtlZXBkaW1zGACgAQIKNhIMbWFza2VkX3ZhbHVl"
    "IghDb25zdGFudCocCgV2YWx1ZSoQEAEiBABAHMZCBm1hc2tlZKABBAo1Cg5jYW5kaWRh"
    "dGVfbWFzawoGc2NvcmVzCgxtYXNrZWRfdmFsdWUSBmxvZ2l0cyIFV2hlcmUSEnRpbnlf"
    "aXRlbV9zZWxlY3RvclonChBjb250ZXh0X2ZlYXR1cmVzEhMKEQgBEg0KBxIFYmF0Y2gK"
    "AggEWi0KEmNhbmRpZGF0ZV9mZWF0dXJlcxIXChUIARIRCgcSBWJhdGNoCgIIBAoCCANa"
    "JQoOY2FuZGlkYXRlX21hc2sSEwoRCAkSDQoHEgViYXRjaAoCCARiHQoGbG9naXRzEhMK"
    "EQgBEg0KBxIFYmF0Y2gKAggEQgQKABAR"
)

# batch 軸を静的 1 に固定した ONNX（M2 static-onnx-axes-pass-loader 回帰用）。
# やさしい説明: 同時プレイ人数を 1 人からしか動かせないモデルを load 時に拒否できるか試す。
_STATIC_BATCH_SELECTOR_ONNX = base64.b64decode(
    "CAg6rwMKOxILcmVkdWNlX2F4ZXMiCENvbnN0YW50KiIKBXZhbHVlKhYIARAHOgr/////"
    "//////8BQgRheGVzoAEECkUKEmNhbmRpZGF0ZV9mZWF0dXJlcwoLcmVkdWNlX2F4ZXMS"
    "BnNjb3JlcyIJUmVkdWNlU3VtKg8KCGtlZXBkaW1zGACgAQIKNhIMbWFza2VkX3ZhbHVl"
    "IghDb25zdGFudCocCgV2YWx1ZSoQEAEiBABAHMZCBm1hc2tlZKABBAo1Cg5jYW5kaWRh"
    "dGVfbWFzawoGc2NvcmVzCgxtYXNrZWRfdmFsdWUSBmxvZ2l0cyIFV2hlcmUSEnRpbnlf"
    "aXRlbV9zZWxlY3RvcloiChBjb250ZXh0X2ZlYXR1cmVzEg4KDAgBEggKAggBCgIIBFoy"
    "ChJjYW5kaWRhdGVfZmVhdHVyZXMSHAoaCAESFgoCCAEKDBIKY2FuZGlkYXRlcwoCCANa"
    "KgoOY2FuZGlkYXRlX21hc2sSGAoWCAkSEgoCCAEKDBIKY2FuZGlkYXRlc2IiCgZsb2dp"
    "dHMSGAoWCAESEgoCCAEKDBIKY2FuZGlkYXRlc0IECgAQEQ=="
)


@dataclass(frozen=True)
class ItemSelectorPackageFixture:
    """変更可能な ItemSelector package と manifest をまとめる。

    やさしい説明: 異常系では正常な箱を複製して一か所だけ壊し、原因を明確にする。
    """

    root: Path
    manifest: dict[str, Any]

    def copy_to(self, destination: Path) -> "ItemSelectorPackageFixture":
        """package を destination へ複製して独立した fixture を返す。

        やさしい説明: 元の正常 package を壊さず、テスト専用のコピーを用意する。
        """
        shutil.copytree(self.root, destination)
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        return ItemSelectorPackageFixture(destination, manifest)

    def write_manifest(self) -> None:
        """保持中の manifest を canonical JSON として package へ書き戻す。

        やさしい説明: フィールド欠損や型違いを作り、loader が拒否するか試せるようにする。
        """
        (self.root / "manifest.json").write_bytes(canonical_json_bytes(self.manifest))

    def replace_onnx(self, content: bytes, *, update_hashes: bool) -> None:
        """ONNX 内容を差し替え、必要なら manifest の binding hash も更新する。

        やさしい説明: hash 改ざんと、hash は正しいが ONNX 自体が壊れた場合を分けて作る。
        """
        (self.root / "model.onnx").write_bytes(content)
        if update_hashes:
            digest = sha256_hex(content)
            self.manifest["onnx_model_hash"] = digest
            self.manifest["files"]["model.onnx"] = digest
            core = dict(self.manifest)
            core.pop("artifact_identity")
            self.manifest["artifact_identity"] = canonical_hash(core)
            self.write_manifest()


def _write_item_selector_package(root: Path) -> ItemSelectorPackageFixture:
    """共有契約を満たす ItemSelector package を root へ書く。

    やさしい説明: ONNX は候補 feature の合計を順番どおり logits にするため、mask と
    permutation の対応を手計算した期待値で検証できる。
    """
    root.mkdir(parents=True)
    installed = NonModelUiPolicyConfigV1.load_default()
    policy_source = inspect.getsourcefile(installed_ui_policy)
    assert policy_source is not None
    files = {
        "model.pt": b"unused-torch-model",
        "model.onnx": _TINY_SELECTOR_ONNX,
        "ui_policy_config.json": canonical_json_bytes(installed.to_wire()),
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    file_hashes = {name: sha256_hex(content) for name, content in files.items()}
    vocabulary = ["knife", "wand", "whip"]
    core: dict[str, Any] = {
        "schema_version": "survivors.item_selector_artifact.v1",
        "target_capability_hash": sha256_hex(b"target-capability"),
        "nmax": 4,
        "context_dim": 4,
        "candidate_dim": 3,
        "feature_schema": "context_only_v1",
        "vocabulary_hash": canonical_hash(vocabulary),
        "item_vocabulary": vocabulary,
        "temperature": 1.0,
        "confidence_threshold": 0.0,
        "student_output_temperature": 1.0,
        "dataset_identity": sha256_hex(b"item-selector-dataset"),
        "model_state_hash": file_hashes["model.pt"],
        "onnx_model_hash": file_hashes["model.onnx"],
        "files": file_hashes,
        "policy_schema_hash": canonical_hash(installed.to_wire()),
        "policy_config_hash": installed.config_hash,
        "policy_impl_hash": sha256_hex(Path(policy_source).read_bytes()),
        "lineage": {
            "source_descriptor_identity": sha256_hex(b"source-descriptor"),
            "teacher_verdict_identity": sha256_hex(b"teacher-verdict"),
            "trace_dataset_identity": sha256_hex(b"trace-dataset"),
            "model_training_run_id": "item-selector-runtime-test",
        },
        "dependency_versions": {
            "numpy": "1.26.4",
            "onnx": "1.16.0",
            "onnxruntime": "1.18.1",
            "torch": "2.11.0",
        },
    }
    core["onnx_input_tensors"] = [
        {"name": "context_features", "shape": ["batch", 4], "dtype": "float32"},
        {
            "name": "candidate_features",
            "shape": ["batch", 4, 3],
            "dtype": "float32",
        },
        {"name": "candidate_mask", "shape": ["batch", 4], "dtype": "bool"},
    ]
    core["onnx_output_tensors"] = [
        {"name": "logits", "shape": ["batch", 4], "dtype": "float32"}
    ]
    manifest = {**core, "artifact_identity": canonical_hash(core)}
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    return ItemSelectorPackageFixture(root, manifest)


@pytest.fixture
def item_selector_package(tmp_path: Path) -> ItemSelectorPackageFixture:
    """正常な ItemSelector package を各テストへ渡す。

    やさしい説明: 埋め込み済みモデルなので外部ダウンロードや重い export は行わない。
    """
    return _write_item_selector_package(tmp_path / "selector")


@dataclass(frozen=True)
class FormalBundleInputs:
    """`RuntimeBundle.load()` を通過する成果物一式。

    やさしい説明: 正常に起動できる状態の「箱・保管庫・系譜・許可証」をまとめたものです。
    テストはここから 1 か所だけ壊して、その 1 点で起動が止まることを確かめます。
    """

    root: Path
    combat_dir: Path
    selector_dir: Path
    store: ArtifactStore
    descriptors: list[ArtifactDescriptor]
    combat_manifest: dict[str, Any]
    selector_manifest: dict[str, Any]
    target_profile: TargetProfileRef
    host_profile: HostRuntimeProfile
    trust_anchor: TrustAnchor
    registry_path: Path
    target_capability_hash: str
    choice_capability_hash: str

    def as_load_kwargs(self, **overrides: Any) -> dict[str, Any]:
        """`RuntimeBundle.load()` へ渡す keyword 引数を作る。

        やさしい説明: 既定は「全部正常」の組み合わせです。`overrides` で 1 項目だけ
        差し替えると、その 1 点だけが異常な入力を簡単に作れます。
        """
        kwargs: dict[str, Any] = {
            "combat_package_dir": self.combat_dir,
            "item_selector_dir": self.selector_dir,
            "artifact_store": self.store,
            "descriptors": self.descriptors,
            "target_profile": self.target_profile,
            "host_profile": self.host_profile,
            "trust_anchor": self.trust_anchor,
        }
        kwargs.update(overrides)
        return kwargs

    def copy_combat_package(self, destination: Path) -> Path:
        """combat package を複製して、壊してよい作業用 directory を返す。"""
        shutil.copytree(self.combat_dir, destination)
        return destination


def _build_formal_inputs(root: Path) -> FormalBundleInputs:
    """正常系の成果物一式を root 配下に組み立てる。

    やさしい説明: package を書き、保管庫へ登録し、系譜を作り、その系譜の identity を
    署名済みの許可証一覧へ載せる、という実運用と同じ順序で用意します。
    """
    capability = fx.hash_of("target-capability")
    choice_capability = fx.hash_of("choice-capability")
    target_profile = fx.default_target_profile()
    host_profile = fx.default_host_profile()

    selector_manifest = fx.write_item_selector_package(
        root / "selector", target_capability_hash=capability
    )
    model, model_config = fx.build_combat_model()
    combat_manifest = fx.write_combat_package(root / "combat", model, model_config)

    store = ArtifactStore(root / "store")
    descriptors = fx.build_formal_descriptors(
        store=store,
        combat_manifest=combat_manifest,
        item_selector_manifest=selector_manifest,
        target_capability_hash=capability,
        choice_capability_hash=choice_capability,
    )
    entry = fx.release_entry(
        descriptors=descriptors,
        combat_manifest=combat_manifest,
        item_selector_manifest=selector_manifest,
        target_profile=target_profile,
        host_profile=host_profile,
        target_capability_hash=capability,
        choice_capability_hash=choice_capability,
    )
    registry_path = fx.write_trust_registry(root / "trust" / "releases.json", [entry])
    trust_anchor = TrustAnchor.load(
        registry_path, verification_public_keys=[fx.public_key_hex()]
    )
    return FormalBundleInputs(
        root=root,
        combat_dir=root / "combat",
        selector_dir=root / "selector",
        store=store,
        descriptors=descriptors,
        combat_manifest=combat_manifest,
        selector_manifest=selector_manifest,
        target_profile=target_profile,
        host_profile=host_profile,
        trust_anchor=trust_anchor,
        registry_path=registry_path,
        target_capability_hash=capability,
        choice_capability_hash=choice_capability,
    )


@pytest.fixture(scope="session")
def formal_inputs(tmp_path_factory: pytest.TempPathFactory) -> FormalBundleInputs:
    """検証を通過する formal 成果物一式を session 単位で 1 度だけ組み立てる。"""
    return _build_formal_inputs(tmp_path_factory.mktemp("formal"))

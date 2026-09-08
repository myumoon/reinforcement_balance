"""ItemSelector runtime 単体テスト用の最小 package fixture を提供する。

やさしい説明: Torch や署名基盤を使わず、小さな埋め込み ONNX と共有 UI policy だけで
ロード・推論・改ざん拒否を試せる箱を各テストへ作る。
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

"""ItemSelector package (02-03) の共有 manifest / binding 契約。

Training の export/evaluator と Deployment の runtime adapter が同じ package を
読めるように、schema version・manifest field 集合・hash 検証・UI policy binding を
このインストール可能な Common package だけで定義します。

やさしい説明:
「アイテム選択AIの入った箱（package）が本物かどうかを確かめるルール」を1か所に
まとめたファイルです。学習側（Training）と実行側（Deployment）が別々にコピーを
持つと片方だけ古くなって食い違うので、両方がこのファイルだけを見るようにします。
torch や onnxruntime には依存せず、JSON と hash の検証だけを行います。
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any, Mapping

from . import ui_policy as installed_ui_policy
from .canonical_json import canonical_hash, canonical_json_bytes, sha256_hex
from .ui_policy import NonModelUiPolicyConfigV1

__all__ = [
    "ITEM_SELECTOR_ARTIFACT_SCHEMA_VERSION",
    "ITEM_SELECTOR_PACKAGE_FILES",
    "ITEM_SELECTOR_MANIFEST_FIELDS",
    "ItemSelectorPackageError",
    "policy_schema_hash",
    "installed_policy_impl_hash",
    "assert_single_installed_ui_policy_distribution",
    "artifact_binding_payload",
    "expected_onnx_tensor_manifest",
    "read_item_selector_manifest",
    "validate_item_selector_manifest",
    "verify_item_selector_package_files",
    "load_verified_ui_policy_config",
    "verify_ui_policy_binding",
]

ITEM_SELECTOR_ARTIFACT_SCHEMA_VERSION = "survivors.item_selector_artifact.v1"
ITEM_SELECTOR_PACKAGE_FILES = frozenset({"model.pt", "model.onnx", "ui_policy_config.json"})
ITEM_SELECTOR_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_identity",
        "target_capability_hash",
        "nmax",
        "context_dim",
        "candidate_dim",
        "feature_schema",
        "vocabulary_hash",
        "item_vocabulary",
        "temperature",
        "confidence_threshold",
        "student_output_temperature",
        "dataset_identity",
        "model_state_hash",
        "onnx_model_hash",
        "files",
        "policy_schema_hash",
        "policy_config_hash",
        "policy_impl_hash",
        "lineage",
        "dependency_versions",
        "onnx_input_tensors",
        "onnx_output_tensors",
    }
)
_LINEAGE_FIELDS = frozenset(
    {
        "source_descriptor_identity",
        "teacher_verdict_identity",
        "trace_dataset_identity",
        "model_training_run_id",
    }
)
_DEPENDENCY_FIELDS = frozenset({"torch", "onnx", "onnxruntime", "numpy"})


class ItemSelectorPackageError(ValueError):
    """artifact package の schema / content / distribution binding 違反。

    やさしい説明: 箱の中身が manifest と食い違うときに投げる例外です。
    警告で続行せず、推論を始める前に必ず止めます。
    """


def _is_sha256(value: Any) -> bool:
    """小文字 SHA-256 16 進文字列だけを True にする。

    やさしい説明: 「64桁の小文字16進数か？」を確かめるだけの関数です。
    """
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def policy_schema_hash(config: NonModelUiPolicyConfigV1) -> str:
    """共有 wire そのものから NonModelUiPolicyConfigV1 schema binding を返す。

    やさしい説明: UI ルール設定の「形」を表すハッシュ値を計算します。
    """
    return canonical_hash(config.to_wire())


def installed_policy_impl_hash() -> str:
    """import された共有 ui_policy 実装 source の content hash を返す。

    やさしい説明: 実際に読み込まれている UI ルールの Python ファイルそのものの
    ハッシュです。ルールの実装が差し替えられていないかを確認できます。
    """
    source_name = inspect.getsourcefile(installed_ui_policy)
    if not source_name:
        raise ItemSelectorPackageError("installed policy implementation source is unavailable")
    try:
        return sha256_hex(Path(source_name).read_bytes())
    except OSError as exc:
        raise ItemSelectorPackageError(
            f"cannot read installed policy implementation: {exc}"
        ) from exc


def assert_single_installed_ui_policy_distribution() -> Path:
    """共有 UI policy が単一の install 済み distribution から来ていることを確認する。

    やさしい説明: 同じ名前のルールモジュールがプロジェクト内に複数コピーされて
    いると、どれが使われるか分からなくなります。読み込み経路が 1 つだけで、
    決定関数がその配布物の中にあることを確かめます。
    """
    package = installed_ui_policy.__package__ or "reinbalance_survivors_contracts"
    module = __import__(package, fromlist=["__path__"])
    search_locations = list(getattr(module, "__path__", []) or [])
    if len(search_locations) != 1:
        raise ItemSelectorPackageError(
            f"shared UI policy distribution must resolve to exactly one location, got {search_locations}"
        )
    distribution_root = Path(search_locations[0]).resolve()
    decide = getattr(installed_ui_policy, "decide_non_model_ui_intent", None)
    source_file = inspect.getsourcefile(decide) if decide is not None else None
    if source_file is None:
        raise ItemSelectorPackageError("shared UI policy rule source is unavailable")
    try:
        Path(source_file).resolve().relative_to(distribution_root)
    except ValueError as exc:
        raise ItemSelectorPackageError(
            "shared UI policy rule is not provided by the installed distribution"
        ) from exc
    return distribution_root


def artifact_binding_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """TorchScript へ封入する model/data/feature binding の固定部分を返す。

    やさしい説明: モデルファイルの中に埋め込んでおく「身分証」の項目一覧です。
    manifest と埋め込みの両方が一致して初めて同じ artifact とみなします。
    """
    fields = (
        "target_capability_hash",
        "nmax",
        "context_dim",
        "candidate_dim",
        "feature_schema",
        "vocabulary_hash",
        "dataset_identity",
        "temperature",
        "confidence_threshold",
        "student_output_temperature",
    )
    return {field: manifest[field] for field in fields}


def expected_onnx_tensor_manifest(
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """manifest の次元から期待される ONNX 入出力 tensor 仕様を返す。

    やさしい説明: ONNX モデルが持つべき入力・出力の名前・形・型の一覧です。
    """
    nmax = manifest["nmax"]
    context_dim = manifest["context_dim"]
    candidate_dim = manifest["candidate_dim"]
    expected_inputs = [
        {"name": "context_features", "shape": ["batch", context_dim], "dtype": "float32"},
        {
            "name": "candidate_features",
            "shape": ["batch", nmax, candidate_dim],
            "dtype": "float32",
        },
        {"name": "candidate_mask", "shape": ["batch", nmax], "dtype": "bool"},
    ]
    expected_outputs = [{"name": "logits", "shape": ["batch", nmax], "dtype": "float32"}]
    return expected_inputs, expected_outputs


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """JSON object ファイルを読み、object 以外を拒否する。

    やさしい説明: JSON を読み込み、辞書でなければエラーにします。
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ItemSelectorPackageError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ItemSelectorPackageError(f"{label} must be a JSON object")
    return value


def read_item_selector_manifest(package_dir: Path) -> dict[str, Any]:
    """package directory の manifest.json を読み、field 集合を検証する。

    やさしい説明: 箱に入っている説明書を読み、項目の過不足がないか確認します。
    """
    manifest = _read_json_object(Path(package_dir) / "manifest.json", label="artifact manifest")
    if frozenset(manifest) != ITEM_SELECTOR_MANIFEST_FIELDS:
        raise ItemSelectorPackageError("artifact manifest fields mismatch")
    return manifest


def _validate_scalar_manifest(manifest: Mapping[str, Any]) -> None:
    """manifest のスカラー値（hash・次元・温度・閾値）を検証する。

    やさしい説明: 数値や ID が正しい形式・範囲になっているかを見ます。
    """
    if manifest.get("schema_version") != ITEM_SELECTOR_ARTIFACT_SCHEMA_VERSION:
        raise ItemSelectorPackageError("unsupported artifact schema version")
    for field in (
        "artifact_identity",
        "target_capability_hash",
        "vocabulary_hash",
        "dataset_identity",
        "model_state_hash",
        "onnx_model_hash",
        "policy_schema_hash",
        "policy_config_hash",
        "policy_impl_hash",
    ):
        if not _is_sha256(manifest.get(field)):
            raise ItemSelectorPackageError(f"{field} must be lowercase SHA-256")
    for field in ("nmax", "context_dim", "candidate_dim"):
        if type(manifest.get(field)) is not int or int(manifest[field]) <= 0:
            raise ItemSelectorPackageError(f"{field} must be a positive integer")
    feature_schema = manifest.get("feature_schema")
    if not isinstance(feature_schema, str) or not feature_schema:
        raise ItemSelectorPackageError("feature schema must be non-empty")
    for field in ("temperature", "student_output_temperature"):
        value = manifest.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ItemSelectorPackageError(f"{field} must be positive and finite")
    threshold = manifest.get("confidence_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ItemSelectorPackageError("confidence_threshold must be in [0, 1]")


def _validate_vocabulary(manifest: Mapping[str, Any]) -> frozenset[str]:
    """item vocabulary の整列・一意性と hash 束縛を検証する。

    やさしい説明: 扱えるアイテム名の一覧が並び順どおりで重複がないかを見ます。
    """
    vocabulary = manifest.get("item_vocabulary")
    if (
        not isinstance(vocabulary, list)
        or not vocabulary
        or any(not isinstance(item, str) or not item for item in vocabulary)
        or len(set(vocabulary)) != len(vocabulary)
        or vocabulary != sorted(vocabulary)
    ):
        raise ItemSelectorPackageError("item vocabulary must be a sorted unique string list")
    if canonical_hash(vocabulary) != manifest["vocabulary_hash"]:
        raise ItemSelectorPackageError("item vocabulary hash mismatch")
    return frozenset(vocabulary)


def _validate_tensor_manifest(manifest: Mapping[str, Any]) -> None:
    """manifest 内の ONNX 入出力宣言が期待仕様と一致するか検証する。

    やさしい説明: 説明書に書かれた入出力の形が決まりどおりかを見ます。
    """
    expected_inputs, expected_outputs = expected_onnx_tensor_manifest(manifest)
    if manifest.get("onnx_input_tensors") != expected_inputs:
        raise ItemSelectorPackageError("ONNX input tensor manifest mismatch")
    if manifest.get("onnx_output_tensors") != expected_outputs:
        raise ItemSelectorPackageError("ONNX output tensor manifest mismatch")


def _validate_metadata_objects(manifest: Mapping[str, Any]) -> None:
    """files / lineage / dependency_versions の構造を検証する。

    やさしい説明: 同梱ファイル一覧、来歴、依存ライブラリ版の書式を確認します。
    """
    files = manifest.get("files")
    if not isinstance(files, Mapping) or frozenset(files) != ITEM_SELECTOR_PACKAGE_FILES:
        raise ItemSelectorPackageError("package files manifest mismatch")
    if any(not _is_sha256(value) for value in files.values()):
        raise ItemSelectorPackageError("package file hash must be lowercase SHA-256")
    lineage = manifest.get("lineage")
    if not isinstance(lineage, Mapping) or frozenset(lineage) != _LINEAGE_FIELDS:
        raise ItemSelectorPackageError("lineage fields mismatch")
    for field in _LINEAGE_FIELDS - {"model_training_run_id"}:
        if not _is_sha256(lineage.get(field)):
            raise ItemSelectorPackageError(f"lineage {field} must be lowercase SHA-256")
    if not isinstance(lineage.get("model_training_run_id"), str) or not lineage[
        "model_training_run_id"
    ]:
        raise ItemSelectorPackageError("model training run ID must be non-empty")
    dependencies = manifest.get("dependency_versions")
    if (
        not isinstance(dependencies, Mapping)
        or frozenset(dependencies) != _DEPENDENCY_FIELDS
        or any(not isinstance(value, str) or not value for value in dependencies.values())
    ):
        raise ItemSelectorPackageError("dependency versions manifest mismatch")


def validate_item_selector_manifest(manifest: Mapping[str, Any]) -> frozenset[str]:
    """manifest 全体を検証し、artifact identity 一致まで確認して vocabulary を返す。

    やさしい説明: 説明書の中身をすべて点検し、説明書自身の指紋（identity）が
    内容と一致することまで確かめます。
    """
    _validate_scalar_manifest(manifest)
    _validate_metadata_objects(manifest)
    _validate_tensor_manifest(manifest)
    expected_identity = canonical_hash(
        {key: value for key, value in manifest.items() if key != "artifact_identity"}
    )
    if manifest["artifact_identity"] != expected_identity:
        raise ItemSelectorPackageError("artifact identity mismatch")
    return _validate_vocabulary(manifest)


def verify_item_selector_package_files(
    package_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    """package の実ファイル集合と全 byte hash を manifest と照合する。

    やさしい説明: 箱の中のファイルを実際に読み、説明書に書かれた指紋と
    一致するかを 1 つずつ確認します。ここを通るまでモデルは読み込みません。
    """
    package = Path(package_dir)
    try:
        package_entries = {path.name for path in package.iterdir()}
    except OSError as exc:
        raise ItemSelectorPackageError(f"cannot enumerate artifact package: {exc}") from exc
    expected_entries = set(ITEM_SELECTOR_PACKAGE_FILES) | {"manifest.json"}
    if package_entries != expected_entries:
        raise ItemSelectorPackageError("artifact package contains missing or unlisted files")

    actual_hashes: dict[str, str] = {}
    for name in sorted(ITEM_SELECTOR_PACKAGE_FILES):
        path = package / name
        if path.is_symlink() or not path.is_file():
            raise ItemSelectorPackageError(f"package file must be a regular file: {name}")
        try:
            actual_hashes[name] = sha256_hex(path.read_bytes())
        except OSError as exc:
            raise ItemSelectorPackageError(f"cannot read package file {name}: {exc}") from exc
        if actual_hashes[name] != manifest["files"][name]:
            raise ItemSelectorPackageError(f"package file hash mismatch: {name}")
    if actual_hashes["model.pt"] != manifest["model_state_hash"]:
        raise ItemSelectorPackageError("model state hash mismatch")
    if actual_hashes["model.onnx"] != manifest["onnx_model_hash"]:
        raise ItemSelectorPackageError("ONNX model hash mismatch")
    return actual_hashes


def load_verified_ui_policy_config(
    package_dir: Path, manifest: Mapping[str, Any]
) -> NonModelUiPolicyConfigV1:
    """同梱 ui_policy_config.json を install 済み config と対称検証して返す。

    やさしい説明: 箱に入っている UI ルール設定が、いま入っている共有ルールと
    完全に同じかを確認します。違えば起動しません。
    """
    package = Path(package_dir)
    config_wire = _read_json_object(package / "ui_policy_config.json", label="UI policy config")
    try:
        config = NonModelUiPolicyConfigV1.from_wire(config_wire)
        installed = NonModelUiPolicyConfigV1.load_default()
    except ValueError as exc:
        raise ItemSelectorPackageError(f"UI policy config is invalid: {exc}") from exc
    if canonical_json_bytes(config_wire) != (package / "ui_policy_config.json").read_bytes():
        raise ItemSelectorPackageError("UI policy config must use canonical JSON bytes")
    if config.to_wire() != installed.to_wire():
        raise ItemSelectorPackageError("installed policy config mismatch")
    verify_ui_policy_binding(manifest, installed=installed)
    return config


def verify_ui_policy_binding(
    manifest: Mapping[str, Any],
    *,
    installed: NonModelUiPolicyConfigV1 | None = None,
) -> None:
    """manifest の policy schema / config / impl hash を install 済み実体と照合する。

    やさしい説明: 「どの UI ルールで作られた箱か」が、いま動く UI ルールと
    一致しているかを 3 種類のハッシュで確認します。
    """
    resolved = installed if installed is not None else NonModelUiPolicyConfigV1.load_default()
    assert_single_installed_ui_policy_distribution()
    if manifest.get("policy_schema_hash") != policy_schema_hash(resolved):
        raise ItemSelectorPackageError("policy schema hash mismatch")
    if manifest.get("policy_config_hash") != resolved.config_hash:
        raise ItemSelectorPackageError("policy config hash mismatch")
    if manifest.get("policy_impl_hash") != installed_policy_impl_hash():
        raise ItemSelectorPackageError("policy implementation hash mismatch")

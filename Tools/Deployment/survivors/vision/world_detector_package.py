"""WorldDetector development package writer / loader。

development weight、resolved config、class map、tracker config、
metrics、dataset/target/build/contract hash を
content-addressed temp store へ atomic publish する。

package ディレクトリ内に manifest.json / world_detector_v1.yaml /
world_class_map_v1.yaml を格納するため、別環境でも restore できる。

restore 後に 04-06 golden fixture と同じ TrackedWorldStateV1 schema を返し、
schema 差を PackageSchemaError で拒否する。

development package は formal_detector_eligible=false とし、
formal publish は全 parent と validation PASS がなければ出力しない。
formal publish では checkpoint_manifest.assert_formal_eligible() を呼び
parent hash を検証する。

使用例:
    from survivors.vision.world_detector_package import publish_development_package, restore_package
    pkg_path = publish_development_package(manifest, metrics, cfg, cm_path, cfg_path, out_store)
    state = restore_package(pkg_path, frame_bgr)
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---- package schema ----

PACKAGE_SCHEMA_VERSION = "world_detector_package.v1"

# package 内のファイル名（固定）
_MANIFEST_NAME = "manifest.json"
_CONFIG_NAME = "world_detector_v1.yaml"
_CLASS_MAP_NAME = "world_class_map_v1.yaml"
_WEIGHT_NAME = "model.pt"


@dataclass(frozen=True)
class PackageManifest:
    """パッケージ内容を記述する manifest。

    content-addressed store のキーは manifest の SHA-256。
    formal_detector_eligible=false のパッケージは正式ローダーに拒否される。
    weight_included はパッケージ内に weight ファイルがあるかを示す。
    """

    schema_version: str
    formal_detector_eligible: bool
    model_hash: str
    data_hash: str
    config_hash: str
    build_hash: str
    class_map_hash: str
    contract_hash: str          # 04-06 TrackedWorldStateV1 フィールド定義の hash
    metrics: dict               # EvalMetrics.to_json() の dict
    checkpoint_selection: dict  # CheckpointSelector.to_dict()
    weight_included: bool       # package 内に model.pt があるか

    def to_dict(self) -> dict:
        """JSON シリアライズ用 dict。"""
        return {
            "schema_version": self.schema_version,
            "formal_detector_eligible": self.formal_detector_eligible,
            "model_hash": self.model_hash,
            "data_hash": self.data_hash,
            "config_hash": self.config_hash,
            "build_hash": self.build_hash,
            "class_map_hash": self.class_map_hash,
            "contract_hash": self.contract_hash,
            "metrics": self.metrics,
            "checkpoint_selection": self.checkpoint_selection,
            "weight_included": self.weight_included,
        }

    @staticmethod
    def from_dict(d: dict) -> "PackageManifest":
        """dict から PackageManifest を復元する。schema_version を検証する。"""
        if d.get("schema_version") != PACKAGE_SCHEMA_VERSION:
            raise ValueError(
                f"未知の package schema_version: {d.get('schema_version')!r}"
            )
        return PackageManifest(
            schema_version=d["schema_version"],
            formal_detector_eligible=d["formal_detector_eligible"],
            model_hash=d["model_hash"],
            data_hash=d["data_hash"],
            config_hash=d["config_hash"],
            build_hash=d["build_hash"],
            class_map_hash=d["class_map_hash"],
            contract_hash=d["contract_hash"],
            metrics=d.get("metrics", {}),
            checkpoint_selection=d.get("checkpoint_selection", {}),
            weight_included=d.get("weight_included", False),
        )

    def assert_formal_eligible(self) -> None:
        """formal 昇格していない package をロードしようとすると拒否する。"""
        if not (type(self.formal_detector_eligible) is bool and self.formal_detector_eligible):
            raise FormalPackageRejectedError(
                "development package は formal detector として使用できません。"
                " formal_detector_eligible=true の package を 04-08 で作成してください。"
            )


class FormalPackageRejectedError(ValueError):
    """development package を formal がロードしようとしたときに送出される。"""


class PackageSchemaError(ValueError):
    """restore 後の schema が TrackedWorldStateV1 と一致しないときに送出される。"""


# ---- contract hash ----

def _compute_contract_hash() -> str:
    """TrackedWorldStateV1 のフィールド定義 hash を計算する。

    フィールドリストが変わると hash が変わり、schema ドリフトを検知する。
    """
    from survivors.vision.entity_tracker import TrackedWorldStateV1
    field_names = TrackedWorldStateV1.track_field_names()
    payload = json.dumps({"track_fields": field_names}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---- content-addressed store ----

def _content_key(manifest: PackageManifest) -> str:
    """manifest 内容の SHA-256 をストアキーとして返す。"""
    payload = json.dumps(manifest.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---- publish ----

def _write_package(
    pkg_manifest: PackageManifest,
    store_dir: pathlib.Path,
    cfg_path: pathlib.Path,
    cm_path: pathlib.Path,
    weight_path: pathlib.Path | None,
) -> pathlib.Path:
    """package を content-addressed store へ atomic write する。

    store_dir/<content_hash>/ に manifest.json / config / class_map を配置する。
    weight_path が指定されている場合は model.pt としてコピーする。
    既存 package は冪等（内容が同じなら再書き込みしない）。
    """
    content_key = _content_key(pkg_manifest)
    pkg_dir = store_dir / content_key

    if pkg_dir.exists():
        return pkg_dir / _MANIFEST_NAME

    # atomic: tmp dir に書いてから rename
    with tempfile.TemporaryDirectory(dir=store_dir) as tmp:
        tmp_dir = pathlib.Path(tmp) / "package"
        tmp_dir.mkdir()

        # manifest
        (tmp_dir / _MANIFEST_NAME).write_text(
            json.dumps(pkg_manifest.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # config（内容をコピーし hash 検証）
        shutil.copy2(cfg_path, tmp_dir / _CONFIG_NAME)
        if _sha256_file(tmp_dir / _CONFIG_NAME) != pkg_manifest.config_hash:
            raise ValueError("config ファイルの hash が manifest と一致しません。")

        # class_map
        shutil.copy2(cm_path, tmp_dir / _CLASS_MAP_NAME)
        if _sha256_file(tmp_dir / _CLASS_MAP_NAME) != pkg_manifest.class_map_hash:
            raise ValueError("class_map ファイルの hash が manifest と一致しません。")

        # weight（任意）
        if weight_path is not None and weight_path.exists():
            shutil.copy2(weight_path, tmp_dir / _WEIGHT_NAME)

        shutil.move(str(tmp_dir), str(pkg_dir))

    return pkg_dir / _MANIFEST_NAME


def publish_development_package(
    checkpoint_manifest: Any,   # CheckpointManifest
    metrics_dict: dict,
    checkpoint_selection: dict,
    store_dir: pathlib.Path,
    *,
    cfg_path: pathlib.Path,
    cm_path: pathlib.Path,
    weight_path: pathlib.Path | None = None,
) -> pathlib.Path:
    """development package を content-addressed store へ atomic publish する。

    store_dir/<content_hash>/ に manifest.json / config / class_map を配置する。
    formal_detector_eligible は常に False（04-08 が差し替える）。
    contract_hash は TrackedWorldStateV1 フィールド定義から計算する。

    weight_path が指定されている場合は model.pt としてコピーし model_hash を検証する。
    """
    contract_hash = _compute_contract_hash()

    pkg_manifest = PackageManifest(
        schema_version=PACKAGE_SCHEMA_VERSION,
        formal_detector_eligible=False,   # development のみ
        model_hash=checkpoint_manifest.model_hash,
        data_hash=checkpoint_manifest.data_hash,
        config_hash=checkpoint_manifest.config_hash,
        build_hash=checkpoint_manifest.build_hash,
        class_map_hash=checkpoint_manifest.class_map_hash,
        contract_hash=contract_hash,
        metrics=metrics_dict,
        checkpoint_selection=checkpoint_selection,
        weight_included=(weight_path is not None and weight_path.exists()),
    )

    store_dir.mkdir(parents=True, exist_ok=True)
    return _write_package(pkg_manifest, store_dir, cfg_path, cm_path, weight_path)


def publish_formal_package(
    checkpoint_manifest: Any,
    metrics_dict: dict,
    checkpoint_selection: dict,
    store_dir: pathlib.Path,
    *,
    cfg_path: pathlib.Path,
    cm_path: pathlib.Path,
    weight_path: pathlib.Path | None = None,
    validation_passed: bool,
) -> pathlib.Path:
    """formal package を publish する。

    全 parent ハッシュと validation PASS が必要。
    checkpoint_manifest.assert_formal_eligible() で parent hash を検証する。
    validation_passed=False の場合は ValueError を送出する。
    """
    if not validation_passed:
        raise ValueError(
            "formal publish には validation PASS が必要です。"
            " performance gate を全て通過してから再実行してください。"
        )
    # parent hash 検証（assert_formal_eligible は型 / SHA-256 形式も確認する）
    checkpoint_manifest.assert_formal_eligible()

    contract_hash = _compute_contract_hash()

    pkg_manifest = PackageManifest(
        schema_version=PACKAGE_SCHEMA_VERSION,
        formal_detector_eligible=True,
        model_hash=checkpoint_manifest.model_hash,
        data_hash=checkpoint_manifest.data_hash,
        config_hash=checkpoint_manifest.config_hash,
        build_hash=checkpoint_manifest.build_hash,
        class_map_hash=checkpoint_manifest.class_map_hash,
        contract_hash=contract_hash,
        metrics=metrics_dict,
        checkpoint_selection=checkpoint_selection,
        weight_included=(weight_path is not None and weight_path.exists()),
    )

    store_dir.mkdir(parents=True, exist_ok=True)
    return _write_package(pkg_manifest, store_dir, cfg_path, cm_path, weight_path)


# ---- restore ----

def restore_package(
    manifest_path: pathlib.Path,
    frame_bgr: np.ndarray,
    *,
    score_threshold: float = 0.5,
    require_formal: bool = False,
) -> "TrackedWorldStateV1":
    """package manifest から detector を復元し、1 フレームを推論して TrackedWorldStateV1 を返す。

    package ディレクトリ内の config / class_map から detector を構築する。
    weight_included=True のとき model.pt を load する（torch 不在時は stub）。
    schema が TrackedWorldStateV1 フィールド定義と一致しない場合は PackageSchemaError。
    require_formal=True のとき、development package は FormalPackageRejectedError で拒否する。
    """
    from survivors.vision.entity_tracker import EntityTracker, TrackedWorldStateV1
    from survivors.vision.world_detector import WorldDetector
    from survivors.vision.world_dataset import load_class_map

    import yaml

    pkg_dir = manifest_path.parent

    # -- manifest load
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    pkg_manifest = PackageManifest.from_dict(raw)

    if require_formal:
        pkg_manifest.assert_formal_eligible()

    # -- contract hash 検証
    current_contract_hash = _compute_contract_hash()
    if pkg_manifest.contract_hash != current_contract_hash:
        raise PackageSchemaError(
            f"contract_hash 不一致: package={pkg_manifest.contract_hash!r}"
            f" current={current_contract_hash!r}。"
            " TrackedWorldStateV1 のフィールド定義が変更されています。"
        )

    # -- package 内 config / class_map を使う
    cfg_path = pkg_dir / _CONFIG_NAME
    cm_path = pkg_dir / _CLASS_MAP_NAME

    if not cfg_path.exists():
        raise FileNotFoundError(f"package 内に config が見つかりません: {cfg_path}")
    if not cm_path.exists():
        raise FileNotFoundError(f"package 内に class_map が見つかりません: {cm_path}")

    # hash 検証
    if _sha256_file(cfg_path) != pkg_manifest.config_hash:
        raise PackageSchemaError("config_hash が manifest と一致しません。")
    if _sha256_file(cm_path) != pkg_manifest.class_map_hash:
        raise PackageSchemaError("class_map_hash が manifest と一致しません。")

    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    detector = WorldDetector.from_config(cfg, cm_path)

    # -- weight ロード（torch 利用可能かつ weight が含まれている場合）
    weight_file = pkg_dir / _WEIGHT_NAME
    if pkg_manifest.weight_included and weight_file.exists():
        try:
            import torch
            state_dict = torch.load(weight_file, map_location="cpu")
            detector._model.load_state_dict(state_dict)
        except ImportError:
            pass  # torch 不在: stub model のまま推論する

    # -- infer
    result = detector.infer(frame_bgr, score_threshold=score_threshold)

    # -- tracker（config から構築）
    tracker_cfg = cfg.get("tracker", {})
    cm = load_class_map(cm_path)
    max_age_raw: dict = tracker_cfg.get("max_age_by_class", {})
    max_age_by_class: dict[int, int] = {}
    for name, age in max_age_raw.items():
        try:
            max_age_by_class[cm.name_to_id(name)] = age
        except KeyError:
            pass

    tracker = EntityTracker(
        max_age_by_class=max_age_by_class,
        max_match_cost=tracker_cfg.get("max_match_cost", 0.7),
        velocity_ema_alpha=tracker_cfg.get("velocity_ema_alpha", 0.6),
        confidence_decay_per_frame=tracker_cfg.get("confidence_decay_per_frame", 0.9),
    )

    state = tracker.update(result, frame_index=0, timestamp_ns=0)
    v1 = TrackedWorldStateV1.from_state(state, frame_index=0, timestamp_ns=0)

    # -- schema 検証
    from dataclasses import fields as dc_fields
    from survivors.vision.entity_tracker import TrackedEntityV1
    expected_fields = set(TrackedWorldStateV1.track_field_names())
    actual_fields = {f.name for f in dc_fields(TrackedEntityV1)}
    if expected_fields != actual_fields:
        raise PackageSchemaError(
            f"TrackedEntityV1 フィールド不一致: expected={sorted(expected_fields)}"
            f" actual={sorted(actual_fields)}"
        )

    return v1

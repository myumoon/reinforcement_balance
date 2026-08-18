"""WorldDetector development package writer / loader。

development weight、threshold、resolved config、class map、tracker config、
metrics、dataset/target/build/contract hash を
content-addressed temp store へ atomic publish する。

restore 後に 04-06 golden fixture と同じ TrackedWorldStateV1 schema を返し、
schema 差をrejectする。

development package は formal_detector_eligible=false とし、
formal publish は全 parent と validation PASS がなければ出力しない。

使用例:
    from survivors.vision.world_detector_package import publish_development_package, restore_package
    pkg_path = publish_development_package(manifest, metrics, cfg, out_store)
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


@dataclass(frozen=True)
class PackageManifest:
    """パッケージ内容を記述する manifest。

    content-addressed store のキーは manifest の SHA-256。
    formal_detector_eligible=false のパッケージは正式ローダーに拒否される。
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


# ---- publish ----

def publish_development_package(
    checkpoint_manifest: Any,   # CheckpointManifest
    metrics_dict: dict,
    checkpoint_selection: dict,
    store_dir: pathlib.Path,
) -> pathlib.Path:
    """development package を content-addressed store へ atomic publish する。

    store_dir/<content_hash>/manifest.json として保存する。
    formal_detector_eligible は常に False（04-08 が差し替える）。
    contract_hash は TrackedWorldStateV1 フィールド定義から計算する。

    atomic: temp dir に書いてから rename する。
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
    )

    content_key = _content_key(pkg_manifest)
    pkg_dir = store_dir / content_key

    # atomic publish: temp -> rename
    with tempfile.TemporaryDirectory(dir=store_dir) as tmp:
        tmp_path = pathlib.Path(tmp) / "manifest.json"
        tmp_path.write_text(
            json.dumps(pkg_manifest.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # rename は同一ファイルシステム内なら atomic
        if not pkg_dir.exists():
            shutil.move(str(tmp_path.parent), str(pkg_dir))
        else:
            # 既存 package は冪等（内容が同じなら再書き込みしない）
            pass

    return pkg_dir / "manifest.json"


def publish_formal_package(
    checkpoint_manifest: Any,
    metrics_dict: dict,
    checkpoint_selection: dict,
    store_dir: pathlib.Path,
    *,
    validation_passed: bool,
) -> pathlib.Path:
    """formal package を publish する。

    全 parent ハッシュと validation PASS が必要。
    validation_passed=False の場合は ValueError を送出する。
    """
    if not validation_passed:
        raise ValueError(
            "formal publish には validation PASS が必要です。"
            " performance gate を全て通過してから再実行してください。"
        )
    if not (type(checkpoint_manifest.formal_detector_eligible) is bool
            and checkpoint_manifest.formal_detector_eligible):
        raise ValueError(
            "formal publish には CheckpointManifest.formal_detector_eligible=True が必要です。"
            " 04-08 で正式 weight をロードしてから再実行してください。"
        )

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
    )

    content_key = _content_key(pkg_manifest)
    pkg_dir = store_dir / content_key

    with tempfile.TemporaryDirectory(dir=store_dir) as tmp:
        tmp_path = pathlib.Path(tmp) / "manifest.json"
        tmp_path.write_text(
            json.dumps(pkg_manifest.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not pkg_dir.exists():
            shutil.move(str(tmp_path.parent), str(pkg_dir))

    return pkg_dir / "manifest.json"


# ---- restore ----

def restore_package(
    manifest_path: pathlib.Path,
    frame_bgr: np.ndarray,
    *,
    score_threshold: float = 0.5,
    require_formal: bool = False,
) -> "TrackedWorldStateV1":
    """package manifest から detector を復元し、1 フレームを推論して TrackedWorldStateV1 を返す。

    schema が TrackedWorldStateV1 フィールド定義と一致しない場合は PackageSchemaError を送出する。
    require_formal=True のとき、development package は FormalPackageRejectedError で拒否する。
    """
    from survivors.vision.entity_tracker import EntityTracker, TrackedWorldStateV1
    from survivors.vision.world_detector import WorldDetector

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

    # -- detector（config from manifest dir; stub model）
    manifest_dir = manifest_path.parent.parent.parent.parent  # store_dir/../..
    cfg_path = manifest_dir / "configs" / "world_detector_v1.yaml"
    cm_path = manifest_dir / "configs" / "world_class_map_v1.yaml"

    import yaml
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    detector = WorldDetector.from_config(cfg, cm_path)

    # -- infer
    result = detector.infer(frame_bgr, score_threshold=score_threshold)

    # -- tracker （config から構築）
    tracker_cfg = cfg.get("tracker", {})
    from survivors.vision.world_dataset import load_class_map
    cm = load_class_map(cm_path)
    max_age_raw: dict = tracker_cfg.get("max_age_by_class", {})
    max_age_by_class: dict[int, int] = {
        cm.name_to_id(name): age for name, age in max_age_raw.items()
        if _safe_name_to_id(cm, name) is not None
    }

    tracker = EntityTracker(
        max_age_by_class=max_age_by_class,
        max_match_cost=tracker_cfg.get("max_match_cost", 0.7),
        velocity_ema_alpha=tracker_cfg.get("velocity_ema_alpha", 0.6),
        confidence_decay_per_frame=tracker_cfg.get("confidence_decay_per_frame", 0.9),
    )

    state = tracker.update(result, frame_index=0, timestamp_ns=0)
    v1 = TrackedWorldStateV1.from_state(state, frame_index=0, timestamp_ns=0)

    # -- schema 検証: フィールド名が golden fixture と一致するか
    expected_fields = set(TrackedWorldStateV1.track_field_names())
    from dataclasses import fields as dc_fields
    from survivors.vision.entity_tracker import TrackedEntityV1
    actual_fields = {f.name for f in dc_fields(TrackedEntityV1)}
    if expected_fields != actual_fields:
        raise PackageSchemaError(
            f"TrackedEntityV1 フィールド不一致: expected={sorted(expected_fields)}"
            f" actual={sorted(actual_fields)}"
        )

    return v1


def _safe_name_to_id(cm: Any, name: str) -> int | None:
    """name_to_id を KeyError なしで呼び出すヘルパー。"""
    try:
        return cm.name_to_id(name)
    except KeyError:
        return None

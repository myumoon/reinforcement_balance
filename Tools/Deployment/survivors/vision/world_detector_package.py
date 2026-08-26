"""WorldDetector development package writer / loader（PR#315 development-only）。

development weight、resolved config、class map、tracker config、
metrics、dataset/target/build/contract hash を
content-addressed temp store へ atomic publish する。

package ディレクトリ内に manifest.json / world_detector_v1.yaml /
world_class_map_v1.yaml を格納するため、別環境でも restore できる。

restore 後に 04-06 golden fixture と同じ TrackedWorldStateV1 schema を返し、
schema 差を PackageSchemaError で拒否する。

PR#315 では development package のみ提供する。formal publish は 04-08 に委譲する。
publish_formal_package() は呼び出すと常に FormalPackageRejectedError を送出する。

使用例:
    from survivors.vision.world_detector_package import publish_development_package, restore_package
    pkg_path = publish_development_package(
        manifest, metrics, checkpoint_selection, out_store,
        cfg_path=cfg_path, cm_path=cm_path, weight_path=weight_path,
    )
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
    score_threshold: float      # restore_package の推論スコア閾値（再現性保証）
    development_only: bool = True   # 常に True。04-08 以外で False にはならない。
    training_mode: str = "development"  # "smoke" | "development"

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
            "score_threshold": self.score_threshold,
            "development_only": self.development_only,
            "training_mode": self.training_mode,
        }

    @staticmethod
    def from_dict(d: dict) -> "PackageManifest":
        """dict から PackageManifest を復元する。schema_version を検証する。

        04-07 境界: formal_detector_eligible は必ず False へ上書き、development_only は True、
        training_mode は "smoke" | "development" のみ許容する。
        """
        if d.get("schema_version") != PACKAGE_SCHEMA_VERSION:
            raise ValueError(
                f"未知の package schema_version: {d.get('schema_version')!r}"
            )
        raw_mode = d.get("training_mode", "development")
        if raw_mode not in ("smoke", "development"):
            raise ValueError(
                f"training_mode の許容値は 'smoke' | 'development' です。got: {raw_mode!r}"
            )
        return PackageManifest(
            schema_version=d["schema_version"],
            formal_detector_eligible=False,    # 04-07 は常に False
            model_hash=d["model_hash"],
            data_hash=d["data_hash"],
            config_hash=d["config_hash"],
            build_hash=d["build_hash"],
            class_map_hash=d["class_map_hash"],
            contract_hash=d["contract_hash"],
            metrics=d.get("metrics", {}),
            checkpoint_selection=d.get("checkpoint_selection", {}),
            weight_included=d.get("weight_included", False),
            score_threshold=float(d.get("score_threshold", 0.5)),
            development_only=True,             # 04-07 は常に True
            training_mode=raw_mode,
        )

    def assert_formal_eligible(self) -> None:
        """04-07 の package は caller 指定フラグに関わらず常に formal 拒否する。

        formal_detector_eligible の値に関係なく例外を送出する。
        formal 昇格は 04-08 の validated verdict によってのみ可能。
        """
        raise FormalPackageRejectedError(
            "04-07 package は formal detector として使用できません。"
            " formal 昇格は 04-08 の検証済み verdict によってのみ行われます。"
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


def _verify_weight_hash(weight_path: pathlib.Path, expected_hash: str) -> None:
    """weight の存在と model_hash 一致を fail-closed で検証する。"""
    if not weight_path.is_file():
        raise ValueError(f"weight ファイルが見つかりません: {weight_path}")
    actual_hash = _sha256_file(weight_path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"weight の model_hash が manifest と一致しません: "
            f"expected={expected_hash}, actual={actual_hash}"
        )


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
    weight_path が指定されている場合は copy 前後に model_hash を検証する。
    既存 package は冪等（内容が同じなら再書き込みしない）。
    """
    if weight_path is not None:
        weight_path = pathlib.Path(weight_path)
        _verify_weight_hash(weight_path, pkg_manifest.model_hash)
    if pkg_manifest.weight_included != (weight_path is not None):
        raise ValueError("manifest の weight_included と weight_path が一致しません。")

    content_key = _content_key(pkg_manifest)
    pkg_dir = store_dir / content_key

    if pkg_dir.exists():
        # 既存 package の完全性を検証する。欠落・破損があれば明示的に失敗させる。
        _manifest_path = pkg_dir / _MANIFEST_NAME
        if not _manifest_path.exists():
            raise PackageSchemaError(
                f"既存 package ディレクトリ {pkg_dir} に manifest が存在しません。"
                " 破損しています。ディレクトリを削除して再 publish してください。"
            )
        # manifest が読める状態かを確認し、content key が dir 名と一致するかを検証する
        try:
            _cached_pkg = PackageManifest.from_dict(json.loads(_manifest_path.read_text(encoding="utf-8")))
            if _content_key(_cached_pkg) != pkg_dir.name:
                raise PackageSchemaError(
                    f"既存 package の manifest content key が dir 名と一致しません: {pkg_dir}"
                )
        except (json.JSONDecodeError, KeyError, ValueError) as _e:
            raise PackageSchemaError(
                f"既存 package の manifest 読み込みに失敗しました: {_manifest_path}: {_e}"
            ) from _e
        _cfg_cached = pkg_dir / _CONFIG_NAME
        if not _cfg_cached.exists() or _sha256_file(_cfg_cached) != pkg_manifest.config_hash:
            raise PackageSchemaError(
                f"既存 package の config が欠落または改ざんされています: {_cfg_cached}"
            )
        _cm_cached = pkg_dir / _CLASS_MAP_NAME
        if not _cm_cached.exists() or _sha256_file(_cm_cached) != pkg_manifest.class_map_hash:
            raise PackageSchemaError(
                f"既存 package の class_map が欠落または改ざんされています: {_cm_cached}"
            )
        if pkg_manifest.weight_included:
            _verify_weight_hash(pkg_dir / _WEIGHT_NAME, pkg_manifest.model_hash)
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

        # weight（指定時は copy 前後で同じ model_hash を要求）
        if weight_path is not None:
            _verify_weight_hash(weight_path, pkg_manifest.model_hash)
            copied_weight = tmp_dir / _WEIGHT_NAME
            shutil.copy2(weight_path, copied_weight)
            _verify_weight_hash(copied_weight, pkg_manifest.model_hash)

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
    score_threshold: float = 0.5,
) -> pathlib.Path:
    """development package を content-addressed store へ atomic publish する。

    store_dir/<content_hash>/ に manifest.json / config / class_map を配置する。
    formal_detector_eligible は常に False（04-08 が差し替える）。
    contract_hash は TrackedWorldStateV1 フィールド定義から計算する。

    weight_path が指定されている場合は model.pt としてコピーし、copy 前後で
    model_hash を検証する。指定した path の欠落や不一致は publish を失敗させる。
    score_threshold は restore_package での推論スコア閾値として manifest に保存する。
    """
    # ---- 書き込み境界バリデーション（fail-closed）----
    import math as _math
    raw_mode = getattr(checkpoint_manifest, "training_mode", "development")
    if raw_mode not in ("smoke", "development"):
        raise ValueError(
            f"publish_development_package: training_mode の許容値は 'smoke' | 'development' です。"
            f" got: {raw_mode!r}"
        )
    raw_dev_only = getattr(checkpoint_manifest, "development_only", True)
    if raw_dev_only is not True:
        raise ValueError(
            "publish_development_package: development_only は True でなければなりません。"
        )
    if (
        isinstance(score_threshold, bool)
        or not isinstance(score_threshold, (int, float))
        or not _math.isfinite(score_threshold)
        or not (0.0 <= score_threshold <= 1.0)
    ):
        raise ValueError(
            f"publish_development_package: score_threshold は [0, 1] の有限数値である必要があります。"
            f" got: {score_threshold!r}"
        )

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
        weight_included=(weight_path is not None),
        score_threshold=score_threshold,
        development_only=getattr(checkpoint_manifest, "development_only", True),
        training_mode=getattr(checkpoint_manifest, "training_mode", "development"),
    )

    store_dir.mkdir(parents=True, exist_ok=True)
    return _write_package(pkg_manifest, store_dir, cfg_path, cm_path, weight_path)


def publish_formal_package(*_args: Any, **_kwargs: Any) -> pathlib.Path:
    """formal package publish は PR#315 の責務外。常に FormalPackageRejectedError を送出する。

    formal weight / threshold / gate の確定は 04-08 に委譲する。
    """
    raise FormalPackageRejectedError(
        "publish_formal_package() は PR#315 では無効です。"
        " formal package の発行は 04-08 で実施してください。"
    )


# ---- restore ----

def restore_package(
    manifest_path: pathlib.Path,
    frame_bgr: np.ndarray,
    *,
    require_formal: bool = False,
) -> "TrackedWorldStateV1":
    """package manifest から detector を復元し、1 フレームを推論して TrackedWorldStateV1 を返す。

    package ディレクトリ内の config / class_map から detector を構築する。
    weight_included=True のとき model.pt を load する（torch 不在時は stub）。
    schema が TrackedWorldStateV1 フィールド定義と一致しない場合は PackageSchemaError。
    require_formal=True のとき、development package は FormalPackageRejectedError で拒否する。
    推論スコア閾値は manifest の score_threshold を使用する（呼び出し側引数不可）。
    """
    from survivors.vision.entity_tracker import EntityTracker, TrackedWorldStateV1
    from survivors.vision.world_detector import WorldDetector
    from survivors.vision.world_dataset import load_class_map

    import yaml

    pkg_dir = manifest_path.parent

    # -- manifest load
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    pkg_manifest = PackageManifest.from_dict(raw)

    # content-addressed ディレクトリ名が manifest 内容と一致するか検証（改ざん検知）
    expected_content_key = _content_key(pkg_manifest)
    actual_dir_name = manifest_path.parent.name
    if actual_dir_name != expected_content_key:
        raise PackageSchemaError(
            f"package ディレクトリ名 {actual_dir_name!r} が manifest の content key と一致しません。"
            " manifest が改ざんされている可能性があります。"
        )

    if require_formal:
        pkg_manifest.assert_formal_eligible()
    if pkg_manifest.formal_detector_eligible and not pkg_manifest.weight_included:
        raise PackageSchemaError("formal package に必須の weight が含まれていません。")

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

    # -- weight hash 検証・ロード（hash は torch の有無にかかわらず必須）
    weight_file = pkg_dir / _WEIGHT_NAME
    if pkg_manifest.weight_included:
        if not weight_file.is_file():
            raise PackageSchemaError(f"package 内に weight が見つかりません: {weight_file}")
        if _sha256_file(weight_file) != pkg_manifest.model_hash:
            raise PackageSchemaError("weight の model_hash が manifest と一致しません。")
        try:
            import torch
        except ModuleNotFoundError as e:
            if e.name != "torch":
                raise
        else:
            # torch が利用可能な場合のみ weight をロードする（内部エラーは伝播）
            try:
                state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(weight_file, map_location="cpu")
            detector._model.load_state_dict(state_dict)

    # -- infer（manifest の score_threshold で再現性を保証）
    result = detector.infer(frame_bgr, score_threshold=pkg_manifest.score_threshold)

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
    v1 = TrackedWorldStateV1.from_state(state, frame_index=0, timestamp_ns=0, class_map_path=cm_path)

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

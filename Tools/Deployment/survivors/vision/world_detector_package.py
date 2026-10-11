"""WorldDetector development/formal package writer and restore loader。


development weight、resolved config、class map、tracker config、
metrics、dataset/target/build/contract hash を
content-addressed temp store へ atomic publish する。

package ディレクトリ内に manifest.json / world_detector_v2.yaml /
world_class_map_v2.yaml を格納するため、別環境でも restore できる。

restore 後に 04-06 golden fixture と同じ TrackedWorldStateV2 schema を返し、
schema 差を PackageSchemaError で拒否する。

formal package は validation verdict と checkpoint identity が一致した場合だけ発行する。
restore_package(require_formal=True) は verdict、weight、resolved config の hash を再検証する。

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
import math
import pathlib
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any

from survivors.vision.world_detector import validate_detector_config

_SHA256_RE = re.compile(r"[0-9a-f]{64}")

import numpy as np


# ---- package schema ----

PACKAGE_SCHEMA_VERSION = "world_detector_package.v1"

# package 内のファイル名（固定）
_MANIFEST_NAME = "manifest.json"
_CONFIG_NAME = "world_detector_v2.yaml"
_CLASS_MAP_NAME = "world_class_map_v2.yaml"
_WEIGHT_NAME = "model.pt"
_VALIDATION_VERDICT_NAME = "validation_verdict.json"


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
    contract_hash: str          # 04-06 TrackedWorldStateV2 フィールド定義の hash
    metrics: dict               # EvalMetrics.to_json() の dict
    checkpoint_selection: dict  # CheckpointSelector.to_dict()
    weight_included: bool       # package 内に model.pt があるか
    score_threshold: float      # restore_package の推論スコア閾値（再現性保証）
    development_only: bool = True
    training_mode: str = "development"  # "smoke" | "development" | "formal"
    validation_verdict_hash: str = ""

    def to_dict(self) -> dict:
        """package manifest を JSON wire mapping に変換する。\n\n未設定の verdict hash は省き、既存 development package の content key を保つ。"""
        result = {
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
        if self.validation_verdict_hash:
            result["validation_verdict_hash"] = self.validation_verdict_hash
        return result

    @staticmethod
    def from_dict(d: dict) -> "PackageManifest":
        """dict から PackageManifest を復元する。

        formal mode は verdict hash を必須にし、eligibility flags をそのまま検証する。
        disk 由来の manifest はすべて型・値・hash 形式を検証してから構築する。
        score_threshold は必須フィールドで暗黙の変換や default は使わない。
        """
        if not isinstance(d, dict):
            raise ValueError(
                f"PackageManifest.from_dict: dict が必要。got: {type(d).__name__!r}"
            )
        if d.get("schema_version") != PACKAGE_SCHEMA_VERSION:
            raise ValueError(
                f"未知の package schema_version: {d.get('schema_version')!r}"
            )
        raw_mode = d.get("training_mode", "development")
        if raw_mode not in ("smoke", "development", "formal"):
            raise ValueError(
                f"training_mode の許容値は 'smoke' | 'development' | 'formal' です。got: {raw_mode!r}"
            )

        # metrics / checkpoint_selection は dict が必要
        metrics = d.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError(
                f"PackageManifest: metrics は dict が必要。got: {type(metrics).__name__!r}"
            )
        ckpt_sel = d.get("checkpoint_selection", {})
        if not isinstance(ckpt_sel, dict):
            raise ValueError(
                f"PackageManifest: checkpoint_selection は dict が必要。"
                f" got: {type(ckpt_sel).__name__!r}"
            )

        # weight_included は bool が必要
        wi = d.get("weight_included", False)
        if not isinstance(wi, bool):
            raise ValueError(
                f"PackageManifest: weight_included は bool が必要。got: {type(wi).__name__!r}"
            )

        # score_threshold は必須フィールド: bool 不可・有限・[0, 1]
        if "score_threshold" not in d:
            raise ValueError("PackageManifest: score_threshold が欠落しています。")
        st = d["score_threshold"]
        if isinstance(st, bool) or not isinstance(st, (int, float)):
            raise ValueError(
                f"PackageManifest: score_threshold は数値が必要 (bool / 文字列は不可)。"
                f" got: {st!r}"
            )
        if not math.isfinite(st):
            raise ValueError(
                f"PackageManifest: score_threshold は有限値が必要 (NaN / Infinity は不可)。"
                f" got: {st!r}"
            )
        if not (0.0 <= st <= 1.0):
            raise ValueError(
                f"PackageManifest: score_threshold は [0, 1] の範囲が必要。got: {st!r}"
            )

        raw_verdict_hash = d.get("validation_verdict_hash", "")
        if raw_mode == "formal":
            if d.get("development_only") is not False or d.get("formal_detector_eligible") is not True:
                raise ValueError("formal package は development_only=false / formal_detector_eligible=true が必要です")
            if wi is not True:
                raise ValueError("formal package は model weight を必須にします")
            if not isinstance(raw_verdict_hash, str) or not _SHA256_RE.fullmatch(raw_verdict_hash):
                raise ValueError("formal package は validation_verdict_hash に SHA-256 が必要です")
            formal_eligible = True
            development_only = False
        else:
            formal_eligible = False
            development_only = True
            raw_verdict_hash = ""
        pm = PackageManifest(
            schema_version=d["schema_version"],
            formal_detector_eligible=formal_eligible,
            model_hash=d["model_hash"],
            data_hash=d["data_hash"],
            config_hash=d["config_hash"],
            build_hash=d["build_hash"],
            class_map_hash=d["class_map_hash"],
            contract_hash=d["contract_hash"],
            metrics=metrics,
            checkpoint_selection=ckpt_sel,
            weight_included=wi,
            score_threshold=float(st),
            development_only=development_only,
            training_mode=raw_mode,
            validation_verdict_hash=raw_verdict_hash,
        )
        for fname in ("model_hash", "data_hash", "config_hash", "build_hash", "class_map_hash", "contract_hash"):
            h = getattr(pm, fname)
            if not _SHA256_RE.fullmatch(h):
                raise ValueError(
                    f"PackageManifest: {fname} は SHA-256 hex 64 文字が必要です。got: {h!r}"
                )
        return pm

    def assert_formal_eligible(self) -> None:
        """formal identity が揃った package だけを formal runtime に許可する。

        development mode や欠けた verdict hash はすべて fail-closed で拒否する。
        """
        if (
            self.training_mode == "formal"
            and self.development_only is False
            and self.formal_detector_eligible is True
            and isinstance(self.validation_verdict_hash, str)
            and _SHA256_RE.fullmatch(self.validation_verdict_hash)
        ):
            return
        raise FormalPackageRejectedError("formal package の eligibility または verdict hash が不正です。")


class FormalPackageRejectedError(ValueError):
    """formal loader が不適格 package を拒否したことを示す。\n\n不足した formal verdict や eligibility の理由を呼出側へ伝える。"""


class PackageSchemaError(ValueError):
    """package schema が runtime contract と一致しないことを示す。\n\nrestore 前に model や config を不整合な wire format から守る。"""


# ---- contract hash ----

def _compute_contract_hash() -> str:
    """TrackedWorldStateV2 契約全体の SHA-256 を計算する。

    TrackedWorldStateV2・TrackedEntityV2・PlayerAnchorState の
    フィールド名と型注釈を含む canonical descriptor を JSON 化してハッシュする。
    トップレベルフィールドの追加・削除・型変更でもハッシュが変わり、
    schema ドリフトを確実に検知する。
    """
    import dataclasses as _dc
    from survivors.vision.entity_tracker import (
        TrackedWorldStateV2, TrackedEntityV2, PlayerAnchorState,
    )

    def _descriptor(cls: type) -> dict:
        """dataclass の field 名と型注釈を辞書へまとめる。\n\ncontract hash に型の変更も反映できるよう string 表現を保存する。"""
        return {f.name: str(f.type) for f in _dc.fields(cls)}

    descriptor = {
        "TrackedWorldStateV2": _descriptor(TrackedWorldStateV2),
        "TrackedEntityV2": _descriptor(TrackedEntityV2),
        "PlayerAnchorState": _descriptor(PlayerAnchorState),
    }
    payload = json.dumps(descriptor, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ---- content-addressed store ----

def _content_key(manifest: PackageManifest) -> str:
    """manifest 内容の SHA-256 を store directory key として返す。\n\nwire mapping のみを hash し、package path と manifest identity を一致させる。"""
    payload = json.dumps(manifest.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    """file の全 byte から SHA-256 を計算する。\n\npackage weight と verdict が manifest に束縛された内容か照合する。"""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _verify_weight_hash(weight_path: pathlib.Path, expected_hash: str) -> None:
    """weight file の存在と manifest hash の一致を検証する。\n\n欠落または差し替えを publish と restore の境界で拒否する。"""
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
    validation_verdict_path: pathlib.Path | None = None,
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
    if pkg_manifest.training_mode == "formal":
        if validation_verdict_path is None or _sha256_file(validation_verdict_path) != pkg_manifest.validation_verdict_hash:
            raise ValueError("formal validation verdict hash が manifest と一致しません。")
    elif validation_verdict_path is not None:
        raise ValueError("development package に validation verdict は格納できません。")

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
        if pkg_manifest.training_mode == "formal":
            _verdict_path = pkg_dir / _VALIDATION_VERDICT_NAME
            if not _verdict_path.is_file() or _sha256_file(_verdict_path) != pkg_manifest.validation_verdict_hash:
                raise PackageSchemaError("既存 package の validation verdict が欠落または改ざんされています。")
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
        if validation_verdict_path is not None:
            shutil.copy2(validation_verdict_path, tmp_dir / _VALIDATION_VERDICT_NAME)
            if _sha256_file(tmp_dir / _VALIDATION_VERDICT_NAME) != pkg_manifest.validation_verdict_hash:
                raise ValueError("validation verdict の hash が manifest と一致しません。")

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
    contract_hash は TrackedWorldStateV2 フィールド定義から計算する。

    weight_path が指定されている場合は model.pt としてコピーし、copy 前後で
    model_hash を検証する。指定した path の欠落や不一致は publish を失敗させる。
    score_threshold は restore_package での推論スコア閾値として manifest に保存する。
    """
    # ---- cfg_path の config を共有 validator で検証（model / tracker 構築より前）----
    import yaml as _yaml
    with cfg_path.open(encoding="utf-8") as _f:
        _cfg_to_validate = _yaml.safe_load(_f)
    validate_detector_config(_cfg_to_validate)

    # ---- 書き込み境界バリデーション（fail-closed）----
    import math as _math
    for _fname in ("model_hash", "data_hash", "config_hash", "build_hash", "class_map_hash"):
        _h = getattr(checkpoint_manifest, _fname, None)
        if not isinstance(_h, str) or not _SHA256_RE.fullmatch(_h):
            raise ValueError(
                f"publish_development_package: {_fname} は SHA-256 hex 64 文字が必要です。got: {_h!r}"
            )
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

    # metrics_dict / checkpoint_selection は dict が必要（PackageManifest.from_dict() と同じ制約）
    # content hash 計算・store 書込みより前に拒否することで publish 成功だが restore 不能な状態を防ぐ
    if not isinstance(metrics_dict, dict):
        raise ValueError(
            f"publish_development_package: metrics_dict は dict が必要。"
            f" got: {type(metrics_dict).__name__!r}"
        )
    if not isinstance(checkpoint_selection, dict):
        raise ValueError(
            f"publish_development_package: checkpoint_selection は dict が必要。"
            f" got: {type(checkpoint_selection).__name__!r}"
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


def publish_formal_package(
    *,
    verdict_path: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    out_root: pathlib.Path,
    class_map_path: pathlib.Path,
    config_path: pathlib.Path,
    tracker_config: dict,
) -> pathlib.Path:
    """PASS verdict が checkpoint と一致する場合だけ formal package を保存する。

    package config は training run manifest の resolved config を再構成する。
    """
    import yaml
    from survivors.vision.world_detector import CheckpointManifest
    from survivors.vision.world_formal_release import ValidationVerdict, resolved_config_hash

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    manifest_path = checkpoint_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    checkpoint = CheckpointManifest.load(manifest_path)
    if (
        checkpoint.training_mode != "formal"
        or checkpoint.development_only is not False
        or checkpoint.formal_detector_eligible is not False
    ):
        raise FormalPackageRejectedError("development checkpoint は formal package に昇格できません。")
    verdict_bytes = pathlib.Path(verdict_path).read_bytes()
    verdict = ValidationVerdict.from_dict(json.loads(verdict_bytes))
    if verdict.status != "PASS":
        raise FormalPackageRejectedError("PASS 以外の validation verdict は publish できません。")
    weight_path = checkpoint_dir / _WEIGHT_NAME
    run_manifest = json.loads((checkpoint_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(run_manifest, dict):
        raise FormalPackageRejectedError("checkpoint run manifest が不正です。")
    resolved_cfg = run_manifest.get("resolved_config")
    actual_weight_hash = _sha256_file(weight_path)
    checkpoint_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if (
        verdict.weight_hash != actual_weight_hash
        or checkpoint.model_hash != actual_weight_hash
        or verdict.checkpoint_manifest_hash != checkpoint_hash
        or verdict.dataset_identity != checkpoint.data_hash
        or verdict.resolved_config_hash != checkpoint.resolved_config_hash
        or verdict.build_identity != checkpoint.build_hash
        or run_manifest.get("schema_version") != "world_detector_training_run.v1"
        or run_manifest.get("preflight_canonical_hash") != verdict.preflight_canonical_hash
        or run_manifest.get("dataset_identity") != verdict.dataset_identity
        or run_manifest.get("build_identity") != verdict.build_identity
        or run_manifest.get("code_hashes") != dict(verdict.code_hashes)
        or not isinstance(resolved_cfg, dict)
        or resolved_config_hash(resolved_cfg) != checkpoint.resolved_config_hash
        or not isinstance(tracker_config, dict)
        or resolved_cfg.get("tracker") != tracker_config
    ):
        raise FormalPackageRejectedError("checkpoint、resolved config、weight と verdict の identity が一致しません。")
    if _sha256_file(config_path) != checkpoint.config_hash or _sha256_file(class_map_path) != checkpoint.class_map_hash:
        raise FormalPackageRejectedError("checkpoint の config/class map hash と入力 file が一致しません。")
    if not isinstance(resolved_cfg.get("model"), dict) or resolved_cfg["model"].get("architecture") != "ssdlite320":
        raise FormalPackageRejectedError("formal package は ssdlite320 config のみ発行できます。")

    serialized_cfg = yaml.safe_dump(resolved_cfg, sort_keys=True, allow_unicode=True).encode("utf-8")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config = pathlib.Path(temp_dir) / _CONFIG_NAME
        temp_config.write_bytes(serialized_cfg)
        config_hash = hashlib.sha256(serialized_cfg).hexdigest()
        raw_manifest = json.loads(manifest_bytes)
        selection = raw_manifest.get("checkpoint_selection", run_manifest.get("selection_rule", {}))
        if not isinstance(selection, dict):
            raise FormalPackageRejectedError("checkpoint selection rule が不正です。")
        manifest = PackageManifest(
            schema_version=PACKAGE_SCHEMA_VERSION,
            formal_detector_eligible=True,
            model_hash=actual_weight_hash,
            data_hash=checkpoint.data_hash,
            config_hash=config_hash,
            build_hash=checkpoint.build_hash,
            class_map_hash=checkpoint.class_map_hash,
            contract_hash=_compute_contract_hash(),
            metrics=verdict.metrics,
            checkpoint_selection=selection,
            weight_included=True,
            score_threshold=verdict.score_threshold,
            development_only=False,
            training_mode="formal",
            validation_verdict_hash=hashlib.sha256(verdict_bytes).hexdigest(),
        )
        out_root = pathlib.Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        return _write_package(
            manifest,
            out_root,
            temp_config,
            pathlib.Path(class_map_path),
            weight_path,
            pathlib.Path(verdict_path),
        )


# ---- restore ----

def restore_package(
    manifest_path: pathlib.Path,
    frame_bgr: np.ndarray,
    *,
    require_formal: bool = False,
) -> "TrackedWorldStateV2":
    """package manifest から detector を復元し、1 フレームを推論して TrackedWorldStateV2 を返す。

    package ディレクトリ内の config / class_map から detector を構築する。
    weight_included=True のとき model.pt を load する（torch 不在時は stub）。
    schema が TrackedWorldStateV2 フィールド定義と一致しない場合は PackageSchemaError。
    require_formal=True のとき、development package は FormalPackageRejectedError で拒否する。
    推論スコア閾値は manifest の score_threshold を使用する（呼び出し側引数不可）。
    """
    from survivors.vision.entity_tracker import EntityTracker, TrackedWorldStateV2
    from survivors.vision.world_detector import WorldDetector
    from survivors.vision.world_dataset import load_class_map

    import yaml

    pkg_dir = manifest_path.parent

    if require_formal:
        detector, tracker, pkg_manifest = load_formal_detector(pkg_dir)
        result = _infer_formal_detections(detector, frame_bgr, pkg_manifest.score_threshold)
        state = tracker.update(result, frame_index=0, timestamp_ns=0)
        return TrackedWorldStateV2.from_state(
            state, frame_index=0, timestamp_ns=0,
            class_map_path=pkg_dir / _CLASS_MAP_NAME,
        )

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

    formal_verdict = None
    if pkg_manifest.training_mode == "formal":
        verdict_path = pkg_dir / _VALIDATION_VERDICT_NAME
        if not verdict_path.is_file() or _sha256_file(verdict_path) != pkg_manifest.validation_verdict_hash:
            raise PackageSchemaError("validation verdict が manifest と一致しません。")
        from survivors.vision.world_formal_release import ValidationVerdict
        try:
            formal_verdict = ValidationVerdict.from_dict(json.loads(verdict_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise PackageSchemaError(f"validation verdict が不正です: {exc}") from exc
        if (
            formal_verdict.status != "PASS"
            or formal_verdict.score_threshold != pkg_manifest.score_threshold
            or formal_verdict.weight_hash != pkg_manifest.model_hash
            or formal_verdict.dataset_identity != pkg_manifest.data_hash
            or formal_verdict.build_identity != pkg_manifest.build_hash
        ):
            raise PackageSchemaError("validation verdict と package identity が一致しません。")

    # -- contract hash 検証
    current_contract_hash = _compute_contract_hash()
    if pkg_manifest.contract_hash != current_contract_hash:
        raise PackageSchemaError(
            f"contract_hash 不一致: package={pkg_manifest.contract_hash!r}"
            f" current={current_contract_hash!r}。"
            " TrackedWorldStateV2 のフィールド定義が変更されています。"
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

    # 共有 validator を明示的に呼ぶ（model / tracker 構築より前）
    validate_detector_config(cfg)
    if formal_verdict is not None:
        from survivors.vision.world_formal_release import resolved_config_hash
        if resolved_config_hash(cfg) != formal_verdict.resolved_config_hash:
            raise PackageSchemaError("resolved config が validation verdict と一致しません。")

    detector = WorldDetector.from_config(cfg, cm_path)

    # -- weight hash 検証・ロード（hash は torch の有無にかかわらず必須）
    weight_file = pkg_dir / _WEIGHT_NAME
    if pkg_manifest.weight_included:
        if not weight_file.is_file():
            raise PackageSchemaError(f"package 内に weight が見つかりません: {weight_file}")
        if _sha256_file(weight_file) != pkg_manifest.model_hash:
            raise PackageSchemaError("weight の model_hash が manifest と一致しません。")
        if formal_verdict is not None and _sha256_file(weight_file) != formal_verdict.weight_hash:
            raise PackageSchemaError("weight hash が validation verdict と一致しません。")
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
    if pkg_manifest.training_mode == "formal" and hasattr(detector._model, "score_thresh"):
        detector._model.score_thresh = 0.0
    result = detector.infer(frame_bgr, score_threshold=pkg_manifest.score_threshold)
    if pkg_manifest.training_mode == "formal":
        from survivors.vision.world_detector import drop_ignored_detections
        result = drop_ignored_detections(result, cfg["input"].get("ignore_regions", []))

    # -- tracker（config から構築）
    tracker_cfg = cfg.get("tracker", {})
    cm = load_class_map(cm_path)
    max_age_raw: dict = tracker_cfg.get("max_age_by_class", {})
    max_age_by_class: dict[int, int] = {}
    for name, age in max_age_raw.items():
        # validate_detector_config が既に class 名を検証済み。KeyError は伝播させる。
        max_age_by_class[cm.name_to_id(name)] = age

    tracker = EntityTracker(
        max_age_by_class=max_age_by_class,
        max_match_cost=tracker_cfg.get("max_match_cost", 0.7),
        velocity_ema_alpha=tracker_cfg.get("velocity_ema_alpha", 0.6),
        confidence_decay_per_frame=tracker_cfg.get("confidence_decay_per_frame", 0.9),
        coarse_by_class_id=cm.coarse_by_class_id(),
    )

    state = tracker.update(result, frame_index=0, timestamp_ns=0)
    v2 = TrackedWorldStateV2.from_state(state, frame_index=0, timestamp_ns=0, class_map_path=cm_path)

    # -- schema 検証は上の contract_hash 比較で実施済み（_compute_contract_hash が3クラス全体をカバー）
    return v2


def _infer_formal_detections(
    detector: Any,
    frame_bgr: np.ndarray,
    score_threshold: float,
) -> Any:
    """formal detector を verdict threshold と ignore region で推論する。

    model 内 threshold を無効化した後に明示 threshold を適用し、検証済み region を除外する。
    """
    model = getattr(detector, "_model", None)
    if model is not None and hasattr(model, "score_thresh"):
        model.score_thresh = 0.0
    result = detector.infer(frame_bgr, score_threshold=score_threshold)
    from survivors.vision.world_detector import drop_ignored_detections

    return drop_ignored_detections(result, getattr(detector, "_formal_ignore_regions", []))


def _formal_manifest_from_dict(wire: object) -> PackageManifest:
    """formal detector manifest を共通 schema 検証で復元する。

    formal 必須項目と eligibility は PackageManifest の既存検証に委ねる。
    """
    manifest = PackageManifest.from_dict(wire)
    manifest.assert_formal_eligible()
    return manifest


def load_formal_detector(
    package_dir: pathlib.Path | str,
) -> tuple["WorldDetector", "EntityTracker", PackageManifest]:
    """検証済み formal detector package から model と tracker を復元する。

    content key、verdict、schema、config/class map/weight/contract hash を照合してから返す。
    """
    import yaml

    from survivors.vision.entity_tracker import EntityTracker
    from survivors.vision.world_dataset import load_class_map
    from survivors.vision.world_detector import WorldDetector

    pkg_dir = pathlib.Path(package_dir)
    manifest_path = pkg_dir / _MANIFEST_NAME
    raw = json.loads(manifest_path.read_bytes())
    manifest = _formal_manifest_from_dict(raw)
    if pkg_dir.name != _content_key(manifest):
        raise PackageSchemaError("formal detector directory does not match its content key")
    if manifest.contract_hash != _compute_contract_hash():
        raise PackageSchemaError("formal detector contract hash does not match current schema")
    verdict_path = pkg_dir / _VALIDATION_VERDICT_NAME
    if not verdict_path.is_file() or _sha256_file(verdict_path) != manifest.validation_verdict_hash:
        raise PackageSchemaError("formal detector validation verdict is missing or does not match its hash")
    cfg_path, class_map_path = pkg_dir / _CONFIG_NAME, pkg_dir / _CLASS_MAP_NAME
    if not cfg_path.is_file() or not class_map_path.is_file():
        raise FileNotFoundError("formal detector config or class map is missing")
    if _sha256_file(cfg_path) != manifest.config_hash or _sha256_file(class_map_path) != manifest.class_map_hash:
        raise PackageSchemaError("formal detector config/class-map hash mismatch")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    validate_detector_config(cfg)
    detector = WorldDetector.from_config(cfg, class_map_path)
    detector._formal_ignore_regions = cfg["input"].get("ignore_regions", [])
    weight_path = pkg_dir / _WEIGHT_NAME
    if not manifest.weight_included or not weight_path.is_file() or _sha256_file(weight_path) != manifest.model_hash:
        raise PackageSchemaError("formal detector weight is missing or does not match its hash")
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise FormalPackageRejectedError("torch is required for formal detector inference") from exc
    try:
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weight_path, map_location="cpu")
    detector._model.load_state_dict(state_dict)
    class_map = load_class_map(class_map_path)
    tracker_cfg = cfg.get("tracker", {})
    max_age_by_class = {
        class_map.name_to_id(name): age
        for name, age in tracker_cfg.get("max_age_by_class", {}).items()
    }
    tracker = EntityTracker(
        max_age_by_class=max_age_by_class,
        max_match_cost=tracker_cfg.get("max_match_cost", 0.7),
        velocity_ema_alpha=tracker_cfg.get("velocity_ema_alpha", 0.6),
        confidence_decay_per_frame=tracker_cfg.get("confidence_decay_per_frame", 0.9),
        coarse_by_class_id=class_map.coarse_by_class_id(),
    )
    return detector, tracker, manifest

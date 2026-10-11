"""formal_gates.v1 YAML の schema・identity・metrics 再計算を担う。"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

from reinbalance_survivors_contracts.canonical_json import canonical_hash


SCHEMA_VERSION = "formal_gates.v1"
_METRIC_FIELDS = {"values", "canonical_hash", "yaml_sha256"}
_VALUE_FIELDS = {
    "schema_version", "min_session_seconds", "slice_count_floors",
    "slice_session_floors", "occurrence_dependent_slices", "band_floor_mode",
}


class _UniqueSafeLoader(yaml.SafeLoader):
    """YAML mapping key の重複を拒否する SafeLoader。
    PyYAML の safe constructor は維持したまま mapping だけ厳格化する。
    """


def _construct_unique_mapping(loader: _UniqueSafeLoader, node: Any) -> dict:
    """mapping node を展開し、同じ YAML key の上書きを防ぐ。
    重複 key は gate identity の解釈が曖昧になるため reject する。
    """
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _sha256(value: object, label: str) -> str:
    """gate identity に使う lowercase SHA-256 を確認する。
    値の canonical hash と bytes hash を同じ字種規則に揃える。
    """
    if (
        type(value) is not str or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


@lru_cache(maxsize=1)
def _baseline_keys() -> tuple[set[str], set[str], set[str]]:
    """repository default YAML から formal slice schema の閉集合を取得する。
    世代設定で slice 名の追加や削除が起きないようにする。
    """
    path = Path(__file__).resolve().parents[1] / "configs" / "formal_gates_default_v1.yaml"
    baseline = yaml.safe_load(path.read_bytes())
    return (
        set(baseline["slice_count_floors"]),
        set(baseline["slice_session_floors"]),
        set(baseline["occurrence_dependent_slices"]),
    )


@dataclass(frozen=True, slots=True)
class FormalGates:
    """検証済み gate 値と YAML/canonical identity を保持する。
    descriptor bytes と verdict metrics の双方で同じ値を使う。
    """

    min_session_seconds: int | float
    slice_count_floors: Mapping[str, int]
    slice_session_floors: Mapping[str, int]
    occurrence_dependent_slices: tuple[str, ...]
    band_floor_mode: str
    canonical_hash: str
    yaml_sha256: str

    def values_wire(self) -> dict[str, Any]:
        """hash に含めた gate 値を canonical な wire mapping へ戻す。
        mapping key と occurrence list を安定順にして再計算可能にする。
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "min_session_seconds": self.min_session_seconds,
            "slice_count_floors": dict(sorted(self.slice_count_floors.items())),
            "slice_session_floors": dict(sorted(self.slice_session_floors.items())),
            "occurrence_dependent_slices": list(self.occurrence_dependent_slices),
            "band_floor_mode": self.band_floor_mode,
        }

    def metrics_wire(self) -> dict[str, Any]:
        """verdict metrics に記録する値と YAML 原文 hash を返す。
        canonical hash と bytes hash は別フィールドで追跡する。
        """
        return {
            "values": self.values_wire(),
            "canonical_hash": self.canonical_hash,
            "yaml_sha256": self.yaml_sha256,
        }


def _validated_values(value: object) -> dict[str, Any]:
    """YAML または metrics に現れる gate 値を一つの schema で検証する。
    YAML loader と verdict loader の validation 規則を共通化する。
    """
    if not isinstance(value, Mapping) or set(value) != _VALUE_FIELDS:
        raise ValueError("formal_gates.v1 values have an invalid field set")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported formal_gates schema_version")
    minimum = value["min_session_seconds"]
    if (
        isinstance(minimum, bool) or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum)) or minimum <= 0
    ):
        raise ValueError("min_session_seconds must be a positive finite number")
    count_floors = value["slice_count_floors"]
    session_floors = value["slice_session_floors"]
    counts_expected, sessions_expected, occurrence_expected = _baseline_keys()
    if not isinstance(count_floors, Mapping) or set(count_floors) != counts_expected:
        raise ValueError("slice_count_floors do not match formal slice schema")
    if not isinstance(session_floors, Mapping) or set(session_floors) != sessions_expected:
        raise ValueError("slice_session_floors do not match formal slice schema")
    for label, floors in (("slice_count_floors", count_floors), ("slice_session_floors", session_floors)):
        if any(type(floor) is not int or floor < 0 for floor in floors.values()):
            raise ValueError(f"{label} must contain non-negative integer floors")
    occurrence = value["occurrence_dependent_slices"]
    if (
        not isinstance(occurrence, list)
        or any(type(name) is not str for name in occurrence)
        or len(occurrence) != len(set(occurrence))
        or not set(occurrence).issubset(counts_expected)
        or not occurrence_expected.issubset(set(occurrence))
    ):
        raise ValueError("occurrence_dependent_slices do not match formal slice schema")
    mode = value["band_floor_mode"]
    if type(mode) is not str or mode not in {"all", "reached_only"}:
        raise ValueError("band_floor_mode must be 'all' or 'reached_only'")
    return {
        "schema_version": SCHEMA_VERSION,
        "min_session_seconds": minimum,
        "slice_count_floors": dict(sorted(count_floors.items())),
        "slice_session_floors": dict(sorted(session_floors.items())),
        "occurrence_dependent_slices": sorted(occurrence),
        "band_floor_mode": mode,
    }


def load_formal_gates_bytes(data: bytes) -> FormalGates:
    """bytes の YAML を strict schema で読み、原文 hash と共に返す。
    重複 key と未知 schema は gate 判定に使わず拒否する。
    """
    if not isinstance(data, bytes):
        raise TypeError("formal gates YAML must be bytes")
    try:
        raw = yaml.load(data, Loader=_UniqueSafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError("invalid formal gates YAML") from exc
    values = _validated_values(raw)
    return FormalGates(
        min_session_seconds=values["min_session_seconds"],
        slice_count_floors=values["slice_count_floors"],
        slice_session_floors=values["slice_session_floors"],
        occurrence_dependent_slices=tuple(values["occurrence_dependent_slices"]),
        band_floor_mode=values["band_floor_mode"],
        canonical_hash=canonical_hash(values),
        yaml_sha256=hashlib.sha256(data).hexdigest(),
    )


def load_formal_gates(path: str | Path) -> FormalGates:
    """formal_gates.v1 YAML file を bytes hash を保持して読む。
    descriptor binding に使う元 bytes の identity を保持する。
    """
    return load_formal_gates_bytes(Path(path).read_bytes())


def default_formal_gates() -> FormalGates:
    """既定閾値を宣言した repository YAML を load して返す。

    descriptor のない synthetic/legacy 経路だけがこの既定ファイルへ戻る。
    """
    path = Path(__file__).resolve().parents[1] / "configs" / "formal_gates_default_v1.yaml"
    return load_formal_gates(path)


def formal_gates_from_metrics(value: object) -> FormalGates:
    """保存 metrics の gate 値・canonical hash・YAML hash 形式を再検証する。
    再計算 loader で gate の値だけを差し替えた verdict を拒否する。
    """
    if not isinstance(value, Mapping) or set(value) != _METRIC_FIELDS:
        raise ValueError("formal_gates metrics have an invalid field set")
    values = _validated_values(value["values"])
    claimed_hash = _sha256(value["canonical_hash"], "formal_gates.canonical_hash")
    if canonical_hash(values) != claimed_hash:
        raise ValueError("formal_gates canonical hash mismatch")
    yaml_hash = _sha256(value["yaml_sha256"], "formal_gates.yaml_sha256")
    return FormalGates(
        min_session_seconds=values["min_session_seconds"],
        slice_count_floors=values["slice_count_floors"],
        slice_session_floors=values["slice_session_floors"],
        occurrence_dependent_slices=tuple(values["occurrence_dependent_slices"]),
        band_floor_mode=values["band_floor_mode"],
        canonical_hash=claimed_hash,
        yaml_sha256=yaml_hash,
    )

"""Named estimate を共有 DeployObsV1 tensor へ解決する release adapter。

実画面 parser と simulator wrapper が、同じ欠損・鮮度・可視性の規則を
利用するための純粋な変換処理を提供します。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.deploy_obs import (
    DeployObservation, DeployObsField, DeployObsSchema,
)
from reinbalance_survivors_contracts.ui_intent import ContractValidationError, ensure, is_strict_number

_ESTIMATE_KEYS = frozenset({"value", "timestamp_ns", "validity"})
_TRACK_KEYS = frozenset({"screen_x", "screen_y", "visible", "occluded", "clipped", "timestamp_ns"})
WEAPON_VOCABULARY = ("aura", "projectile", "melee", "unknown")


@dataclass(frozen=True)
class NamedEstimate:
    """source timestamp 付きの名前付き推定値。

    flat array ではなく意味のある名前で adapter 境界へ渡し、取得時刻と信頼度も保持します。
    """

    value: tuple[float, ...]
    timestamp_ns: int
    validity: float = 1.0

    def __post_init__(self) -> None:
        """推定値の型・有限性・信頼度範囲を検証する。

        parser の異常値が tensor に混ざる前に入力境界で拒否します。
        """
        ensure(isinstance(self.value, tuple) and bool(self.value), "estimate value must be non-empty tuple")
        ensure(all(is_strict_number(x) and np.isfinite(x) for x in self.value), "estimate values must be finite numbers")
        ensure(isinstance(self.timestamp_ns, int) and not isinstance(self.timestamp_ns, bool) and self.timestamp_ns >= 0, "invalid estimate timestamp")
        ensure(is_strict_number(self.validity) and np.isfinite(self.validity) and 0 <= self.validity <= 1, "invalid estimate validity")

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "NamedEstimate":
        """厳密な wire mapping から named estimate を作る。

        未知キーや欠落キー、scalar/list の取り違えを黙認しません。
        """
        ensure(isinstance(data, Mapping) and set(data) == _ESTIMATE_KEYS, "estimate keys mismatch")
        ensure(isinstance(data["value"], (list, tuple)), "estimate value must be sequence")
        return cls(tuple(data["value"]), data["timestamp_ns"], data["validity"])


def load_schema(path: str | Path) -> DeployObsSchema:
    """YAML の順序付き schema を厳密に読み込む。

    YAML は設定の見やすさだけに使い、内容の検証は共有 schema 契約へ委譲します。
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return DeployObsSchema.from_wire(data)


def normalized_category(value: str, vocabulary: Sequence[str] = WEAPON_VOCABULARY) -> float:
    """未知値を含む categorical id を [0,1] へ正規化する。

    vocabulary に無い表示は末尾の unknown として扱い、enum 増減で例外にしません。
    """
    ensure(isinstance(value, str), "category must be str")
    ensure(len(vocabulary) >= 2 and vocabulary[-1] == "unknown" and len(set(vocabulary)) == len(vocabulary), "invalid vocabulary")
    index = vocabulary.index(value) if value in vocabulary else len(vocabulary) - 1
    return index / (len(vocabulary) - 1)


def screen_to_centered(x: float, y: float, width: float, height: float) -> tuple[float, float]:
    """viewport pixel 座標を中心基準 [-1,1] へ変換する。

    clipping 後の画面内座標だけを受け、camera scale や world unit を出力へ持ち込みません。
    """
    for value in (x, y, width, height):
        ensure(is_strict_number(value) and np.isfinite(value), "screen coordinates must be finite")
    ensure(width > 0 and height > 0 and 0 <= x <= width and 0 <= y <= height, "point outside viewport")
    return (2.0 * x / width - 1.0, 2.0 * y / height - 1.0)


def _resolve(
    field: DeployObsField,
    estimates: Mapping[str, NamedEstimate],
    now_ns: int,
    *,
    include_unobservable: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一つの segment を value・validity・age へ解決する。

    source class が違っても欠損表現と stale decay はこの共通経路を必ず通ります。
    """
    neutral = np.full(field.size, field.neutral, dtype=np.float32)
    if field.source_class == "unobservable" and not include_unobservable:
        return neutral, np.zeros(field.size, np.float32), np.ones(field.size, np.float32)
    if field.source_class == "constant":
        return np.ones(field.size, np.float32), np.ones(field.size, np.float32), np.zeros(field.size, np.float32)
    estimate = estimates.get(field.name)
    if estimate is None:
        return neutral, np.zeros(field.size, np.float32), np.ones(field.size, np.float32)
    ensure(len(estimate.value) == field.size, f"{field.name} estimate shape mismatch")
    ensure(estimate.timestamp_ns <= now_ns, f"{field.name} timestamp is in future")
    age_ms = (now_ns - estimate.timestamp_ns) / 1_000_000
    age = min(age_ms / field.max_age_ms, 1.0)
    validity = estimate.validity
    if age_ms > field.stale_after_ms:
        span = field.max_age_ms - field.stale_after_ms
        validity *= max(0.0, 1.0 - (age_ms - field.stale_after_ms) / span) if span else 0.0
    if validity == 0:
        return neutral, np.zeros(field.size, np.float32), np.ones(field.size, np.float32)
    return np.asarray(estimate.value, np.float32), np.full(field.size, validity, np.float32), np.full(field.size, age, np.float32)


def validate_output(observation: DeployObservation, schema: DeployObsSchema) -> DeployObservation:
    """adapter 出力の hash・shape・有限性・範囲を一か所で検証する。

    real/simulator のどちらが生成しても、policy に渡す直前の同じ gate を通します。
    """
    observation.validate_for(schema)
    return observation


def assert_release_artifact_allowed(
    observation: DeployObservation,
    schema: DeployObsSchema,
) -> DeployObservation:
    """release artifact 境界で provenance と観測契約を再検証する。

    oracle 診断値は tensor 自体が有限でも本番成果物へ出せないため、
    保存・シリアライズ直前に内在マーカーを確認して拒否します。
    """
    ensure(isinstance(observation, DeployObservation), "observation must be DeployObservation")
    observation.validate_for(schema)
    ensure(observation.provenance == "release", "oracle_diagnostic cannot create release artifacts")
    return observation


def release_policy_tensor(observation: DeployObservation, schema: DeployObsSchema) -> np.ndarray:
    """release 用 policy tensor を guarded observation から生成する。

    adapter を直接使う保存経路でも wrapper と同じ provenance gate を通し、
    oracle observation の bare な artifact 化を防ぎます。
    """
    return assert_release_artifact_allowed(observation, schema).as_policy_tensor(schema)


def deploy_obs_producer_hash(schema: DeployObsSchema, release_adapter_hash: str) -> str:
    """schema と release adapter を束ねた producer identity を計算する。

    どちらかが変われば fidelity baseline と比較する identity も必ず変わります。
    """
    ensure(isinstance(release_adapter_hash, str) and len(release_adapter_hash) == 64, "invalid release adapter hash")
    return canonical_hash({"schema_hash": schema.schema_hash, "release_adapter_hash": release_adapter_hash})


def build_deploy_observation(
    schema: DeployObsSchema,
    estimates: Mapping[str, NamedEstimate],
    timestamp_ns: int,
) -> DeployObservation:
    """release 用 named estimates から DeployObservation を構築する。

    privileged estimate を渡されても unobservable segment は必ず欠損にし、
    公開 release 経路から oracle 診断値を有効化できないようにします。
    """
    return _build_observation(
        schema, estimates, timestamp_ns, include_unobservable=False, provenance="release",
    )


def build_oracle_diagnostic_observation(
    schema: DeployObsSchema,
    estimates: Mapping[str, NamedEstimate],
    timestamp_ns: int,
) -> DeployObservation:
    """artifact 化禁止の oracle 診断 Observation を構築する。

    privileged segment を有効にする唯一の公開入口です。呼び出し側は
    oracle 専用 constructor と release artifact gate を必ず併用します。
    """
    return _build_observation(
        schema, estimates, timestamp_ns,
        include_unobservable=True, provenance="oracle_diagnostic",
    )


def _build_observation(
    schema: DeployObsSchema,
    estimates: Mapping[str, NamedEstimate],
    timestamp_ns: int,
    *,
    include_unobservable: bool,
    provenance: str,
) -> DeployObservation:
    """mode 固定済みの estimates を schema 順に tensor 化する。

    release/oracle の公開入口が選んだ能力だけを内部 resolver へ渡し、
    bare な公開 bool で privileged segment を開放させません。
    """
    ensure(isinstance(estimates, Mapping), "estimates must be mapping")
    ensure(isinstance(timestamp_ns, int) and not isinstance(timestamp_ns, bool) and timestamp_ns >= 0, "invalid timestamp")
    names = {field.name for field in schema.fields}
    ensure(set(estimates) <= names and all(isinstance(x, NamedEstimate) for x in estimates.values()), "unknown or invalid estimate")
    ensure(type(include_unobservable) is bool, "include_unobservable must be bool")
    ensure(
        (include_unobservable, provenance) in {
            (False, "release"), (True, "oracle_diagnostic"),
        },
        "observation capability and provenance mismatch",
    )
    planes = [
        _resolve(field, estimates, timestamp_ns, include_unobservable=include_unobservable)
        for field in schema.fields
    ]
    observation = DeployObservation(
        np.concatenate([x[0] for x in planes]), np.concatenate([x[1] for x in planes]),
        np.concatenate([x[2] for x in planes]), schema.schema_hash, timestamp_ns, provenance,
    )
    return validate_output(observation, schema)


def _validated_viewport(viewport: Any) -> tuple[int, int]:
    """viewport の wire 表現を検証して不変 tuple へ正規化する。

    Python 内の tuple と JSON decode 後の list を同じ入力として扱い、
    画面寸法ではない文字列や bool、非正の整数を入口で拒否します。
    """
    ensure(isinstance(viewport, (tuple, list)) and len(viewport) == 2, "viewport must be pair sequence")
    ensure(
        all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in viewport),
        "viewport values must be positive int",
    )
    return viewport[0], viewport[1]


def visible_track_estimates(tracks: Sequence[Mapping[str, Any]], viewport: tuple[int, int] | list[int], timestamp_ns: int) -> dict[str, NamedEstimate]:
    """可視 track だけから敵 offset と count の named estimates を作る。

    off-screen・occluded・clipped track は全体数にも最近傍にも含めず leakage を防ぎます。
    """
    width, height = _validated_viewport(viewport)
    visible: list[tuple[float, float, int]] = []
    for track in tracks:
        ensure(isinstance(track, Mapping) and set(track) == _TRACK_KEYS, "track keys mismatch")
        ensure(all(isinstance(track[k], bool) for k in ("visible", "occluded", "clipped")), "track flags must be bool")
        ensure(all(is_strict_number(track[k]) and np.isfinite(track[k]) for k in ("screen_x", "screen_y")), "track coordinates must be finite")
        ensure(isinstance(track["timestamp_ns"], int) and not isinstance(track["timestamp_ns"], bool) and 0 <= track["timestamp_ns"] <= timestamp_ns, "invalid track timestamp")
        if track["visible"] and not track["occluded"] and not track["clipped"]:
            x, y = screen_to_centered(track["screen_x"], track["screen_y"], width, height)
            visible.append((x, y, track["timestamp_ns"]))
    result = {"visible_enemy_count": NamedEstimate((min(len(visible) / 20.0, 1.0),), timestamp_ns)}
    if visible:
        nearest = min(visible, key=lambda item: item[0] ** 2 + item[1] ** 2)
        result["nearest_enemy_offset"] = NamedEstimate(nearest[:2], nearest[2])
    return result

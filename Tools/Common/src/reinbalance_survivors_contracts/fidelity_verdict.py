"""Simulator-to-Real fidelity verdict の検証・失効判定・原子的な発行。

本モジュールは、監査の説明情報と再監査を要求する producer identity を分離します。
利用者は読み込んだ verdict を必ず再検証し、現在の producer hash と一致する場合だけ
後続処理へ進めます。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from .canonical_json import canonical_hash, canonical_json_bytes
from .ui_intent import ContractValidationError, ensure

FIDELITY_SCHEMA_VERSION = "survivors.sim_real_fidelity.v2"
PRODUCER_ALLOWLIST_VERSION = "fidelity_producer_paths.v1"
GATING_KEYS = (
    "logic_public", "logic_private", "game_facade", "http_service",
    "sim_runtime_config", "content_schema", "action_time_schema",
    "external_decision_schema", "preview_schema", "deploy_obs_schema",
    "deploy_release_adapter", "target_profile", "target_build_attestation",
)
STAGES = ("baseline", "integration", "post_curriculum")
_TOP_KEYS = frozenset({
    "schema_version", "verdict_stage", "subject", "metrics",
    "blocking_reasons", "provenance", "gating_producer_hashes",
})
_PROVENANCE_KEYS = frozenset({
    "git_commit", "workspace_dirty_summary", "audit_tool_version",
    "dependency_versions", "operator", "timestamp",
})
_SUBJECT_KEYS = frozenset({
    "target_profile_hash", "target_build_attestation_hash", "report_scope",
    "producer_allowlist_version", "producer_manifest_hash", "resolved_producers",
})


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    """wire object を厳密な mapping として検証する。

    JSON object 以外を早い段階で拒否し、後続の型変換による値の丸め込みを防ぎます。
    """
    ensure(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str] | frozenset[str], label: str) -> None:
    """object の未知 key と欠落 key を同時に拒否する。

    producer の追加や typo を黙って無視せず、schema 更新を明示的に要求します。
    """
    actual = set(value)
    ensure(actual == set(keys), f"{label} keys mismatch: missing={sorted(set(keys)-actual)}, unknown={sorted(actual-set(keys))}")


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    """文字列フィールドを暗黙変換なしで検証する。

    数値などを文字列化せず、空文字を許す箇所だけ呼び出し側が明示します。
    """
    ensure(isinstance(value, str), f"{label} must be a string")
    ensure(allow_empty or bool(value), f"{label} must not be empty")
    return value


def _json_value(value: Any, label: str) -> Any:
    """nested JSON 値が有限かつ canonical 化可能であることを検証する。

    metrics や provenance の内部にも NaN や非 JSON 型が入り込まないようにします。
    """
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{label} is not canonical JSON: {exc}") from exc
    return value


@dataclass(frozen=True)
class FidelityMetric:
    """単一の fidelity 計測結果。

    名前、有限な値、単位、計測可否を保持し、未計測値を推測値で補うことを禁止します。
    """
    name: str
    value: float | None
    unit: str
    measurable: bool
    accepted_uncertainty: str | None = None

    def __post_init__(self) -> None:
        """構築経路を問わず metric の型・有限性・整合性を検証する。

        measurable=false の値は None に限定し、理由のない uncertainty を拒否します。
        """
        _text(self.name, "metric.name")
        _text(self.unit, "metric.unit", allow_empty=True)
        ensure(type(self.measurable) is bool, "metric.measurable must be bool")
        if self.measurable:
            ensure(isinstance(self.value, (int, float)) and not isinstance(self.value, bool), "measurable metric.value must be numeric")
            ensure(math.isfinite(float(self.value)), "metric.value must be finite")
            ensure(self.accepted_uncertainty is None, "measurable metric cannot have accepted_uncertainty")
        else:
            ensure(self.value is None, "unmeasurable metric.value must be null")
            _text(self.accepted_uncertainty, "metric.accepted_uncertainty")

    @classmethod
    def from_wire(cls, value: Any) -> "FidelityMetric":
        """wire object から未知 key を拒否して metric を生成する。

        任意フィールドも明示的に列挙し、schema 外の値を保持しません。
        """
        data = _mapping(value, "metric")
        allowed = {"name", "value", "unit", "measurable", "accepted_uncertainty"}
        ensure(set(data) <= allowed, f"metric unknown keys: {sorted(set(data)-allowed)}")
        ensure({"name", "value", "unit", "measurable"} <= set(data), "metric missing required keys")
        return cls(data["name"], data["value"], data["unit"], data["measurable"], data.get("accepted_uncertainty"))

    def to_wire(self) -> dict[str, Any]:
        """metric を canonical wire object に変換する。

        uncertainty は存在するときだけ出力し、意味のない null field を増やしません。
        """
        result = {"name": self.name, "value": self.value, "unit": self.unit, "measurable": self.measurable}
        if self.accepted_uncertainty is not None:
            result["accepted_uncertainty"] = self.accepted_uncertainty
        return result


@dataclass(frozen=True)
class BlockingReason:
    """下流処理を止める監査行。

    category と理由を必須にし、baseline の action/offer/terminal gate を明示します。
    """
    category: str
    reason: str

    def __post_init__(self) -> None:
        """直接構築時にも空の blocking reason を拒否する。

        空文字による見かけ上の blocking 行を作れないようにします。
        """
        _text(self.category, "blocking_reason.category")
        _text(self.reason, "blocking_reason.reason")

    @classmethod
    def from_wire(cls, value: Any) -> "BlockingReason":
        """wire object を exact-key で blocking reason に変換する。

        未知 key は監査表示に紛れ込ませず拒否します。
        """
        data = _mapping(value, "blocking_reason")
        _exact_keys(data, {"category", "reason"}, "blocking_reason")
        return cls(data["category"], data["reason"])

    def to_wire(self) -> dict[str, str]:
        """blocking reason の wire 表現を返す。

        出力フィールドは category と reason の二つに固定します。
        """
        return {"category": self.category, "reason": self.reason}


@dataclass(frozen=True)
class FidelityVerdict:
    """不変な v2 fidelity verdict。

    provenance は表示用、gating_producer_hashes は失効判定用として別 object に保持します。
    """
    verdict_stage: Literal["baseline", "integration", "post_curriculum"]
    subject: Mapping[str, Any]
    metrics: tuple[FidelityMetric, ...]
    blocking_reasons: tuple[BlockingReason, ...]
    provenance: Mapping[str, Any]
    gating_producer_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        """全 nested object と stage 固有 gate を対称に検証する。

        baseline は DeployObs を absent とし、action/offer/terminal を常に blocking にします。
        """
        ensure(self.verdict_stage in STAGES, "invalid verdict_stage")
        subject = dict(_mapping(self.subject, "subject"))
        provenance = dict(_mapping(self.provenance, "provenance"))
        _exact_keys(subject, _SUBJECT_KEYS, "subject")
        _exact_keys(provenance, _PROVENANCE_KEYS, "provenance")
        _json_value(subject, "subject")
        _json_value(provenance, "provenance")
        ensure(subject["producer_allowlist_version"] == PRODUCER_ALLOWLIST_VERSION, "producer allowlist version mismatch")
        ensure(subject["report_scope"] in {"exact_target", "all_content_generalization"}, "invalid report_scope")
        for key in ("target_profile_hash", "target_build_attestation_hash", "producer_manifest_hash"):
            digest = subject[key]
            ensure(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), f"invalid subject hash: {key}")
        ensure(isinstance(subject["resolved_producers"], Mapping), "resolved_producers must be an object")
        _exact_keys(subject["resolved_producers"], set(GATING_KEYS), "resolved_producers")
        for key, records in subject["resolved_producers"].items():
            ensure(isinstance(records, list), f"resolved_producers.{key} must be an array")
            for record in records:
                ensure(isinstance(record, Mapping) and set(record) == {"path", "sha256"}, f"resolved producer record invalid: {key}")
                _text(record["path"], f"resolved_producers.{key}.path")
                digest = record["sha256"]
                ensure(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), f"invalid resolved producer hash: {key}")
        ensure(all(isinstance(x, FidelityMetric) for x in self.metrics), "metrics must contain FidelityMetric")
        ensure(all(isinstance(x, BlockingReason) for x in self.blocking_reasons), "blocking_reasons must contain BlockingReason")
        hashes = _mapping(self.gating_producer_hashes, "gating_producer_hashes")
        _exact_keys(hashes, set(GATING_KEYS), "gating_producer_hashes")
        for key, digest in hashes.items():
            _text(digest, f"gating_producer_hashes.{key}")
            ensure(digest == "absent" or (len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)), f"invalid producer hash: {key}")
        categories = {row.category for row in self.blocking_reasons}
        if self.verdict_stage == "baseline":
            ensure(hashes["deploy_obs_schema"] == hashes["deploy_release_adapter"] == "absent", "baseline DeployObs hashes must be absent")
            ensure({"action", "offer", "terminal"} <= categories, "baseline requires action/offer/terminal blocking rows")
        else:
            ensure("deploy_obs_visibility" in categories, "integration/post_curriculum requires DeployObs visibility blocking row")

    @classmethod
    def from_wire(cls, value: Any) -> "FidelityVerdict":
        """未知・欠落 field を拒否して v2 verdict を生成する。

        nested metric と blocking row もそれぞれの factory で再検証します。
        """
        data = _mapping(value, "verdict")
        _exact_keys(data, _TOP_KEYS, "verdict")
        ensure(data["schema_version"] == FIDELITY_SCHEMA_VERSION, "unsupported fidelity schema_version")
        ensure(isinstance(data["metrics"], list), "metrics must be an array")
        ensure(isinstance(data["blocking_reasons"], list), "blocking_reasons must be an array")
        return cls(
            data["verdict_stage"], dict(_mapping(data["subject"], "subject")),
            tuple(FidelityMetric.from_wire(x) for x in data["metrics"]),
            tuple(BlockingReason.from_wire(x) for x in data["blocking_reasons"]),
            dict(_mapping(data["provenance"], "provenance")),
            dict(_mapping(data["gating_producer_hashes"], "gating_producer_hashes")),
        )

    def to_wire(self) -> dict[str, Any]:
        """検証済み verdict の canonical wire object を返す。

        schema version を必ず付け、provenance と gating map を混合しません。
        """
        return {
            "schema_version": FIDELITY_SCHEMA_VERSION, "verdict_stage": self.verdict_stage,
            "subject": dict(self.subject), "metrics": [x.to_wire() for x in self.metrics],
            "blocking_reasons": [x.to_wire() for x in self.blocking_reasons],
            "provenance": dict(self.provenance), "gating_producer_hashes": dict(self.gating_producer_hashes),
        }

    @property
    def identity_hash(self) -> str:
        """verdict artifact の canonical identity hash を返す。

        再現用 artifact identity として全 wire bytes を共有 canonical 経路で hash します。
        """
        return canonical_hash(self.to_wire())


def verify_current_fidelity(verdict: FidelityVerdict | Mapping[str, Any], current_gating_producer_hashes: Mapping[str, str], required_stage: str) -> FidelityVerdict:
    """verdict を利用直前に再検証し、current hash と stage を fail-closed 比較する。

    provenance は比較対象にせず、baseline の下流流用、key 差、hash 差を例外にします。
    """
    checked = FidelityVerdict.from_wire(verdict.to_wire() if isinstance(verdict, FidelityVerdict) else verdict)
    ensure(required_stage in STAGES, "invalid required_stage")
    current = dict(_mapping(current_gating_producer_hashes, "current_gating_producer_hashes"))
    _exact_keys(current, set(GATING_KEYS), "current_gating_producer_hashes")
    ensure(STAGES.index(checked.verdict_stage) >= STAGES.index(required_stage), "fidelity verdict stage is stale")
    ensure(not (required_stage != "baseline" and checked.verdict_stage == "baseline"), "baseline verdict cannot unlock downstream")
    ensure(dict(checked.gating_producer_hashes) == current, "gating producer hashes differ")
    return checked


def downstream_release_allowed(
    verdict: FidelityVerdict | Mapping[str, Any],
    current_gating_producer_hashes: Mapping[str, str],
    required_stage: str,
) -> bool:
    """current verdict が下流処理を解禁できるかを返す。

    verdict の妥当性・鮮度は例外で検証し、blocking row の有無だけを解禁可否として
    分離するため、必須の visibility 行を持つ昇格 verdict 自体は current と判定できます。
    """
    checked = verify_current_fidelity(verdict, current_gating_producer_hashes, required_stage)
    return not checked.blocking_reasons


def _pair_commit_path(json_path: Path, report_path: Path) -> Path:
    """artifact pair 専用の commit marker path を決定する。

    file 名の組を marker 名へ含め、同じ directory に複数の pair があっても世代を
    取り違えないようにします。
    """
    return json_path.parent / f".{json_path.name}.{report_path.name}.commit"


def read_verdict_pair_atomic(json_path: Path, report_path: Path) -> tuple[FidelityVerdict, str]:
    """commit marker が指す同一世代の JSON と Markdown を読み込む。

    呼び出し側は個別 path を直接開かず、最後に確定した marker から二つの payload を
    同時に解決することで half-committed generation を観測しません。
    """
    ensure(json_path.parent == report_path.parent, "artifact pair must share a directory")
    marker_path = _pair_commit_path(json_path, report_path)
    try:
        marker = json.loads(marker_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"invalid fidelity pair commit marker: {exc}") from exc
    marker_data = _mapping(marker, "artifact_pair_marker")
    _exact_keys(marker_data, {"generation", "json_name", "report_name"}, "artifact_pair_marker")
    ensure(marker_data["json_name"] == json_path.name and marker_data["report_name"] == report_path.name, "artifact pair marker names mismatch")
    generation_relative = _text(marker_data["generation"], "artifact_pair_marker.generation")
    generation_path = Path(generation_relative)
    ensure(not generation_path.is_absolute() and ".." not in generation_path.parts, "artifact pair generation escapes directory")
    generation = json_path.parent / generation_path
    try:
        verdict_wire = json.loads((generation / json_path.name).read_bytes())
        report = (generation / report_path.name).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"invalid committed fidelity pair: {exc}") from exc
    return FidelityVerdict.from_wire(verdict_wire), report


def write_verdict_pair_atomic(verdict: FidelityVerdict, json_path: Path, report_path: Path, report_markdown: str) -> None:
    """JSON と Markdown を単一 commit marker で同一世代として確定する。

    完成済み bundle は reader から不可視の staging area に置き、最後の marker rename
    一回だけを外部可視な commit 点にします。電源断 durability は保証範囲外です。
    """
    checked = FidelityVerdict.from_wire(verdict.to_wire())
    ensure(json_path.parent == report_path.parent, "artifact pair must share a directory")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".fidelity-pair-", dir=json_path.parent))
    bundle_root = json_path.parent / ".fidelity-pairs"
    bundle_root.mkdir(exist_ok=True)
    generation_id = canonical_hash({"verdict_identity": checked.identity_hash, "report_markdown": report_markdown})
    generation = bundle_root / generation_id
    marker = _pair_commit_path(json_path, report_path)
    marker_temp: Path | None = None
    try:
        for name, payload in ((json_path.name, canonical_json_bytes(checked.to_wire())), (report_path.name, report_markdown.encode("utf-8"))):
            with (staging / name).open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        if not generation.exists():
            staging.rename(generation)
        marker_payload = canonical_json_bytes({
            "generation": generation.relative_to(json_path.parent).as_posix(),
            "json_name": json_path.name,
            "report_name": report_path.name,
        })
        fd, marker_name = tempfile.mkstemp(prefix=f".{marker.name}.", dir=json_path.parent)
        marker_temp = Path(marker_name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(marker_payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(marker_temp, marker)
        marker_temp = None
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if marker_temp is not None:
            marker_temp.unlink(missing_ok=True)

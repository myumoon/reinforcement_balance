"""ArtifactStore-backed Survivors perception formal benchmark runner。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from Tools.Artifacts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.artifact_dag import validate_artifact_dag
from reinbalance_survivors_contracts.artifact_identity import (
    ArtifactDescriptor,
    ArtifactRef,
    artifact_uri,
)
from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes

from survivors.capture.captured_frame import CapturedFrame
from survivors.perception_snapshot import FormalReplayEvidence
from survivors.capture_dataset import (
    DatasetWriter,
    SessionManifest,
    SplitManifest,
    _split_manifest_from_bytes,
)
from survivors.perception_benchmark import (
    ExpectedTick,
    SnapshotReplayTick,
    _formalize_benchmark_report,
    run_benchmark,
)
from survivors.perception_error_fit import (
    CalibrationResidual,
    FittedPerceptionErrorProfile,
    PerceptionFinalVerdict,
    _create_formal_final_verdict,
    _create_formal_lineage_seal,
    _reserve_final_session,
    _write_formal_calibration_profile,
    _write_formal_final_verdict,
    fit_error_profile,
)
from survivors.perception_session_split import SessionSplit, validate_split


CalibrationProvider = Callable[
    [tuple[SessionManifest, ...], Mapping[str, bytes]], Sequence[CalibrationResidual]
]


class FormalAssembler(Protocol):
    """runner が raw frame から正解と予測を別々に生成するアセンブラプロトコル。

    runner が frame の選択（dataset から）と detector_artifact_hash の設定（restored package から）
    を制御し、assembler は frame の解釈のみを担当します。外部 provider の自己申告を排除します。
    raw 証拠は V1 型を変えず FormalReplayEvidence として分離して返します。
    """

    def assemble_frame(
        self, frame: CapturedFrame, session_id: str, frame_index: int
    ) -> tuple[
        PerceptionSnapshot | None,
        PerceptionSnapshot | None,
        float,
        FormalReplayEvidence | None,
    ]:
        """(ground_truth, predicted, latency_ms, evidence) を返す。正解 None はスキップ。"""
        ...


FormalAssemblerFactory = Callable[[Mapping[str, bytes]], FormalAssembler]


@dataclass(frozen=True, slots=True)
class FormalBenchmarkRequest:
    """formal benchmark 実行リクエスト。

    provider/assembler はここに持たない。run_formal_pipeline が _CLI_PROVIDER_FACTORY
    から hash-bound entrypoint をロードする。callback 注入は development-only API に限定する。
    """

    store: ArtifactStore
    dependency_refs: Mapping[str, ArtifactRef]
    capture_store_root: Path
    calibration_logical_id: str = "perception/calibration/profile.json"
    verdict_logical_id: str = "perception/final/verdict.json"


@dataclass(frozen=True, slots=True)
class FormalBenchmarkResult:
    split: SessionSplit
    profile: FittedPerceptionErrorProfile
    verdict: PerceptionFinalVerdict
    calibration_ref: ArtifactRef
    verdict_ref: ArtifactRef
    descriptors: tuple[ArtifactDescriptor, ...]


_DEPENDENCY_NAMES = frozenset(
    {
        "capture_dataset", "parser_package", "detector_package",
        "assembler_config", "target_config",
    }
)
# パッケージ種別ごとの required field 集合。JSON 内容を exact schema で検証する。
_PACKAGE_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "parser_package": frozenset({"schema_version", "development_only", "formal_eligible", "artifact_hash"}),
    "detector_package": frozenset({"schema_version", "development_only", "formal_eligible", "artifact_hash"}),
    "assembler_config": frozenset({
        "schema_version", "development_only", "formal_eligible",
        "assembler_schema_hash", "ui_presentation_schema_hash",
        "atlas_vocabulary_hash", "assembler_impl_hash", "roi_resolver_input_hash",
    }),
    "target_config": frozenset({"schema_version", "development_only", "formal_eligible", "threshold_hash"}),
}
def _load_cli_provider_factory() -> (
    Callable[[Mapping[str, bytes]], tuple[CalibrationProvider, FormalAssemblerFactory]] | None
):
    """インストール済みパッケージから hash-bound formal provider を動的にロードする。

    survivors.formal_perception_provider が未インストールの場合は None を返す。
    本番環境では parser/detector package とともにインストールし、
    None 以外の factory が返るようにすること。
    """
    try:
        from survivors.formal_perception_provider import build_formal_provider  # type: ignore[import]
        return build_formal_provider
    except ImportError:
        return None


_CLI_PROVIDER_FACTORY: Callable[[Mapping[str, bytes]], tuple[CalibrationProvider, FormalAssemblerFactory]] | None = _load_cli_provider_factory()


def _restore_dependencies(request: FormalBenchmarkRequest) -> dict[str, bytes]:
    """全 dependency を content store から restore → hash/size → JSON 内容検証する。"""
    if not isinstance(request.store, ArtifactStore):
        raise TypeError("formal runner requires ArtifactStore")
    refs = dict(request.dependency_refs)
    if set(refs) != _DEPENDENCY_NAMES:
        raise ValueError("formal dependency ref set is incomplete or contains unknown entries")
    restored: dict[str, bytes] = {}
    for name in sorted(_DEPENDENCY_NAMES):
        ref = refs[name]
        if not isinstance(ref, ArtifactRef):
            raise TypeError(f"{name} must be an ArtifactRef")
        verification = request.store.verify(ref)
        if not verification.ok:
            raise ValueError(f"{name} ArtifactStore verification failed: {verification.reason}")
        data = request.store.object_path(ref.store_uri).read_bytes()
        if len(data) != ref.size_bytes:
            raise ValueError(f"{name} restored size changed after verification")
        # capture split は専用 typed parser で内容検証する。他 package/config は
        # exact JSON mapping と formal marker を検証し、任意ファイルを Verified としない。
        if name != "capture_dataset":
            try:
                payload = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{name} is not a valid JSON package") from exc
            if (
                not isinstance(payload, dict)
                or type(payload.get("schema_version")) is not str
                or not payload["schema_version"]
                or payload.get("development_only") is not False
                or payload.get("formal_eligible") is not True
            ):
                raise ValueError(f"{name} package content is not formal eligible")
            # パッケージ種別別の exact required field 検証。
            required_fields = _PACKAGE_REQUIRED_FIELDS[name]
            missing = required_fields - set(payload)
            if missing:
                raise ValueError(f"{name} is missing required fields: {sorted(missing)}")
            # parser/detector パッケージ: artifact_hash が 64 char hex であることを確認する。
            # replay との照合は run_formal_pipeline で行う。
            if name in {"parser_package", "detector_package"}:
                decl_hash = payload.get("artifact_hash")
                if not isinstance(decl_hash, str) or len(decl_hash) != 64 or not all(
                    c in "0123456789abcdef" for c in decl_hash
                ):
                    raise ValueError(f"{name} artifact_hash must be a 64-char lowercase SHA-256 hex string")
        restored[name] = data
        # TOCTOU を閉じるため内容を読んだ後にも object を再検証する。
        if not request.store.verify(ref).ok:
            raise ValueError(f"{name} changed during restore")
    return restored


def _restore_split_and_sessions(
    request: FormalBenchmarkRequest, capture_bytes: bytes
) -> tuple[SplitManifest, dict[str, SessionManifest], SessionSplit]:
    split_ref = request.dependency_refs["capture_dataset"]
    split = _split_manifest_from_bytes(
        request.store.object_path(split_ref.store_uri), capture_bytes
    )
    if split.manifest_sha256 != split_ref.sha256:
        raise ValueError("capture split manifest is not exact-hash bound to ArtifactRef")
    # Phase 1: final session は metadata_only で restore してから reservation し、
    # その後に PNG デコード（full restore）する。競合 runner が reservation 前に
    # 同じ PNG を読まないよう、デコードと予約の順序を逆転させる。
    # calibration session は reservation 不要なので最初から full restore する。
    final_session_ids: set[str] = {
        unit.session_id
        for unit in split.splits.get("final_e2e_test", ())
    }

    manifests: dict[str, SessionManifest] = {}
    for split_name, units in split.splits.items():
        for unit in units:
            if unit.session_id in manifests:
                raise ValueError(f"duplicate session in split: {unit.session_id}")
            is_final = unit.session_id in final_session_ids
            manifests[unit.session_id] = DatasetWriter.restore(
                request.capture_store_root, unit.session_id,
                metadata_only=is_final,  # final は予約前に PNG をデコードしない
            )
    validated = validate_split(split, manifests)
    for session in validated.sessions:
        if session.kind in {"error_calibration", "final_e2e_test"}:
            manifest = manifests[session.session_id]
            if manifest.formal_dataset_eligible is not True:
                raise ValueError(
                    f"session {session.session_id!r} is not formal_dataset_eligible"
                )
    return split, manifests, validated


def _package_payloads(restored: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for name in _DEPENDENCY_NAMES - {"capture_dataset"}:
        payload = json.loads(restored[name])
        assert isinstance(payload, dict)
        payloads[name] = payload
    return payloads


def _subject_hashes(
    request: FormalBenchmarkRequest,
    restored_payloads: Mapping[str, Mapping[str, Any]],
    profile: FittedPerceptionErrorProfile,
    *, lineage_seal_hash: str,
) -> dict[str, str]:
    assembler = restored_payloads["assembler_config"]
    target = restored_payloads["target_config"]
    values = {
        "parser_artifact_hash": request.dependency_refs["parser_package"].sha256,
        "detector_artifact_hash": request.dependency_refs["detector_package"].sha256,
        "assembler_schema_hash": assembler.get("assembler_schema_hash"),
        "ui_presentation_schema_hash": assembler.get("ui_presentation_schema_hash"),
        "config_hash": request.dependency_refs["target_config"].sha256,
        "capture_dataset_hash": request.dependency_refs["capture_dataset"].sha256,
        "calibration_profile_hash": profile.artifact_hash,
        "threshold_hash": target.get("threshold_hash"),
        "atlas_vocabulary_hash": assembler.get("atlas_vocabulary_hash"),
        "assembler_impl_hash": assembler.get("assembler_impl_hash"),
        "roi_resolver_input_hash": assembler.get("roi_resolver_input_hash"),
        "benchmark_fit_code_hash": profile.fit_code_hash,
        "lineage_seal_hash": lineage_seal_hash,
    }
    for name, value in values.items():
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"formal subject {name} is not a SHA-256")
    return values  # type: ignore[return-value]


def _predicted_ref(logical_id: str, data: bytes) -> ArtifactRef:
    import hashlib

    digest = hashlib.sha256(data).hexdigest()
    return ArtifactRef(
        logical_id=logical_id, sha256=digest, size_bytes=len(data),
        media_type="application/json", store_uri=artifact_uri(digest),
    )


def _descriptor_chain(
    request: FormalBenchmarkRequest,
    profile: FittedPerceptionErrorProfile,
    verdict: PerceptionFinalVerdict,
) -> tuple[ArtifactDescriptor, ...]:
    source = ArtifactDescriptor(
        logical_id="perception/capture/source",
        node_kind="source_descriptor",
        producer_id="benchmark_survivors_perception",
        producer_version="v2",
        identity_metadata={"split_manifest_hash": request.dependency_refs["capture_dataset"].sha256},
        files=(request.dependency_refs["capture_dataset"],),
    )
    # consumer-compatible wire（to_wire()）が canonical ref のコンテンツ（#8対応）
    profile_bytes = canonical_json_bytes(profile.to_wire())
    profile_node = ArtifactDescriptor(
        logical_id=request.calibration_logical_id,
        node_kind="perception_calibration_profile",
        producer_id="perception_error_fit",
        producer_version="v2",
        identity_metadata={
            "profile_hash": profile.profile_hash,
            "fit_code_hash": profile.fit_code_hash,
        },
        parents=(source.node_ref(),),
        files=(_predicted_ref(request.calibration_logical_id, profile_bytes),),
    )
    verdict_bytes = canonical_json_bytes(verdict.to_wire())
    final_node = ArtifactDescriptor(
        logical_id=request.verdict_logical_id,
        node_kind="perception_final_verdict",
        producer_id="benchmark_survivors_perception",
        producer_version="v2",
        identity_metadata={"verdict_id": verdict.verdict_id, "seal_id": verdict.seal_id},
        parents=(profile_node.node_ref(),),
        files=(_predicted_ref(request.verdict_logical_id, verdict_bytes),),
    )
    descriptors = (source, profile_node, final_node)
    validate_artifact_dag(descriptors)
    return descriptors


def run_formal_pipeline(request: FormalBenchmarkRequest) -> FormalBenchmarkResult:
    """formal runner の実経路。依存検証失敗時は ArtifactStore へ何も書かない。"""
    if _CLI_PROVIDER_FACTORY is None:
        raise ValueError(
            "formal pipeline requires an installed provider factory; "
            "install survivors.formal_perception_provider or patch _CLI_PROVIDER_FACTORY in tests"
        )
    # dependency/session restore は read-only で完了させ、その後 final marker だけを先行予約する。
    # 予約後の失敗では session は消費済みだが、seal/profile/verdict は publish されない。
    restored = _restore_dependencies(request)
    calibration_provider, assembler_factory = _CLI_PROVIDER_FACTORY(restored)
    _, manifests, validated_split = _restore_split_and_sessions(
        request, restored["capture_dataset"]
    )
    payloads = _package_payloads(restored)
    calibration_manifests = tuple(
        manifests[session.session_id]
        for session in validated_split.sessions
        if session.kind == "error_calibration"
    )
    final_manifests = tuple(
        manifests[session.session_id]
        for session in validated_split.sessions
        if session.kind == "final_e2e_test"
    )
    # restore 済み manifest の実ファイル hash を再検証してから final 集合を予約する。
    # create-once 競合を provider/staging/canonical publish より先に確定させる。
    for manifest in final_manifests:
        _reserve_final_session(
            request.store, manifest.session_id,
            manifest.manifest_sha256, manifest.session_path / "session_manifest.json",
        )
    # reservation 完了後に final session を full restore（PNG デコード）する。
    # metadata_only=True で restore した manifest を PNG 付きで差し替える。
    final_manifests = tuple(
        DatasetWriter.restore(request.capture_store_root, manifest.session_id)
        for manifest in final_manifests
    )
    residuals = list(calibration_provider(calibration_manifests, restored))
    calibration_hashes = {
        manifest.session_id: manifest.manifest_sha256
        for manifest in calibration_manifests
    }
    final_hashes = {
        manifest.session_id: manifest.manifest_sha256 for manifest in final_manifests
    }
    profile = fit_error_profile(
        residuals,
        [manifest.session_id for manifest in calibration_manifests],
        [manifest.session_id for manifest in final_manifests],
        calibration_session_hashes=calibration_hashes,
        formal=True,
    )

    # seal identity の lineage_seal_hash 自身は含めず、seal wire hash を verdict subject にする。
    preseal_subjects = _subject_hashes(
        request, payloads, profile, lineage_seal_hash="0" * 64
    )
    seal_subjects = {
        name: value for name, value in preseal_subjects.items()
        if name != "lineage_seal_hash"
    }
    # まず pure staged seal を構築する。replay/gate/DAG が失敗しても store は未変更。
    seal = _create_formal_lineage_seal(
        final_session_hashes=final_hashes,
        store=request.store,
        _publish=False,
        **seal_subjects,
    )
    lineage_seal_hash = __import__("hashlib").sha256(
        canonical_json_bytes(seal.to_wire())
    ).hexdigest()
    subjects = _subject_hashes(
        request, payloads, profile, lineage_seal_hash=lineage_seal_hash
    )

    # runner が dataset の raw frame から tick を生成する（外部 provider を排除）。
    # runner が制御するもの: どの frame を処理するか（dataset から）、detector hash（restored package から）、
    # session_kind / source_policy（validated split から）。
    # assembler が制御するもの: frame の解釈のみ（観測値）。
    assembler = assembler_factory(restored)
    # 検証済み store object の content hash を snapshot identity に固定する。
    # package JSON 内の自己申告 artifact_hash は formal identity の根拠にしない。
    parser_content_hash = request.dependency_refs["parser_package"].sha256
    detector_content_hash = request.dependency_refs["detector_package"].sha256
    final_session_records = {
        s.session_id: s for s in validated_split.sessions if s.kind == "final_e2e_test"
    }
    replay_ticks: list[SnapshotReplayTick] = []
    expected_ticks: list[ExpectedTick] = []
    formal_evidence: dict[str, FormalReplayEvidence] = {}
    for manifest in final_manifests:
        record = final_session_records[manifest.session_id]
        for frame in manifest.frames:
            fid = f"{manifest.session_id}:{frame.session_frame_index}:{frame.session_frame_index}"
            expected_ticks.append(ExpectedTick(manifest.session_id, fid))
            ground_truth, predicted, latency_ms, evidence = assembler.assemble_frame(
                frame, manifest.session_id, frame.session_frame_index
            )
            if ground_truth is None:
                continue
            # runner が gt/pred 双方の detector identity を固定する。
            # assembler の自己申告値を formal gate の根拠として受理しない。
            ground_truth = replace(
                ground_truth, detector_artifact_hash=detector_content_hash
            )
            if predicted is not None:
                predicted = replace(
                    predicted, detector_artifact_hash=detector_content_hash
                )
            if evidence is None:
                raise ValueError(
                    f"assembler returned no formal evidence: "
                    f"session={manifest.session_id} frame={frame.session_frame_index}"
                )
            snapshots = (ground_truth,) if predicted is None else (ground_truth, predicted)
            if any(
                snapshot.ui_presentation.parser_artifact_hash != parser_content_hash
                for snapshot in snapshots
            ):
                raise ValueError(
                    "snapshot parser_artifact_hash does not match "
                    f"parser_package content ({parser_content_hash[:8]}…)"
                )
            # 同一 tick の gt/pred は V1 契約上同じ frame_id に束縛される。
            # replay_snapshots がこの evidence を双方の UI hash に対して検証する。
            formal_evidence[ground_truth.frame_id] = evidence
            replay_ticks.append(SnapshotReplayTick(
                manifest.session_id, record.kind, record.source_policy,
                ground_truth.frame_id, ground_truth, predicted, latency_ms,
            ))
    report = _formalize_benchmark_report(
        run_benchmark(replay_ticks, expected_ticks=expected_ticks, formal_evidence=formal_evidence)
    )
    # final session は既に消費済みだが、gate 不合格 verdict は ArtifactStore へ公開しない。
    # staging/batch commit/canonical publish は gate 成功が確定した後だけ開始する。
    if not report.passed:
        raise ValueError(
            "formal perception gate failed; verdict was not published: "
            + "; ".join(report.blocking_reasons)
        )
    verdict = _create_formal_final_verdict(
        report, seal_id=seal.seal_id,
        final_session_ids=[manifest.session_id for manifest in final_manifests],
        **subjects,
    )
    descriptors = _descriptor_chain(request, profile, verdict)

    # Phase B: stage → batch commit（single atomic）→ seal → canonical logical_id の順序で publish する。
    # ・batch commit が唯一の commit 点（単一 put_bytes）：これが存在すれば cal + verdict が確定。
    # ・staging 後に失敗 → batch commit 未発行 → canonical logical_id 不可視（clean state）。
    # ・batch commit 後に失敗 → 再実行は全操作が idempotent なので安全に回復できる。

    # 1) content を staging logical_id（content hash ベース・canonical 名前空間外）へ書く。
    staged_cal_ref = _write_formal_calibration_profile(
        profile, store=request.store,
        logical_id=f"perception/staging/calibration/{descriptors[1].files[0].sha256}",
    )
    staged_ver_ref = _write_formal_final_verdict(
        verdict, store=request.store,
        logical_id=f"perception/staging/verdict/{descriptors[2].files[0].sha256}",
    )
    if (
        staged_cal_ref.sha256 != descriptors[1].files[0].sha256
        or staged_ver_ref.sha256 != descriptors[2].files[0].sha256
    ):
        raise ValueError("staged refs sha256 differ from prevalidated descriptor refs")
    validate_artifact_dag(descriptors)

    # 2a0) descriptor manifest を immutable object として publish する。
    for desc in descriptors:
        desc_bytes = canonical_json_bytes(desc.to_wire())
        request.store.put_bytes(
            logical_id=f"perception/staging/descriptor/{desc.logical_id}",
            data=desc_bytes, media_type="application/json",
        )

    # 2a) seal を batch commit より前にステージングする。
    #     batch commit 直後の consumer が seal を見つけられるよう commit 点の前に公開する。
    seal_bytes = canonical_json_bytes(seal.to_wire())
    staged_seal_sha = lineage_seal_hash  # seal wire の sha256 = lineage_seal_hash
    request.store.put_bytes(
        logical_id=f"perception/staging/seal/{seal.seal_id}",
        data=seal_bytes, media_type="application/json",
    )

    # 2b) batch commit manifest を単一 put_bytes で発行する（= commit 点）。
    #     calibration / verdict / seal の sha256 を同時に可視化する。
    #     同一 lineage の再実行は同一 sha256 → idempotent。
    batch_commit_payload = canonical_json_bytes({
        "schema_version": "perception_formal_batch_commit.v1",
        "seal_id": seal.seal_id,
        "seal_artifact_sha256": staged_seal_sha,
        "calibration_artifact_sha256": staged_cal_ref.sha256,
        "verdict_artifact_sha256": staged_ver_ref.sha256,
    })
    request.store.put_bytes(
        logical_id=f"perception/batch_commit/{seal.seal_id}",
        data=batch_commit_payload,
        media_type="application/json",
    )

    # 3) 以降はすべて idempotent（batch commit 済みのため re-run 安全）。
    persisted_seal = _create_formal_lineage_seal(
        final_session_hashes=final_hashes,
        store=request.store,
        **seal_subjects,
    )
    if persisted_seal.seal_id != seal.seal_id:
        raise ValueError("persisted lineage seal identity changed after validation")
    calibration_ref = _write_formal_calibration_profile(
        profile, store=request.store, logical_id=request.calibration_logical_id,
        subject_hashes=subjects,
    )
    verdict_ref = _write_formal_final_verdict(
        verdict, store=request.store, logical_id=request.verdict_logical_id
    )
    return FormalBenchmarkResult(
        validated_split, profile, verdict, calibration_ref, verdict_ref, descriptors
    )


def _run_dry() -> int:
    """公開 synthetic 入口は常に development-only。"""
    from survivors.perception_benchmark import BenchmarkRecord

    records = []
    for session_id in ("synthetic-a", "synthetic-b"):
        for frame_id in ("0", "1"):
            records.append(
                BenchmarkRecord(
                    frame_id, session_id, "error_calibration", "raw",
                    "screen_state", "gameplay", "gameplay", 1.0, 1.0,
                )
            )
    report = run_benchmark(records)
    print(json.dumps({
        "development_only": report.development_only,
        "formal_perception_verdict_eligible": report.formal_perception_verdict_eligible,
        "passed": report.passed,
        "blocking_reasons": report.blocking_reasons,
    }, indent=2))
    return 0


def _load_ref(path_str: str) -> ArtifactRef:
    path = Path(path_str).expanduser().resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ArtifactRef file: {path}") from exc
    return ArtifactRef.from_wire(data)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Survivors perception formal benchmark")
    for option in (
        "capture-dataset", "parser-package", "detector-package",
        "assembler-config", "target-config",
    ):
        parser.add_argument(f"--{option}", help=f"{option} ArtifactRef JSON")
    parser.add_argument("--artifact-store", help="ArtifactStore root")
    parser.add_argument("--capture-store", help="restorable capture session store root")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.dry_run:
        return _run_dry()
    required = {
        "capture_dataset": args.capture_dataset,
        "parser_package": args.parser_package,
        "detector_package": args.detector_package,
        "assembler_config": args.assembler_config,
        "target_config": args.target_config,
        "artifact_store": args.artifact_store,
        "capture_store": args.capture_store,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        print(f"BLOCKED: missing formal inputs: {', '.join(sorted(missing))}", file=sys.stderr)
        return 2
    try:
        refs = {name: _load_ref(required[name]) for name in _DEPENDENCY_NAMES}
        store = ArtifactStore(required["artifact_store"])
        result = run_formal_pipeline(
            FormalBenchmarkRequest(
                store=store, dependency_refs=refs,
                capture_store_root=Path(required["capture_store"]),
            )
        )
    except Exception as exc:  # fail closed: dependency failure never prints Verified
        print(f"BLOCKED: formal benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "verdict_id": result.verdict.verdict_id,
        "passed": result.verdict.passed,
        "verdict_artifact": result.verdict_ref.store_uri,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

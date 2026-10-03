"""Survivors controller(shadow/live)を実機ウィンドウに対して起動する CLI。

detector・tracker・HUD parser・obs assembler・policy runtime・state machine・health・telemetry を
組み立てて ``SurvivorsController`` に渡し、``run()`` の戻り値をそのままプロセス終了コードにします。
既定は shadow mode(入力を一切送らない観測専用)です。live mode は
``--live --target-profile <path> --ack-risk`` の3点がそろったときだけ受け付け、さらに正式な
live 許可(runtime bundle と detector の formal verdict)が無ければ入力系を作る前に拒否します。
live 中の arm/disarm は別プロセスの input helper が Ctrl+Shift+F12 の edge で切り替えます
(この CLI は hotkey を監視しません)。hold-to-run の dead-man 操作は導入していません。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
from typing import NoReturn

import numpy as np
import torch
import yaml

from survivors.action_semantics import load_action_contract
from survivors.capture import CaptureSession, CtypesWin32Api, DxcamCaptureBackend, TargetWindowPolicy, WindowLocator
from survivors.controller import controller as controller_module
from survivors.controller.controller import EXIT_ERROR, SurvivorsController
from survivors.controller.health_monitor import HealthMonitor
from survivors.controller.state_machine import CampaignRunMode, StateMachine
from survivors.controller.telemetry import TelemetrySessionHeader, TelemetryWriter
from survivors.input.controller import InputLeaseController
from survivors.real_obs_assembler import RealObsAssembler
from survivors.runtime.agent_runtime import AgentRuntime
from survivors.runtime.artifact_bundle import BundleLoadError, RuntimeBundle, _load_combat_package
from survivors.target_profile import load_runtime_profile
from survivors.vision import hud_parser as hud_parser_module
from survivors.vision.entity_tracker import EntityTracker
from survivors.vision.hud_parser import HudParser
from survivors.vision.world_dataset import load_class_map
from survivors.vision.world_detector import (
    CheckpointManifest,
    FormalDetectorRejectedError,
    WorldDetector,
    load_detector_config,
)

# 正式 live verdict が無いため --live を拒否したときの終了コード(controller の 0/2/3/4 と重ならない)。
EXIT_LIVE_REJECTED = 5


def _sha256_file(path: Path) -> str:
    """ファイル内容の sha256 hex を返す。

    telemetry の artifact hash と weight の改ざん検出に使います。
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """コマンドライン引数を読み、live 起動の3点セットを検証する。

    ``--live`` は ``--target-profile`` と ``--ack-risk`` の両方が無ければ argparse エラー(exit 2)です。
    ``--ack-risk`` を ``--live`` なしで渡すのも誤操作として拒否します。
    ``--target-profile`` は shadow でも使えます(省略時は .env/ の確定済み profile)。
    """
    parser = argparse.ArgumentParser(description="Survivors controller (default: shadow mode, no OS input)")
    parser.add_argument("--telemetry", required=True, type=Path, help="新規作成する telemetry JSONL のパス")
    parser.add_argument("--combat-package", required=True, type=Path)
    parser.add_argument("--detector-config", required=True, type=Path)
    parser.add_argument("--class-map", required=True, type=Path)
    parser.add_argument("--detector-weights", required=True, type=Path)
    parser.add_argument("--detector-manifest", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument(
        "--session-id", default=datetime.now(timezone.utc).strftime("controller-%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--campaign-run-mode", choices=[mode.value for mode in CampaignRunMode],
        default=CampaignRunMode.FORMAL_SINGLE_ATTEMPT.value,
    )
    parser.add_argument("--target-profile", type=Path, default=None)
    parser.add_argument("--live", action="store_true", help="実入力を送る(--target-profile と --ack-risk が必須)")
    parser.add_argument("--ack-risk", action="store_true", help="live mode の実入力リスクを了承する")
    args = parser.parse_args(argv)
    if args.live and (args.target_profile is None or not args.ack_risk):
        parser.error("--live requires both --target-profile <path> and --ack-risk")
    if args.ack_risk and not args.live:
        parser.error("--ack-risk is only valid together with --live")
    return args


def _load_detector(args: argparse.Namespace) -> tuple[WorldDetector, EntityTracker, CheckpointManifest]:
    """detector config/class map/weight/manifest から detector と tracker を組み立てる。

    weight/config/class map の sha256 が manifest の model_hash/config_hash/class_map_hash と
    1つでも違えば読み込まずに止めます(world_detector_package.restore_package と同じ照合)。
    tracker の設定は world_detector_package の復元経路と同じく config の ``tracker`` 節から作ります。
    """
    manifest = CheckpointManifest.load(args.detector_manifest)
    for path, expected, message in (
        (args.detector_weights, manifest.model_hash, "detector weight hash does not match the checkpoint manifest model_hash"),
        (args.detector_config, manifest.config_hash, "detector config hash does not match the checkpoint manifest config_hash"),
        (args.class_map, manifest.class_map_hash, "class map hash does not match the checkpoint manifest class_map_hash"),
    ):
        if _sha256_file(path) != expected:
            raise ValueError(message)
    config = load_detector_config(args.detector_config)
    detector = WorldDetector.from_config(config, args.class_map)
    state_dict = torch.load(args.detector_weights, map_location="cpu", weights_only=True)
    detector._model.load_state_dict(state_dict)
    tracker_cfg = config.get("tracker", {})
    class_map = load_class_map(args.class_map)
    tracker = EntityTracker(
        max_age_by_class={class_map.name_to_id(name): age for name, age in tracker_cfg.get("max_age_by_class", {}).items()},
        max_match_cost=tracker_cfg.get("max_match_cost", 0.7),
        velocity_ema_alpha=tracker_cfg.get("velocity_ema_alpha", 0.6),
        confidence_decay_per_frame=tracker_cfg.get("confidence_decay_per_frame", 0.9),
        coarse_by_class_id=class_map.coarse_by_class_id(),
    )
    return detector, tracker, manifest


def _load_artifacts(args: argparse.Namespace) -> SimpleNamespace:
    """combat package・detector・HUD parser を読み、artifact hash 一覧と一緒に返す。

    combat package は hash 検証付きで読み、development bundle(live 不可)に包みます。
    HUD parser は学習済み重みを持たないため、parser のソースコードの hash を artifact hash とします。
    """
    # ponytail: development bundle のみ。正式 RuntimeBundle.load() 経路は formal artifact(03-05/04-10)が揃ったら配線する。
    combat_policy, combat_manifest = _load_combat_package(args.combat_package)
    bundle = RuntimeBundle.from_golden_fixture(combat_policy)
    detector, tracker, detector_manifest = _load_detector(args)
    parser_hash = _sha256_file(Path(hud_parser_module.__file__))
    return SimpleNamespace(
        bundle=bundle, detector=detector, tracker=tracker, detector_manifest=detector_manifest,
        hud_parser=HudParser(parser_artifact_hash=parser_hash),
        artifact_hashes={
            "combat_model": combat_manifest["model_sha256"],
            "detector_model": detector_manifest.model_hash,
            # manifest の自己申告ではなく、実際に読んだファイルの hash を記録する。
            "detector_class_map": _sha256_file(args.class_map),
            "hud_parser": parser_hash,
            "deploy_obs_schema": bundle.deploy_schema_hash,
        },
    )


def _live_gate(bundle: RuntimeBundle, detector_manifest: CheckpointManifest) -> None:
    """runtime bundle と detector の両方が正式 live 許可を持つか確認する(M10)。

    どちらかが development 成果物なら例外を送出します。呼び出し側はここを通るまで
    capture も入力 helper も作りません。
    """
    bundle.assert_live_eligible()
    detector_manifest.assert_formal_eligible()


def _open_capture(profile) -> tuple[CaptureSession, object]:
    """対象ウィンドウを特定し、未開始の CaptureSession と特定結果を返す。

    capture の開始・停止は controller の ``run()`` と shutdown が行います。
    特定結果の pid/hwnd は live mode の入力 helper の固定先に使います。
    """
    policy = TargetWindowPolicy(
        process_executable="VampireSurvivors.exe", window_class="YYGameMakerYY", window_title="Vampire Survivors",
    )
    locator = WindowLocator(CtypesWin32Api(), profile, policy)
    target = locator.locate()
    backend = DxcamCaptureBackend.create(
        output_idx=target.monitor.dxgi_output_idx,
        device_idx=target.monitor.dxgi_device_idx,
        expected_client_rect=target.client_rect_screen_px,
    )
    return CaptureSession(locator, target, backend), target


def _header(args: argparse.Namespace, profile, artifact_hashes: dict[str, str]) -> TelemetrySessionHeader:
    """telemetry 先頭に書く session identity と実行環境を作る。

    profile/build/artifact hash と host/依存バージョンを固定し、後から出所を辿れるようにします。
    """
    return TelemetrySessionHeader(
        session_id=args.session_id,
        mode="live" if args.live else "shadow",
        target_profile_hash=profile.target_hash,
        game_build_id=str(profile.sections["build"]["build_id"]),
        controller_build_id=_sha256_file(Path(controller_module.__file__)),
        artifact_hashes=artifact_hashes,
        host={"platform": platform.platform(), "node": platform.node()},
        device={"inference": "cpu"},
        dependency_versions={"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
    )


def _write_abort(telemetry: TelemetryWriter, session_id: str, stage: str, payload: dict) -> None:
    """controller 起動前の中断理由を telemetry へ1行書いて閉じる。

    controller が動き出す前の失敗でも、理由が記録に残るようにします。
    """
    try:
        telemetry.write_stage(
            stage, correlation_id=f"{session_id}:controller", timestamp_ns=time.perf_counter_ns(),
            latency_ns=0, queue_depth=0, payload=payload,
        )
    finally:
        telemetry.close()


def _run(args: argparse.Namespace) -> int:
    """部品を組み立てて controller を実行し、その終了コードを返す。

    live は正式 verdict の確認(失敗なら EXIT_LIVE_REJECTED)→ capture 特定 → 入力 helper 起動の順です。
    controller 起動前の例外は telemetry に記録して EXIT_ERROR を返します。
    起動後は controller が shutdown と telemetry close まで責任を持ちます。
    """
    profile = load_runtime_profile(args.target_profile) if args.target_profile else load_runtime_profile()
    parts = _load_artifacts(args)
    telemetry = TelemetryWriter(args.telemetry, _header(args, profile, parts.artifact_hashes))
    if args.live:
        try:
            _live_gate(parts.bundle, parts.detector_manifest)
        except (BundleLoadError, FormalDetectorRejectedError) as exc:
            _write_abort(
                telemetry, args.session_id, "live_gate",
                {"eligible": False, "type": type(exc).__name__, "reason": str(exc)},
            )
            print(f"--live rejected (fail closed): {exc}", file=sys.stderr)
            return EXIT_LIVE_REJECTED
    controller = None

    def build(capture, runtime, input_controller) -> int:
        """controller を作って arm し、run() の終了コードを返す(shadow/live 共通経路)。"""
        nonlocal controller
        controller = SurvivorsController(
            mode="live" if args.live else "shadow", session_id=args.session_id, capture=capture,
            detector=parts.detector, tracker=parts.tracker, hud_parser=parts.hud_parser,
            assembler=RealObsAssembler(), runtime=runtime,
            state_machine=StateMachine(), health=HealthMonitor(), telemetry=telemetry,
            schema=parts.bundle.deploy_schema, model_hashes=parts.artifact_hashes,
            input_controller=input_controller, ui_config=parts.bundle.ui_policy_config,
            class_map_path=args.class_map, score_threshold=args.score_threshold,
        )
        controller.arm(
            campaign_run_mode=CampaignRunMode(args.campaign_run_mode),
            run_id=args.session_id, gameplay_attempt_id=f"{args.session_id}:attempt-1",
        )
        return controller.run(max_frames=args.max_frames)

    try:
        # require_live は gate の二重化: live 不可の bundle なら入力 helper を作る前にここで止まる。
        runtime = AgentRuntime(parts.bundle, require_live=args.live)
        capture, target = _open_capture(profile)
        if not args.live:
            return build(capture, runtime, None)
        with InputLeaseController(
            target_hash=profile.target_hash, action_hash=load_action_contract().contract_hash,
            target_pid=target.pid, target_hwnd=target.hwnd,
            audit_path=args.telemetry.with_name(args.telemetry.stem + ".input_audit.jsonl"),
        ) as lease:
            return build(capture, runtime, lease)
    except Exception as exc:
        if controller is not None and controller.exit_code is not None:
            raise
        _write_abort(telemetry, args.session_id, "error", {"stage": "startup", "type": type(exc).__name__, "message": str(exc)})
        print(f"controller startup failed: {exc}", file=sys.stderr)
        return EXIT_ERROR


def main(argv: list[str] | None = None) -> NoReturn:
    """CLI エントリポイント。controller の終了コードをそのままプロセス終了コードにする。

    ``SurvivorsController.run()`` は終了コードを返すだけなので、ここで ``sys.exit`` します。
    health STOP(2)・例外(3)・terminal failure(4)・live 拒否(5)はすべて非0で終わります。
    """
    sys.exit(_run(parse_args(argv)))


if __name__ == "__main__":
    main()

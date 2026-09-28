"""記録済み capture session を実 SurvivorsController へ仮想時計で再生する E2E replay CLI(06-01)。

``run_survivors_controller.py`` と同じ手順で combat package・detector・HUD parser を hash 検証付きで読み、
capture manifest の target profile / game build と照合してから ``run_recorded_replay`` を1回実行します。
controller は常に shadow mode で、入力 helper も OS 入力 backend も作りません。
出力(telemetry・discrete/numeric・effects・manifest)は ``--output-dir`` へ書きます。
終了コードは controller の終了コード(0/2/3/4)をそのまま返します。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from run_survivors_controller import _load_artifacts
from survivors.controller.state_machine import CampaignRunMode
from survivors.replay.e2e_replay import CaptureManifest, run_recorded_replay
from survivors.replay.recorded_frame_source import DeterminismManifest
from survivors.target_profile import load_runtime_profile


def build_parser() -> argparse.ArgumentParser:
    """CLI の引数解析器を組み立てる。

    artifact 系の引数名は run_survivors_controller.py と同じにし、同じ loader をそのまま使います。
    """
    parser = argparse.ArgumentParser(description="記録済み capture session を shadow controller で決定的に再生する。")
    parser.add_argument("--capture-manifest", required=True, type=Path, help="recorded capture manifest(JSON)")
    parser.add_argument("--output-dir", required=True, type=Path, help="replay 出力の保存先(新規 telemetry を作る)")
    parser.add_argument("--combat-package", required=True, type=Path)
    parser.add_argument("--detector-config", required=True, type=Path)
    parser.add_argument("--class-map", required=True, type=Path)
    parser.add_argument("--detector-weights", required=True, type=Path)
    parser.add_argument("--detector-manifest", required=True, type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--target-profile", type=Path, default=None, help="省略時は .env/ の確定済み profile")
    parser.add_argument(
        "--campaign-run-mode", choices=[mode.value for mode in CampaignRunMode],
        default=CampaignRunMode.FORMAL_SINGLE_ATTEMPT.value,
    )
    parser.add_argument("--device", default="cpu", help="推論 device(記録時の determinism manifest と照合)")
    parser.add_argument("--nms-backend", default="torchvision.ops.nms", help="NMS 実装名(記録時と照合)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI エントリポイント。capture 照合 → artifact 読み込み → replay の順に実行し終了コードを返す。

    capture の profile/build が合わなければ重い artifact を読む前に ReplayIntegrityError で止まります。
    """
    args = build_parser().parse_args(argv)
    profile = load_runtime_profile(args.target_profile) if args.target_profile else load_runtime_profile()
    identity = {"target_profile_hash": profile.target_hash, "game_build_id": str(profile.sections["build"]["build_id"])}
    capture = CaptureManifest.load(args.capture_manifest)
    capture.require_identity(**identity)
    parts = _load_artifacts(args)
    result = run_recorded_replay(
        capture, args.output_dir,
        detector=parts.detector, tracker=parts.tracker, hud_parser=parts.hud_parser, bundle=parts.bundle,
        artifact_hashes=parts.artifact_hashes, **identity,
        replay_determinism=DeterminismManifest.from_torch(device=args.device, nms_backend=args.nms_backend),
        detector_manifest=parts.detector_manifest, class_map_path=args.class_map,
        score_threshold=args.score_threshold, campaign_run_mode=CampaignRunMode(args.campaign_run_mode),
    )
    print(f"exit_code={result.exit_code} reason={result.exit_reason} "
          f"formal_replay_eligible={result.manifest['formal_replay_eligible']} manifest={result.paths['manifest.json']}")
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())

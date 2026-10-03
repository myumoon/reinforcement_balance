"""記録済み capture session を実 SurvivorsController へ仮想時計で再生する E2E replay CLI(06-01)。

サブコマンドは4つです。
- ``replay``: combat package・detector・HUD parser を hash 検証付きで読み、capture manifest の
  target profile / game build と照合してから ``run_recorded_replay`` を実行します。``--runs 3`` なら
  同じ bundle を3回再生し、1回目と各回を ``compare_replays`` で比べます(local full suite)。
- ``compare``: 2つの replay 出力、または golden と replay 出力を比べ、最初の分岐点を tree で出します。
- ``update-golden``: 旧/新 run の集計 metrics・比較 metrics・artifact hashes を揃えて golden を書き直します。
- ``publish-formal-verdict``: 同一 bundle の3回分から formal replay verdict を書きます。正式 parent が
  欠けている(development_only を含む)ときは ``FormalReplayRejectedError`` で publish を拒否します。
controller は常に shadow mode で、入力 helper も OS 入力 backend も作りません。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from run_survivors_controller import _load_artifacts
from survivors.controller.state_machine import CampaignRunMode
from survivors.replay.e2e_replay import (
    NMS_BACKEND,
    CaptureManifest,
    FormalReplayRejectedError,
    compare_replays,
    golden_mismatches,
    inference_device,
    publish_formal_replay_verdict,
    replay_metrics,
    run_recorded_replay,
    update_golden,
)
from survivors.replay.recorded_frame_source import DeterminismManifest
from survivors.target_profile import load_runtime_profile

# compare / update-golden は artifact を読まないので、runtime bundle と同じ既定 DeployObs schema で segment 名を付ける。
_OBS_LAYOUT = DeployObsSchema.default_v2().layout


def _add_artifact_args(parser: argparse.ArgumentParser) -> None:
    """artifact 系の引数を足す(名前は run_survivors_controller.py と同じで、同じ loader をそのまま使う)。"""
    parser.add_argument("--capture-manifest", required=True, type=Path, help="recorded capture manifest(JSON)")
    parser.add_argument("--combat-package", required=True, type=Path)
    parser.add_argument("--detector-config", required=True, type=Path)
    parser.add_argument("--class-map", required=True, type=Path)
    parser.add_argument("--detector-weights", required=True, type=Path)
    parser.add_argument("--detector-manifest", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    """CLI の引数解析器(replay / compare / update-golden / publish-formal-verdict)を組み立てる。

    サブコマンドごとに必要な引数だけを受け取ります。
    """
    parser = argparse.ArgumentParser(description="記録済み capture session を shadow controller で決定的に再生する。")
    commands = parser.add_subparsers(dest="command", required=True)

    replay = commands.add_parser("replay", help="capture session を再生する(--runs 3 で決定性確認)")
    _add_artifact_args(replay)
    replay.add_argument("--output-dir", required=True, type=Path, help="replay 出力の保存先(--runs>1 なら run-<k>/)")
    replay.add_argument("--runs", type=int, default=1, help="同じ bundle を再生する回数(formal verdict は3)")
    replay.add_argument("--score-threshold", type=float, default=0.5)
    replay.add_argument("--target-profile", type=Path, default=None, help="省略時は .env/ の確定済み profile")
    replay.add_argument(
        "--campaign-run-mode", choices=[mode.value for mode in CampaignRunMode],
        default=CampaignRunMode.FORMAL_SINGLE_ATTEMPT.value,
    )
    replay.add_argument(
        "--device", default=None,
        help="推論 device の申告値。省略時は読み込んだ detector/policy の device を使い、指定時は実測と食い違えば拒否する",
    )
    replay.add_argument(
        "--nms-backend", choices=[NMS_BACKEND], default=NMS_BACKEND, help="NMS 実装名(detector が実際に使うものだけ)",
    )

    compare = commands.add_parser("compare", help="replay 出力を旧 run / golden と比べる")
    compare.add_argument("--new", required=True, type=Path, help="比べる replay 出力ディレクトリ")
    compare.add_argument("--old", type=Path, default=None, help="基準の replay 出力ディレクトリ")
    compare.add_argument("--golden", type=Path, default=None, help="基準の golden JSON")
    compare.add_argument("--report", type=Path, default=None, help="比較結果 JSON の保存先")

    golden = commands.add_parser("update-golden", help="expected improvement を golden として書き直す")
    golden.add_argument("--golden", required=True, type=Path)
    golden.add_argument("--old-run", required=True, type=Path, help="旧 golden 側の replay 出力")
    golden.add_argument("--new-run", required=True, type=Path, help="新しい golden にする replay 出力")
    golden.add_argument("--artifact-hashes", required=True, type=Path, help="新 run の artifact hashes(JSON)")
    golden.add_argument("--approved-by", default=None, help="artifact 版が変わるときの独立検証担当者")

    verdict = commands.add_parser("publish-formal-verdict", help="3回分の replay から formal verdict を書く")
    _add_artifact_args(verdict)
    verdict.add_argument("--run", required=True, type=Path, action="append", help="replay 出力(3回分を順に指定)")
    verdict.add_argument("--verdict", required=True, type=Path, help="formal replay verdict の保存先")
    return parser


def _run_paths(directory: Path) -> dict[str, Path]:
    """replay 出力ディレクトリを compare_replays などが受け取る paths dict にする。"""
    return {path.name: path for path in directory.iterdir() if path.is_file()}


def _replay(args: argparse.Namespace) -> int:
    """capture 照合 → artifact 読み込み → replay(--runs 回)→ 回ごとの比較の順に実行する。

    capture の profile/build が合わなければ重い artifact を読む前に ReplayIntegrityError で止まります。
    1回なら controller の終了コード、複数回なら比較が1つでも失敗すると 1 を返します。
    """
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    profile = load_runtime_profile(args.target_profile) if args.target_profile else load_runtime_profile()
    identity = {"target_profile_hash": profile.target_hash, "game_build_id": str(profile.sections["build"]["build_id"])}
    capture = CaptureManifest.load(args.capture_manifest)
    capture.require_identity(**identity)
    results = []
    for run in range(1, args.runs + 1):
        # run ごとに読み直す(controller は tracker を reset しないので、内部状態を持つ部品を run 間で共有しない)。
        parts = _load_artifacts(args)
        output_dir = args.output_dir if args.runs == 1 else args.output_dir / f"run-{run}"
        # device は読み込んだ部品から実測する。--device の申告が実測と違えば run_recorded_replay が拒否する。
        device = args.device if args.device is not None else inference_device(parts.detector, parts.bundle)
        result = run_recorded_replay(
            capture, output_dir,
            detector=parts.detector, tracker=parts.tracker, hud_parser=parts.hud_parser, bundle=parts.bundle,
            artifact_hashes=parts.artifact_hashes, **identity,
            replay_determinism=DeterminismManifest.from_torch(device=device, nms_backend=args.nms_backend),
            detector_manifest=parts.detector_manifest, class_map_path=args.class_map,
            score_threshold=args.score_threshold, campaign_run_mode=CampaignRunMode(args.campaign_run_mode),
        )
        print(f"run={run} exit_code={result.exit_code} reason={result.exit_reason} "
              f"formal_replay_eligible={result.manifest['formal_replay_eligible']} manifest={result.paths['manifest.json']}")
        results.append(result)
    if args.runs == 1:
        return results[0].exit_code
    layout = parts.bundle.deploy_schema.layout
    diffs = [compare_replays(results[0], result, obs_layout=layout) for result in results[1:]]
    report = [{"old": "run-1", "new": f"run-{k}", "passed": diff.passed, "metrics": diff.metrics,
               "first_divergence": diff.first_divergence} for k, diff in enumerate(diffs, start=2)]
    (args.output_dir / "comparisons.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    passed = all(diff.passed for diff in diffs)
    print(f"runs={args.runs} deterministic={passed} comparisons={args.output_dir / 'comparisons.json'}")
    return results[0].exit_code if passed else 1


def _compare(args: argparse.Namespace) -> int:
    """新 run を旧 run と golden の指定された方と比べる(どちらも quantized hash → tolerance と first divergence)。

    golden は golden に固定した参照 run 出力と compare_replays と同じ規則で比べ、artifact hashes は exact に比べます。
    どちらも合格なら 0、1つでも食い違えば 1 を返します。
    """
    if args.old is None and args.golden is None:
        raise SystemExit("compare needs --old and/or --golden")
    new = _run_paths(args.new)
    report: dict = {}
    if args.old is not None:
        diff = compare_replays(_run_paths(args.old), new, obs_layout=_OBS_LAYOUT)
        report.update(passed=diff.passed, metrics=diff.metrics, first_divergence=diff.first_divergence)
    if args.golden is not None:
        mismatches, golden_diff = golden_mismatches(args.golden, new, obs_layout=_OBS_LAYOUT)
        report.update(
            golden_mismatches=mismatches, golden_first_divergence=golden_diff.first_divergence,
            golden_metrics=golden_diff.metrics, passed=report.get("passed", True) and not mismatches,
        )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


def _update_golden(args: argparse.Namespace) -> int:
    """旧/新 run の集計 metrics と比較 metrics、明示の artifact hashes で golden を書き直す。"""
    old, new = _run_paths(args.old_run), _run_paths(args.new_run)
    golden = update_golden(
        args.golden, new,
        old_metrics=replay_metrics(old), new_metrics=replay_metrics(new),
        comparison=compare_replays(old, new, obs_layout=_OBS_LAYOUT).metrics,
        artifact_hashes=json.loads(args.artifact_hashes.read_text(encoding="utf-8")),
        approved_by=args.approved_by, obs_layout=_OBS_LAYOUT,
    )
    print(f"golden={args.golden} discrete_sha256={golden['discrete_sha256']} approved_by={golden['approved_by']}")
    return 0


def _publish_formal_verdict(args: argparse.Namespace) -> int:
    """同一 bundle の3回分の replay 出力と正式 parent から formal replay verdict を書く。

    run の artifact hashes が今読んだ artifact と違えば、正式 parent と別物の再生なので拒否します。
    それ以外の拒否条件(development_only・run 数不足など)は publish_formal_replay_verdict が判定します。
    """
    capture = CaptureManifest.load(args.capture_manifest)
    parts = _load_artifacts(args)
    runs = [_run_paths(directory) for directory in args.run]
    manifests = [json.loads(run["manifest.json"].read_text(encoding="utf-8")) for run in runs]
    if any(manifest["artifact_hashes"] != parts.artifact_hashes for manifest in manifests):
        raise FormalReplayRejectedError("replay runs were not produced by the loaded formal artifacts")
    layout = parts.bundle.deploy_schema.layout
    verdict = publish_formal_replay_verdict(
        args.verdict, diffs=[compare_replays(runs[0], run, obs_layout=layout) for run in runs[1:]],
        manifests=manifests, bundle=parts.bundle, detector_manifest=parts.detector_manifest, capture=capture,
    )
    print(f"verdict={args.verdict} passed={verdict['passed']} runs={verdict['runs']}")
    return 0 if verdict["passed"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    """CLI エントリポイント。サブコマンドを実行して終了コードを返す。"""
    args = build_parser().parse_args(argv)
    handlers = {
        "replay": _replay, "compare": _compare,
        "update-golden": _update_golden, "publish-formal-verdict": _publish_formal_verdict,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

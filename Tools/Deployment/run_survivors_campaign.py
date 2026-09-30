"""Survivors live canary campaign を C0〜C4 の順に進める CLI です。

`run` は campaign plan を組み立てて 06-04 の CampaignRunner / SaveLifecycle を呼ぶだけで、
launch・activation・report の意味論は 06-02 / 06-03 / runner に任せます(この file では複製しません)。
現在動くのは development dry-run です。06-03 の実 LaunchBroker が harmless な target helper
(`python.exe -c "sleep"`)を起動し、fake controller / helper / target / operator が即座に結果を返すので、
C0〜C4 の virtual schedule を数秒で最後まで流せます。dry-run の成果物は必ず
`development_only=true` / `formal_campaign_eligible=false` です。
`--live` は formal parents(06-02 prerequisite)が無い場合に何も書かずに拒否します(fail closed)。
`restore` は backup store へ複製した artifact bundle を hash 検証してから空の root へ戻します。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from reinbalance_survivors_contracts.canonical_json import canonical_hash, canonical_json_bytes

from survivors.campaign.campaign_runner import (
    STAGES,
    SUMMARY,
    ArtifactStore,
    CampaignBlocked,
    CampaignPlan,
    CampaignRunner,
    ReleaseObservation,
    RunnerError,
    RunnerState,
    RunResult,
)
from survivors.campaign.durable_launch_store import DurableLaunchStore, LedgerError
from survivors.campaign.save_lifecycle import SaveLifecycle, SaveLifecycleError
from survivors.campaign.win32_launch_broker import LaunchBroker, probe_identity

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "mad_forest_canary_v1.yaml"
CONFIG_SCHEMA = "survivors_canary_campaign.v1"
CONFIG_KEYS = frozenset({"schema_version", "grace_seconds", "thresholds", "threshold_parent_hash"})
# development dry-run で broker が起動する harmless target helper。broker close 時に Job ごと終了します。
HARMLESS_TARGET = "import time; time.sleep(600)"
BUNDLE_DIR = "bundle"
BUNDLE_INDEX = "bundle_index.json"
EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_NOT_PASSED = 3
_DIGEST = re.compile(r"[0-9a-f]{64}")


class CliRefusal(RuntimeError):
    """CLI が fail closed で処理を拒否したことを表します。

    この例外で止まった場合、artifact・ledger・save はまだ変更されていません
    (restore の検証失敗も、root へ書く前に止まります)。
    """


def _sha256(data: bytes) -> str:
    """bytes の SHA-256 hex を返します。

    plan の各 hash と bundle index で使います。
    """
    return hashlib.sha256(data).hexdigest()


def load_config(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    """campaign config を読み込み、内容と file hash を返します。

    既知の key だけを持つ `survivors_canary_campaign.v1` 以外は拒否します。
    file hash は plan の config_hash になり、config を変えると新しい campaign id が必要になります。
    """
    data = Path(path).read_bytes()
    config = yaml.safe_load(data)
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS or config["schema_version"] != CONFIG_SCHEMA:
        raise CliRefusal(f"{path} must be a {CONFIG_SCHEMA} config with keys {sorted(CONFIG_KEYS)}")
    return config, _sha256(data)


def refuse_live(formal_parents: str | None, config: Mapping[str, Any]) -> None:
    """`--live` の前提を順に検査し、この build では必ず拒否します。

    formal parents(06-02 prerequisite と parent hash)が無い・development-only である、
    metric floor が未解決、のいずれかなら理由を示して拒否します。そろっていても
    live 用の controller / helper / target adapter がまだ無いので拒否します。
    どの場合も artifact・ledger・save には一切触れません。
    """
    if formal_parents is None:
        raise CliRefusal("--live requires --formal-parents (06-02 prerequisite evidence); refusing to publish "
                         "a formal campaign manifest without formal parents")
    parents = json.loads(Path(formal_parents).read_bytes())
    prerequisites = parents.get("prerequisites") if isinstance(parents, dict) else None
    parent_hash = parents.get("prerequisite_parent_hash") if isinstance(parents, dict) else None
    if (not isinstance(prerequisites, dict) or prerequisites.get("development_only") is not False
            or not isinstance(parent_hash, str) or _DIGEST.fullmatch(parent_hash) is None):
        raise CliRefusal("--formal-parents must hold non-development prerequisites and a prerequisite_parent_hash")
    if not config["thresholds"] or config["threshold_parent_hash"] is None:
        raise CliRefusal("metric floors are unresolved (thresholds / threshold_parent_hash); --live refused")
    # ponytail: live adapters are not wired; add production controller/helper/target/operator ports with D06-LIVE-CAMPAIGN.
    raise CliRefusal("live controller/helper/target adapters are not wired in this build (D06-LIVE-CAMPAIGN)")


class DevelopmentHarness:
    """development dry-run 用の fake controller / helper / target / operator です。

    1つの object が4つの port を兼ね、operator checkpoint は全て承認、target は常に停止済み、
    helper release は即確認、controller run は confirmed success を即座に返します。
    `development_only=True` なので、formal plan の runner はこの harness を拒否します。
    """

    development_only = True

    def checkpoint(self, name: str, context: Mapping[str, Any]) -> bool:
        """operator checkpoint を無条件に承認します。

        arm / focus / manual start / cloud sync を人が確認する代わりです。
        """
        return True

    def stopped(self) -> bool:
        """game / launcher が停止済みだと答えます。

        save 差し替え前の停止確認に使われます。
        """
        return True

    def stop(self, spec: Mapping[str, Any]) -> bool:
        """target の停止を確認済みとして返します。

        harmless target helper 自体は broker close 時に Job ごと終了します。
        """
        return True

    def confirm_release(self, helper_pid: int | None, spec: Mapping[str, Any]) -> ReleaseObservation:
        """helper lease release を確認済みとして返します。

        observer_ref には development であることを残します。
        """
        return ReleaseObservation(True, f"development:{helper_pid}", 0.0)

    def start(self, spec: Any) -> _DevelopmentRun:
        """fake controller run を開始します。

        confirmed identity を受けた RunSpec をそのまま run に渡します。
        """
        return _DevelopmentRun(spec)


class _DevelopmentRun:
    """fake controller の run handle です。

    wait は stage 秒数を待たずに、決まった値の confirmed success を返します。
    """

    def __init__(self, spec: Any) -> None:
        """RunSpec を保持し、controller / helper PID に自 process を使います。"""
        self.spec = spec
        self.controller_pid = self.helper_pid = os.getpid()

    def wait(self, timeout_s: float) -> RunResult:
        """run 結果を即座に返します。

        値は固定の合成値で、実 gameplay の測定ではありません。
        """
        return RunResult(
            gameplay_attempt_id=self.spec.gameplay_attempt_id, gameplay_entries=1, target_success="confirmed",
            died=False, level=21, gems=340, kills=812, choices=20, unknown_frames=0, fallback_decisions=0,
            latency_ms={"p50": 9.5, "p95": 14.0}, menu_inputs_sent=0,
        )

    def terminate(self) -> None:
        """timeout 時の terminate です。

        即座に結果を返すので何もしません。
        """


def latest_stage_state(artifacts: ArtifactStore, campaign_id: str, stage: str) -> RunnerState | None:
    """stage の最新実行の確定 state を返します。

    実行が無いか途中(execution.json 未作成)なら None です。runner の artifact layout を読むだけで書きません。
    """
    number, state = 1, None
    while artifacts.exists(f"stages/{campaign_id}.{stage}.x{number}/manifest.json"):
        record = f"stages/{campaign_id}.{stage}.x{number}/execution.json"
        state = RunnerState(artifacts.read_json(record)["state"]) if artifacts.exists(record) else None
        number += 1
    return state


def drive(runner: CampaignRunner, artifacts: ArtifactStore, campaign_id: str,
          max_slots: int | None) -> tuple[dict[str, Any], int]:
    """C0 から順に stage を実行・再開し、campaign が確定したら finish します。

    pass 済み stage は飛ばし、STOPPED(--max-slots)と PREFLIGHT_BLOCKED では finish せずに返します
    (次回の起動で再開・新 execution id で再実行できます)。それ以外で stage が pass しなければ、
    その時点で campaign は確定なので finish で元 save を復元して summary を作ります。
    """
    states: dict[str, str] = {}
    for stage in STAGES:
        previous = latest_stage_state(artifacts, campaign_id, stage)
        if previous is RunnerState.STAGE_PASSED:
            states[stage] = previous.value
            continue
        if previous not in (None, RunnerState.PREFLIGHT_BLOCKED):
            states[stage] = previous.value
            break
        try:
            result = runner.run_stage(stage, max_slots=max_slots)
        except CampaignBlocked:
            states[stage] = RunnerState.CAMPAIGN_BLOCKED.value
            break
        states[stage] = result.state.value
        if result.state in (RunnerState.STOPPED, RunnerState.PREFLIGHT_BLOCKED):
            code = EXIT_OK if result.state is RunnerState.STOPPED else EXIT_NOT_PASSED
            return {"status": result.state.value, "stage": stage, "stages": states}, code
        if result.state is not RunnerState.STAGE_PASSED:
            break
    summary = runner.finish()
    passed = all(states.get(stage) == RunnerState.STAGE_PASSED.value for stage in STAGES)
    outcome = {
        "status": "finished", "stages": states, "all_stages_passed": passed,
        **{key: summary[key] for key in ("development_only", "formal_campaign_eligible", "formal_evidence_eligible")},
    }
    return outcome, EXIT_OK if passed else EXIT_NOT_PASSED


def run_campaign(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """`run` subcommand の本体です。

    `--live` は refuse_live で必ず拒否します。dry-run は synthetic plan を組み立て、
    実 DurableLaunchStore と in-process の実 LaunchBroker、DevelopmentHarness で runner を動かします。
    `--backup-root` があれば終了後に artifact bundle を backup store へ複製します。
    """
    config, config_hash = load_config(args.config)
    if args.live:
        refuse_live(args.formal_parents, config)
    if args.formal_parents is not None:
        raise CliRefusal("development dry-run cannot carry formal parents; formal manifests are published only by --live")
    canonical = Path(args.canonical_save).read_bytes()
    executable = sys.executable
    executable_hash = _sha256(Path(executable).read_bytes())
    thresholds = config["thresholds"]
    plan = CampaignPlan(
        campaign_id=args.campaign_id, mode="synthetic", executable_path=executable, executable_hash=executable_hash,
        build_hash=executable_hash, config_hash=config_hash, argv=(executable, "-c", HARMLESS_TARGET),
        canonical_save_hash=_sha256(canonical), thresholds=thresholds,
        threshold_parent_hash=config["threshold_parent_hash"] or canonical_hash({"development_thresholds": thresholds}),
        grace_seconds=config["grace_seconds"],
    )
    artifacts = ArtifactStore(args.artifacts)
    if artifacts.exists(SUMMARY):
        raise CliRefusal("campaign is already finalized; start a new campaign id from C0")
    harness = DevelopmentHarness()
    save = SaveLifecycle(args.save, artifacts, processes_stopped=harness.stopped)
    save.register_canonical(canonical)
    with DurableLaunchStore(args.ledger) as ledger:
        broker = LaunchBroker(ledger)
        try:
            runner = CampaignRunner(
                plan, artifacts=artifacts, ledger=ledger, broker=broker.handle_message, probe=probe_identity,
                controller=harness, helper=harness, target=harness, operator=harness, save=save,
            )
            runner.begin()
            outcome, code = drive(runner, artifacts, plan.campaign_id, args.max_slots)
        finally:
            broker.close()
    if args.backup_root is not None:
        outcome["backup_files"] = replicate_bundle(args.artifacts, args.backup_root)
    return outcome, code


def _atomic_write(path: Path, data: bytes) -> None:
    """一時 file へ書いて fsync してから置き換えます。

    途中で落ちても中途半端な file を残しません。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def replicate_bundle(primary: str | os.PathLike[str], backup: str | os.PathLike[str]) -> int:
    """primary artifact root を backup store へ複製し、file 数を返します。

    `bundle/` 配下へ全 file を写し、各 file の SHA-256 を `bundle_index.json` に記録します。
    artifact は write-once / append-only なので、毎回の上書き複製で最新の mirror になります。
    """
    primary, backup = Path(primary), Path(backup)
    index = {}
    for path in sorted(primary.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        name = path.relative_to(primary).as_posix()
        data = path.read_bytes()
        _atomic_write(backup / BUNDLE_DIR / name, data)
        index[name] = _sha256(data)
    _atomic_write(backup / BUNDLE_INDEX, canonical_json_bytes({"files": index}))
    return len(index)


def restore_bundle(backup: str | os.PathLike[str], root: str | os.PathLike[str]) -> int:
    """backup store の bundle を空の artifact root へ復元し、file 数を返します。

    root が空でない、bundle の file 集合が index と違う、hash が合わない、のいずれかなら
    root へ何も書かずに拒否します。検証済みの bytes だけを書きます。
    """
    backup, root = Path(backup), Path(root)
    if root.exists() and any(root.iterdir()):
        raise CliRefusal(f"restore target {root} must be empty")
    index = json.loads((backup / BUNDLE_INDEX).read_bytes())["files"]
    source = backup / BUNDLE_DIR
    present = {path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()}
    if present != set(index):
        raise CliRefusal("backup bundle files do not match bundle_index.json")
    verified = {}
    for name, digest in index.items():
        data = (source / name).read_bytes()
        if _sha256(data) != digest:
            raise CliRefusal(f"backup bundle file {name} fails hash verification")
        verified[name] = data
    for name, data in verified.items():
        _atomic_write(root / name, data)
    return len(verified)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """`run` と `restore` の2つの subcommand を解釈します。

    run は campaign の実行・再開、restore は backup bundle の空 root への復元です。
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="campaign を C0 から実行・再開する")
    run.add_argument("--campaign-id", required=True, help="code/model/config を変えたら新しい id にする")
    run.add_argument("--artifacts", required=True, help="primary artifact root")
    run.add_argument("--ledger", required=True, help="06-03 durable launch ledger directory")
    run.add_argument("--save", required=True, help="game が読む save file(元 save は backup 後に復元される)")
    run.add_argument("--canonical-save", required=True, help="各 slot の開始時に置く canonical save")
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--max-slots", type=int, default=None, help="現在の stage で N slot 処理したら停止する")
    run.add_argument("--backup-root", default=None, help="終了後に artifact bundle を複製する backup store")
    run.add_argument("--live", action="store_true", help="formal campaign(formal parents が必須)")
    run.add_argument("--formal-parents", default=None, help="06-02 prerequisite と prerequisite_parent_hash の JSON")
    restore = commands.add_parser("restore", help="backup bundle を空の artifact root へ復元する")
    restore.add_argument("--backup-root", required=True)
    restore.add_argument("--artifacts", required=True, help="復元先(空であること)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI の entry point です。結果を1行の JSON で stdout へ出し、終了コードを返します。

    0 は全 stage pass で finish 済み・停止(再開可能)・restore 成功、2 は fail-closed の拒否、
    3 は stage が pass せずに確定(または preflight blocked)したことを表します。
    """
    args = parse_args(argv)
    try:
        if args.command == "restore":
            outcome, code = {"status": "restored", "files": restore_bundle(args.backup_root, args.artifacts)}, EXIT_OK
        else:
            outcome, code = run_campaign(args)
    except (CliRefusal, RunnerError, SaveLifecycleError, LedgerError, ValueError, KeyError, OSError) as exc:
        outcome, code = {"status": "refused", "error": f"{type(exc).__name__}: {exc}"}, EXIT_REFUSED
    print(json.dumps(outcome, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())

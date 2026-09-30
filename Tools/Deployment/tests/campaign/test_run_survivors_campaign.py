"""run_survivors_campaign CLI の development dry-run・live fail-closed・bundle 復元を検証します。

dry-run は実 DurableLaunchStore と実 06-03 LaunchBroker(harmless target helper を起動)を使い、
C0〜C4 の virtual schedule を停止 → backup から空 root へ復元 → 再開 → finalize まで流します。
`--live` と formal manifest の発行は formal parents が無ければ何も書かずに拒否されることを確認します。
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import run_survivors_campaign as cli
from survivors.campaign.campaign_runner import STAGES, ArtifactStore
from survivors.campaign.campaign_schema import STAGE_POLICIES, CampaignEvent
from survivors.campaign.durable_launch_store import DurableLaunchStore, UnsupportedStorageError

ORIGINAL = b"original-save"
CANONICAL = b"canonical-save"


@pytest.fixture
def paths(tmp_path):
    """元 save・canonical save・ledger・backup store の path を用意します。

    元 save は finalize 時に復元されるべき内容です。
    """
    game = tmp_path / "game"
    game.mkdir()
    save = game / "SaveData.sav"
    save.write_bytes(ORIGINAL)
    canonical = tmp_path / "canonical.sav"
    canonical.write_bytes(CANONICAL)
    return SimpleNamespace(save=save, canonical=canonical, ledger=tmp_path / "ledger", backup=tmp_path / "backup")


def _run(capsys, paths, artifacts, *extra):
    """`run` subcommand を実行して終了コードと JSON 出力を返します。"""
    code = cli.main(["run", "--campaign-id", "canary-dev", "--artifacts", str(artifacts), "--ledger",
                     str(paths.ledger), "--save", str(paths.save), "--canonical-save", str(paths.canonical), *extra])
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def _restore(capsys, backup, artifacts):
    """`restore` subcommand を実行して終了コードと JSON 出力を返します。"""
    code = cli.main(["restore", "--backup-root", str(backup), "--artifacts", str(artifacts)])
    return code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def _files(root):
    """root 配下の file を相対 path → bytes で返します。"""
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_dry_run_virtual_schedule_stops_resumes_and_finalizes(tmp_path, capsys, paths):
    """C0〜C4 を停止・空 root への復元・再開・finalize の順に流します。

    stage は C0 から順に parent を持ち、全 run・manifest・summary が development-only で、
    finalize 後は元 save が戻り、同じ campaign id では再実行できません。
    """
    if os.name != "nt":
        with pytest.raises(UnsupportedStorageError):
            DurableLaunchStore(tmp_path / "probe-ledger")
        pytest.skip("formal_ineligible: platform_not_windows (verdict asserted)")
    primary = tmp_path / "primary"
    code, out = _run(capsys, paths, primary, "--max-slots", "1", "--backup-root", str(paths.backup))
    assert (code, out["status"], out["stage"]) == (0, "stopped", "C0")
    assert not ArtifactStore(primary).exists("campaign/summary.json")
    assert paths.save.read_bytes() == CANONICAL  # 元 save の復元は finalize 時だけです。

    restored = tmp_path / "restored"
    code, out = _restore(capsys, paths.backup, restored)
    assert code == 0 and _files(restored) == _files(primary) and out["files"] == len(_files(primary))

    code, out = _run(capsys, paths, restored, "--backup-root", str(paths.backup))
    assert code == 0 and out["status"] == "finished" and out["all_stages_passed"] is True
    assert out["stages"] == {stage: "stage_passed" for stage in STAGES}
    assert (out["development_only"], out["formal_campaign_eligible"], out["formal_evidence_eligible"]) == (
        True, False, False)
    store = ArtifactStore(restored)
    plan = store.read_json("campaign/plan.json")
    assert (plan["mode"], plan["development_only"], plan["formal_campaign_eligible"]) == ("synthetic", True, False)
    for index, stage in enumerate(STAGES):
        manifest = store.read_json(f"stages/canary-dev.{stage}.x1/manifest.json")
        assert manifest["parent_execution_id"] == (None if index == 0 else f"canary-dev.{STAGES[index - 1]}.x1")
        assert (manifest["development_only"], manifest["formal_campaign_eligible"]) == (True, False)
        for slot in range(STAGE_POLICIES[stage].slot_count):
            run = store.read_json(f"stages/canary-dev.{stage}.x1/runs/slot-{slot:02d}.json")
            assert (run["development_only"], run["formal_campaign_eligible"]) == (True, False)
    outcomes = [(record["stage_execution_id"], CampaignEvent.from_wire(record["event"]).slot_id)
                for record in store.stream("launched_outcomes")]
    assert len(outcomes) == len(set(outcomes)) == sum(policy.slot_count for policy in STAGE_POLICIES.values())
    assert [execution.split(".")[1] for execution, _ in outcomes] == sorted(
        (execution.split(".")[1] for execution, _ in outcomes), key=STAGES.index)
    summary = store.read_json("campaign/summary.json")
    assert summary["restore_verdict"]["status"] == "PASS" and summary["formal_evidence_eligible"] is False
    assert paths.save.read_bytes() == ORIGINAL

    code, out = _run(capsys, paths, restored)
    assert code == 2 and "already finalized" in out["error"]
    final = tmp_path / "final"
    assert _restore(capsys, paths.backup, final)[0] == 0 and _files(final) == _files(restored)


def test_live_and_formal_publish_fail_closed_without_formal_parents(tmp_path, capsys, paths):
    """`--live` は formal parents・metric floor・live adapter が欠ければ何も書かずに拒否します。

    dry-run に formal parents を渡して formal manifest を発行することもできません。
    """
    artifacts = tmp_path / "artifacts"
    parents = tmp_path / "parents.json"
    parents.write_text(json.dumps({"prerequisites": {"development_only": False}, "prerequisite_parent_hash": "f" * 64}))
    development_parents = tmp_path / "development-parents.json"
    development_parents.write_text(json.dumps({"prerequisites": {"development_only": True},
                                               "prerequisite_parent_hash": "f" * 64}))
    resolved = tmp_path / "resolved.yaml"
    resolved.write_text(cli.DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
        "thresholds: {}", "thresholds: {perception_recall: 0.9}").replace(
        "threshold_parent_hash: null", f"threshold_parent_hash: '{'a' * 64}'"), encoding="utf-8")
    cases = [
        (("--live",), "requires --formal-parents"),
        (("--live", "--formal-parents", str(development_parents)), "non-development prerequisites"),
        (("--live", "--formal-parents", str(parents)), "metric floors are unresolved"),
        (("--live", "--formal-parents", str(parents), "--config", str(resolved)), "D06-LIVE-CAMPAIGN"),
        (("--formal-parents", str(parents)), "dry-run cannot carry formal parents"),
    ]
    for extra, message in cases:
        code, out = _run(capsys, paths, artifacts, *extra)
        assert (code, out["status"]) == (2, "refused") and message in out["error"]
    assert not artifacts.exists() and not paths.ledger.exists() and paths.save.read_bytes() == ORIGINAL


def test_bundle_restore_rejects_tampered_or_non_empty_root(tmp_path):
    """restore は空でない root・index にない file・hash 不一致を root へ書く前に拒否します。

    正常な bundle は index どおりの内容で空 root に戻ります。
    """
    primary = tmp_path / "primary"
    store = ArtifactStore(primary)
    store.put_json("campaign/plan.json", {"mode": "synthetic"})
    store.append("launched_outcomes", [{"slot": 0}])
    backup = tmp_path / "backup"
    assert cli.replicate_bundle(primary, backup) == 2
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "stale.json").write_text("{}")
    with pytest.raises(cli.CliRefusal, match="must be empty"):
        cli.restore_bundle(backup, occupied)
    clean = tmp_path / "clean"
    assert cli.restore_bundle(backup, clean) == 2 and _files(clean) == _files(primary)

    extra = backup / "bundle" / "extra.json"
    extra.write_text("{}")
    empty = tmp_path / "empty"
    with pytest.raises(cli.CliRefusal, match="do not match"):
        cli.restore_bundle(backup, empty)
    extra.unlink()
    (backup / "bundle" / "campaign" / "plan.json").write_bytes(b'{"mode":"formal"}')
    with pytest.raises(cli.CliRefusal, match="hash verification"):
        cli.restore_bundle(backup, empty)
    assert not empty.exists()

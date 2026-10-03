"""collect_survivors_combat_distillation.py の synthetic 経路と正式経路の終了コードを検証する。

synthetic は DeployObs v2 の dataset を保存し、正式経路は前提 artifact や検証が欠けると
終了コード 3 で UE5 へ接続せずに止まることを、fake env / fake teacher で確かめます。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import collect_survivors_combat_distillation as cli
from games.survivors import distillation_collector as collector
from games.survivors.combat_distillation_dataset import CombatDistillationDataset
from reinbalance_survivors_contracts.artifact_identity import ArtifactDescriptor
from reinbalance_survivors_contracts.artifact_store import ArtifactStore, ArtifactStoreError
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from test_distillation_collector import FakeSurvivorsEnv, FakeTeacher, _formal_inputs, _verdict
from test_documented_formal_cli import _documented_argv


def test_synthetic_path_saves_v2_dataset(tmp_path):
    """--source-descriptor 無しは v2 schema hash の synthetic dataset を保存して 0 を返す。

    5 episode のうち train の 4 本だけが保存されます（既存の synthetic 規則）。
    """
    output = tmp_path / "dev"
    assert cli.main(["--output", str(output), "--episodes", "5", "--sequence-length", "8", "--burn-in", "2"]) == 0
    dataset = CombatDistillationDataset.load(output)
    assert dataset.deploy_schema_hash == DeployObsSchema.default_v2().schema_hash
    assert dataset.observations.shape == (4, 8, DeployObsSchema.default_v2().dim * 3)


def _formal_argv(tmp_path, descriptor, verdict_path, *, drop=(), output="out", episodes=5):
    """正式経路の argv を作る。

    drop に入れた flag は省き、前提 artifact の指定漏れを再現します。output 名と episode 数も変えられます。
    """
    values = {
        "--source-descriptor": descriptor, "--fidelity-verdict": verdict_path,
        "--artifact-store": tmp_path / "store", "--generated-input-descriptor": tmp_path / "generated-inputs.json",
        "--ubt-action-graph": tmp_path / "ubt-action-graph.json",
    }
    argv = [
        "--output", str(tmp_path / output), "--episodes", str(episodes), "--sequence-length", "4", "--burn-in", "2",
    ]
    for flag, value in values.items():
        if flag not in drop:
            argv += [flag, str(value)]
    return argv


@pytest.fixture
def fake_formal(monkeypatch):
    """current hash 解決・UE5 env・教師ロードを fake に差し替え、作られた env を記録する。

    戻り値は (current hash に合う verdict, 作られた env の一覧) です。
    """
    verdict, hashes = _verdict()
    envs = []
    monkeypatch.setattr(
        "reinbalance_survivors_contracts.current_fidelity.resolve_current_gating_producer_hashes",
        lambda *a: SimpleNamespace(current_gating_producer_hashes=hashes),
    )
    monkeypatch.setattr(cli, "_make_env", lambda args: envs.append(FakeSurvivorsEnv()) or envs[-1])
    monkeypatch.setattr(collector, "load_teacher", lambda path: FakeTeacher())
    return verdict, envs


def test_formal_path_collects_and_saves_with_all_prerequisites(tmp_path, fake_formal):
    """前提 artifact が揃えば正式収集して 0 を返し、train/validation と descriptor を保存する。

    収集後に env が閉じられることも確かめます。
    """
    verdict, envs = fake_formal
    descriptor, verdict_path = _formal_inputs(tmp_path, verdict)
    assert cli.main(_formal_argv(tmp_path, descriptor, verdict_path)) == 0
    assert len(envs) == 1 and envs[0].closed
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"train", "validation", "artifact_descriptor.json"}


def test_formal_path_second_collection_in_same_store_succeeds(tmp_path, fake_formal):
    """同じ store・教師・verdict・seed で episodes を変えた 2 回目の CLI 収集も 0 で、別 dataset として保存される。

    logical id の衝突で収集後に終了コード 3 になる不具合の回帰テストです。
    """
    verdict, envs = fake_formal
    descriptor, verdict_path = _formal_inputs(tmp_path, verdict)
    assert cli.main(_formal_argv(tmp_path, descriptor, verdict_path, output="out1", episodes=5)) == 0
    assert cli.main(_formal_argv(tmp_path, descriptor, verdict_path, output="out2", episodes=10)) == 0
    ids = {
        ArtifactDescriptor.from_wire(json.loads((tmp_path / name / "artifact_descriptor.json").read_bytes())).logical_id
        for name in ("out1", "out2")
    }
    assert len(ids) == 2 and len(envs) == 2


def test_formal_path_store_failure_exits_3_without_partial_output(tmp_path, fake_formal, monkeypatch):
    """収集後の store 登録が失敗すると終了コード 3 で、--output には何も残らない。

    train/data.npz の put だけを失敗させ、一時 directory も片付けられることを確かめます。
    """
    verdict, _ = fake_formal
    descriptor, verdict_path = _formal_inputs(tmp_path, verdict)
    original = ArtifactStore.put_bytes

    def failing_put(self, *, logical_id, data, media_type):
        """train/data.npz の登録だけ失敗させる。

        それ以外の put は本物の store に通します。
        """
        if logical_id.endswith("/train/data.npz"):
            raise ArtifactStoreError("injected store failure")
        return original(self, logical_id=logical_id, data=data, media_type=media_type)

    monkeypatch.setattr(ArtifactStore, "put_bytes", failing_put)
    assert cli.main(_formal_argv(tmp_path, descriptor, verdict_path)) == 3
    assert not (tmp_path / "out").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".out.")]


def test_formal_path_unusable_store_exits_3_without_connecting(tmp_path, fake_formal):
    """--artifact-store を開けない場合は UE5 に接続する前に終了コード 3 で止まる。

    store の場所にファイルを置き、収集を始める前に検出されることを確かめます。
    """
    verdict, envs = fake_formal
    descriptor, verdict_path = _formal_inputs(tmp_path, verdict)
    (tmp_path / "store").write_bytes(b"not a directory")
    assert cli.main(_formal_argv(tmp_path, descriptor, verdict_path)) == 3
    assert envs == [] and not (tmp_path / "out").exists()


@pytest.mark.parametrize("flag", ["--fidelity-verdict", "--artifact-store", "--generated-input-descriptor", "--ubt-action-graph"])
def test_formal_path_without_prerequisite_exits_3_without_connecting(tmp_path, fake_formal, flag, capsys):
    """前提 artifact の指定が 1 つでも欠ければ終了コード 3 で、UE5 に接続せず何も保存しない。

    stderr には欠けた flag 名が出ます。
    """
    verdict, envs = fake_formal
    descriptor, verdict_path = _formal_inputs(tmp_path, verdict)
    assert cli.main(_formal_argv(tmp_path, descriptor, verdict_path, drop=(flag,))) == 3
    assert envs == [] and not (tmp_path / "out").exists()
    assert flag in capsys.readouterr().err


@pytest.mark.parametrize("case", ["missing_descriptor", "invalid_descriptor", "blocked_verdict"])
def test_formal_path_with_invalid_prerequisite_exits_3(tmp_path, fake_formal, monkeypatch, case):
    """descriptor 不在・descriptor 検証失敗・blocking 付き verdict はどれも終了コード 3 で収集しない。

    例外は CLI が捕まえて終了コードに変え、UE5 への接続も dataset の保存も起きません。
    """
    verdict, envs = fake_formal
    if case == "blocked_verdict":
        verdict, _ = _verdict(blocked=True)
    descriptor, verdict_path = _formal_inputs(tmp_path, verdict)
    if case == "missing_descriptor":
        descriptor.unlink()
    if case == "invalid_descriptor":
        monkeypatch.setattr(collector, "load_teacher", lambda path: (_ for _ in ()).throw(ValueError("bad descriptor")))
    assert cli.main(_formal_argv(tmp_path, descriptor, verdict_path)) == 3
    assert envs == [] and not (tmp_path / "out").exists()


def test_documented_formal_collection_command_parses():
    """docs/train/overview.md の正式収集コマンドが現在の CLI 引数で解釈できる。

    引数名を変えたら運用手順の docs も直す必要があることを、このテストで気づけるようにします。
    """
    args = cli._parser().parse_args(_documented_argv("collect_survivors_combat_distillation.py"))
    assert args.source_descriptor is not None and args.fidelity_verdict is not None
    assert args.generated_input_descriptor.name == "generated-inputs.json"

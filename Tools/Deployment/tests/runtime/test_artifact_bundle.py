"""artifact bundle: loader の hash / DAG / restore / action / recurrent 境界を検証する。

golden fixture では formal artifact なしで全 loader テストを実行でき、
development_only bundle が正式起動を拒否することを確認する。formal 経路では
immutable parent、exact subject hash、artifact store restore、action semantics、
SB3 RecurrentPPO と deploy VecNormalize の全 gate を 1 つずつ壊して拒否を確認する。
"""
from __future__ import annotations

import inspect
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.target_action import ActionSemantics
from reinbalance_survivors_contracts.ui_policy import NonModelUiPolicyConfigV1

from survivors.runtime import artifact_bundle as artifact_bundle_module
from survivors.runtime.artifact_bundle import (
    BUNDLE_DEVELOPMENT_SENTINEL,
    BundleLoadError,
    RecurrentCombatPolicy,
    RuntimeBundle,
)

from . import _runtime_fixtures as fx


def _stage(
    formal,
    tmp_path: Path,
    *,
    mutate_combat: Callable[[dict[str, Any]], None] | None = None,
    mutate_selector_dir: Callable[[Path], None] | None = None,
    mutate_combat_dir: Callable[[Path], None] | None = None,
    descriptor_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """formal package を tmp_path へ複製し、指定箇所だけを壊して load 引数を返す。

    descriptor は複製後の manifest から作り直すため、狙った gate だけが発火する。
    """
    combat_dir = tmp_path / "combat"
    selector_dir = tmp_path / "selector"
    shutil.copytree(formal.combat_dir, combat_dir)
    shutil.copytree(formal.selector_dir, selector_dir)

    combat_manifest = dict(formal.combat_manifest)
    if mutate_combat is not None:
        combat_manifest = {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in combat_manifest.items()
        }
        mutate_combat(combat_manifest)
        fx.rewrite_combat_manifest(combat_dir, combat_manifest)
    if mutate_combat_dir is not None:
        mutate_combat_dir(combat_dir)
    if mutate_selector_dir is not None:
        mutate_selector_dir(selector_dir)

    store = ArtifactStore(tmp_path / "store")
    descriptors = fx.build_formal_descriptors(
        store=store,
        combat_manifest=combat_manifest,
        item_selector_manifest=formal.selector_manifest,
        target_capability_hash=formal.target_capability_hash,
        choice_capability_hash=formal.choice_capability_hash,
        **(descriptor_kwargs or {}),
    )
    return {
        "combat_package_dir": combat_dir,
        "item_selector_dir": selector_dir,
        "artifact_store": store,
        "descriptors": descriptors,
        "target_profile": formal.target_profile,
    }


def _load(kwargs: dict[str, Any]) -> RuntimeBundle:
    """_stage() が返した引数で RuntimeBundle.load を呼ぶ。"""
    return RuntimeBundle.load(
        kwargs["combat_package_dir"],
        kwargs["item_selector_dir"],
        artifact_store=kwargs["artifact_store"],
        descriptors=kwargs["descriptors"],
        target_profile=kwargs["target_profile"],
    )


class TestNoTrainingImport:
    """[指摘1] Deployment loader が Training package へ依存しないことを確認する。"""

    def test_module_does_not_reference_training_proxy(self):
        source = inspect.getsource(artifact_bundle_module)
        assert "training_proxy" not in source
        assert "games.survivors" not in source

    def test_module_does_not_mutate_sys_path(self):
        """sys.path へ Training を挿し込む fallback が残っていないこと。"""
        source = inspect.getsource(artifact_bundle_module)
        assert "sys.path" not in source

    def test_item_selector_loads_through_onnx_adapter(self, formal_packages, tmp_path):
        from survivors.runtime.item_selector_runtime import OnnxItemSelector

        bundle = _load(_stage(formal_packages, tmp_path))
        assert isinstance(bundle.item_selector, OnnxItemSelector)


class TestGoldenFixtureBundle:
    """golden fixture は development_only であり正式起動できない。"""

    def test_development_only_is_true(self, golden_combat_policy):
        bundle = RuntimeBundle.from_golden_fixture(golden_combat_policy)
        assert bundle.development_only is True

    def test_live_eligible_is_false(self, golden_combat_policy):
        bundle = RuntimeBundle.from_golden_fixture(golden_combat_policy)
        assert bundle.live_eligible is False

    def test_startup_report_kind_is_sentinel(self, golden_combat_policy):
        bundle = RuntimeBundle.from_golden_fixture(golden_combat_policy)
        assert bundle.startup_report["bundle_kind"] == BUNDLE_DEVELOPMENT_SENTINEL

    def test_assert_live_eligible_raises(self, golden_combat_policy):
        bundle = RuntimeBundle.from_golden_fixture(golden_combat_policy)
        with pytest.raises(BundleLoadError, match="development_only"):
            bundle.assert_live_eligible()

    def test_non_policy_object_rejected(self):
        with pytest.raises(BundleLoadError, match="RecurrentCombatPolicy"):
            RuntimeBundle.from_golden_fixture(object())  # type: ignore[arg-type]

    def test_deploy_schema_is_populated(self, golden_combat_policy):
        bundle = RuntimeBundle.from_golden_fixture(golden_combat_policy)
        assert isinstance(bundle.deploy_schema, DeployObsSchema)
        assert bundle.deploy_schema_hash == bundle.deploy_schema.schema_hash

    def test_ui_policy_config_defaults_to_installed(self, golden_combat_policy):
        bundle = RuntimeBundle.from_golden_fixture(golden_combat_policy)
        assert isinstance(bundle.ui_policy_config, NonModelUiPolicyConfigV1)


class TestFormalBundleHappyPath:
    """全 gate を通過した bundle だけが live eligible になる。"""

    def test_formal_bundle_is_live_eligible(self, formal_packages, tmp_path):
        bundle = _load(_stage(formal_packages, tmp_path))
        assert bundle.live_eligible is True
        assert bundle.development_only is False
        bundle.assert_live_eligible()

    def test_startup_report_lists_device_and_profile(self, formal_packages, tmp_path):
        """[指摘2] startup report に device / profile / schema / dependency を出す。"""
        report = _load(_stage(formal_packages, tmp_path)).startup_report
        for key in (
            "os_build", "gpu_name", "driver_version", "cuda_version", "capture_backend",
            "decision_hz", "key_lease_duration_ms", "choice_capability_hash",
            "action_semantics_hash", "action_semantics_actions",
            "deploy_schema_version", "combat_deploy_schema_hash",
            "combat_formal_dependency_identities", "perception_subject_hashes",
            "artifact_store_verified_objects", "target_profile_ref_hash",
        ):
            assert key in report, key
        assert report["decision_hz"] == 15
        assert len(report["perception_subject_hashes"]) == 16

    def test_combat_policy_is_recurrent_with_lstm_state(self, formal_packages, tmp_path):
        """[指摘9] SB3 RecurrentPPO を直接 load し `[n_layers,1,hidden]` state を持つ。"""
        bundle = _load(_stage(formal_packages, tmp_path))
        assert isinstance(bundle.combat_policy, RecurrentCombatPolicy)
        assert bundle.combat_policy.lstm_state_shape == (
            bundle.combat_policy.n_lstm_layers, 1, bundle.combat_policy.lstm_hidden_size
        )
        assert bundle.combat_policy.model.policy.lstm_actor is not None

    def test_deploy_vecnormalize_is_inference_only(self, formal_packages, tmp_path):
        """[指摘9] deploy VecNormalize は training=False / norm_reward=False。"""
        bundle = _load(_stage(formal_packages, tmp_path))
        assert bundle.combat_policy.vecnormalize.training is False
        assert bundle.combat_policy.vecnormalize.norm_reward is False


class TestTrustedLoadingBoundary:
    """[指摘3] 検証済み artifact を先に確定してからでないと model を読まない。"""

    def test_tampered_policy_zip_rejected(self, formal_packages, tmp_path):
        def tamper(root: Path) -> None:
            (root / "policy.zip").write_bytes(b"not-a-real-policy-archive")

        with pytest.raises(BundleLoadError, match="hash mismatch"):
            _load(_stage(formal_packages, tmp_path, mutate_combat_dir=tamper))

    def test_tampered_vecnormalize_rejected(self, formal_packages, tmp_path):
        def tamper(root: Path) -> None:
            (root / "vecnormalize.pkl").write_bytes(b"\x80\x04arbitrary-pickle")

        with pytest.raises(BundleLoadError, match="hash mismatch"):
            _load(_stage(formal_packages, tmp_path, mutate_combat_dir=tamper))

    def test_hash_check_precedes_deserialization(self):
        """hash 照合を通らない file が torch / pickle へ渡らない実装であること。"""
        source = inspect.getsource(artifact_bundle_module._load_combat_package)
        verify_pos = source.index("_verified_bytes")
        load_pos = source.index("_load_recurrent_policy")
        assert verify_pos < load_pos

    def test_extra_file_in_package_rejected(self, formal_packages, tmp_path):
        def add_file(root: Path) -> None:
            (root / "unlisted.bin").write_bytes(b"x")

        with pytest.raises(BundleLoadError, match="directory contents mismatch"):
            _load(_stage(formal_packages, tmp_path, mutate_combat_dir=add_file))

    def test_missing_vecnormalize_rejected(self, formal_packages, tmp_path):
        def remove(root: Path) -> None:
            (root / "vecnormalize.pkl").unlink()

        with pytest.raises(BundleLoadError, match="directory contents mismatch|is missing"):
            _load(_stage(formal_packages, tmp_path, mutate_combat_dir=remove))

    def test_package_file_name_with_traversal_rejected(self, formal_packages, tmp_path):
        root = tmp_path / "combat"
        shutil.copytree(formal_packages.combat_dir, root)
        with pytest.raises(BundleLoadError, match="plain file name"):
            artifact_bundle_module._resolve_package_file(root, "../policy.zip")

    def test_development_only_package_rejected(self, formal_packages, tmp_path):
        with pytest.raises(BundleLoadError, match="development_only"):
            _load(
                _stage(
                    formal_packages, tmp_path,
                    mutate_combat=lambda m: m.__setitem__("development_only", True),
                )
            )

    def test_not_student_eligible_rejected(self, formal_packages, tmp_path):
        with pytest.raises(BundleLoadError, match="formal_student_eligible"):
            _load(
                _stage(
                    formal_packages, tmp_path,
                    mutate_combat=lambda m: m.__setitem__("formal_student_eligible", False),
                )
            )


class TestActionSemanticsBinding:
    """[指摘8] action_dim と ActionSemantics の順序・hash を bundle で検証する。"""

    def test_action_dim_two_is_rejected(self, formal_packages, tmp_path):
        def shrink(manifest: dict[str, Any]) -> None:
            manifest["model_config"]["action_dim"] = 2

        with pytest.raises(BundleLoadError, match="action_dim must be 9"):
            _load(_stage(formal_packages, tmp_path, mutate_combat=shrink))

    def test_real_two_action_policy_is_rejected(self, tmp_path):
        """実際に 2-action の RecurrentPPO package を作っても move を返さない。"""
        policy = fx.build_golden_combat_policy(action_dim=2)
        profile = fx.default_target_profile()
        capability = fx._hash_of("target-capability")
        choice = fx._hash_of("choice-capability")
        manifest = fx.write_combat_package(
            tmp_path / "combat2", policy, target_profile=profile,
            target_capability_hash=capability, choice_capability_hash=choice,
        )
        selector_manifest = fx.write_item_selector_package(
            tmp_path / "selector2", target_capability_hash=capability
        )
        store = ArtifactStore(tmp_path / "store2")
        descriptors = fx.build_formal_descriptors(
            store=store, combat_manifest=manifest,
            item_selector_manifest=selector_manifest,
            target_capability_hash=capability, choice_capability_hash=choice,
        )
        with pytest.raises(BundleLoadError, match="action_dim must be 9"):
            RuntimeBundle.load(
                tmp_path / "combat2", tmp_path / "selector2",
                artifact_store=store, descriptors=descriptors, target_profile=profile,
            )

    def test_action_semantics_hash_mismatch_rejected(self, formal_packages, tmp_path):
        with pytest.raises(BundleLoadError, match="action_semantics_hash mismatch"):
            _load(
                _stage(
                    formal_packages, tmp_path,
                    mutate_combat=lambda m: m.__setitem__("action_semantics_hash", "0" * 64),
                )
            )

    def test_default_action_semantics_has_nine_actions(self):
        assert ActionSemantics.default_v1().num_actions == 9


class TestFormalDependencyAndDagGates:
    """[指摘2] formal dependency / immutable parent / exact subject hash を強制する。"""

    def test_missing_formal_dependency_identities_rejected(self, formal_packages, tmp_path):
        def drop(manifest: dict[str, Any]) -> None:
            manifest["formal_dependency_identities"] = {}

        with pytest.raises(BundleLoadError, match="formal dependency identities"):
            _load(_stage(formal_packages, tmp_path, mutate_combat=drop))

    def test_failed_perception_verdict_rejected(self, formal_packages, tmp_path):
        with pytest.raises(BundleLoadError, match="formal runtime DAG rejected"):
            _load(
                _stage(
                    formal_packages, tmp_path,
                    descriptor_kwargs={"verdict_passed": False},
                )
            )

    def test_development_only_verdict_rejected(self, formal_packages, tmp_path):
        with pytest.raises(BundleLoadError, match="formal runtime DAG rejected"):
            _load(
                _stage(
                    formal_packages, tmp_path,
                    descriptor_kwargs={"verdict_development_only": True},
                )
            )

    def test_runtime_subject_hash_mismatch_rejected(self, formal_packages, tmp_path):
        """04-10 exact subject hash が 1 つでも違えば起動しない。"""
        subjects = fx.perception_subject_hashes()
        subjects["parser_artifact_hash"] = "1" * 64
        with pytest.raises(BundleLoadError, match="formal runtime DAG rejected"):
            _load(
                _stage(
                    formal_packages, tmp_path,
                    descriptor_kwargs={"runtime_subject_hashes": subjects},
                )
            )

    def test_combat_manifest_hash_must_match_descriptor(self, formal_packages, tmp_path):
        """package を差し替えても descriptor 側 identity と一致しなければ拒否。"""
        kwargs = _stage(formal_packages, tmp_path)
        manifest = dict(formal_packages.combat_manifest)
        manifest["runtime_profile"] = dict(manifest["runtime_profile"])
        manifest["runtime_profile"]["gpu_name"] = "swapped-gpu"
        fx.rewrite_combat_manifest(kwargs["combat_package_dir"], manifest)
        with pytest.raises(BundleLoadError, match="combat package manifest hash"):
            _load(kwargs)

    def test_corrupt_store_object_rejected(self, formal_packages, tmp_path):
        """artifact store restore 検証に失敗した bundle は live 起動しない。"""
        kwargs = _stage(formal_packages, tmp_path)
        store: ArtifactStore = kwargs["artifact_store"]
        target = next(iter(store.objects_root.rglob("*")))
        while target.is_dir():
            target = next(iter(target.iterdir()))
        target.write_bytes(b"corrupted")
        with pytest.raises(BundleLoadError, match="restore check failed|verification failed"):
            _load(kwargs)

    def test_target_profile_mismatch_rejected(self, formal_packages, tmp_path):
        from reinbalance_survivors_contracts.target_action import TargetProfileRef

        kwargs = _stage(formal_packages, tmp_path)
        kwargs["target_profile"] = TargetProfileRef(
            build_id="vs-9.9.9",
            canonical_save_hash=fx._hash_of("other-save"),
            hardware_profile_id="other-hw",
        )
        with pytest.raises(BundleLoadError, match="target hardware profile mismatch"):
            _load(kwargs)

    def test_capability_hash_mismatch_rejected(self, formal_packages, tmp_path):
        with pytest.raises(BundleLoadError, match="target_capability_hash mismatch"):
            _load(
                _stage(
                    formal_packages, tmp_path,
                    mutate_combat=lambda m: m.__setitem__("target_capability_hash", "2" * 64),
                )
            )

    def test_decision_hz_must_be_fifteen(self, formal_packages, tmp_path):
        def retune(manifest: dict[str, Any]) -> None:
            manifest["runtime_profile"]["decision_hz"] = 30

        with pytest.raises(BundleLoadError, match="decision_hz must be 15"):
            _load(_stage(formal_packages, tmp_path, mutate_combat=retune))


class TestCorruptItemSelectorPackage:
    """[指摘1/6] corrupt ONNX / 差し替え package を load 時に拒否する。"""

    def test_corrupt_onnx_rejected(self, formal_packages, tmp_path):
        def corrupt(root: Path) -> None:
            (root / "model.onnx").write_bytes(b"not-an-onnx-graph")

        with pytest.raises(BundleLoadError, match="ItemSelector package rejected"):
            _load(_stage(formal_packages, tmp_path, mutate_selector_dir=corrupt))

    def test_unlisted_selector_file_rejected(self, formal_packages, tmp_path):
        def add(root: Path) -> None:
            (root / "extra.json") .write_bytes(b"{}")

        with pytest.raises(BundleLoadError, match="ItemSelector package rejected"):
            _load(_stage(formal_packages, tmp_path, mutate_selector_dir=add))

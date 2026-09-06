"""runtime test 用の共有 pytest fixture。

SB3 RecurrentPPO の構築と ONNX export は 1 件あたり数秒かかるため、session scope
で 1 度だけ組み立てて全テストで使い回す。formal 一式 (combat package /
ItemSelector package / artifact store / descriptor) も同様に共有する。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from reinbalance_survivors_contracts.artifact_store import ArtifactStore
from reinbalance_survivors_contracts.target_action import TargetProfileRef

from . import _runtime_fixtures as fx


@dataclass(frozen=True)
class FormalPackages:
    """formal bundle テストに必要な成果物一式。

    package directory、artifact store、descriptor、identity hash をまとめて渡す。
    """

    combat_dir: Path
    selector_dir: Path
    store: ArtifactStore
    descriptors: list[Any]
    combat_manifest: dict[str, Any]
    selector_manifest: dict[str, Any]
    target_profile: TargetProfileRef
    target_capability_hash: str
    choice_capability_hash: str


@pytest.fixture(scope="session")
def golden_combat_policy():
    """golden fixture 用の RecurrentPPO + VecNormalize を 1 度だけ構築する。"""
    return fx.build_golden_combat_policy()


@pytest.fixture(scope="session")
def formal_packages(tmp_path_factory, golden_combat_policy) -> FormalPackages:
    """検証を通過する formal package 一式を組み立てる。

    異常系テストは、この一式を tmp_path へコピーしてから壊して使う。
    """
    root = tmp_path_factory.mktemp("formal")
    capability = fx._hash_of("target-capability")
    choice_capability = fx._hash_of("choice-capability")
    profile = fx.default_target_profile()

    selector_manifest = fx.write_item_selector_package(
        root / "selector", target_capability_hash=capability
    )
    combat_manifest = fx.write_combat_package(
        root / "combat",
        golden_combat_policy,
        target_profile=profile,
        target_capability_hash=capability,
        choice_capability_hash=choice_capability,
    )
    store = ArtifactStore(root / "store")
    descriptors = fx.build_formal_descriptors(
        store=store,
        combat_manifest=combat_manifest,
        item_selector_manifest=selector_manifest,
        target_capability_hash=capability,
        choice_capability_hash=choice_capability,
    )
    return FormalPackages(
        combat_dir=root / "combat",
        selector_dir=root / "selector",
        store=store,
        descriptors=descriptors,
        combat_manifest=combat_manifest,
        selector_manifest=selector_manifest,
        target_profile=profile,
        target_capability_hash=capability,
        choice_capability_hash=choice_capability,
    )

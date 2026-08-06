"""NonModelUiPolicyV1 がartifactへinstalled distribution bindingされることを検証する。"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from reinbalance_survivors_contracts.canonical_json import sha256_hex
from reinbalance_survivors_contracts import ui_policy
from reinbalance_survivors_contracts.ui_policy import NonModelUiPolicyConfigV1

from games.survivors.item_selector_artifact import ArtifactBindingError, ItemSelectorArtifact
from test_item_selector_artifact import _mutate_manifest, export_fixture


def test_policy_bundled_in_package(tmp_path: Path) -> None:
    package = export_fixture(tmp_path)
    assert (package / "ui_policy_config.json").is_file()


def test_policy_config_hash_matches_installed(tmp_path: Path) -> None:
    package = export_fixture(tmp_path)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy_config_hash"] == NonModelUiPolicyConfigV1.load_default().config_hash


def test_policy_impl_hash_stable(tmp_path: Path) -> None:
    package = export_fixture(tmp_path)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    source = Path(inspect.getsourcefile(ui_policy) or "")
    assert manifest["policy_impl_hash"] == sha256_hex(source.read_bytes())


def test_same_hash_loads_correctly_then_tamper_rejects(tmp_path: Path) -> None:
    package = export_fixture(tmp_path)
    ItemSelectorArtifact.load(package)
    _mutate_manifest(package, lambda value: value.__setitem__("policy_impl_hash", "0" * 64))
    with pytest.raises(ArtifactBindingError, match="policy implementation"):
        ItemSelectorArtifact.load(package)

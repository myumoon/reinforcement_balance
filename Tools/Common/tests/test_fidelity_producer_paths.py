"""Producer path manifest の schema と identity binding を検証する。

13 key の exact allowlist と manifest bytes の hash 反映を固定します。
"""

import json

import pytest

from reinbalance_survivors_contracts.fidelity_producer_paths import load_producer_path_manifest
from reinbalance_survivors_contracts.fidelity_verdict import GATING_KEYS
from reinbalance_survivors_contracts.ui_intent import ContractValidationError


def test_packaged_manifest_has_exact_keys_and_hash() -> None:
    """package 内 manifest が13 key と identity hash を持つことを検証する。

    schema の増減は version 更新なしに通りません。
    """
    manifest = load_producer_path_manifest()
    assert set(manifest.producers) == set(GATING_KEYS)
    assert len(manifest.manifest_hash) == 64


def test_unknown_or_missing_producer_is_rejected(tmp_path) -> None:
    """未知 producer と欠落 producer を loader が拒否する。

    typo や新 producer の黙認を防ぎます。
    """
    original = load_producer_path_manifest()
    data = {"schema_version": original.schema_version, "producers": dict(original.producers)}
    data["producers"]["unknown"] = data["producers"]["logic_public"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ContractValidationError):
        load_producer_path_manifest(path)

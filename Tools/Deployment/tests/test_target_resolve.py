"""target_resolve.py の純粋関数と自己検証(fail-closed)の回帰テスト。

証跡JSONから実測値を組み立てて resolved profile を作る一連の変換が、既知の3バグ
(ハッシュ取り違え・build_idの型崩れ・executable_versionの捏造値)を再発させないことを固定する。
"""

from __future__ import annotations

import pytest
import yaml

from survivors.target_audit import AuditError
from survivors.target_profile import load_target_profile
from survivors.target_resolve import (
    Measurements,
    build_measurements,
    build_resolved_wire,
    find_steam_appmanifest,
    latest_evidence_manifest,
    manual_evidence_bytes,
    manual_evidence_doc,
    parse_env_file,
    parse_steam_build_id,
    validate_resolved_wire,
    _read_json_bom,
)

_MANIFEST = {
    "executable": {"basename": "vampire_survivors_20260919.exe", "canonical_hash": "5a4e7c2a" + "0" * 56},
    "canonical_save": {"basename": "SaveData", "canonical_hash": "7dcae1b9" + "0" * 56},
}
_MACHINE_INFO = {
    "gpu": [{"Name": "reference-gpu", "VRAM_GiB": 12, "DriverVersion": "32.0.15.6094", "DriverDate": "2026-01-01", "Source": "wmi"}],
    "os": {"Caption": "Windows 11 Pro", "Version": "10.0.26200", "BuildNumber": "26200"},
}
_EXE_METADATA = {
    "executable": {
        "path": "C:/Steam/steamapps/common/Vampire Survivors/VampireSurvivors.exe",
        "file_version": "6000.0.62.16359173",
        "product_version": "6000.0.62f1 (f99f05b3e950)",
        "sha256": "7ecc56cd" + "0" * 56,
    }
}
_KWARGS = dict(build_id=25016043, save_format_version=11105, pytorch_version="2.11.0", cuda_version="12.4", operator="neko", date_str="2026-09-20")


def test_parse_env_file_handles_quotes_and_comments():
    text = '# comment\nFOO="bar"\nBAZ=\'qux\'\n\nBARE=plain\n'
    assert parse_env_file(text) == {"FOO": "bar", "BAZ": "qux", "BARE": "plain"}


def test_parse_steam_build_id_extracts_tab_separated_value():
    acf = '"AppState"\n{\n\t"appid"\t\t"1794680"\n\t"buildid"\t\t"25016043"\n}\n'
    assert parse_steam_build_id(acf) == "25016043"


def test_parse_steam_build_id_raises_when_absent():
    with pytest.raises(ValueError):
        parse_steam_build_id('"AppState"\n{\n\t"appid"\t\t"1794680"\n}\n')


def test_find_steam_appmanifest_walks_up_to_steamapps(tmp_path):
    exe = tmp_path / "steamapps" / "common" / "Vampire Survivors" / "VampireSurvivors.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    manifest = tmp_path / "steamapps" / "appmanifest_1794680.acf"
    manifest.write_text('"buildid"\t\t"1"\n', encoding="utf-8")
    assert find_steam_appmanifest(exe) == manifest


def test_find_steam_appmanifest_raises_when_not_found(tmp_path):
    exe = tmp_path / "a" / "b" / "VampireSurvivors.exe"
    exe.parent.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        find_steam_appmanifest(exe, max_levels=2)


def test_build_measurements_does_not_swap_executable_and_save_hash():
    # 既知バグ回帰: save_artifact_hash が executable_hash と取り違えられていた。
    m = build_measurements(_MANIFEST, _MACHINE_INFO, _EXE_METADATA, **_KWARGS)
    assert m.executable_hash == _MANIFEST["executable"]["canonical_hash"]
    assert m.save_artifact_hash == _MANIFEST["canonical_save"]["canonical_hash"]
    assert m.executable_hash != m.save_artifact_hash


def test_build_measurements_build_id_is_always_str():
    m = build_measurements(_MANIFEST, _MACHINE_INFO, _EXE_METADATA, **_KWARGS)
    assert m.build_id == "25016043" and isinstance(m.build_id, str)


def test_build_measurements_prefers_file_version_over_product_version():
    # 既知バグ回帰: 過去の実装は file_version にも product_version にも一致しない捏造値だった。
    m = build_measurements(_MANIFEST, _MACHINE_INFO, _EXE_METADATA, **_KWARGS)
    assert m.executable_version == "6000.0.62.16359173"


def test_build_measurements_falls_back_to_product_version_when_file_version_missing():
    metadata = {"executable": {k: v for k, v in _EXE_METADATA["executable"].items() if k != "file_version"}}
    m = build_measurements(_MANIFEST, _MACHINE_INFO, metadata, **_KWARGS)
    assert m.executable_version == "6000.0.62f1 (f99f05b3e950)"


def test_build_measurements_vram_mb_is_gib_times_1024():
    m = build_measurements(_MANIFEST, _MACHINE_INFO, _EXE_METADATA, **_KWARGS)
    assert m.vram_mb == 12288 and isinstance(m.vram_mb, int)


def _measurements():
    return build_measurements(_MANIFEST, _MACHINE_INFO, _EXE_METADATA, **_KWARGS)


def test_manual_evidence_doc_matches_audit_claimed_shape():
    doc = manual_evidence_doc(_measurements())
    assert set(doc) == {"schema_version", "measurements"}
    assert doc["schema_version"] == "survivors_manual_attestation.v1"
    assert set(doc["measurements"]) == {
        "build_id", "executable_version", "save_format_version",
        "os_build", "gpu_name", "vram_mb", "driver_version", "cuda_version", "pytorch_version",
    }


def test_manual_evidence_bytes_has_no_crlf():
    body = manual_evidence_bytes(_measurements())
    assert b"\r\n" not in body
    assert yaml.safe_load(body.decode("utf-8")) == manual_evidence_doc(_measurements())


def test_build_resolved_wire_passes_full_self_validation():
    m = _measurements()
    evidence_hash = "a" * 64
    template_wire = load_target_profile().to_wire()
    wire = build_resolved_wire(template_wire, m, evidence_hash)
    validate_resolved_wire(wire)  # raises on failure
    assert wire["provenance"] == "operator-attested"
    assert wire["build"]["build_id"] == "25016043"
    assert wire["build"]["distribution"] == template_wire["build"]["distribution"]  # 未対象フィールドは温存


def test_build_resolved_wire_build_id_survives_yaml_roundtrip_as_str():
    # 既知バグ回帰: 過去は build_id がクォートなしでダンプされ int 化していた。
    m = _measurements()
    wire = build_resolved_wire(load_target_profile().to_wire(), m, "a" * 64)
    reloaded = yaml.safe_load(yaml.safe_dump(wire, sort_keys=True, allow_unicode=True))
    assert isinstance(reloaded["build"]["build_id"], str) and reloaded["build"]["build_id"] == "25016043"


def test_validate_resolved_wire_rejects_leftover_placeholder():
    m = _measurements()
    wire = build_resolved_wire(load_target_profile().to_wire(), m, "a" * 64)
    wire["hardware"]["gpu_name"] = "TEST_FIXTURE_ONLY"
    with pytest.raises(AuditError):
        validate_resolved_wire(wire)


def test_read_json_bom_strips_byte_order_mark(tmp_path):
    path = tmp_path / "with_bom.json"
    path.write_bytes(b"\xef\xbb\xbf" + b'{"a": 1}')
    assert _read_json_bom(path) == {"a": 1}


def test_latest_evidence_manifest_picks_newest_name(tmp_path):
    old = tmp_path / "target-evidence-20260101_000000_000.json"
    new = tmp_path / "target-evidence-20260919_210009_203.json"
    old.write_text("{}", encoding="utf-8")
    new.write_text("{}", encoding="utf-8")
    assert latest_evidence_manifest(tmp_path) == new


def test_latest_evidence_manifest_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        latest_evidence_manifest(tmp_path)

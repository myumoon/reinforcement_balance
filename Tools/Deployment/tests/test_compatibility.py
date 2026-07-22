import pytest
from survivors.compatibility import invalidation_for,validate_manifest,load_matrix
from reinbalance_survivors_contracts.launch_lifecycle import _verify_audit_evidence
from reinbalance_survivors_contracts.canonical_json import canonical_hash
from survivors.target_profile import SECTIONS

@pytest.mark.parametrize(("fields","expected"),[
 ({"ui_revision"},{"parser","replay"}),
 ({"content_revision"},{"taxonomy","selector","fidelity"}),
 ({"physics_hz"},{"fidelity","policy","canary"}),
])
def test_invalidation_matrix(fields,expected):assert set(invalidation_for(fields)["invalidate"])==expected

def test_unknown_difference_and_missing_audit_parent_fail_closed():
    with pytest.raises(ValueError):invalidation_for({"future_field"})
    with pytest.raises(ValueError):validate_manifest({})
    evidence=b"audit"; verdict=_verify_audit_evidence(attempt_id="a",target_identity_hash="a"*64,canonical_save_hash="b"*64,save_semantics_hash="c"*64,evidence_bytes=evidence,expected_evidence_hash=canonical_hash({"bytes_hex":evidence.hex()}))
    validate_manifest({"schema_version":"artifact_manifest.v1","artifact_kind":"capture","target_audit_hash":verdict.canonical_hash,"target_audit_verdict":verdict})
    with pytest.raises(ValueError): validate_manifest({"schema_version":"artifact_manifest.v1","artifact_kind":"capture","target_audit_hash":"a"*64,"target_audit_verdict":verdict})
    with pytest.raises(ValueError): validate_manifest({"schema_version":"artifact_manifest.v1","artifact_kind":"capture","target_audit_hash":"a"*64,"target_audit_verdict":{"status":"PASS"}})

def test_matrix_semantics_are_immutable(tmp_path):
    import yaml
    data=load_matrix(); data["classes"]["ui_only"]["invalidate"]=["policy"]
    path=tmp_path/"matrix.yaml"; path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError): load_matrix(path)

def test_every_target_profile_field_is_classified_fail_closed(tmp_path):
    import yaml
    data=load_matrix()
    classified=[field for rule in data["classes"].values() for field in rule["changed_fields"]]
    assert set(classified)=={"schema_version",*(field for fields in SECTIONS.values() for field in fields)}
    assert len(classified)==len(set(classified))
    data["classes"]["ui_only"]["changed_fields"].remove("ui_revision")
    path=tmp_path/"matrix.yaml"; path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError,match="coverage"):
        load_matrix(path)

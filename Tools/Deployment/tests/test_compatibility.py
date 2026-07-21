import pytest
from survivors.compatibility import invalidation_for,validate_manifest,load_matrix
from reinbalance_survivors_contracts.launch_lifecycle import AuditVerdict

@pytest.mark.parametrize(("fields","expected"),[
 ({"ui_revision"},{"parser","replay"}),
 ({"content_revision"},{"taxonomy","selector","fidelity"}),
 ({"physics_hz"},{"fidelity","policy","canary"}),
])
def test_invalidation_matrix(fields,expected):assert set(invalidation_for(fields)["invalidate"])==expected

def test_unknown_difference_and_missing_audit_parent_fail_closed():
    with pytest.raises(ValueError):invalidation_for({"future_field"})
    with pytest.raises(ValueError):validate_manifest({})
    verdict=AuditVerdict._verified("a"*64,"b"*64)
    validate_manifest({"schema_version":"artifact_manifest.v1","artifact_kind":"capture","target_audit_hash":verdict.canonical_hash,"target_audit_verdict":verdict})
    with pytest.raises(ValueError): validate_manifest({"schema_version":"artifact_manifest.v1","artifact_kind":"capture","target_audit_hash":"a"*64,"target_audit_verdict":verdict})
    with pytest.raises(ValueError): validate_manifest({"schema_version":"artifact_manifest.v1","artifact_kind":"capture","target_audit_hash":"a"*64,"target_audit_verdict":{"status":"PASS"}})

def test_matrix_semantics_are_immutable(tmp_path):
    import yaml
    data=load_matrix(); data["classes"]["ui_only"]["invalidate"]=["policy"]
    path=tmp_path/"matrix.yaml"; path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError): load_matrix(path)

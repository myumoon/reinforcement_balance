import pytest
from survivors.compatibility import invalidation_for,validate_manifest
from reinbalance_survivors_contracts.canonical_json import canonical_hash

@pytest.mark.parametrize(("fields","expected"),[
 ({"ui_revision"},{"parser","replay"}),
 ({"content_revision"},{"taxonomy","selector","fidelity"}),
 ({"physics_hz"},{"fidelity","policy","canary"}),
])
def test_invalidation_matrix(fields,expected):assert set(invalidation_for(fields)["invalidate"])==expected

def test_unknown_difference_and_missing_audit_parent_fail_closed():
    with pytest.raises(ValueError):invalidation_for({"future_field"})
    with pytest.raises(ValueError):validate_manifest({})
    report={"schema_version":"target_audit.v1","target_hash":"a"*64,"status":"PASS"}
    validate_manifest({"schema_version":"artifact_manifest.v1","artifact_kind":"capture","target_audit_hash":canonical_hash(report),"target_audit_report":report})
    with pytest.raises(ValueError): validate_manifest({"schema_version":"artifact_manifest.v1","artifact_kind":"capture","target_audit_hash":"a"*64,"target_audit_report":report})

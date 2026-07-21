import pytest
from survivors.compatibility import invalidation_for,validate_manifest

@pytest.mark.parametrize(("fields","expected"),[
 ({"ui_revision"},{"parser","replay"}),
 ({"content_revision"},{"taxonomy","selector","fidelity"}),
 ({"physics_hz"},{"fidelity","policy","canary"}),
])
def test_invalidation_matrix(fields,expected):assert set(invalidation_for(fields)["invalidate"])==expected

def test_unknown_difference_and_missing_audit_parent_fail_closed():
    with pytest.raises(ValueError):invalidation_for({"future_field"})
    with pytest.raises(ValueError):validate_manifest({})
    validate_manifest({"target_audit_hash":"a"*64})

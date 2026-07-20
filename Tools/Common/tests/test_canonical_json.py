"""決定的な canonical JSON バイト列とハッシュのテスト。"""

import math

import pytest

from reinbalance_survivors_contracts import (
    CanonicalJsonError,
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)


def test_key_order_independent():
    a = {"b": 1, "a": 2, "c": {"z": 1, "y": 2}}
    b = {"c": {"y": 2, "z": 1}, "a": 2, "b": 1}
    assert canonical_json_bytes(a) == canonical_json_bytes(b)
    assert canonical_hash(a) == canonical_hash(b)


def test_compact_and_sorted_output():
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_tuple_encodes_as_array():
    assert canonical_json_bytes((1, 2, 3)) == canonical_json_bytes([1, 2, 3])


def test_non_ascii_is_utf8_stable():
    out = canonical_json_bytes({"name": "コイン"})
    assert out == '{"name":"コイン"}'.encode("utf-8")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_float_rejected(bad):
    with pytest.raises(CanonicalJsonError):
        canonical_json_bytes({"x": bad})


def test_nested_non_finite_rejected():
    with pytest.raises(CanonicalJsonError):
        canonical_json_bytes({"a": [1, {"b": math.inf}]})


def test_non_str_key_rejected():
    with pytest.raises(CanonicalJsonError):
        canonical_json_bytes({1: "a"})


def test_unsupported_type_rejected():
    with pytest.raises(CanonicalJsonError):
        canonical_json_bytes({"x": {1, 2, 3}})  # set は JSON ネイティブではない


def test_bool_is_preserved_not_int():
    assert canonical_json_bytes({"x": True}) == b'{"x":true}'


def test_hash_matches_sha256_of_bytes():
    obj = {"a": 1, "b": [2, 3]}
    assert canonical_hash(obj) == sha256_hex(canonical_json_bytes(obj))

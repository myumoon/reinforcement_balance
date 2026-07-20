"""共有 Survivors 契約のための canonical JSON シリアライズと内容ハッシュ。

全ての共有 wire 型は :func:`canonical_json_bytes` を通してシリアライズされる。これに
より、Training と Deployment は、プロセス・プラットフォーム・dict の挿入順に関係なく、
同じ論理値に対して byte-identical なバイト列（＝同一の SHA-256 ハッシュ）を生成する。

ルール（プラン 00-01 タスク2 参照）:

- UTF-8 エンコード。
- オブジェクトのキーは辞書順にソート（``sort_keys=True``）。
- コンパクトな区切り（余分な空白なし）。
- 有限な数値のみ許可する。``NaN`` / ``Infinity`` は fail-closed で拒否する。厳密な
  JSON では表現できず、言語間の parity を壊すため。
- ``ensure_ascii=False`` により非 ASCII は UTF-8 のまま保持する（決定的なまま）。
- schema identity に volatile なメタデータ（timestamp・run id・hostname）を含めては
  ならない。呼び出し側は identity フィールドのみを :func:`canonical_hash` に渡す。
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

__all__ = [
    "CanonicalJsonError",
    "canonical_json_bytes",
    "sha256_hex",
    "canonical_hash",
]


class CanonicalJsonError(ValueError):
    """値を決定的に canonical 化できないときに送出される例外。"""


def _assert_canonicalizable(obj: Any) -> None:
    """決定的な parity を壊す値に対して fail-closed で拒否する。"""
    if isinstance(obj, bool):
        # bool は int のサブクラス。許可し、int の分岐より前で処理する。
        return
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise CanonicalJsonError(f"non-finite float is not allowed: {obj!r}")
        return
    if isinstance(obj, (int, str)) or obj is None:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise CanonicalJsonError(
                    f"object keys must be str, got {type(key).__name__}"
                )
            _assert_canonicalizable(value)
        return
    if isinstance(obj, (list, tuple)):
        for value in obj:
            _assert_canonicalizable(value)
        return
    raise CanonicalJsonError(
        f"type {type(obj).__name__} is not canonicalizable to JSON"
    )


def canonical_json_bytes(obj: Any) -> bytes:
    """``obj`` の決定的な canonical JSON バイト列を返す。

    ``obj`` は JSON ネイティブ型（dict / list / tuple / str / int / float / bool /
    None）のみで構成されていなければならない。tuple は JSON 配列としてエンコードする。
    """
    _assert_canonicalizable(obj)
    try:
        text = json.dumps(
            obj,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - 防御的
        raise CanonicalJsonError(str(exc)) from exc
    return text.encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """``data`` の SHA-256 16進ダイジェストを返す。"""
    return hashlib.sha256(data).hexdigest()


def canonical_hash(obj: Any) -> str:
    """``obj`` の canonical JSON バイト列の SHA-256 16進ダイジェストを返す。"""
    return sha256_hex(canonical_json_bytes(obj))

"""C++ `/content_schema` export に監査 annotation だけを付与する。

武器・item・level・進化 pair は入力 JSON から読み取り、Python 側へ同じ集合を
ハードコードしません。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AnnotatedFidelitySchema:
    """C++ schema と target relevance の不変な組。

    content identity を改変せず保持し、監査固有の注釈を別 object に分離します。
    """
    cpp_schema: Mapping[str, Any]
    annotations: Mapping[str, Mapping[str, Any]]


def annotate_cpp_content_schema(export: Mapping[str, Any], annotations: Mapping[str, Mapping[str, Any]]) -> AnnotatedFidelitySchema:
    """C++ export の id 集合に対する annotation を厳格に付与する。

    annotation に未知 id があれば拒否し、ids/levels/pairs を Python から生成しません。
    """
    if not isinstance(export, Mapping) or set(export) != {"schema_version", "content", "action_time"}:
        raise ValueError("content schema top-level keys mismatch")
    if export["schema_version"] != "survivors.content_schema.v1":
        raise ValueError("unsupported content schema")
    content = export["content"]
    if not isinstance(content, Mapping):
        raise ValueError("content must be an object")
    ids: set[str] = set()
    for collection in ("weapons", "passives", "gems"):
        values = content.get(collection)
        if not isinstance(values, list):
            raise ValueError(f"{collection} must be an array")
        for row in values:
            if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
                raise ValueError(f"{collection} row requires string id")
            ids.add(row["id"])
    if not isinstance(annotations, Mapping) or not set(annotations) <= ids:
        raise ValueError("annotations contain unknown C++ content ids")
    for key, value in annotations.items():
        if not isinstance(value, Mapping) or set(value) - {"target_relevant", "note"}:
            raise ValueError(f"invalid annotation for {key}")
        if type(value.get("target_relevant")) is not bool:
            raise ValueError(f"target_relevant must be bool for {key}")
    return AnnotatedFidelitySchema(dict(export), {key: dict(value) for key, value in annotations.items()})

"""旧 Value Source maturity record を読み書きする互換 wrapper。

過去の分析用 field は維持しますが、label-ready 判定は teacher validation へ移ったため
この module から正式 gate field を出力しません。
"""

from __future__ import annotations

def make_value_source_maturity_record(
    *,
    run_name: str,
    item_stage_key: str,
    bootstrap_complete: bool,
    passive_coverage_count: int,
    evolution_coverage_count: int,
    union_coverage_count: int,
    model_path: str,
    vecnormalize_path: str,
) -> dict:
    """deprecated maturity 情報を gate を含まない互換 record として返す。

    ``ready_for_value_labels`` を復元せず、呼び出し元を immutable descriptor の
    ``ready_for_probe`` と後続 teacher verdict へ段階的に移行させます。
    """
    return {
        "run_name": run_name,
        "item_stage_key": item_stage_key,
        "bootstrap_complete": bool(bootstrap_complete),
        "passive_coverage_count": int(passive_coverage_count),
        "evolution_coverage_count": int(evolution_coverage_count),
        "union_coverage_count": int(union_coverage_count),
        "model_path": model_path,
        "vecnormalize_path": vecnormalize_path,
    }

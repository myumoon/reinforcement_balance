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
    ready = (
        item_stage_key == "IS2"
        and bootstrap_complete
        and passive_coverage_count > 0
        and evolution_coverage_count > 0
        and union_coverage_count > 0
    )
    return {
        "run_name": run_name,
        "item_stage_key": item_stage_key,
        "bootstrap_complete": bool(bootstrap_complete),
        "passive_coverage_count": int(passive_coverage_count),
        "evolution_coverage_count": int(evolution_coverage_count),
        "union_coverage_count": int(union_coverage_count),
        "model_path": model_path,
        "vecnormalize_path": vecnormalize_path,
        "ready_for_value_labels": ready,
    }

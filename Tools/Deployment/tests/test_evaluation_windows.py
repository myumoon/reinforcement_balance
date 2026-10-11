"""evaluation_windows.v1 の規則生成と封印を検証するテスト。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from survivors.evaluation_windows import (
    EvaluationWindowPlan,
    generate_windows,
    validate_plan_wire,
    validate_windows,
)


def _manifest(frame_count: int) -> SimpleNamespace:
    """指定件数の等間隔 frame を持つ synthetic session を作る。
    timestamp と frame id の対応を固定して規則生成を再現可能にする。
    """
    records = tuple(
        SimpleNamespace(frame_id=index, captured_monotonic_ns=index * 1_000_000_000)
        for index in range(frame_count)
    )
    return SimpleNamespace(session_id="synthetic-session", frame_records=records)


def _rule() -> dict[str, object]:
    """小さな窓で固定 offset・tail・event 規則を組み合わせる。
    hash と threshold を固定し、生成内容を人が選ぶ余地を除く。
    """
    return {
        "fixed_offsets_per_time_band": {
            "window_frames": 4,
            "offsets_seconds": {"early": 2, "mid": 8, "late": 14},
        },
        "run_tail": {"seconds": 3},
        "event_windows": {
            "detector": "development_hud_parser",
            "states": ["level_up_items", "level_up_fallback", "chest"],
            "pre_frames": 1,
            "post_frames": 1,
            "max_per_band": 4,
            "parser_artifact_hash": "a" * 64,
        },
        "boss_windows": {
            "detector": "development_world_detector",
            "checkpoint_hash": "b" * 64,
            "score_threshold": 0.5,
            "pre_frames": 1,
            "post_frames": 2,
            "max_per_session": 8,
        },
    }


def test_generate_windows_merges_rule_windows_and_seals_plan() -> None:
    """規則で拾った重複窓を併合し、session と hash に固定する。
    plan wire の session 改変が canonical hash 検証で拒否されることも確認する。
    """
    manifest = _manifest(20)

    def load_event(record: SimpleNamespace) -> dict[str, object]:
        """frame 4 で level-up、frame 10 で boss が現れる。
        その他の frame は event が無い固定 fixture とする。
        """
        return {
            "screen_state": "level_up_items" if record.frame_id == 4 else "gameplay",
            "boss_score": 0.9 if record.frame_id == 10 else 0.0,
        }

    windows = generate_windows(manifest, _rule(), frame_loader=load_event)
    plan = EvaluationWindowPlan.create(
        {manifest.session_id: windows}, rule=_rule(), stride=1,
        generated_at_utc="2026-10-11T00:00:00Z",
    )

    assert windows == [(2, 5), (8, 12), (14, 19)]
    assert validate_plan_wire(plan.to_wire()).plan_canonical_hash == plan.plan_canonical_hash
    changed = plan.to_wire()
    changed["sessions"][manifest.session_id][0][0] = 0
    with pytest.raises(ValueError, match="hash"):
        validate_plan_wire(changed)


def test_validate_windows_rejects_overlapping_or_out_of_session_ranges() -> None:
    """閉区間の重複と manifest の終端外を拒否する。
    window boundary の曖昧さを persisted plan に残さない。
    """
    with pytest.raises(ValueError, match="overlap"):
        validate_windows([(0, 3), (3, 4)], frame_count=5)
    with pytest.raises(ValueError, match="range"):
        validate_windows([(0, 5)], frame_count=5)

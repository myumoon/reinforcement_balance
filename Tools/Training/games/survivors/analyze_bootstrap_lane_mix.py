"""task_cell_episode_metrics.jsonl の selected bootstrap lane 監査ヘルパー。

TaskCellSamplerCallback が書き出す JSONL を解析し、
設定した sample_mix と実際に選ばれた lane 配分のズレを監査する。

出力:
- selected lane の count / ratio
- selected lane × task_kind クロス集計
- weapon × phase × selected lane の count
- 直近 N episodes の selected lane ratio

CLI:
    python -m games.survivors.analyze_bootstrap_lane_mix \
        --run-dir runs/survivors/v12/train/run07 \
        --window-episodes 200 \
        --weapon-id 7 \
        --phase 2
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


JSONL_NAME = "task_cell_episode_metrics.jsonl"


def resolve_jsonl_path(run_dir: str | Path) -> Path:
    """run-dir から task_cell_episode_metrics.jsonl のパスを解決する。

    run_dir が JSONL ファイルそのものを指す場合はそのまま返す。
    それ以外は run_dir 直下、または run_dir/work 配下を探索する。
    """
    p = Path(run_dir)
    if p.is_file():
        return p
    candidates = [p / JSONL_NAME, p / "work" / JSONL_NAME, p / "logs" / JSONL_NAME]
    for c in candidates:
        if c.exists():
            return c
    # 見つからない場合は再帰探索（最初にヒットしたもの）
    matches = sorted(p.rglob(JSONL_NAME))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"{JSONL_NAME} が見つかりません: {run_dir}\n"
        f"探索した候補: {[str(c) for c in candidates]}"
    )


def load_records(jsonl_path: str | Path) -> list[dict]:
    """JSONL を読み込んで record の list を返す（壊れた行はスキップ）。"""
    records: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _selected_lane(record: dict) -> str | None:
    """record から selected lane を取り出す。

    新フィールド selected_bootstrap_lane を優先。無ければ legacy bootstrap_lane に
    フォールバックする（旧 run 互換）。
    """
    lane = record.get("selected_bootstrap_lane")
    if lane is not None:
        return lane
    return record.get("bootstrap_lane")


def filter_records(
    records: Iterable[dict],
    *,
    weapon_id: int | None = None,
    phase: int | None = None,
) -> list[dict]:
    """weapon_id / enemy_phase_idx で record を絞り込む。"""
    out: list[dict] = []
    for r in records:
        if weapon_id is not None and r.get("first_weapon_id") != weapon_id:
            continue
        if phase is not None and r.get("enemy_phase_idx") != phase:
            continue
        out.append(r)
    return out


def selected_lane_counts(records: Iterable[dict]) -> Counter:
    """selected lane ごとの count を返す。"""
    counter: Counter = Counter()
    for r in records:
        lane = _selected_lane(r)
        if lane is None:
            continue
        counter[lane] += 1
    return counter


def selected_lane_ratios(records: Iterable[dict]) -> dict[str, float]:
    """selected lane ごとの ratio を返す。"""
    counts = selected_lane_counts(records)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {lane: c / total for lane, c in counts.items()}


def lane_task_kind_crosstab(records: Iterable[dict]) -> dict[str, Counter]:
    """selected lane × task_kind のクロス集計を返す。

    { selected_lane: Counter({task_kind: count}) }
    """
    table: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        lane = _selected_lane(r)
        if lane is None:
            continue
        task_kind = r.get("bootstrap_task_kind") or r.get("task_kind")
        table[lane][task_kind] += 1
    return dict(table)


def weapon_phase_lane_counts(records: Iterable[dict]) -> dict[tuple, Counter]:
    """(weapon_id, phase) × selected lane の count を返す。

    { (first_weapon_id, enemy_phase_idx): Counter({selected_lane: count}) }
    """
    table: dict[tuple, Counter] = defaultdict(Counter)
    for r in records:
        lane = _selected_lane(r)
        if lane is None:
            continue
        key = (r.get("first_weapon_id"), r.get("enemy_phase_idx"))
        table[key][lane] += 1
    return dict(table)


def recent_lane_ratios(records: list[dict], window_episodes: int) -> dict[str, float]:
    """直近 N episodes の selected lane ratio を返す。"""
    tail = records[-window_episodes:] if window_episodes > 0 else records
    return selected_lane_ratios(tail)


def _format_ratio_block(title: str, counts: Counter, ratios: dict[str, float]) -> str:
    lines = [title]
    total = sum(counts.values())
    if total == 0:
        lines.append("  (no records)")
        return "\n".join(lines)
    for lane, count in counts.most_common():
        lines.append(f"  {lane:<16} count={count:>6}  ratio={ratios.get(lane, 0.0):.3f}")
    lines.append(f"  {'TOTAL':<16} count={total:>6}")
    return "\n".join(lines)


def analyze(
    run_dir: str | Path,
    *,
    window_episodes: int = 200,
    weapon_id: int | None = None,
    phase: int | None = None,
) -> str:
    """run-dir を解析してレポート文字列を返す。"""
    jsonl_path = resolve_jsonl_path(run_dir)
    records = load_records(jsonl_path)
    records = filter_records(records, weapon_id=weapon_id, phase=phase)

    out: list[str] = []
    out.append(f"JSONL: {jsonl_path}")
    out.append(f"records (filtered): {len(records)}  weapon_id={weapon_id}  phase={phase}")
    out.append("")

    counts = selected_lane_counts(records)
    ratios = selected_lane_ratios(records)
    out.append(_format_ratio_block("[selected lane count/ratio (all)]", counts, ratios))
    out.append("")

    out.append("[selected lane x task_kind]")
    crosstab = lane_task_kind_crosstab(records)
    if not crosstab:
        out.append("  (no records)")
    for lane, tk_counter in crosstab.items():
        detail = ", ".join(f"{tk}={c}" for tk, c in tk_counter.most_common())
        out.append(f"  {lane:<16} {detail}")
    out.append("")

    out.append("[weapon x phase x selected lane count]")
    wp = weapon_phase_lane_counts(records)
    if not wp:
        out.append("  (no records)")
    for (wid, ph), lane_counter in sorted(wp.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        detail = ", ".join(f"{lane}={c}" for lane, c in lane_counter.most_common())
        out.append(f"  weapon={wid} phase={ph}: {detail}")
    out.append("")

    recent = recent_lane_ratios(records, window_episodes)
    recent_counts = selected_lane_counts(records[-window_episodes:] if window_episodes > 0 else records)
    out.append(_format_ratio_block(
        f"[recent {window_episodes} episodes selected lane count/ratio]", recent_counts, recent))

    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="run ディレクトリ、または JSONL ファイルパス")
    parser.add_argument("--window-episodes", type=int, default=200, help="直近 N episodes の ratio 集計窓")
    parser.add_argument("--weapon-id", type=int, default=None, help="first_weapon_id フィルタ")
    parser.add_argument("--phase", type=int, default=None, help="enemy_phase_idx フィルタ")
    args = parser.parse_args(argv)

    report = analyze(
        args.run_dir,
        window_episodes=args.window_episodes,
        weapon_id=args.weapon_id,
        phase=args.phase,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

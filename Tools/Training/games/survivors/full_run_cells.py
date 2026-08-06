"""Survivors 30分 Full Run curriculum の FR0〜FR4 セルを定義する。

初心者向け: 既存 TaskCell の保存形式は変えず、時間帯と ItemSelector 接続可否を
外側の immutable spec に保持して短時間練習・後半再開・完走を組み立てます。
"""

from __future__ import annotations

from dataclasses import dataclass

from games.survivors.modules.task_cell_sampler_module import TaskCell
from games.survivors.survivors_weapon_curriculum import ALL_BASE_ATTACK_WEAPONS


@dataclass(frozen=True)
class FullRunCellSpec:
    """TaskCell に full-run 固有の時間帯と selector flag を束ねる。

    初心者向け: TaskCell 自体へ field を足さないため、既存 checkpoint と key を
    そのまま読み書きしながら 30分 curriculum の情報を利用できます。
    """

    band: str
    elapsed_min: float
    elapsed_max: float
    task_cell: TaskCell
    item_selector_enabled: bool = False

    def __post_init__(self) -> None:
        """band 名、時間境界、TaskCell の対応を構築時に検証する。

        初心者向け: typo や逆転した時間帯を sampler へ渡す前に止めます。
        """
        if self.band not in {"FR0", "FR1", "FR2", "FR3", "FR4"}:
            raise ValueError(f"unknown full-run band: {self.band!r}")
        if not (0.0 <= self.elapsed_min < self.elapsed_max <= 30.0):
            raise ValueError("full-run elapsed bounds must be inside 0..30 minutes")
        if self.task_cell.combo_key != self.band.lower():
            raise ValueError("TaskCell combo_key must match the full-run band")
        if self.task_cell.task_kind != "full_run":
            raise ValueError("full-run specs require task_kind='full_run'")

    @property
    def lane(self) -> str:
        """既定 sample mix で使う lane 名を返す。

        初心者向け: FR0 は短い基礎、FR1〜FR3 は途中再開、FR4 は最初から完走です。
        """
        if self.band == "FR0":
            return "short_skill"
        if self.band == "FR4":
            return "full_run"
        return "late_rsi"

    def to_env_params(self) -> dict:
        """既存 /params 名を使った episode 初期化値を返す。

        初心者向け: 時間は分から秒へ変換し、固定装備と ItemSelector mode を
        一つの request にまとめます。
        """
        cell = self.task_cell
        return {
            "initial_elapsed_time": self.elapsed_min * 60.0,
            "MaxEpisodeTime": self.elapsed_max * 60.0,
            "initial_weapon_slots": [
                {"weapon_id": weapon_id, "level": level}
                for weapon_id, level in cell.initial_weapon_slots
            ],
            "initial_passive_slots": [
                {"passive_id": passive_id, "level": level}
                for passive_id, level in cell.initial_passive_slots
            ],
            "allowed_weapon_types": list(cell.allowed_weapon_ids),
            "item_selection_mode": (
                "external" if self.item_selector_enabled else "auto"
            ),
        }


def _cell(
    band: str,
    *,
    enemy_phase_idx: int,
    weapon_slots: tuple[tuple[int, int], ...],
    passive_slots: tuple[tuple[int, int], ...],
    item_selector_enabled: bool = False,
) -> TaskCell:
    """full-run 用 TaskCell を既存 field だけで構築する。

    初心者向け: band は combo_key に入り、従来の key 生成と serialization を再利用します。
    """
    first_weapon_id = weapon_slots[0][0] if weapon_slots else ALL_BASE_ATTACK_WEAPONS[0]
    return TaskCell(
        weapon_unlock_stage_key="WU12",
        first_weapon_id=first_weapon_id,
        enemy_phase_idx=enemy_phase_idx,
        task_kind="full_run",
        build_policy="fresh_progression" if item_selector_enabled else "combination_slots",
        combo_key=band.lower(),
        initial_weapon_slots=weapon_slots,
        initial_passive_slots=passive_slots,
        allowed_weapon_ids=tuple(ALL_BASE_ATTACK_WEAPONS),
    )


FULL_RUN_BANDS: dict[str, FullRunCellSpec] = {
    "FR0": FullRunCellSpec(
        "FR0", 0.0, 5.0,
        _cell("FR0", enemy_phase_idx=3, weapon_slots=((1, 1),), passive_slots=()),
    ),
    "FR1": FullRunCellSpec(
        "FR1", 5.0, 10.0,
        _cell(
            "FR1", enemy_phase_idx=9,
            weapon_slots=((1, 4), (7, 4)), passive_slots=((4, 2), (8, 2)),
        ),
    ),
    "FR2": FullRunCellSpec(
        "FR2", 10.0, 20.0,
        _cell(
            "FR2", enemy_phase_idx=12,
            weapon_slots=((1, 8), (7, 8), (3, 6), (9, 6)),
            passive_slots=((4, 5), (8, 5), (5, 4), (11, 4)),
        ),
    ),
    "FR3": FullRunCellSpec(
        "FR3", 20.0, 30.0,
        _cell(
            "FR3", enemy_phase_idx=15,
            weapon_slots=((16, 1), (22, 1), (18, 1), (24, 1), (25, 1), (26, 1)),
            passive_slots=((4, 5), (8, 5), (5, 5), (11, 5), (2, 5), (9, 2)),
        ),
    ),
    "FR4": FullRunCellSpec(
        "FR4", 0.0, 30.0,
        _cell(
            "FR4", enemy_phase_idx=15, weapon_slots=(), passive_slots=(),
            item_selector_enabled=True,
        ),
        item_selector_enabled=True,
    ),
}


def build_full_run_cells() -> tuple[FullRunCellSpec, ...]:
    """FR0〜FR4 を昇順の immutable tuple で返す。

    初心者向け: 呼び出し側が global 定義を変更できない形で sampler へ渡します。
    """
    return tuple(FULL_RUN_BANDS[f"FR{index}"] for index in range(5))


"""30分 Full Run curriculum のセル境界と sampling 契約を検証する。

初心者向け: 短時間練習、後半再開、最初からの完走が所定の比率と装備で
選ばれ、ItemSelector が FR4 だけで有効になることを確認します。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.full_run_cells import FULL_RUN_BANDS, build_full_run_cells
from games.survivors.modules.task_cell_sampler_module import (
    DEFAULT_FULL_RUN_SAMPLE_MIX,
    TaskCell,
    TaskCellSamplerStateModule,
)


def test_fr0_to_fr4_have_fixed_time_enemy_and_loadout_contract() -> None:
    """FR0〜FR4 の時間帯、敵 phase、初期 loadout を固定表と照合する。

    初心者向け: curriculum の意味がコード整理で変わらないよう、全 band を一括で確認します。
    """
    expected = {
        "FR0": (0.0, 5.0, 3, ((1, 1),), ()),
        "FR1": (5.0, 10.0, 9, ((1, 4), (7, 4)), ((4, 2), (8, 2))),
        "FR2": (
            10.0,
            20.0,
            12,
            ((1, 8), (7, 8), (3, 6), (9, 6)),
            ((4, 5), (8, 5), (5, 4), (11, 4)),
        ),
        "FR3": (
            20.0,
            30.0,
            15,
            ((16, 1), (22, 1), (18, 1), (24, 1), (25, 1), (26, 1)),
            ((4, 5), (8, 5), (5, 5), (11, 5), (2, 5), (9, 2)),
        ),
        "FR4": (0.0, 30.0, 15, (), ()),
    }

    assert set(FULL_RUN_BANDS) == set(expected)
    for key, spec in FULL_RUN_BANDS.items():
        elapsed_min, elapsed_max, enemy_phase, weapons, passives = expected[key]
        assert isinstance(spec.task_cell, TaskCell)
        assert (
            spec.elapsed_min,
            spec.elapsed_max,
            spec.task_cell.enemy_phase_idx,
            spec.task_cell.initial_weapon_slots,
            spec.task_cell.initial_passive_slots,
        ) == (elapsed_min, elapsed_max, enemy_phase, weapons, passives)
        assert spec.task_cell.combo_key == key.lower()
        assert spec.task_cell.key().startswith("full_run/WU12/")


def test_only_fr4_enables_item_selector() -> None:
    """ItemSelector の接続フラグが FR4 だけで有効なことを確認する。

    初心者向け: 後半 RSI の固定装備に選択モデルが混ざる事故を防ぎます。
    """
    enabled = {key for key, spec in FULL_RUN_BANDS.items() if spec.item_selector_enabled}
    assert enabled == {"FR4"}


def test_full_run_cells_are_deterministic_and_keep_task_cell_contract() -> None:
    """builder が既存 TaskCell を band 順で返し key を変更しないことを確認する。

    初心者向け: resume や統計保存が従来の TaskCell 形式をそのまま利用できることを守ります。
    """
    first = build_full_run_cells()
    second = build_full_run_cells()
    assert first == second
    assert tuple(spec.band for spec in first) == ("FR0", "FR1", "FR2", "FR3", "FR4")
    assert len({spec.task_cell.key() for spec in first}) == 5


def test_full_run_sample_mix_defaults_and_validation() -> None:
    """full-run lane mix の既定値と不正入力の拒否を確認する。

    初心者向け: 比率の typo や負値を無視せず、訓練開始前に止めます。
    """
    assert DEFAULT_FULL_RUN_SAMPLE_MIX == {
        "short_skill": 0.45,
        "late_rsi": 0.25,
        "full_run": 0.30,
    }
    sampler = TaskCellSamplerStateModule()
    assert sampler.full_run_sample_mix == DEFAULT_FULL_RUN_SAMPLE_MIX

    custom = {"short_skill": 1.0, "late_rsi": 2.0, "full_run": 1.0}
    sampler = TaskCellSamplerStateModule(full_run_sample_mix=custom)
    assert sampler.full_run_sample_mix == custom

    with pytest.raises(ValueError, match="keys"):
        TaskCellSamplerStateModule(full_run_sample_mix={"full_run": 1.0})
    with pytest.raises(ValueError, match="finite non-negative"):
        TaskCellSamplerStateModule(
            full_run_sample_mix={"short_skill": -1.0, "late_rsi": 1.0, "full_run": 1.0}
        )


def test_sampler_accepts_fr_band_specs_with_default_mix(monkeypatch) -> None:
    """sampler が full-run spec を受けて三つの lane から選択できることを確認する。

    初心者向け: TaskCell 本体を拡張せず、band metadata を sampler 境界で利用します。
    """
    sampler = TaskCellSamplerStateModule()
    monkeypatch.setattr("numpy.random.choice", lambda count, p: count - 1)
    chosen = sampler.sample_full_run_cell(build_full_run_cells())
    assert chosen.band == "FR4"


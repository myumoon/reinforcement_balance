"""PerceptionErrorProfile の厳格な wire 契約と bootstrap 設定を検証する。

設定ファイルの入力間違いや評価データの混入を学習開始前に止め、
同じ設定内容が常に同じ識別 hash になることを初心者にも確認できるテストです。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.perception_error import (
    PERCEPTION_ERROR_SCHEMA_VERSION,
    PerceptionErrorProfile,
)

BOOTSTRAP = Path(__file__).parents[2] / "configs" / "perception_error_bootstrap_v1.json"


def _clean_wire() -> dict:
    """全 corruption が無効な正規 wire mapping を作る。

    各テストはこの辞書の一項目だけを壊すことで、どの入力境界を
    検証しているのかを読みやすくします。
    """
    return {
        "latency_mean_frames": 0.0,
        "latency_std_frames": 0.0,
        "burst_enter_prob": 0.0,
        "burst_exit_prob": 1.0,
        "burst_dropout_prob": 0.0,
        "coord_noise_std": 0.0,
        "coord_quantization_px": 0.0,
        "count_clip_max": 32,
        "item_confusion_matrix": [],
        "enemy_confusion_matrix": [],
        "hud_timer_stale_prob": 0.0,
        "hud_hp_misread_std": 0.0,
        "hud_xp_stale_prob": 0.0,
        "hud_inventory_stale_prob": 0.0,
        "unknown_screen_collapse_prob": 0.0,
        "unknown_screen_collapse_duration_frames": 0.0,
        "calibration_session_ids": [],
        "final_e2e_session_ids": [],
        "schema_version": PERCEPTION_ERROR_SCHEMA_VERSION,
    }


def test_clean_and_bootstrap_profiles_load_and_hash_canonically():
    """clean と配布 bootstrap の両 profile を厳密 loader で読む。

    読み戻した wire と共有 canonical hash が一致し、独自の JSON hash
    実装へ逸脱していないことも同時に保証します。
    """
    clean = PerceptionErrorProfile.from_wire(_clean_wire())
    bootstrap_data = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))
    bootstrap = PerceptionErrorProfile.from_wire(bootstrap_data)

    assert clean.is_clean
    assert not bootstrap.is_clean
    assert clean.profile_hash == canonical_hash(clean.to_wire())
    assert bootstrap.profile_hash == canonical_hash(bootstrap_data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latency_mean_frames", -0.1),
        ("latency_std_frames", float("inf")),
        ("burst_enter_prob", -0.1),
        ("burst_exit_prob", 1.1),
        ("burst_dropout_prob", "0.2"),
        ("coord_noise_std", float("nan")),
        ("coord_quantization_px", -1.0),
        ("count_clip_max", 0),
        ("count_clip_max", True),
        ("hud_timer_stale_prob", 2.0),
        ("hud_hp_misread_std", -0.1),
        ("hud_xp_stale_prob", None),
        ("hud_inventory_stale_prob", -0.1),
        ("unknown_screen_collapse_prob", 1.1),
        ("unknown_screen_collapse_duration_frames", -1.0),
        ("calibration_session_ids", ["ok", 1]),
        ("final_e2e_session_ids", "session"),
        ("item_confusion_matrix", ((1.0,),)),
        ("enemy_confusion_matrix", ((1.0,),)),
        ("calibration_session_ids", ("session",)),
    ],
)
def test_profile_rejects_wrong_types_nonfinite_and_out_of_range(field, value):
    """全 scalar/list field の型・有限性・範囲違反を拒否する。

    bool を整数として受理したり、NaN を後段へ流したりしない
    fail-closed な入力契約を一つずつ反例で確認します。
    """
    data = _clean_wire()
    data[field] = value
    with pytest.raises(ValueError):
        PerceptionErrorProfile.from_wire(data)


@pytest.mark.parametrize(
    "matrix",
    [
        [[0.5, 0.6], [0.0, 1.0]],
        [[0.5], [0.5, 0.5]],
        [[0.5, -0.1], [0.0, 1.0]],
        [[0.5, float("nan")], [0.0, 1.0]],
        [[0.5, True], [0.0, 1.0]],
        "not-a-matrix",
    ],
)
def test_confusion_matrices_are_square_finite_probabilities(matrix):
    """confusion matrix を有限な正方確率行列へ制限する。

    各行の確率和が 1 以下でなければ、残余確率を「元カテゴリ維持」
    として安全に解釈できないためロード時点で拒否します。
    """
    data = _clean_wire()
    data["item_confusion_matrix"] = matrix
    with pytest.raises(ValueError):
        PerceptionErrorProfile.from_wire(data)


def test_profile_rejects_unknown_missing_version_and_session_overlap():
    """未知 field・version 不一致・calibration/E2E 重複を拒否する。

    typo や将来 schema の黙認に加え、最終評価 session を calibration に
    混ぜて精度を過大評価するデータリークも fail closed にします。
    """
    unknown = _clean_wire()
    unknown["future_field"] = 1
    with pytest.raises(ValueError):
        PerceptionErrorProfile.from_wire(unknown)

    non_string_key = _clean_wire()
    non_string_key[1] = "invalid"
    with pytest.raises(ValueError):
        PerceptionErrorProfile.from_wire(non_string_key)

    wrong_version = _clean_wire()
    wrong_version["schema_version"] = "perception_error.v2"
    with pytest.raises(ValueError):
        PerceptionErrorProfile.from_wire(wrong_version)

    overlap = _clean_wire()
    overlap["calibration_session_ids"] = ["cal-1", "shared"]
    overlap["final_e2e_session_ids"] = ["shared", "eval-2"]
    with pytest.raises(ValueError):
        PerceptionErrorProfile.from_wire(overlap)


def test_profile_requires_all_wire_fields_and_unique_session_ids():
    """欠落 field と重複 session id を曖昧に補完しない。

    manifest の意味を schema version ごとに完全固定し、同じ session を
    重複カウントする設定も入力境界で止めます。
    """
    missing = _clean_wire()
    del missing["coord_noise_std"]
    with pytest.raises(ValueError):
        PerceptionErrorProfile.from_wire(missing)

    duplicate = _clean_wire()
    duplicate["calibration_session_ids"] = ["cal-1", "cal-1"]
    with pytest.raises(ValueError):
        PerceptionErrorProfile.from_wire(duplicate)

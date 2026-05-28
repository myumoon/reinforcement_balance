"""Survivors ゲームパラメーター難易度スコア算出ユーティリティ。

CurriculumCallback・SpalfCallback 両方から使用できる汎用関数を提供する。
"""
from __future__ import annotations

# パラメーター範囲: (min, max) のタプル。SPALF の探索範囲かつ difficulty_score の正規化基準。
# min: PHASES[0]（入門）の値 → difficulty_score = 0.0 相当
# max: PHASES[-1]（Mad Forest）の値 → difficulty_score = 1.0 相当
# PHASES を変更した場合はここも合わせて更新すること。
# spalf_callback.py がこのテーブルを _PARAM_BOUNDS としてインポートするため、
# 両ファイルで探索範囲・スコア計算範囲が常に一致する。
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "min_enemies":        ( 4.0,  40.0),
    "max_enemies":        ( 6.0, 150.0),
    "speed_mult":         ( 0.8,   1.2),
    "spawn_rate_mult":    ( 1.0,   4.0),
    "max_enemy_type_id":  ( 1.0,  10.0),
    "enemy_hp_scale":     ( 0.50,  2.00),
    "enemy_damage_scale": ( 0.50,  2.00),
    "time_scaling":       ( 0.0,   1.0),
}

_PARAM_DIFFICULTY_MIN: dict[str, float] = {k: v[0] for k, v in PARAM_BOUNDS.items()}
_PARAM_DIFFICULTY_MAX: dict[str, float] = {k: v[1] for k, v in PARAM_BOUNDS.items()}

# 難易度スコアの重み（正規化後の加重平均に使用）。
PARAM_DIFFICULTY_WEIGHTS: dict[str, float] = {
    "min_enemies":        1.0,
    "max_enemies":        1.0,
    "speed_mult":         1.5,  # 速度は回避難易度に直結
    "spawn_rate_mult":    1.0,
    "max_enemy_type_id":  2.0,  # 敵種の多様性は難易度を大きく左右
    "enemy_hp_scale":     2.0,  # TTK が長くなる → 生存難易度大
    "enemy_damage_scale": 1.0,
    "time_scaling":       1.0,
}


def compute_difficulty_score(params: dict) -> float:
    """ゲームパラメーターから難易度スコアを計算する。

    PHASES[0]（入門）= 0.0、PHASES[-1]（Mad Forest）= 1.0 を基準とする。
    Mad Forest を超える設定（SPALF 探索時など）は 1.0 以上の値になる。

    Args:
        params: SurvivorsUE5Env に渡すパラメーター dict。
                time_scaling は bool / float どちらでも受け付ける。

    Returns:
        重み付き正規化難易度スコア（カリキュラム範囲内なら概ね 0.0〜1.0）。
    """
    weighted_sum = 0.0
    total_weight = 0.0
    for key, (min_val, max_val) in PARAM_BOUNDS.items():
        if max_val <= min_val:
            continue
        val = params.get(key, min_val)
        if isinstance(val, bool):
            val = 1.0 if val else 0.0
        norm = (float(val) - min_val) / (max_val - min_val)
        weight = PARAM_DIFFICULTY_WEIGHTS.get(key, 1.0)
        weighted_sum += weight * norm
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0.0 else 0.0

"""共通ユーティリティ。"""


def _linear_schedule(initial_value: float):
    """PPO 学習率の線形減衰スケジュール（訓練終了時に 0 になる）。"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

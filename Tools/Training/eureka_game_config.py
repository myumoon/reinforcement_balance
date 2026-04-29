"""EUREKA ゲーム設定の抽象基底クラス。"""

from abc import ABC, abstractmethod


def _linear_schedule(initial_value: float):
    """PPO 学習率の線形減衰スケジュール（訓練終了時に 0 になる）。"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class EurekaGameConfig(ABC):
    """ゲーム固有の処理を外部から注入するためのインターフェース。

    新しいゲームに対応するには、このクラスを継承した具体クラスを作成し
    モジュール末尾に `create_config()` ファクトリ関数を定義する。
    """

    def setup(self, host: str, port: int) -> None:
        """サーバー接続・obs_schema 取得など初期化処理（任意 override）。"""
        pass

    @abstractmethod
    def make_env(self, host: str, port: int):
        """gymnasium 環境を生成して返す。"""
        ...

    @abstractmethod
    def build_prompt(self, prev_metrics: dict | None, iteration: int) -> str:
        """LLM へのプロンプト文字列を生成して返す。"""
        ...

    def compute_primary_metric(self, episode_base_rewards: list[float],
                               episode_lengths: list[int]) -> float:
        """最良イテレーション判定に使うスカラー値（大きいほど良い）。

        デフォルト実装: episode_base_rewards の平均値を返す。
        ゲーム固有の指標がある場合は override する。
        """
        if not episode_base_rewards:
            return 0.0
        return sum(episode_base_rewards) / len(episode_base_rewards)

    def compute_extra_metrics(self, episode_base_rewards: list[float],
                              episode_lengths: list[int]) -> dict:
        """プロンプトに追加するゲーム固有の補助メトリクスを返す。

        デフォルト実装は空 dict。ゲーム固有の指標がある場合は override する。
        """
        return {}

    def build_game_context(self) -> str:
        """レビュープロンプト用のゲームコンテキストを返す（任意 override）。

        ゲームルール・物理定数・固定報酬など reward_fn 設計に必要な情報を返す。
        build_prompt() のメトリクス・タスク指示セクションは含めない。
        """
        return ""

    def make_model(self, env):
        """PPO モデルを生成して返す。

        デフォルト実装: MlpPolicy [64, 64] + 改善済みハイパーパラメータ。
        カスタムネットワーク（エンティティアテンションなど）を使う場合は override する。
        """
        from stable_baselines3 import PPO
        return PPO(
            "MlpPolicy", env,
            learning_rate=_linear_schedule(3e-4),
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
        )

    @property
    def primary_metric_name(self) -> str:
        """compute_primary_metric が返す指標の名前（ログ表示用）。"""
        return "primary_metric"

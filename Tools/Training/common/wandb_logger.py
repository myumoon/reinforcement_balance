"""W&B ログ操作を担当するクラス。

train.py で1インスタンス生成して各コールバックに注入する。
ロガー側は Curriculum / SPALF 等の特定アルゴリズムを知らない設計。
"""

from __future__ import annotations


class WandbLogger:
    """W&B ログ操作を担当するクラス。

    train.py で生成し、各コールバックのコンストラクタで受け取る。
    生成時は global_step のみ設定される。Curriculum/SPALF 等は add_metric_prefix() で
    自分のメトリクスグループを後から追加登録する（ロガーはアルゴリズムを知らない）。

    使い方:
        # train.py で生成
        logger = WandbLogger(enabled=args.wandb)
        # wandb.init() 後に setup()
        logger.setup()

        # コールバックで登録・使用
        logger.add_metric_prefix("survivors/")
        logger.add_metric_prefix("spalf/")
        logger.log({"survivors/active_score": 1234}, step=num_timesteps)
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._ready = False

    def setup(self) -> None:
        """wandb.init() の後に呼ぶ。global_step を step metric として定義する。"""
        if not self._enabled:
            return
        try:
            import wandb
            if not wandb.run:
                return
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")
            self._ready = True
        except ImportError:
            pass

    def add_metric_prefix(self, prefix: str) -> None:
        """外部から追加のメトリクスグループを登録する。

        コールバックが有効になったタイミングで呼ぶ。
        例: "survivors/" "spalf/" "curriculum/" "eval/" "expl/"

        ロガー側は prefix の意味を知らない（疎結合）。
        """
        if not self._ready:
            return
        try:
            import wandb
            if wandb.run:
                wandb.define_metric(f"{prefix}*", step_metric="global_step")
        except ImportError:
            pass

    def log(self, metrics: dict, step: int) -> None:
        """global_step を付与して wandb.log を呼ぶ。"""
        if not self._ready:
            return
        try:
            import wandb
            if wandb.run:
                wandb.log({**metrics, "global_step": step}, step=step)
        except ImportError:
            pass

    @property
    def enabled(self) -> bool:
        return self._enabled and self._ready

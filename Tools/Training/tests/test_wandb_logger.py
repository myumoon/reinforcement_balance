"""WandbLogger のユニットテスト。"""
import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from common.wandb_logger import WandbLogger


class TestWandbLogger:
    def test_disabled_logger_does_nothing(self):
        logger = WandbLogger(enabled=False)
        logger.setup()
        logger.add_metric_prefix("test/")
        logger.log({"test/metric": 1.0}, step=100)
        assert logger.enabled is False

    def test_add_metric_prefix_before_setup(self):
        logger = WandbLogger(enabled=True)
        # setup() を呼ばずに add_metric_prefix() を呼んでもエラーにならないこと
        logger.add_metric_prefix("test/")
        assert logger._ready is False

    def test_log_when_not_ready(self):
        logger = WandbLogger(enabled=True)
        # setup() を呼ばずに log() を呼んでもエラーにならないこと
        logger.log({"metric": 1.0}, step=100)
        assert logger._ready is False

    def test_setup_without_wandb_run(self):
        logger = WandbLogger(enabled=True)
        # wandb.run が None のとき setup() は _ready=True にならない
        logger.setup()
        assert logger._ready is False

    def test_enabled_property(self):
        logger_off = WandbLogger(enabled=False)
        assert logger_off.enabled is False
        logger_on = WandbLogger(enabled=True)
        assert logger_on.enabled is False  # _ready=False のため

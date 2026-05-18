"""_parse_step_shorthand / _parse_resume_spec の単体テスト。"""
import sys
from pathlib import Path

import pytest

# train.py は Tools/Training/ 直下にあるため、そこをパスに追加する
_TRAINING_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_TRAINING_DIR))

from train import _parse_step_shorthand, _parse_resume_spec


class TestParseStepShorthand:
    def test_plain_int(self):
        assert _parse_step_shorthand("200000") == 200_000
    def test_k_lower(self):
        assert _parse_step_shorthand("200k") == 200_000
    def test_k_upper(self):
        assert _parse_step_shorthand("200K") == 200_000
    def test_m_lower(self):
        assert _parse_step_shorthand("2m") == 2_000_000
    def test_m_upper(self):
        assert _parse_step_shorthand("2M") == 2_000_000
    def test_fractional_m(self):
        assert _parse_step_shorthand("2.5M") == 2_500_000
    def test_fractional_k(self):
        assert _parse_step_shorthand("1.5k") == 1_500
    def test_underscore_k(self):
        assert _parse_step_shorthand("2_000k") == 2_000_000
    def test_underscore_plain(self):
        assert _parse_step_shorthand("200_000") == 200_000
    def test_underscore_large(self):
        assert _parse_step_shorthand("2_000_000") == 2_000_000
    def test_zero(self):
        assert _parse_step_shorthand("0") == 0
    def test_small_k(self):
        assert _parse_step_shorthand("50k") == 50_000
    def test_invalid_raises(self):
        import pytest
        with pytest.raises((ValueError, TypeError)):
            _parse_step_shorthand("abc")
    def test_invalid_float_no_suffix_raises(self):
        import pytest
        with pytest.raises((ValueError, TypeError)):
            _parse_step_shorthand("2.5")


class TestParseResumeSpec:
    def test_path_only(self):
        path, step = _parse_resume_spec("runs/survivors/v06/train/run-base")
        assert path == Path("runs/survivors/v06/train/run-base")
        assert step is None
    def test_path_with_step_k(self):
        path, step = _parse_resume_spec("runs/survivors/v06/train/run-base@200k")
        assert path == Path("runs/survivors/v06/train/run-base")
        assert step == 200_000
    def test_path_with_step_M(self):
        path, step = _parse_resume_spec("runs/survivors/v06/train/run-base@2M")
        assert path == Path("runs/survivors/v06/train/run-base")
        assert step == 2_000_000
    def test_path_with_step_fractional_M(self):
        path, step = _parse_resume_spec("runs/survivors/v06/train/run-base@2.5M")
        assert path == Path("runs/survivors/v06/train/run-base")
        assert step == 2_500_000
    def test_path_with_step_plain(self):
        path, step = _parse_resume_spec("runs/survivors/v06/train/run-base@2000000")
        assert path == Path("runs/survivors/v06/train/run-base")
        assert step == 2_000_000
    def test_path_with_underscore_step(self):
        path, step = _parse_resume_spec("runs/survivors/v06/train/run-base@2_000k")
        assert path == Path("runs/survivors/v06/train/run-base")
        assert step == 2_000_000
    def test_last_at_used_for_split(self):
        path, step = _parse_resume_spec("runs/@special/train/run-base@500k")
        assert path == Path("runs/@special/train/run-base")
        assert step == 500_000
    def test_zero_step_raises_value_error(self):
        import pytest
        with pytest.raises(ValueError):
            _parse_resume_spec("runs/survivors/v06/train/run-base@0")
    def test_negative_step_raises_value_error(self):
        import pytest
        with pytest.raises(ValueError):
            _parse_resume_spec("runs/survivors/v06/train/run-base@-100k")
    def test_absolute_path(self):
        path, step = _parse_resume_spec("/home/user/runs/run-base@1M")
        assert path == Path("/home/user/runs/run-base")
        assert step == 1_000_000
    def test_absolute_path_no_step(self):
        path, step = _parse_resume_spec("/home/user/runs/run-base")
        assert path == Path("/home/user/runs/run-base")
        assert step is None

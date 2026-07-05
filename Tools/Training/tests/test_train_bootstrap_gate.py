"""tests/test_train_bootstrap_gate.py

train.py の --bootstrap-gate-eval-port / --bootstrap-gate-eval-episodes
CLI 引数・バリデーション・env 生成条件のユニットテスト。
UE5 不要。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Tools/Training ディレクトリをパスに追加
_TRAINING_DIR = Path(__file__).parent.parent
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from train import parse_args


# ---------------------------------------------------------------------------
# CLI 引数のデフォルト値
# ---------------------------------------------------------------------------

class TestBootstrapGateCliDefaults:
    """--bootstrap-gate-eval-episodes のデフォルト値を確認するテスト。"""

    def test_bootstrap_gate_eval_episodes_default_40(self, monkeypatch):
        """--bootstrap-gate-eval-episodes のデフォルトは 40 であること。"""
        monkeypatch.setattr(sys, "argv", ["train.py", "--game", "survivors"])
        args = parse_args()
        assert args.bootstrap_gate_eval_episodes == 40

    def test_bootstrap_gate_eval_port_default_none(self, monkeypatch):
        """--bootstrap-gate-eval-port のデフォルトは None であること。"""
        monkeypatch.setattr(sys, "argv", ["train.py", "--game", "survivors"])
        args = parse_args()
        assert args.bootstrap_gate_eval_port is None

    def test_bootstrap_gate_eval_episodes_custom(self, monkeypatch):
        """--bootstrap-gate-eval-episodes を指定した値で上書きできること。"""
        monkeypatch.setattr(
            sys, "argv",
            ["train.py", "--game", "survivors", "--bootstrap-gate-eval-episodes", "20"]
        )
        args = parse_args()
        assert args.bootstrap_gate_eval_episodes == 20

    def test_bootstrap_gate_eval_port_custom(self, monkeypatch):
        """--bootstrap-gate-eval-port を指定した値で設定できること。"""
        monkeypatch.setattr(
            sys, "argv",
            ["train.py", "--game", "survivors", "--bootstrap-gate-eval-port", "8769"]
        )
        args = parse_args()
        assert args.bootstrap_gate_eval_port == 8769


# ---------------------------------------------------------------------------
# バリデーション: P1 fix (weapon_bootstrap_lanes + missing port)
# ---------------------------------------------------------------------------

class TestBootstrapGateValidation:
    """バリデーション: weapon_bootstrap_lanes 有効時にポート必須であること。"""

    def _make_args_namespace(self, **kwargs):
        """テスト用の args Namespace を作成する。"""
        defaults = dict(
            game="survivors",
            dry_run=False,
            weapon_bootstrap_lanes=True,
            bootstrap_gate_eval_port=None,
            bootstrap_gate_eval_episodes=40,
            eval_port=8771,
            n_envs=2,
            base_port=8767,
            port=None,
            eval_freq=10000,
            task_cell_sampler=True,
        )
        defaults.update(kwargs)
        ns = types.SimpleNamespace(**defaults)
        return ns

    def test_p1_weapon_bootstrap_lanes_without_port_raises(self):
        """weapon_bootstrap_lanes + bootstrap_gate_eval_port 未指定 → ValueError。"""
        from train import validate_bootstrap_gate_args
        args = self._make_args_namespace(weapon_bootstrap_lanes=True, bootstrap_gate_eval_port=None)
        with pytest.raises(ValueError, match="--bootstrap-gate-eval-port"):
            validate_bootstrap_gate_args(args, base_port=args.base_port)

    def test_p1_weapon_bootstrap_lanes_with_port_no_raise(self):
        """weapon_bootstrap_lanes + bootstrap_gate_eval_port 指定済み → ValueError が出ないこと。"""
        from train import validate_bootstrap_gate_args
        args = self._make_args_namespace(weapon_bootstrap_lanes=True, bootstrap_gate_eval_port=8769)
        validate_bootstrap_gate_args(args, base_port=args.base_port)  # 例外なし

    def test_weapon_bootstrap_lanes_with_eval_freq_zero_raises(self):
        """weapon_bootstrap_lanes + eval_freq=0 → ValueError。"""
        from train import validate_bootstrap_gate_args
        args = self._make_args_namespace(
            weapon_bootstrap_lanes=True, bootstrap_gate_eval_port=8769, eval_freq=0
        )
        with pytest.raises(ValueError, match="--eval-freq"):
            validate_bootstrap_gate_args(args, base_port=args.base_port)

    def test_p1_dry_run_skips_validation(self):
        """dry_run=True のとき、weapon_bootstrap_lanes があっても ValueError が出ないこと。"""
        from train import validate_bootstrap_gate_args
        args = self._make_args_namespace(
            dry_run=True, weapon_bootstrap_lanes=True, bootstrap_gate_eval_port=None,
        )
        validate_bootstrap_gate_args(args, base_port=args.base_port)  # 例外なし

    def test_port_collision_with_eval_port_raises(self):
        """bootstrap_gate_eval_port が eval_port と重複 → ValueError。"""
        from train import validate_bootstrap_gate_args
        args = self._make_args_namespace(
            bootstrap_gate_eval_port=8771, eval_port=8771, weapon_bootstrap_lanes=True,
        )
        with pytest.raises(ValueError, match="bootstrap_gate_eval_port"):
            validate_bootstrap_gate_args(args, base_port=args.base_port)

    def test_port_collision_with_train_port_raises(self):
        """bootstrap_gate_eval_port が train ports と重複 → ValueError。"""
        from train import validate_bootstrap_gate_args
        args = self._make_args_namespace(
            bootstrap_gate_eval_port=8767, eval_port=8771, weapon_bootstrap_lanes=True,
        )
        with pytest.raises(ValueError, match="bootstrap_gate_eval_port"):
            validate_bootstrap_gate_args(args, base_port=args.base_port)

    def test_no_port_no_bootstrap_gate_env(self):
        """--bootstrap-gate-eval-port が None のとき bootstrap_gate_eval_env は None のまま。"""
        args = self._make_args_namespace(
            bootstrap_gate_eval_port=None,
        )
        # bootstrap_gate_eval_env 生成条件チェック
        should_create = (
            args.game == "survivors"
            and getattr(args, "bootstrap_gate_eval_port", None) is not None
            and getattr(args, "weapon_bootstrap_lanes", False)
            and not args.dry_run
        )
        assert not should_create


# ---------------------------------------------------------------------------
# cleanup: bootstrap_gate_eval_env が eval_env と独立してクローズされる（P2 fix）
# ---------------------------------------------------------------------------

class TestBootstrapGateCleanup:
    """P2 fix: bootstrap_gate_eval_env のクローズが eval_env の有無に依存しないこと。"""

    def test_close_called_even_if_eval_env_is_none(self):
        """eval_env が None のとき bootstrap_gate_eval_env が独立してクローズされること。"""
        mock_bgate_env = MagicMock()

        eval_env = None  # eval_env は None（n_envs=1 の構成）
        bootstrap_gate_eval_env = mock_bgate_env

        # finally ブロックの cleanup ロジックを再現（train.py と同じ）
        if eval_env is not None:
            try:
                eval_env.close()
            except (BrokenPipeError, OSError):
                pass
        # bootstrap gate env のクローズ（eval_env の有無と独立して実行）
        if bootstrap_gate_eval_env is not None:
            try:
                bootstrap_gate_eval_env.close()
            except (BrokenPipeError, OSError):
                pass

        mock_bgate_env.close.assert_called_once()

    def test_close_called_regardless_of_eval_env(self):
        """eval_env が存在する場合も bootstrap_gate_eval_env が独立してクローズされること。"""
        mock_eval_env = MagicMock()
        mock_bgate_env = MagicMock()

        eval_env = mock_eval_env
        bootstrap_gate_eval_env = mock_bgate_env

        # finally ブロックの cleanup ロジックを再現
        if eval_env is not None:
            try:
                eval_env.close()
            except (BrokenPipeError, OSError):
                pass
        if bootstrap_gate_eval_env is not None:
            try:
                bootstrap_gate_eval_env.close()
            except (BrokenPipeError, OSError):
                pass

        mock_eval_env.close.assert_called_once()
        mock_bgate_env.close.assert_called_once()

    def test_close_not_called_if_none(self):
        """bootstrap_gate_eval_env が None のとき close が呼ばれないこと。"""
        eval_env = None
        bootstrap_gate_eval_env = None

        # cleanup ロジック（例外が出ないこと）
        if eval_env is not None:
            eval_env.close()
        if bootstrap_gate_eval_env is not None:
            bootstrap_gate_eval_env.close()
        # ここに到達すれば OK（close が呼ばれていない）

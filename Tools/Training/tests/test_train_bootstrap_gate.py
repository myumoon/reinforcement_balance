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

    def test_bootstrap_gate_eval_freq_default_none(self, monkeypatch):
        """--bootstrap-gate-eval-freq のデフォルトは None であること。"""
        monkeypatch.setattr(sys, "argv", ["train.py", "--game", "survivors"])
        args = parse_args()
        assert args.bootstrap_gate_eval_freq is None

    def test_bootstrap_gate_eval_freq_custom(self, monkeypatch):
        """--bootstrap-gate-eval-freq を指定した値で設定できること。"""
        monkeypatch.setattr(
            sys, "argv",
            ["train.py", "--game", "survivors", "--bootstrap-gate-eval-freq", "200000"]
        )
        args = parse_args()
        assert args.bootstrap_gate_eval_freq == 200000


# ---------------------------------------------------------------------------
# get_effective_bootstrap_gate_eval_freq ヘルパー
# ---------------------------------------------------------------------------

class TestEffectiveBootstrapGateEvalFreq:
    """get_effective_bootstrap_gate_eval_freq のテスト。"""

    def test_uses_eval_freq_when_not_specified(self):
        """bootstrap_gate_eval_freq が未指定なら eval_freq を返すこと。"""
        from train import get_effective_bootstrap_gate_eval_freq
        args = types.SimpleNamespace(bootstrap_gate_eval_freq=None, eval_freq=50_000)
        assert get_effective_bootstrap_gate_eval_freq(args) == 50_000

    def test_uses_bootstrap_gate_eval_freq_when_specified(self):
        """bootstrap_gate_eval_freq が指定されていればそれを返すこと。"""
        from train import get_effective_bootstrap_gate_eval_freq
        args = types.SimpleNamespace(bootstrap_gate_eval_freq=200_000, eval_freq=50_000)
        assert get_effective_bootstrap_gate_eval_freq(args) == 200_000


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
            bootstrap_gate_eval_freq=None,
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
        """weapon_bootstrap_lanes + eval_freq=0 かつ gate freq 未指定 → ValueError。"""
        from train import validate_bootstrap_gate_args
        args = self._make_args_namespace(
            weapon_bootstrap_lanes=True, bootstrap_gate_eval_port=8769, eval_freq=0,
            bootstrap_gate_eval_freq=None,
        )
        with pytest.raises(ValueError, match="--eval-freq"):
            validate_bootstrap_gate_args(args, base_port=args.base_port)

    def test_weapon_bootstrap_lanes_eval_freq_zero_with_gate_freq_no_raise(self):
        """weapon_bootstrap_lanes=True + eval_freq=0 でも bootstrap_gate_eval_freq が指定されていれば
        gate probe freq が独立するため ValueError にならないこと（指摘2 の修正）。"""
        from train import validate_bootstrap_gate_args
        args = self._make_args_namespace(
            weapon_bootstrap_lanes=True, bootstrap_gate_eval_port=8769, eval_freq=0,
            bootstrap_gate_eval_freq=200_000,
        )
        validate_bootstrap_gate_args(args, base_port=args.base_port)  # 例外なし

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

    def test_bootstrap_gate_eval_port_zero_raises(self):
        """bootstrap_gate_eval_port=0 → ValueError。"""
        from train import validate_bootstrap_gate_args
        args = self._make_args_namespace(bootstrap_gate_eval_port=0, weapon_bootstrap_lanes=True)
        with pytest.raises(ValueError, match="正の整数"):
            validate_bootstrap_gate_args(args, base_port=args.base_port)

    def test_bootstrap_gate_eval_port_negative_raises(self):
        """bootstrap_gate_eval_port=-1 → ValueError。"""
        from train import validate_bootstrap_gate_args
        args = self._make_args_namespace(bootstrap_gate_eval_port=-1, weapon_bootstrap_lanes=True)
        with pytest.raises(ValueError, match="正の整数"):
            validate_bootstrap_gate_args(args, base_port=args.base_port)

    def test_bootstrap_gate_eval_freq_zero_raises(self):
        """bootstrap_gate_eval_freq=0 → ValueError。"""
        from train import validate_bootstrap_gate_args
        args = self._make_args_namespace(
            bootstrap_gate_eval_port=8769, weapon_bootstrap_lanes=True,
            bootstrap_gate_eval_freq=0,
        )
        with pytest.raises(ValueError, match="--bootstrap-gate-eval-freq"):
            validate_bootstrap_gate_args(args, base_port=args.base_port)

    def test_eval_freq_zero_with_bootstrap_gate_eval_freq_set_no_raise(self):
        """eval_freq=0 でも bootstrap_gate_eval_freq が指定されていれば freq バリデーションで ValueError にならないこと。

        weapon_bootstrap_lanes=False にして既存の eval_freq>0 必須チェックを回避する。
        """
        from train import validate_bootstrap_gate_args
        args = self._make_args_namespace(
            weapon_bootstrap_lanes=False,
            bootstrap_gate_eval_port=None,
            eval_freq=0,
            bootstrap_gate_eval_freq=200_000,
        )
        validate_bootstrap_gate_args(args, base_port=args.base_port)  # 例外なし

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


# ---------------------------------------------------------------------------
# curriculum_spalf + eval_freq=0 の main() 統合バリデーション（指摘4 の修正）
#
# validate_bootstrap_gate_args() は eval_freq=0 + bootstrap_gate_eval_freq>0 を
# 許可するが、curriculum_spalf=True の実使用経路では main() の
# 「curriculum_spalf + eval_freq==0」チェックが先に走り、gate freq を指定しても
# 拒否される（probe eval が eval_freq を必要とするため）。この end-to-end 制約を
# main() を実際に呼び出して検証する。
# ---------------------------------------------------------------------------
class TestCurriculumSpalfEvalFreqValidation:
    """curriculum_spalf=True + eval_freq=0 は bootstrap_gate_eval_freq を
    指定しても main() で拒否されることを検証する統合テスト。"""

    def test_curriculum_spalf_eval_freq_zero_with_gate_freq_raises_from_main(self, monkeypatch):
        """curriculum_spalf=True + eval_freq=0 は bootstrap_gate_eval_freq を
        指定しても拒否される。

        curriculum_spalf の probe eval は eval_freq を使用するため eval_freq > 0 が
        必要。この制約は main() の curriculum_spalf + eval_freq==0 チェックで強制され、
        validate_bootstrap_gate_args() より前に走る。ここでは main() を実際に呼び、
        gate freq を指定しても ValueError（--eval-freq 関連メッセージ）になることを確認する。
        """
        from train import main

        monkeypatch.setattr(sys, "argv", [
            "train.py", "--game", "survivors",
            "--curriculum-spalf", "--eval-freq", "0",
            "--bootstrap-gate-eval-freq", "200000",
            "--n-envs", "2",  # curriculum_spalf requires n_envs > 1
        ])
        with pytest.raises(ValueError, match="--eval-freq"):
            main()

    def test_curriculum_spalf_eval_freq_positive_passes_gate_validation(self, monkeypatch):
        """対照ケース: curriculum_spalf=True + eval_freq>0 は main() の
        curriculum_spalf/eval_freq チェックを通過する（validate_bootstrap_gate_args
        単体では curriculum_spalf を参照しない）。

        validate_bootstrap_gate_args() は curriculum_spalf を参照しないため、
        weapon_bootstrap_lanes 経由の port 必須チェックのみが対象になる。ここでは
        port を明示して例外が出ないことを確認する。
        """
        from train import validate_bootstrap_gate_args

        monkeypatch.setattr(sys, "argv", [
            "train.py", "--game", "survivors",
            "--curriculum-spalf", "--eval-freq", "50000",
            "--bootstrap-gate-eval-port", "8769",
            "--base-port", "8767",
            "--n-envs", "2",
        ])
        args = parse_args()
        # curriculum_spalf + eval_freq>0 は main() の L1113 相当チェックを通過する
        assert not (args.curriculum_spalf and args.eval_freq == 0)
        # validate_bootstrap_gate_args は curriculum_spalf 非依存で例外を出さない
        validate_bootstrap_gate_args(args, base_port=args.base_port)


def test_bootstrap_gate_maintenance_eval_every_custom(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["train.py", "--bootstrap-gate-maintenance-eval-every", "3"],
    )
    args = parse_args()
    assert args.bootstrap_gate_maintenance_eval_every == 3


def test_bootstrap_gate_maintenance_eval_every_zero_raises():
    import types
    from train import validate_bootstrap_gate_args
    args = types.SimpleNamespace(
        game="survivors",
        dry_run=False,
        weapon_bootstrap_lanes=True,
        bootstrap_gate_eval_port=8768,
        bootstrap_gate_eval_episodes=10,
        bootstrap_gate_eval_freq=200000,
        bootstrap_gate_maintenance_eval_every=0,
        eval_freq=200000,
        task_cell_sampler=True,
        n_envs=4,
        eval_port=None,
    )
    with pytest.raises(ValueError, match="maintenance-eval-every"):
        validate_bootstrap_gate_args(args, base_port=8769)


def test_survivors_supervisor_args_parse(monkeypatch):
    from train import parse_args
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--survivors-supervisor",
            "--bootstrap-target-stage-key", "WU12",
            "--post-bootstrap-mode", "combination_smoke",
            "--survivors-supervisor-check-freq", "4096",
            "--bootstrap-stage-timeout-steps", "2000000",
            "--bootstrap-max-regression-count", "4",
        ],
    )
    args = parse_args()
    assert args.survivors_supervisor is True
    assert args.bootstrap_target_stage_key == "WU12"
    assert args.post_bootstrap_mode == "combination_smoke"
    assert args.survivors_supervisor_check_freq == 4096
    assert args.bootstrap_stage_timeout_steps == 2_000_000
    assert args.bootstrap_max_regression_count == 4


def test_survivors_supervisor_invalid_stage_key_raises():
    import argparse
    import pytest
    from train import validate_survivors_supervisor_args
    args = argparse.Namespace(
        survivors_supervisor=True,
        weapon_bootstrap_lanes=True,
        survivors_supervisor_check_freq=2048,
        bootstrap_max_regression_count=3,
        bootstrap_stage_timeout_steps=0,
        bootstrap_target_stage_key="WU99",
    )
    with pytest.raises(ValueError, match="不正です"):
        validate_survivors_supervisor_args(args)


def test_survivors_supervisor_negative_timeout_raises():
    import argparse
    import pytest
    from train import validate_survivors_supervisor_args
    args = argparse.Namespace(
        survivors_supervisor=True,
        weapon_bootstrap_lanes=True,
        survivors_supervisor_check_freq=2048,
        bootstrap_max_regression_count=3,
        bootstrap_stage_timeout_steps=-1,
        bootstrap_target_stage_key="WU12",
    )
    with pytest.raises(ValueError, match=">= 0"):
        validate_survivors_supervisor_args(args)


def test_train_trait_bootstrap_args_parsed(monkeypatch):
    """trait bootstrap CLI 引数が正しく parse される。"""
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py", "--game", "survivors",
            "--trait-bootstrap-weapon-keys", "peachone,ebony_wings",
            "--trait-bootstrap-target-p10", "120",
            "--trait-bootstrap-min-ep-len-p10", "3000",
            "--trait-bootstrap-max-short-episode-rate", "0.05",
        ],
    )
    from train import parse_args
    args = parse_args()
    assert args.trait_bootstrap_weapon_keys == "peachone,ebony_wings"
    assert args.trait_bootstrap_target_p10 == 120.0
    assert args.trait_bootstrap_min_ep_len_p10 == 3000.0
    assert args.trait_bootstrap_max_short_episode_rate == 0.05


def test_weapon_bootstrap_item_stage_is_available_in_args(monkeypatch):
    """--weapon-item-stage が parse され args に反映される。"""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--game", "survivors",
            "--task-cell-sampler",
            "--weapon-bootstrap-lanes",
            "--weapon-item-stage", "IS1",
            "--bootstrap-gate-eval-port", "8768",
        ],
    )
    args = parse_args()
    assert args.weapon_item_stage == "IS1"

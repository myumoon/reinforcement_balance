"""tests/survivors/test_bootstrap_gate_eval_callback.py

BootstrapGateEvalCallback のユニットテスト。
UE5 不要。eval_env / model / run_survivors_eval_episodes はモック。
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

_TRAINING_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

import pytest

import time

from games.survivors.bootstrap_gate_eval_callback import (
    BootstrapGateEvalCallback,
    BootstrapGateEvalTarget,
    BootstrapGateEvalResult,
)
from games.survivors.modules.weapon_bootstrap_module import (
    WeaponBootstrapStateModule,
    WeaponBootstrapState,
)
from games.survivors.survivors_weapon_table import WEAPON_UNLOCK_ORDER, WeaponEntry
from games.survivors.survivors_weapon_curriculum import WeaponType


# ---------------------------------------------------------------------------
# テスト用ヘルパー
# ---------------------------------------------------------------------------

def _make_module(
    initial_status: dict[str, str] | None = None,
    solo_bootstrap_target_p10: float = 300.0,
    integration_target_p10: float = 300.0,
) -> WeaponBootstrapStateModule:
    """テスト用の WeaponBootstrapStateModule を作成する。"""
    return WeaponBootstrapStateModule(
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        initial_status=initial_status,
        solo_bootstrap_target_p10=solo_bootstrap_target_p10,
        integration_target_p10=integration_target_p10,
    )


def _make_eval_env(training: bool = True) -> MagicMock:
    """モック eval_env を作成する。set_params は True を返す。

    venv=None を設定して _sync_vecnormalize() の venv チェーン走査で無限ループしないようにする。
    """
    env = MagicMock()
    env.training = training
    env.venv = None  # _sync_vecnormalize() の while ループを止める
    env.env_method.return_value = [True]  # set_params の成功を示す
    return env


def _make_model() -> MagicMock:
    """モック SB3 モデルを作成する。"""
    return MagicMock()


def _make_fake_ep_results(n: int = 3, active_score: float = 350.0) -> list[dict]:
    """ダミーのエピソード結果リストを作成する。"""
    return [
        {
            "active_score": active_score,
            "ep_length": 1500,
            "terminated": 0,
        }
        for _ in range(n)
    ]


def _make_callback(
    module: WeaponBootstrapStateModule,
    eval_env=None,
    probe_freq: int = 10_000,
    n_probe_episodes: int = 3,
    enemy_phase_idx: int = 2,
    stage_key_provider=None,
    wandb_logger=None,
    async_eval: bool = False,
) -> BootstrapGateEvalCallback:
    """テスト用の BootstrapGateEvalCallback を作成する。

    デフォルトは async_eval=False（同期）。同期経路のテストが多いため。
    非同期経路のテストは async_eval=True を明示的に渡す。
    """
    if eval_env is None:
        eval_env = _make_eval_env()
    return BootstrapGateEvalCallback(
        weapon_bootstrap_module=module,
        eval_env=eval_env,
        weapon_unlock_order=WEAPON_UNLOCK_ORDER,
        probe_freq=probe_freq,
        n_probe_episodes=n_probe_episodes,
        alive_reward=0.001,
        frame_skip=2,
        enemy_phase_idx=enemy_phase_idx,
        item_stage_key="IS0",
        stage_key_provider=stage_key_provider,
        async_eval=async_eval,
        wandb_logger=wandb_logger,
    )


def _setup_callback(cb: BootstrapGateEvalCallback, model=None, num_timesteps: int = 0):
    """コールバックを初期化して model / num_timesteps をセットする。

    BaseCallback の training_env / logger は property で self.model 経由でアクセスされるため、
    model.get_env() と model.logger が MagicMock を返すよう設定する。

    _sync_vecnormalize() が venv チェーンを辿る際に MagicMock で無限ループしないよう、
    training_env の venv を None に設定する。
    """
    if model is None:
        model = _make_model()
    # training_env は property -> model.get_env() を呼ぶ
    # venv=None にすることで _sync_vecnormalize() の while ループが終了する
    mock_training_env = MagicMock()
    mock_training_env.venv = None
    model.get_env = MagicMock(return_value=mock_training_env)
    # logger は property -> model.logger を返す
    model.logger = MagicMock()
    cb.model = model
    cb.num_timesteps = num_timesteps
    return cb


# ---------------------------------------------------------------------------
# テスト 1: 全ステータスの武器が 0 件のとき eval が走らない
# ---------------------------------------------------------------------------

def test_run_probe_no_targets_skips_eval():
    """solo_bootstrap / integration / maintenance の武器が 0 件のとき eval が走らない。"""
    module = _make_module()  # 全武器が locked 状態
    eval_env = _make_eval_env()
    cb = _make_callback(module, eval_env=eval_env)
    _setup_callback(cb)

    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes"
    ) as mock_run:
        cb._run_probe()
        mock_run.assert_not_called()

    eval_env.env_method.assert_not_called()


# ---------------------------------------------------------------------------
# テスト 2: solo_bootstrap 武器に対して cell_spec = "{key}:solo_bootstrap:2" で eval が走る
# ---------------------------------------------------------------------------

def test_eval_weapon_solo_bootstrap_uses_correct_cell_spec():
    """solo_bootstrap ステータスの武器に対して正しい cell_spec で eval が走ること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    eval_env = _make_eval_env()
    cb = _make_callback(module, eval_env=eval_env, enemy_phase_idx=2)
    _setup_callback(cb, num_timesteps=10_000)

    fake_results = _make_fake_ep_results(3, active_score=350.0)

    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes",
        return_value=(fake_results, {}, None),
    ) as mock_run:
        cb._run_probe()
        mock_run.assert_called_once()
        # module.set_deterministic_result が garlic の weapon_id で呼ばれること
        garlic_id = WeaponType.GARLIC
        state = module._states[garlic_id]
        assert state.deterministic_task_kind == "solo_bootstrap"
        assert state.deterministic_enemy_phase_idx == 2


# ---------------------------------------------------------------------------
# テスト 3: integration 武器に対して cell_spec = "{key}:integration:2" で eval が走る
# ---------------------------------------------------------------------------

def test_eval_weapon_integration_uses_correct_cell_spec():
    """integration ステータスの武器に対して正しい task_kind で eval が走ること。"""
    module = _make_module(initial_status={"garlic": "integration"})
    eval_env = _make_eval_env()
    cb = _make_callback(module, eval_env=eval_env, enemy_phase_idx=2)
    _setup_callback(cb, num_timesteps=20_000)

    fake_results = _make_fake_ep_results(3, active_score=310.0)

    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes",
        return_value=(fake_results, {}, None),
    ):
        cb._run_probe()

    garlic_id = WeaponType.GARLIC
    state = module._states[garlic_id]
    assert state.deterministic_task_kind == "integration"
    assert state.deterministic_enemy_phase_idx == 2
    assert state.deterministic_p10 is not None


# ---------------------------------------------------------------------------
# テスト 4: maintenance 武器に対して cell_spec = "{key}:maintenance:2" で eval が走る
# ---------------------------------------------------------------------------

def test_eval_weapon_maintenance_uses_correct_cell_spec():
    """maintenance ステータスの武器に対して正しい task_kind で eval が走ること。"""
    module = _make_module(initial_status={"garlic": "maintenance"})
    eval_env = _make_eval_env()
    cb = _make_callback(module, eval_env=eval_env, enemy_phase_idx=2)
    _setup_callback(cb, num_timesteps=30_000)

    fake_results = _make_fake_ep_results(3, active_score=320.0)

    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes",
        return_value=(fake_results, {}, None),
    ):
        cb._run_probe()

    garlic_id = WeaponType.GARLIC
    state = module._states[garlic_id]
    assert state.deterministic_task_kind == "maintenance"
    assert state.deterministic_enemy_phase_idx == 2


# ---------------------------------------------------------------------------
# テスト 5: 複数ステータスの複数武器がある場合、全武器分が走り全員 set_deterministic_result が呼ばれる
# ---------------------------------------------------------------------------

def test_run_probe_multiple_weapons_all_evaluated():
    """複数ステータスの複数武器がある場合、全武器分の eval が走ること。"""
    module = _make_module(
        initial_status={
            "garlic": "solo_bootstrap",
            "king_bible": "integration",
            "magic_wand": "maintenance",
        }
    )
    eval_env = _make_eval_env()
    cb = _make_callback(module, eval_env=eval_env)
    _setup_callback(cb, num_timesteps=50_000)

    fake_results = _make_fake_ep_results(3, active_score=300.0)

    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes",
        return_value=(fake_results, {}, None),
    ) as mock_run:
        cb._run_probe()

    # 3武器分の eval が実行されること
    assert mock_run.call_count == 3

    # 全武器の set_deterministic_result が呼ばれること（deterministic_p10 が None でないこと）
    for weapon_key in ["garlic", "king_bible", "magic_wand"]:
        entry = next(e for e in WEAPON_UNLOCK_ORDER if e.key == weapon_key)
        state = module._states[entry.weapon_id]
        assert state.deterministic_p10 is not None, f"{weapon_key} の deterministic_p10 が None"
        assert state.deterministic_task_kind is not None


# ---------------------------------------------------------------------------
# テスト 6: eval_env.training が _run_probe() の前後で復元される
# ---------------------------------------------------------------------------

def test_run_probe_restores_eval_env_training_flag():
    """_run_probe() の前後で eval_env.training が元の値に戻ること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    eval_env = _make_eval_env(training=True)
    cb = _make_callback(module, eval_env=eval_env)
    _setup_callback(cb, num_timesteps=10_000)

    fake_results = _make_fake_ep_results(3)

    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes",
        return_value=(fake_results, {}, None),
    ):
        cb._run_probe()

    # training フラグが True に戻ること
    assert eval_env.training is True


def test_run_probe_restores_training_flag_even_on_exception():
    """eval_weapon が例外を出しても eval_env.training が復元されること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    eval_env = _make_eval_env(training=True)
    cb = _make_callback(module, eval_env=eval_env)
    _setup_callback(cb, num_timesteps=10_000)

    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes",
        side_effect=RuntimeError("テスト用例外"),
    ):
        cb._run_probe()  # 例外が外に出ないこと

    # 例外後も training フラグが復元されること
    assert eval_env.training is True


# ---------------------------------------------------------------------------
# テスト 7: 特定武器の _eval_weapon() が例外を出しても他武器の eval が継続する
# ---------------------------------------------------------------------------

def test_run_probe_continues_on_single_weapon_exception():
    """1武器の eval が例外を出しても、他武器の eval が継続すること。"""
    module = _make_module(
        initial_status={
            "garlic": "solo_bootstrap",
            "king_bible": "solo_bootstrap",
        }
    )
    eval_env = _make_eval_env()
    cb = _make_callback(module, eval_env=eval_env)
    _setup_callback(cb, num_timesteps=10_000)

    fake_results = _make_fake_ep_results(3, active_score=350.0)
    call_count = 0

    def side_effect_first_fails(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("最初の武器で失敗")
        return (fake_results, {}, None)

    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes",
        side_effect=side_effect_first_fails,
    ):
        cb._run_probe()  # 例外が外に出ないこと

    # 2回試みられること（1回失敗、1回成功）
    assert call_count == 2

    # 成功した武器は set_deterministic_result が呼ばれること
    garlic_id = WeaponType.GARLIC
    king_bible_id = WeaponType.KING_BIBLE
    garlic_state = module._states[garlic_id]
    king_bible_state = module._states[king_bible_id]
    # どちらか一方が設定されていること
    assert (
        garlic_state.deterministic_p10 is not None
        or king_bible_state.deterministic_p10 is not None
    )


# ---------------------------------------------------------------------------
# テスト 8: probe_freq 未満のステップでは _run_probe() が呼ばれない
# ---------------------------------------------------------------------------

def test_on_step_probe_freq_not_reached():
    """probe_freq に達していない場合は _run_probe() が呼ばれないこと。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    cb = _make_callback(module, probe_freq=50_000)
    _setup_callback(cb, num_timesteps=0)

    with patch.object(cb, "_run_probe") as mock_probe:
        # 49_999 ステップ（freq 未満）
        cb.num_timesteps = 49_999
        cb._on_step()
        mock_probe.assert_not_called()


def test_on_step_probe_freq_reached():
    """probe_freq に達した場合は _run_probe() が呼ばれること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    cb = _make_callback(module, probe_freq=50_000)
    _setup_callback(cb, num_timesteps=0)

    with patch.object(cb, "_run_probe") as mock_probe:
        cb.num_timesteps = 50_000
        cb._on_step()
        mock_probe.assert_called_once()


def test_on_step_probe_freq_called_twice_at_double_freq():
    """probe_freq の 2 倍に達した場合は 2 回目の _run_probe() が呼ばれること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    cb = _make_callback(module, probe_freq=50_000)
    _setup_callback(cb, num_timesteps=0)

    with patch.object(cb, "_run_probe"):
        cb.num_timesteps = 50_000
        cb._on_step()
        cb.num_timesteps = 99_999
        with patch.object(cb, "_run_probe") as mock_probe2:
            cb._on_step()
            mock_probe2.assert_not_called()

        cb.num_timesteps = 100_000
        with patch.object(cb, "_run_probe") as mock_probe3:
            cb._on_step()
            mock_probe3.assert_called_once()


# ---------------------------------------------------------------------------
# テスト 9: stage_key_provider あり: stage_key_provider() が呼ばれ、返り値が build_eval_params の stage_key に渡される
# ---------------------------------------------------------------------------

def test_stage_key_provider_is_called_and_passed_to_build_eval_params():
    """stage_key_provider がある場合、それが呼ばれ stage_key として build_eval_params に渡されること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    eval_env = _make_eval_env()
    provider_return = "WU3"
    stage_key_provider = MagicMock(return_value=provider_return)

    cb = _make_callback(module, eval_env=eval_env, stage_key_provider=stage_key_provider)
    _setup_callback(cb, num_timesteps=10_000)

    fake_results = _make_fake_ep_results(3)

    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes",
        return_value=(fake_results, {}, None),
    ), patch(
        "games.survivors.bootstrap_gate_eval_callback.build_eval_params",
        wraps=__import__(
            "games.survivors.bootstrap_eval",
            fromlist=["build_eval_params"],
        ).build_eval_params,
    ) as mock_build:
        cb._run_probe()

    # stage_key_provider が呼ばれたこと
    stage_key_provider.assert_called()

    # build_eval_params が stage_key=provider_return で呼ばれたこと
    for call_args in mock_build.call_args_list:
        assert call_args.kwargs.get("stage_key") == provider_return


# ---------------------------------------------------------------------------
# テスト 10: stage_key_provider が None のとき entry.unlock_stage_key がフォールバックとして使われる
# ---------------------------------------------------------------------------

def test_stage_key_provider_none_uses_entry_unlock_stage_key():
    """stage_key_provider が None の場合、entry.unlock_stage_key がフォールバックとして使われ AttributeError が出ないこと。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    eval_env = _make_eval_env()

    cb = _make_callback(module, eval_env=eval_env, stage_key_provider=None)
    _setup_callback(cb, num_timesteps=10_000)

    garlic_entry = next(e for e in WEAPON_UNLOCK_ORDER if e.key == "garlic")
    expected_stage_key = garlic_entry.unlock_stage_key

    fake_results = _make_fake_ep_results(3)

    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes",
        return_value=(fake_results, {}, None),
    ), patch(
        "games.survivors.bootstrap_gate_eval_callback.build_eval_params",
        wraps=__import__(
            "games.survivors.bootstrap_eval",
            fromlist=["build_eval_params"],
        ).build_eval_params,
    ) as mock_build:
        # AttributeError が出ないこと
        cb._run_probe()

    # build_eval_params が entry.unlock_stage_key で呼ばれたこと
    for call_args in mock_build.call_args_list:
        assert call_args.kwargs.get("stage_key") == expected_stage_key


# ---------------------------------------------------------------------------
# テスト 11: set_params が False を返した場合 RuntimeError が発生する
# ---------------------------------------------------------------------------

def test_set_params_failure_raises_runtime_error():
    """set_params が失敗（False を返す）場合に RuntimeError が発生すること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    eval_env = _make_eval_env()
    eval_env.env_method.return_value = [False]  # set_params 失敗

    cb = _make_callback(module, eval_env=eval_env)
    _setup_callback(cb, num_timesteps=10_000)

    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes"
    ) as mock_run:
        # _eval_weapon 内で RuntimeError が発生すること
        with pytest.raises(RuntimeError, match="set_params"):
            cb._eval_weapon("garlic", WeaponType.GARLIC, "solo_bootstrap")

        # RuntimeError が発生したので eval は実行されないこと
        mock_run.assert_not_called()


def test_set_params_failure_is_caught_in_run_probe_and_eval_continues():
    """set_params 失敗（RuntimeError）が _run_probe() 内で握られ、他武器の eval が継続すること。"""
    module = _make_module(
        initial_status={
            "garlic": "solo_bootstrap",
            "king_bible": "solo_bootstrap",
        }
    )
    eval_env = _make_eval_env()

    # garlic の set_params は失敗、king_bible は成功
    garlic_id = WeaponType.GARLIC
    king_bible_id = WeaponType.KING_BIBLE

    set_params_call_count = 0

    def set_params_side_effect(*args, **kwargs):
        nonlocal set_params_call_count
        set_params_call_count += 1
        # 1回目の呼び出しは失敗、2回目は成功（武器の順序依存）
        return [set_params_call_count != 1]

    eval_env.env_method.side_effect = set_params_side_effect

    cb = _make_callback(module, eval_env=eval_env)
    _setup_callback(cb, num_timesteps=10_000)

    fake_results = _make_fake_ep_results(3, active_score=350.0)

    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes",
        return_value=(fake_results, {}, None),
    ) as mock_run:
        cb._run_probe()  # 例外が外に出ないこと

    # 2武器を試みること
    assert set_params_call_count == 2
    # 成功した1武器は eval が実行されること
    assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# Task 3: _apply_result / stale guard / set_params error isolation
# ---------------------------------------------------------------------------

def _make_target(
    weapon_key: str = "garlic",
    weapon_id: int = WeaponType.GARLIC,
    status: str = "solo_bootstrap",
    snapshot_step: int = 12345,
    snapshot_stage_key: str | None = None,
) -> BootstrapGateEvalTarget:
    return BootstrapGateEvalTarget(
        weapon_key=weapon_key,
        weapon_id=weapon_id,
        status=status,
        cell_spec=f"{weapon_key}:{status}:2",
        ue_params={},
        snapshot_step=snapshot_step,
        snapshot_stage_key=snapshot_stage_key,
    )


def _make_summary(p10: float = 350.0) -> dict:
    return {
        "active_score_p10": p10,
        "episode_length_p10": 1500.0,
        "short_episode_rate": 0.0,
    }


def test_apply_result_uses_snapshot_step_for_deterministic_eval_step():
    """_apply_result() が target.snapshot_step を deterministic_eval_step に入れること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    cb = _make_callback(module)
    _setup_callback(cb, num_timesteps=99_999)

    target = _make_target(snapshot_step=42_000)
    result = BootstrapGateEvalResult(target=target, summary=_make_summary(), n_episodes=3)

    cb._apply_result(result, log_step=cb.num_timesteps)

    state = module._states[WeaponType.GARLIC]
    # snapshot_step が eval_step に入り、num_timesteps ではないこと
    assert state.deterministic_eval_step == 42_000
    assert state.deterministic_p10 == 350.0


def test_apply_result_discards_stale_status_change():
    """status が変わった stale result は set_deterministic_result されないこと。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    cb = _make_callback(module)
    _setup_callback(cb, num_timesteps=50_000)

    # eval 時点は solo_bootstrap だったが、apply 時点で maintenance に変化
    target = _make_target(status="solo_bootstrap", snapshot_step=40_000)
    module._states[WeaponType.GARLIC].status = "maintenance"

    result = BootstrapGateEvalResult(target=target, summary=_make_summary(), n_episodes=3)
    cb._apply_result(result, log_step=cb.num_timesteps)

    # 破棄されるので deterministic_p10 は更新されない
    assert module._states[WeaponType.GARLIC].deterministic_p10 is None


def test_apply_result_discards_on_stage_key_change():
    """stage_key_provider の返り値が snapshot_stage_key と違えば破棄されること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    stage_key_provider = MagicMock(return_value="WU5")
    cb = _make_callback(module, stage_key_provider=stage_key_provider)
    _setup_callback(cb, num_timesteps=50_000)

    target = _make_target(snapshot_step=40_000, snapshot_stage_key="WU1")
    result = BootstrapGateEvalResult(target=target, summary=_make_summary(), n_episodes=3)
    cb._apply_result(result, log_step=cb.num_timesteps)

    assert module._states[WeaponType.GARLIC].deterministic_p10 is None


def test_apply_result_error_is_skipped():
    """error を持つ result は set_deterministic_result されずスキップされること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    cb = _make_callback(module)
    _setup_callback(cb, num_timesteps=50_000)

    target = _make_target(snapshot_step=40_000)
    result = BootstrapGateEvalResult(
        target=target, summary=None, n_episodes=0, error=RuntimeError("boom")
    )
    cb._apply_result(result, log_step=cb.num_timesteps)

    assert module._states[WeaponType.GARLIC].deterministic_p10 is None


def test_eval_target_set_params_failure_becomes_error_not_raise():
    """set_params 失敗は target 単位 error になり raise されないこと。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    eval_env = _make_eval_env()
    eval_env.env_method.return_value = [False]  # set_params 失敗
    cb = _make_callback(module, eval_env=eval_env)
    _setup_callback(cb, num_timesteps=10_000)

    target = _make_target()
    model = _make_model()

    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes"
    ) as mock_run:
        result = cb._eval_target(target, model_snapshot=model)

    # raise せず error に入ること
    assert result.error is not None
    assert isinstance(result.error, RuntimeError)
    # eval は実行されないこと
    mock_run.assert_not_called()


def test_eval_target_error_does_not_stop_other_targets():
    """1 target の set_params 失敗が他 target の eval を止めないこと（_eval_target 単位の分離）。"""
    module = _make_module(
        initial_status={"garlic": "solo_bootstrap", "king_bible": "solo_bootstrap"}
    )
    eval_env = _make_eval_env()
    cb = _make_callback(module, eval_env=eval_env)
    _setup_callback(cb, num_timesteps=10_000)

    call_count = 0

    def set_params_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return [call_count != 1]  # 1回目失敗、2回目成功

    eval_env.env_method.side_effect = set_params_side_effect

    t1 = _make_target(weapon_key="garlic", weapon_id=WeaponType.GARLIC)
    t2 = _make_target(weapon_key="king_bible", weapon_id=WeaponType.KING_BIBLE)
    model = _make_model()

    fake_results = _make_fake_ep_results(3, active_score=350.0)
    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes",
        return_value=(fake_results, {}, None),
    ):
        r1 = cb._eval_target(t1, model_snapshot=model)
        r2 = cb._eval_target(t2, model_snapshot=model)

    assert r1.error is not None  # 1つ目失敗
    assert r2.error is None       # 2つ目成功


# ---------------------------------------------------------------------------
# Task 4: async scheduling
# ---------------------------------------------------------------------------

def test_on_step_sync_calls_run_probe_when_async_disabled():
    """async_eval=False では _on_step() が _run_probe() を呼ぶこと。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    cb = _make_callback(module, probe_freq=50_000, async_eval=False)
    _setup_callback(cb, num_timesteps=0)

    with patch.object(cb, "_run_probe") as mock_probe, \
         patch.object(cb, "_start_probe_async_or_skip") as mock_async:
        cb.num_timesteps = 50_000
        cb._on_step()
        mock_probe.assert_called_once()
        mock_async.assert_not_called()


def test_on_step_async_starts_thread_and_returns_immediately():
    """async_eval=True で周期到達時に thread 起動し、_on_step が即返ること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    cb = _make_callback(module, probe_freq=50_000, async_eval=True)
    _setup_callback(cb, num_timesteps=0)

    with patch.object(cb, "_start_probe_async_or_skip") as mock_async, \
         patch.object(cb, "_run_probe") as mock_sync:
        cb.num_timesteps = 50_000
        ret = cb._on_step()
        assert ret is True
        mock_async.assert_called_once()
        mock_sync.assert_not_called()


def test_start_probe_async_starts_worker_thread():
    """_start_probe_async_or_skip() が worker thread を起動すること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    eval_env = _make_eval_env()
    cb = _make_callback(module, eval_env=eval_env, async_eval=True)
    model = _make_model()
    # policy.predict がある model
    model.policy = MagicMock()
    _setup_callback(cb, model=model, num_timesteps=50_000)

    fake_results = _make_fake_ep_results(3, active_score=350.0)
    with patch(
        "games.survivors.bootstrap_gate_eval_callback.run_survivors_eval_episodes",
        return_value=(fake_results, {}, None),
    ):
        cb._start_probe_async_or_skip()
        assert cb._probe_thread is not None
        cb._probe_thread.join(timeout=10)
        assert not cb._probe_thread.is_alive()


def test_async_skip_when_worker_in_flight():
    """worker 走行中に次周期到達しても join されず skip metric がログされること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    wandb_logger = MagicMock()
    wandb_logger.enabled = True
    cb = _make_callback(module, async_eval=True, wandb_logger=wandb_logger)
    _setup_callback(cb, num_timesteps=50_000)

    # 走行中の thread を偽装
    running_thread = MagicMock()
    running_thread.is_alive.return_value = True
    cb._probe_thread = running_thread

    cb._start_probe_async_or_skip()

    # skip metric がログされること
    logged_keys = []
    for c in wandb_logger.log.call_args_list:
        logged_keys.extend(c.args[0].keys() if c.args else c.kwargs.get("data", {}).keys())
    assert any("async_skipped_in_flight" in k for k in logged_keys)
    # join されない（thread はそのまま）
    running_thread.join.assert_not_called()


def test_try_process_pending_applies_results_after_worker_done():
    """worker 完了後 _try_process_pending_probe_result() が module を更新すること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    cb = _make_callback(module, async_eval=True)
    _setup_callback(cb, num_timesteps=60_000)

    # 完了済み thread を偽装
    done_thread = MagicMock()
    done_thread.is_alive.return_value = False
    cb._probe_thread = done_thread
    cb._probe_started_at = time.time()
    cb._probe_snapshot_step = 50_000

    q = __import__("queue").Queue()
    target = _make_target(snapshot_step=50_000)
    q.put([BootstrapGateEvalResult(target=target, summary=_make_summary(320.0), n_episodes=3)])
    cb._probe_result_queue = q

    cb._try_process_pending_probe_result()

    state = module._states[WeaponType.GARLIC]
    assert state.deterministic_p10 == 320.0
    assert state.deterministic_eval_step == 50_000
    # 内部状態がリセットされること
    assert cb._probe_thread is None
    assert cb._probe_result_queue is None


def test_try_process_pending_noop_while_worker_alive():
    """worker 走行中は _try_process_pending_probe_result() が何もしないこと。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    cb = _make_callback(module, async_eval=True)
    _setup_callback(cb, num_timesteps=60_000)

    alive_thread = MagicMock()
    alive_thread.is_alive.return_value = True
    cb._probe_thread = alive_thread

    cb._try_process_pending_probe_result()

    # thread はリセットされない
    assert cb._probe_thread is alive_thread
    assert module._states[WeaponType.GARLIC].deterministic_p10 is None


def test_async_worker_exception_not_reraised():
    """worker 内の例外が外に再 raise されず、result_queue に put されること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    eval_env = _make_eval_env()
    cb = _make_callback(module, eval_env=eval_env, async_eval=True)
    model = _make_model()
    model.policy = MagicMock()
    _setup_callback(cb, model=model, num_timesteps=50_000)

    # _eval_target 自体を例外源にする（worker 全体の try/except を検証）
    with patch.object(
        cb, "_eval_target", side_effect=RuntimeError("worker crash")
    ):
        cb._start_probe_async_or_skip()  # 例外が外に出ないこと
        assert cb._probe_thread is not None
        cb._probe_thread.join(timeout=10)
        assert not cb._probe_thread.is_alive()

    # queue には（空でも）put されていること
    assert cb._probe_result_queue is not None
    assert not cb._probe_result_queue.empty()


def test_on_training_end_joins_thread_and_processes():
    """_on_training_end() が thread を join し残 result を処理すること。"""
    module = _make_module(initial_status={"garlic": "solo_bootstrap"})
    cb = _make_callback(module, async_eval=True)
    _setup_callback(cb, num_timesteps=70_000)

    join_called = {"v": False}

    class _FakeThread:
        def join(self, *a, **k):
            join_called["v"] = True

        def is_alive(self):
            # join 後は完了扱い
            return not join_called["v"]

    cb._probe_thread = _FakeThread()
    cb._probe_started_at = time.time()
    cb._probe_snapshot_step = 60_000

    q = __import__("queue").Queue()
    target = _make_target(snapshot_step=60_000)
    q.put([BootstrapGateEvalResult(target=target, summary=_make_summary(310.0), n_episodes=3)])
    cb._probe_result_queue = q

    cb._on_training_end()

    assert join_called["v"] is True
    assert module._states[WeaponType.GARLIC].deterministic_p10 == 310.0

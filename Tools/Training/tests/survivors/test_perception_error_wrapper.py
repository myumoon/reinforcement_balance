"""DeployObsV1 perception corruption wrapper の順序・統計・再開性を検証する。

画面由来の推定値だけが決定的な乱数列で壊れ、clean 設定では一切変化せず、
並列環境の保存再開後も同じ系列へ戻れることを小さな環境で確かめます。
"""

from __future__ import annotations

from functools import partial
import json
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.perception_error import PerceptionErrorProfile
from survivors.deploy_obs_adapter import visible_track_estimates
from games.survivors.perception_error_wrapper import PerceptionErrorWrapper

BOOTSTRAP = Path(__file__).parents[2] / "configs" / "perception_error_bootstrap_v1.json"
SCHEMA = DeployObsSchema.default_v1()


def _profile(**overrides) -> PerceptionErrorProfile:
    """clean profile に指定項目だけを上書きして構築する。

    production loader と同じ `from_wire` を必ず通すため、テストだけが
    不正な profile を直接生成する抜け道を作りません。
    """
    data = {
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
        "schema_version": "perception_error.v1",
    }
    data.update(overrides)
    return PerceptionErrorProfile.from_wire(data)


def _release_tensor(
    *,
    player_pos=(0.0, 0.0),
    weapon_category=0.0,
    count=0.25,
) -> np.ndarray:
    """有効な release DeployObsV1 tensor を構築する。

    unobservable segment は neutral/invalid/old の欠損表現に固定し、
    observable field だけを統計テスト用の値へ設定します。
    """
    values = np.array(
        [0.8, 0.2, *player_pos, 0.1, -0.2, count, 0.5, -0.5,
         weapon_category, 1.0, 0.0, 0.0],
        dtype=np.float32,
    )
    validity = np.ones(SCHEMA.dim, dtype=np.float32)
    age = np.zeros(SCHEMA.dim, dtype=np.float32)
    for name in ("enemy_hp", "cooldown"):
        offset, size = SCHEMA.layout[name]
        values[offset:offset + size] = next(
            field.neutral for field in SCHEMA.fields if field.name == name
        )
        validity[offset:offset + size] = 0.0
        age[offset:offset + size] = 1.0
    return np.concatenate((values, validity, age)).astype(np.float32)


class StaticDeployEnv(gym.Env):
    """同じ release tensor を返し続ける最小 Gymnasium 環境。

    wrapper の corruption だけを測定し、ゲームロジックや UE5 接続を
    統計結果へ混ぜないためのテスト用環境です。
    """

    metadata = {}

    def __init__(self, observation: np.ndarray | None = None) -> None:
        """固定 observation と互換な Gym space を初期化する。

        action は corruption に影響しない一要素の離散値にします。
        """
        super().__init__()
        self.fixed = np.array(
            _release_tensor() if observation is None else observation,
            dtype=np.float32,
            copy=True,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.fixed.shape, dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        """固定 observation と既存 diagnostics を返す。

        Gymnasium の seed 契約を保ちつつ、値自体は毎回同じにします。
        """
        super().reset(seed=seed)
        return self.fixed.copy(), {"diagnostics": {"env": "static"}}

    def step(self, action):
        """固定 observation と有限な遷移情報を返す。

        learning loop の有限性確認で reward/終了値を一定に保ちます。
        """
        return self.fixed.copy(), 1.0, False, False, {"action": int(action)}


def _wrapper(profile: PerceptionErrorProfile, seed: int) -> PerceptionErrorWrapper:
    """共通 schema・viewport で corruption wrapper を作る。

    各テストが seed 以外の構築条件を共有し、環境間 RNG の比較を
    明確にできるようにします。
    """
    return PerceptionErrorWrapper(
        StaticDeployEnv(), profile, SCHEMA, seed=seed, viewport_size=(1920, 1080),
    )


def test_clean_profile_is_byte_identical_noop_and_preserves_transition_values():
    """clean profile が observation bytes と遷移値を完全維持する。

    乱数器を持つ wrapper を通しても dtype・shape・全 bit が変わらず、
    reward や終了フラグにも副作用がないことを確認します。
    """
    source = _release_tensor()
    wrapper = _wrapper(_profile(), seed=7)
    observation, info = wrapper.reset()
    stepped = wrapper.step(0)

    assert observation.tobytes() == source.tobytes()
    assert stepped[0].tobytes() == source.tobytes()
    assert stepped[1:4] == (1.0, False, False)
    assert info["diagnostics"]["env"] == "static"
    assert info["diagnostics"]["perception_error"]["source_truth"].tobytes() == source.tobytes()


def test_fixed_seed_exact_sequence_and_pipeline_order():
    """同じ seed の出力系列と固定 corruption 順序を完全一致させる。

    latency・dropout・noise・confusion・false entity の順序は
    profile や呼び出し側から並べ替えられないことも diagnostics で確認します。
    """
    profile = _profile(
        latency_mean_frames=1.0,
        burst_enter_prob=0.4,
        burst_exit_prob=0.2,
        burst_dropout_prob=0.3,
        coord_noise_std=0.05,
        item_confusion_matrix=[
            [0.0, 0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        hud_hp_misread_std=0.02,
    )
    first = _wrapper(profile, seed=1234)
    second = _wrapper(profile, seed=1234)

    first_outputs = [first.reset()[0], *(first.step(0)[0] for _ in range(8))]
    second_outputs = [second.reset()[0], *(second.step(0)[0] for _ in range(8))]
    assert [value.tobytes() for value in first_outputs] == [
        value.tobytes() for value in second_outputs
    ]
    _, info = first.reset(seed=1234)
    assert info["diagnostics"]["perception_error"]["stage_order"] == (
        "latency",
        "burst_dropout",
        "coordinate_noise",
        "categorical_confusion",
        "false_entities",
    )


def test_measured_dropout_noise_and_confusion_match_profile_over_100k_samples():
    """10万 sample の dropout・Gaussian・confusion 率を設定値と比較する。

    十分な標本数で乱数実装の系統的なずれを検出しつつ、確率ゆらぎに
    合わせた狭い tolerance 内へ収まることを確認します。
    """
    profile = _profile(
        burst_enter_prob=1.0,
        burst_exit_prob=0.0,
        burst_dropout_prob=0.2,
        coord_noise_std=0.05,
        item_confusion_matrix=[
            [0.0, 0.0, 0.0, 0.3],
            [0.3, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.0],
        ],
    )
    wrapper = _wrapper(profile, seed=2468)
    source = _release_tensor(player_pos=(0.0, 0.0), weapon_category=0.0)
    player_offset, _ = SCHEMA.layout["player_screen_pos"]
    weapon_offset, _ = SCHEMA.layout["weapon_category"]
    value_dim = SCHEMA.dim

    dropout_count = 0
    noise_samples = []
    confusion_count = 0
    valid_count = 0
    for _ in range(100_000):
        corrupted = wrapper.corrupt_observation(source)
        if corrupted[value_dim + player_offset] == 0:
            dropout_count += 1
            continue
        valid_count += 1
        noise_samples.append(float(corrupted[player_offset]))
        confusion_count += corrupted[weapon_offset] == 1.0

    assert dropout_count / 100_000 == pytest.approx(0.2, abs=0.01)
    assert float(np.mean(noise_samples)) == pytest.approx(0.0, abs=0.003)
    assert float(np.std(noise_samples)) == pytest.approx(0.05, abs=0.003)
    assert confusion_count / valid_count == pytest.approx(0.3, abs=0.01)


def test_item_confusion_never_emits_outside_production_vocabulary():
    """全 source category の corruption 出力を production 語彙内へ保つ。

    全ゼロ行の残余確率は元カテゴリを保ち、非ゼロ行から多数回再標本化しても
    正規化済みの 0・1/3・2/3・1 以外を生成しないことを確認します。
    """
    matrix = [[0.25] * 4 for _ in range(4)]
    wrapper = _wrapper(_profile(item_confusion_matrix=matrix), seed=8642)
    zero_wrapper = _wrapper(
        _profile(item_confusion_matrix=[[0.0] * 4 for _ in range(4)]),
        seed=8642,
    )
    weapon_offset, _ = SCHEMA.layout["weapon_category"]
    categories = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)

    for source_category in categories:
        retained = zero_wrapper.corrupt_observation(
            _release_tensor(weapon_category=source_category)
        )
        assert np.isclose(
            retained[weapon_offset], source_category, atol=1e-7
        )
        for _ in range(100):
            output = wrapper.corrupt_observation(
                _release_tensor(weapon_category=source_category)
            )
            assert any(
                np.isclose(output[weapon_offset], allowed, atol=1e-7)
                for allowed in categories
            )


def test_per_env_rng_is_independent_and_reset_seed_replays_sequence():
    """各 env の RNG が独立し、同じ reset seed で系列を再生する。

    一方の環境を余分に進めても他方の乱数位置は変わらず、異なる seed は
    異なる系列になることを byte 比較で検証します。
    """
    profile = _profile(coord_noise_std=0.1)
    source = _release_tensor()
    env_a = _wrapper(profile, seed=10)
    env_b = _wrapper(profile, seed=10)
    env_c = _wrapper(profile, seed=11)

    a_first = env_a.corrupt_observation(source)
    b_first = env_b.corrupt_observation(source)
    c_first = env_c.corrupt_observation(source)
    assert a_first.tobytes() == b_first.tobytes()
    assert a_first.tobytes() != c_first.tobytes()

    env_a.corrupt_observation(source)
    b_second = env_b.corrupt_observation(source)
    replay, _ = env_a.reset(seed=10)
    assert replay.tobytes() == a_first.tobytes()
    assert b_second.tobytes() != replay.tobytes()


def test_corruption_state_pickle_roundtrip_replays_subproc_resume_sequence():
    """spawn した SubprocVecEnv worker 間で corruption 系列を再現する。

    env_method の IPC で worker 別stateを取得・復元し、再開後の全worker出力と
    片workerだけ進めた際の他worker state非干渉を byte 比較します。
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv

    profile = _profile(
        latency_mean_frames=2.0,
        latency_std_frames=0.5,
        burst_enter_prob=0.3,
        burst_exit_prob=0.1,
        burst_dropout_prob=0.4,
        coord_noise_std=0.03,
        hud_xp_stale_prob=0.5,
    )
    original = SubprocVecEnv(
        [partial(_wrapper, profile, seed) for seed in (99, 100)],
        start_method="spawn",
    )
    resumed = SubprocVecEnv(
        [partial(_wrapper, profile, seed) for seed in (123_456, 123_457)],
        start_method="spawn",
    )
    actions = np.array([0, 0])
    try:
        original.reset()
        resumed.reset()
        for _ in range(7):
            original.step(actions)

        worker_states = original.env_method("get_corruption_state")
        assert len(worker_states) == 2
        transferred_states = pickle.loads(pickle.dumps(worker_states))
        for worker_index, state in enumerate(transferred_states):
            resumed.env_method(
                "set_corruption_state", state, indices=worker_index
            )

        for _ in range(12):
            expected, _, _, _ = original.step(actions)
            actual, _, _, _ = resumed.step(actions)
            assert actual.tobytes() == expected.tobytes()

        unaffected_before = original.env_method(
            "get_corruption_state", indices=1
        )[0]
        original.env_method(
            "corrupt_observation", _release_tensor(), indices=0
        )
        unaffected_after = original.env_method(
            "get_corruption_state", indices=1
        )[0]
        assert pickle.dumps(unaffected_after) == pickle.dumps(unaffected_before)
    finally:
        original.close()
        resumed.close()


def test_dropout_latency_stale_and_count_clip_update_matching_metadata():
    """各 corruption value と対応する validity/age/count 上限を更新する。

    latency/stale は age を増やし、dropout は canonical missing へ変更し、
    count clipping は shape や他 field を変えず設定上限へ収めます。
    """
    value_dim = SCHEMA.dim
    player_offset, _ = SCHEMA.layout["player_screen_pos"]
    level_offset, _ = SCHEMA.layout["level"]
    count_offset, _ = SCHEMA.layout["visible_enemy_count"]

    latency = _wrapper(_profile(latency_mean_frames=1.0), seed=3)
    first = _release_tensor(player_pos=(0.1, 0.2))
    second = _release_tensor(player_pos=(0.8, 0.9))
    latency.corrupt_observation(first)
    delayed = latency.corrupt_observation(second)
    assert delayed[player_offset] == first[player_offset]
    assert delayed[value_dim * 2 + player_offset] > first[
        value_dim * 2 + player_offset
    ]

    stale = _wrapper(_profile(hud_xp_stale_prob=1.0), seed=4)
    stale.corrupt_observation(first)
    changed_level = second.copy()
    changed_level[level_offset] = 0.9
    stale_output = stale.corrupt_observation(changed_level)
    assert stale_output[level_offset] == first[level_offset]
    assert stale_output[value_dim * 2 + level_offset] > 0.0

    visible_tracks = [
        {
            "screen_x": 960.0,
            "screen_y": 540.0,
            "visible": True,
            "occluded": False,
            "clipped": False,
            "timestamp_ns": 10,
        }
        for _ in range(15)
    ]
    producer_count = visible_track_estimates(
        visible_tracks, (1920, 1080), 10
    )["visible_enemy_count"].value[0]
    assert producer_count == 0.75

    count_source = _release_tensor(count=producer_count)
    clipped = _wrapper(_profile(count_clip_max=4), seed=5).corrupt_observation(
        count_source
    )
    assert clipped[count_offset] == np.float32(0.2)
    for count_clip_max in (20, 21, 32):
        unchanged = _wrapper(
            _profile(count_clip_max=count_clip_max), seed=5
        ).corrupt_observation(count_source)
        assert unchanged[count_offset].tobytes() == count_source[
            count_offset
        ].tobytes()

    dropped = _wrapper(
        _profile(
            burst_enter_prob=1.0,
            burst_exit_prob=0.0,
            burst_dropout_prob=1.0,
        ),
        seed=6,
    ).corrupt_observation(first)
    assert dropped[player_offset] == 0.0
    assert dropped[value_dim + player_offset] == 0.0
    assert dropped[value_dim * 2 + player_offset] == 1.0


def test_state_import_is_bound_and_atomic_on_invalid_counterexample():
    """profile/schema 不一致 state を拒否し現在系列を部分更新しない。

    binding failure 後の export が byte-equivalent なままであることを確認し、
    RNG だけ先に差し替わるような非 atomic resume を防ぎます。
    """
    wrapper = _wrapper(_profile(coord_noise_std=0.05), seed=71)
    wrapper.corrupt_observation(_release_tensor())
    before = wrapper.get_corruption_state()
    invalid = pickle.loads(pickle.dumps(before))
    invalid["profile_hash"] = "0" * 64

    with pytest.raises(ValueError):
        wrapper.set_corruption_state(invalid)
    assert pickle.dumps(wrapper.get_corruption_state()) == pickle.dumps(before)


def test_invalid_oracle_release_tensor_is_rejected_and_unobservable_stays_missing():
    """privileged oracle 値を observation 経由で渡す試みを拒否する。

    enemy_hp/cooldown は diagnostics 専用で、corruption 後も release tensor の
    neutral・invalid・old 表現から変化しません。
    """
    wrapper = _wrapper(_profile(coord_noise_std=0.2), seed=1)
    source = _release_tensor()
    output = wrapper.corrupt_observation(source)
    for name in ("enemy_hp", "cooldown"):
        offset, size = SCHEMA.layout[name]
        assert np.array_equal(output[offset:offset + size], source[offset:offset + size])
        assert np.array_equal(
            output[SCHEMA.dim + offset:SCHEMA.dim + offset + size],
            source[SCHEMA.dim + offset:SCHEMA.dim + offset + size],
        )

    oracle = source.copy()
    enemy_offset, _ = SCHEMA.layout["enemy_hp"]
    oracle[enemy_offset] = 0.9
    oracle[SCHEMA.dim + enemy_offset] = 1.0
    oracle[SCHEMA.dim * 2 + enemy_offset] = 0.0
    with pytest.raises(ValueError):
        wrapper.corrupt_observation(oracle)


def test_bootstrap_profile_learning_loop_remains_finite():
    """bootstrap profile で反復 step しても全出力を有限に保つ。

    clipping・欠損表現・collapse が何度発生しても NaN/Inf を作らず、
    最小の learning loop が継続できることを確認します。
    """
    profile = PerceptionErrorProfile.from_wire(
        json.loads(BOOTSTRAP.read_text(encoding="utf-8"))
    )
    wrapper = _wrapper(profile, seed=2026)
    observation, _ = wrapper.reset()
    assert np.all(np.isfinite(observation))
    for _ in range(2_048):
        observation, reward, terminated, truncated, _ = wrapper.step(0)
        assert np.all(np.isfinite(observation))
        assert np.isfinite(reward)
        assert not terminated and not truncated

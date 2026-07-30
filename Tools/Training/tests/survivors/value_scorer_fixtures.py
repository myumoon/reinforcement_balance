"""Value scorer の実 PPO/RecurrentPPO fixture を構築する。

小さな固定ネットワークと VecNormalize 統計を保存し、本番と同じロード境界を短時間で
検査できる source descriptor にまとめる。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    sha256_hex,
)

_TRAINING_ROOT = Path(__file__).resolve().parents[2]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from games.survivors.value_source_descriptor import (
    build_value_source_descriptor,
    write_value_source_descriptor,
)
from games.survivors.value_source_loader import policy_state_schema


class TinyValueEnv(gym.Env):
    """固定 shape の連続観測と二値 action を持つ最小環境。

    rollout は不要だが、SB3 の model と VecNormalize を正規の保存 API で作るために使う。
    """

    observation_space = gym.spaces.Box(
        low=-100.0,
        high=100.0,
        shape=(3,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed: int | None = None, options=None):
        """決定的な初期 observation を Gymnasium 形式で返す。

        seed は親実装へ渡し、fixture ごとの policy 初期化を再現可能にする。
        """
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        """保存 fixture では使わない有限 observation を返す。

        万一 SB3 が環境を参照しても episode を進めず、テスト対象を value 計算に限定する。
        """
        del action
        return np.zeros(3, dtype=np.float32), 0.0, False, False, {}


def _source_provenance(
    run_dir: Path,
    source_root: Path,
    *,
    algorithm: str,
    recurrent: bool,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """descriptor の全 probe-ready provenance を生成する。

    artifact と code sibling を全て実ファイルへ結び、loader の fail-closed gate を通す。
    """
    code: dict[str, dict[str, str]] = {}
    for name in (
        "cpp_logic",
        "cpp_base_reward",
        "python_reward",
        "hp_penalty",
        "noveld_config",
        "noveld_callback",
    ):
        path = source_root / f"{name}.txt"
        path.write_text(f"{name}-fixture\n", encoding="utf-8")
        code[name] = {"path": path.name}
    return {
        "source_root": str(source_root),
        "dirty": False,
        "allow_dirty": False,
        "artifacts": {
            "model": {"path": "result/model.zip"},
            "vecnormalize": {"path": "result/vecnormalize.pkl"},
            "package_freeze": {"path": "log/package_freeze.txt"},
        },
        "model_spec": {
            "algorithm": algorithm,
            "policy": "MlpLstmPolicy" if recurrent else "MlpPolicy",
            "recurrent": recurrent,
            "settings": settings,
        },
        "resolved_config": {"path": "log/config_resolved.json"},
        "code": code,
        "runtime": {
            "action_semantics_version": "action_semantics.v1",
            "physics_dt": 1.0 / 60.0,
            "frame_skip": 2,
            "decision_hz": 30.0,
            "ordered_action_map": ["left", "right"],
        },
    }


def build_saved_value_source(
    root: Path,
    *,
    recurrent: bool,
    shared_lstm: bool = False,
    enable_critic_lstm: bool = True,
    seed: int = 7,
) -> tuple[Path, Any, VecNormalize]:
    """tiny model・VecNormalize・descriptor を同じ run に保存する。

    recurrent 三構成を引数で切り替え、全て同じ obs と固定統計で比較できるようにする。
    """
    run_dir = root / "source-run"
    result_dir = run_dir / "result"
    log_dir = run_dir / "log"
    source_root = root / "source"
    result_dir.mkdir(parents=True)
    log_dir.mkdir()
    source_root.mkdir()

    vec_env = DummyVecEnv([TinyValueEnv])
    vecnormalize = VecNormalize(
        vec_env,
        training=True,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    vecnormalize.obs_rms.mean = np.asarray([1.0, -2.0, 0.5], dtype=np.float64)
    vecnormalize.obs_rms.var = np.asarray([4.0, 9.0, 0.25], dtype=np.float64)
    vecnormalize.obs_rms.count = 100.0
    vecnormalize.ret_rms.var = np.asarray(6.25, dtype=np.float64)
    vecnormalize.ret_rms.count = 100.0

    if recurrent:
        from sb3_contrib import RecurrentPPO

        model = RecurrentPPO(
            "MlpLstmPolicy",
            vecnormalize,
            n_steps=4,
            batch_size=4,
            seed=seed,
            device="cpu",
            policy_kwargs={
                "net_arch": {"pi": [8], "vf": [8]},
                "lstm_hidden_size": 4,
                "n_lstm_layers": 2,
                "shared_lstm": shared_lstm,
                "enable_critic_lstm": enable_critic_lstm,
            },
        )
        algorithm = "RecurrentPPO"
    else:
        model = PPO(
            "MlpPolicy",
            vecnormalize,
            n_steps=4,
            batch_size=4,
            seed=seed,
            device="cpu",
            policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
        )
        algorithm = "PPO"

    model.save(result_dir / "model")
    vecnormalize.save(result_dir / "vecnormalize.pkl")
    (log_dir / "package_freeze.txt").write_text(
        "\n".join(
            (
                "numpy==1.26.4",
                "stable-baselines3==2.3.2",
                "sb3-contrib==2.3.0",
                "torch==2.11.0+cpu",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (log_dir / "config_resolved.json").write_text(
        json.dumps({"frame_skip": 2}),
        encoding="utf-8",
    )
    observation_schema = {
        "total_dim": 3,
        "ordered_segments": [{"name": "observation", "dim": 3}],
    }
    state_schema = policy_state_schema(
        model.policy,
        algorithm,
        model_sha256=sha256_hex((result_dir / "model.zip").read_bytes()),
        vecnormalize_sha256=sha256_hex(
            (result_dir / "vecnormalize.pkl").read_bytes()
        ),
        observation_schema_sha256=canonical_hash(observation_schema),
    )
    settings = {
        key: state_schema[key]
        for key in (
            "shared_lstm",
            "enable_critic_lstm",
            "lstm_hidden_size",
            "n_lstm_layers",
            "policy_state_schema_hash",
        )
    }
    provenance = _source_provenance(
        run_dir,
        source_root,
        algorithm=algorithm,
        recurrent=recurrent,
        settings=settings,
    )
    descriptor = build_value_source_descriptor(
        run_dir=run_dir,
        completion={
            "item_stage_key": "IS2",
            "is2_complete": True,
            "weapon_coverage_count": 1,
            "passive_coverage_count": 1,
            "evolution_coverage_count": 1,
            "union_coverage_count": 1,
        },
        obs_schema={
            "total_dim": observation_schema["total_dim"],
            "segments": observation_schema["ordered_segments"],
        },
        git_commit="a" * 40,
        created_at_utc="2026-07-30T00:00:00Z",
        source_provenance=provenance,
    )
    manifest_path = write_value_source_descriptor(run_dir, descriptor)
    state_payload = {
        key: value
        for key, value in state_schema.items()
        if key != "policy_state_schema_hash"
    }
    assert canonical_hash(state_payload) == state_schema["policy_state_schema_hash"]
    return manifest_path, model, vecnormalize

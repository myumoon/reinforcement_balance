"""Immutable source descriptor から value scorer の実行資産を安全にロードする。

descriptor・model・VecNormalize・policy state schema を同時に再検証し、保存時と異なる
critic 経路や observation 契約では formal ranking を開始しない。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)

from games.survivors.value_source_descriptor import (
    ValueSourceDescriptorError,
    validate_value_source_descriptor,
)


class ValueSourceLoadError(ValueError):
    """Value source の同一性または実行契約を証明できない場合の例外。

    不足を warning で継続せず、formal ranking の生成前に一つの fail-closed 境界へ集約する。
    """


@dataclass(frozen=True, slots=True)
class LoadedValueSource:
    """検証済み model・VecNormalize・identity を保持する。

    scorer はこの型だけを受け取り、未検証 path や mutable descriptor を後段へ渡さない。
    """

    manifest_path: Path
    descriptor: Mapping[str, Any]
    manifest_sha256: str
    model: Any
    vecnormalize: Any
    algorithm: str
    observation_dim: int
    policy_state_schema: Mapping[str, Any]

    @property
    def policy(self) -> Any:
        """ロード済み algorithm の policy を返す。

        呼び出し側が model class を再判定せず、検証済みの同じ policy instance を利用する。
        """
        return self.model.policy

    def normalize_raw_obs(self, raw_obs_batch: np.ndarray) -> np.ndarray:
        """raw observation batch の copy だけを VecNormalize へ通す。

        evaluation mode の統計は更新せず、candidate preview の入力配列も変更しない。
        """
        batch = np.asarray(raw_obs_batch, dtype=np.float32)
        return np.asarray(
            self.vecnormalize.normalize_obs(batch.copy()),
            dtype=np.float32,
        )

    def unscale_value(self, value: float) -> float:
        """normalized return value を reward RMS の scale へ戻す。

        順位付けには使わず、ret_rms.var と epsilon による診断値としてのみ公開する。
        """
        variance = float(np.asarray(self.vecnormalize.ret_rms.var).reshape(()))
        scale = float(np.sqrt(variance + float(self.vecnormalize.epsilon)))
        return float(value) * scale


def _policy_schema_payload(
    policy: Any,
    algorithm: str,
    *,
    model_sha256: str,
    vecnormalize_sha256: str,
    observation_schema_sha256: str,
) -> dict[str, Any]:
    """実 policy から recurrent critic 構成を列挙する。

    shared・separate・disabled の三経路を同じ field 集合で表し、hidden size と layer 数も束縛する。
    """
    if algorithm == "RecurrentPPO":
        actor = policy.lstm_actor
        shared_lstm = bool(policy.shared_lstm)
        enable_critic_lstm = bool(policy.enable_critic_lstm)
        hidden_size = int(actor.hidden_size)
        layer_count = int(actor.num_layers)
        vf_semantics = (
            "shared_actor_lstm"
            if shared_lstm
            else "separate_critic_lstm"
            if enable_critic_lstm
            else "actor_state_passthrough"
        )
        shape = [layer_count, 1, hidden_size]
    else:
        shared_lstm = False
        enable_critic_lstm = False
        hidden_size = 0
        layer_count = 0
        vf_semantics = "feedforward"
        shape = []
    return {
        "schema_version": "survivors.policy_state_schema.v1",
        "algorithm": algorithm,
        "model_sha256": model_sha256,
        "vecnormalize_sha256": vecnormalize_sha256,
        "observation_schema_sha256": observation_schema_sha256,
        "shared_lstm": shared_lstm,
        "enable_critic_lstm": enable_critic_lstm,
        "lstm_hidden_size": hidden_size,
        "n_lstm_layers": layer_count,
        "pi_state_shape": shape,
        "vf_state_shape": shape,
        "vf_state_semantics": vf_semantics,
    }


def policy_state_schema(
    policy: Any,
    algorithm: str,
    *,
    model_sha256: str,
    vecnormalize_sha256: str,
    observation_schema_sha256: str,
) -> dict[str, Any]:
    """policy state schema と canonical hash を構築する。

    hash 対象は自己参照 field を除く固定 payload とし、Common の canonical JSON 実装だけを使う。
    """
    payload = _policy_schema_payload(
        policy,
        algorithm,
        model_sha256=model_sha256,
        vecnormalize_sha256=vecnormalize_sha256,
        observation_schema_sha256=observation_schema_sha256,
    )
    return {
        **payload,
        "policy_state_schema_hash": canonical_hash(payload),
    }


def _resolve_artifact(
    run_dir: Path,
    descriptor: Mapping[str, Any],
    name: str,
) -> Path:
    """descriptor の run 相対 artifact path を安全に解決する。

    絶対 path と ``..`` 脱出を再度拒否し、ロード時の filesystem target を run 内へ限定する。
    """
    relative = Path(descriptor["artifacts"][name]["path_relative"])
    candidate = (run_dir / relative).resolve()
    try:
        candidate.relative_to(run_dir)
    except ValueError as exc:
        raise ValueSourceLoadError(f"{name} artifact path escapes source run") from exc
    return candidate


def _verify_artifact(path: Path, expected_hash: str, name: str) -> None:
    """artifact の存在と raw byte hash をロード直前に再検証する。

    manifest 公開後の削除・置換・破損を全 runtime artifact sibling へ同じ規則で適用する。
    """
    if not path.is_file():
        raise ValueSourceLoadError(f"{name} artifact is missing: {path}")
    actual_hash = sha256_hex(path.read_bytes())
    if actual_hash != expected_hash:
        raise ValueSourceLoadError(
            f"{name} artifact hash mismatch: expected={expected_hash}, actual={actual_hash}"
        )


def _detect_algorithm(model_path: Path) -> str:
    """保存 zip の policy_class から PPO/RecurrentPPO を判定する。

    CLI flag や filename から推測せず、SB3 が保存した class identity を唯一の判定材料にする。
    """
    try:
        from sb3_contrib.common.recurrent.policies import (
            RecurrentActorCriticPolicy,
        )
        from stable_baselines3.common.save_util import load_from_zip_file

        data, _, _ = load_from_zip_file(str(model_path), device="cpu")
        policy_class = data.get("policy_class")
    except Exception as exc:
        raise ValueSourceLoadError("model zip policy_class could not be read") from exc
    if not isinstance(policy_class, type):
        raise ValueSourceLoadError("model zip has no concrete policy_class")
    try:
        is_recurrent = issubclass(policy_class, RecurrentActorCriticPolicy)
    except TypeError as exc:
        raise ValueSourceLoadError("model zip policy_class is invalid") from exc
    return "RecurrentPPO" if is_recurrent else "PPO"


def _load_model(model_path: Path, algorithm: str, device: str) -> Any:
    """判定済み algorithm class で model をロードする。

    actor state だけを返す predict API には依存せず、後段が policy.forward を直接使える instance を返す。
    """
    try:
        if algorithm == "RecurrentPPO":
            from sb3_contrib import RecurrentPPO

            return RecurrentPPO.load(str(model_path), device=device)
        from stable_baselines3 import PPO

        return PPO.load(str(model_path), device=device)
    except Exception as exc:
        raise ValueSourceLoadError(f"{algorithm} model could not be loaded") from exc


def _load_vecnormalize(path: Path, model: Any) -> Any:
    """model space に一致する dummy VecEnv へ VecNormalize 統計を接続する。

    HTTP/game 環境は起動せず、normalize_obs に必要な保存統計だけを evaluation mode で復元する。
    """
    try:
        import gymnasium as gym
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        observation_space = model.observation_space
        action_space = model.action_space

        class _InferenceEnv(gym.Env):
            """保存 space だけを提供する非実行 dummy environment。

            scorer は reset/step を呼ばないが、VecNormalize の正規 wrapper 契約を満たす。
            """

            def __init__(self) -> None:
                """保存済み observation/action space を dummy env へ設定する。

                game instance は生成せず、VecNormalize.set_venv の space equality 検証だけを満たす。
                """
                self.observation_space = observation_space
                self.action_space = action_space

            def reset(self, *, seed: int | None = None, options=None):
                """space の zero observation を Gymnasium 形式で返す。

                seed を親へ渡し、外部環境へ接続する副作用は一切持たない。
                """
                super().reset(seed=seed)
                return np.zeros(observation_space.shape, dtype=np.float32), {}

            def step(self, action):
                """scorer が呼び出した場合も有限な terminal-free tuple を返す。

                action は評価せず、policy value 計算と game transition を混在させない。
                """
                del action
                return (
                    np.zeros(observation_space.shape, dtype=np.float32),
                    0.0,
                    False,
                    False,
                    {},
                )

        vec_env = DummyVecEnv([_InferenceEnv])
        vecnormalize = VecNormalize.load(str(path), vec_env)
    except Exception as exc:
        raise ValueSourceLoadError("vecnormalize artifact could not be loaded") from exc
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    return vecnormalize


def _validate_policy_binding(
    descriptor: Mapping[str, Any],
    algorithm: str,
    schema: Mapping[str, Any],
) -> None:
    """descriptor の algorithm と全 recurrent setting を実 policy へ束縛する。

    shared_lstm・critic LSTM・hidden size・layer count・schema hash の全 sibling を対称検証する。
    """
    model_spec = descriptor["model_spec"]
    if model_spec["algorithm"] != algorithm:
        raise ValueSourceLoadError(
            "model algorithm does not match saved policy_class"
        )
    if model_spec["recurrent"] is not (algorithm == "RecurrentPPO"):
        raise ValueSourceLoadError(
            "model recurrent flag does not match saved policy_class"
        )
    settings = model_spec["settings"]
    for key in (
        "shared_lstm",
        "enable_critic_lstm",
        "lstm_hidden_size",
        "n_lstm_layers",
        "policy_state_schema_hash",
    ):
        if settings.get(key) != schema[key]:
            raise ValueSourceLoadError(
                f"model setting {key} does not match loaded policy"
            )


def load_value_source(
    manifest_path: Path,
    device: str = "cpu",
) -> LoadedValueSource:
    """source descriptor と全 runtime artifact を検証してロードする。

    ready gate、canonical manifest hash、artifact raw hash、obs dim、policy schema の順に閉じる。
    """
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise ValueSourceLoadError(f"manifest is missing: {path}")
    try:
        descriptor = json.loads(path.read_text(encoding="utf-8"))
        validate_value_source_descriptor(descriptor)
    except (OSError, json.JSONDecodeError, ValueSourceDescriptorError, TypeError) as exc:
        raise ValueSourceLoadError("manifest descriptor is invalid") from exc
    if descriptor["ready_for_probe"] is not True:
        raise ValueSourceLoadError(
            f"manifest is not probe-ready: {descriptor['blocking_reasons']}"
        )
    manifest_sha256 = sha256_hex(canonical_json_bytes(descriptor))
    run_dir = path.parents[1].resolve()
    artifact_paths = {
        name: _resolve_artifact(run_dir, descriptor, name)
        for name in ("model", "vecnormalize", "package_freeze")
    }
    for name, artifact_path in artifact_paths.items():
        _verify_artifact(
            artifact_path,
            descriptor["artifacts"][name]["sha256"],
            name,
        )
    algorithm = _detect_algorithm(artifact_paths["model"])
    model = _load_model(artifact_paths["model"], algorithm, device)
    schema = policy_state_schema(
        model.policy,
        algorithm,
        model_sha256=descriptor["artifacts"]["model"]["sha256"],
        vecnormalize_sha256=descriptor["artifacts"]["vecnormalize"]["sha256"],
        observation_schema_sha256=descriptor["observation_schema"]["sha256"],
    )
    _validate_policy_binding(descriptor, algorithm, schema)
    observation_space = model.observation_space
    shape = getattr(observation_space, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 1 or shape[0] <= 0:
        raise ValueSourceLoadError("only flat Box observation spaces are supported")
    observation_dim = int(shape[0])
    if descriptor["observation_schema"]["total_dim"] != observation_dim:
        raise ValueSourceLoadError(
            "manifest observation dim does not match loaded model"
        )
    vecnormalize = _load_vecnormalize(artifact_paths["vecnormalize"], model)
    return LoadedValueSource(
        manifest_path=path,
        descriptor=descriptor,
        manifest_sha256=manifest_sha256,
        model=model,
        vecnormalize=vecnormalize,
        algorithm=algorithm,
        observation_dim=observation_dim,
        policy_state_schema=schema,
    )

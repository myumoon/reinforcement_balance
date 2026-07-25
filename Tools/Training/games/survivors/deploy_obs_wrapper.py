"""UE5 raw state を画面投影経由で DeployObsV1 に変換する Gym 互換 wrapper。

raw 配列の slice を避け、release と oracle diagnostic を別 constructor に分けて
実環境と同じ on-screen semantics を訓練側でも守ります。
"""

from __future__ import annotations

from typing import Any, Mapping

from survivors.deploy_obs_adapter import (
    NamedEstimate, build_deploy_observation, normalized_category, visible_track_estimates,
)
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.ui_intent import ensure

_RAW_KEYS = frozenset({"timestamp_ns", "viewport", "hud", "player_screen", "tracks", "temporal", "inventory", "privileged"})
_HUD_KEYS = frozenset({"player_hp", "level"})
_TEMPORAL_KEYS = frozenset({"movement_direction", "timestamp_ns"})
_INVENTORY_KEYS = frozenset({"weapon_category"})
_PRIVILEGED_KEYS = frozenset({"player_pos", "enemy_hp", "cooldown", "all_entity_count", "density"})


class DeployObsWrapper:
    """環境 observation を deploy tensor に置換する軽量 wrapper。

    Gymnasium/SB3 を import せず reset/step を委譲するため、契約 test は単独実行できます。
    """

    def __init__(self, env: Any, schema: DeployObsSchema, mode: str) -> None:
        """検証済み環境・schema・mode を保持する。

        直接構築を禁止し、release/oracle の明示 constructor からだけ作ります。
        """
        ensure(mode in {"release", "oracle_diagnostic"}, "invalid deploy observation mode")
        self.env, self.schema, self.mode = env, schema, mode
        self.run_manifest = {"deploy_obs_mode": mode, "deploy_obs_schema_hash": schema.schema_hash, "vecnormalize": "fresh_outside_deploy_tensor"}

    @classmethod
    def release(cls, env: Any, schema: DeployObsSchema) -> "DeployObsWrapper":
        """実 parser と同じ screen-space semantics の wrapper を作る。

        release artifact を生成できる唯一の mode で、privileged state は読みません。
        """
        return cls(env, schema, "release")

    @classmethod
    def oracle_diagnostic(cls, env: Any, schema: DeployObsSchema) -> "DeployObsWrapper":
        """全 state を比較診断に使える oracle wrapper を作る。

        学習調査専用であり、release artifact の保存は明示的に禁止されます。
        """
        return cls(env, schema, "oracle_diagnostic")

    @property
    def release_artifact_allowed(self) -> bool:
        """現在の mode が release artifact を生成可能か返す。

        oracle の結果を本番成果物と誤認しないための単純な gate です。
        """
        return self.mode == "release"

    def assert_release_artifact_allowed(self) -> None:
        """oracle mode の artifact 出力を fail-closed で拒否する。

        保存処理は実行直前にこの gate を呼び、診断用 truth の混入を防ぎます。
        """
        ensure(self.release_artifact_allowed, "oracle_diagnostic cannot create release artifacts")

    def observation(self, raw: Mapping[str, Any]):
        """raw state を投影・可視性判定・named estimates 経由で tensor 化する。

        release 経路では privileged mapping を一切参照しません。
        """
        ensure(isinstance(raw, Mapping) and set(raw) == _RAW_KEYS, "raw observation keys mismatch")
        now, viewport = raw["timestamp_ns"], raw["viewport"]
        ensure(isinstance(viewport, (list, tuple)) and len(viewport) == 2, "invalid viewport")
        ensure(isinstance(raw["hud"], Mapping) and set(raw["hud"]) <= _HUD_KEYS, "hud keys mismatch")
        ensure(isinstance(raw["temporal"], Mapping) and set(raw["temporal"]) <= _TEMPORAL_KEYS, "temporal keys mismatch")
        ensure(isinstance(raw["inventory"], Mapping) and set(raw["inventory"]) <= _INVENTORY_KEYS, "inventory keys mismatch")
        ensure(isinstance(raw["privileged"], Mapping) and set(raw["privileged"]) <= _PRIVILEGED_KEYS, "privileged keys mismatch")
        estimates = visible_track_estimates(raw["tracks"], tuple(viewport), now)
        for name in ("player_hp", "level"):
            if name in raw["hud"]:
                estimates[name] = NamedEstimate((raw["hud"][name],), now)
        if raw["player_screen"] is not None:
            from survivors.deploy_obs_adapter import screen_to_centered
            estimates["player_screen_pos"] = NamedEstimate(screen_to_centered(*raw["player_screen"], *viewport), now)
        if "movement_direction" in raw["temporal"]:
            ensure(set(raw["temporal"]) == _TEMPORAL_KEYS, "temporal estimate keys mismatch")
            estimates["movement_direction"] = NamedEstimate(tuple(raw["temporal"]["movement_direction"]), raw["temporal"]["timestamp_ns"])
        estimates["weapon_category"] = NamedEstimate((normalized_category(raw["inventory"].get("weapon_category", "unknown")),), now)
        if self.mode == "oracle_diagnostic":
            for name in ("enemy_hp", "cooldown"):
                if name in raw["privileged"]:
                    estimates[name] = NamedEstimate((raw["privileged"][name],), now)
        observation = build_deploy_observation(self.schema, estimates, now)
        observation.validate_for(self.schema)
        return observation.as_policy_tensor()

    def reset(self, **kwargs: Any):
        """下位環境を reset し deploy tensor と info を返す。

        Gymnasium の戻り値形式を保ったまま observation だけを変換します。
        """
        raw, info = self.env.reset(**kwargs)
        return self.observation(raw), info

    def step(self, action: Any):
        """下位環境を step し deploy tensor を含む結果を返す。

        reward・終了フラグ・info は変更せず observation のみ変換します。
        """
        raw, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(raw), reward, terminated, truncated, info


def fresh_vecnormalize(wrapper: DeployObsWrapper, factory: Any) -> Any:
    """deploy tensor の外側へ新規 VecNormalize を構築する。

    privileged source の統計を受け取らず、factory には wrapper だけを渡します。
    """
    ensure(isinstance(wrapper, DeployObsWrapper) and callable(factory), "invalid VecNormalize factory")
    return factory(wrapper, norm_obs=True, training=True)

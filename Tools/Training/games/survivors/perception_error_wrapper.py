"""DeployObsV1 の画面推定値へ perception corruption を加える Gymnasium wrapper。

観測遅延から誤検出までの順序をコード上で固定し、oracle 値を policy 入力へ
混ぜずに、並列 worker ごとの乱数・履歴状態を保存再開できるようにします。
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
from typing import Any, Mapping

import gymnasium as gym
import numpy as np

from reinbalance_survivors_contracts.deploy_obs import (
    DeployObservation,
    DeployObsField,
    DeployObsSchema,
)
from reinbalance_survivors_contracts.perception_error import PerceptionErrorProfile
from reinbalance_survivors_contracts.ui_intent import ensure

_CORRUPTION_STATE_VERSION = "perception_corruption_state.v1"
_STATE_KEYS = frozenset(
    {
        "state_version",
        "schema_hash",
        "profile_hash",
        "rng_state",
        "burst_active",
        "collapse_remaining",
        "latency_buffer",
        "hud_cache",
    }
)
_CACHE_KEYS = frozenset({"values", "validity", "age"})
_STAGE_ORDER = (
    "latency",
    "burst_dropout",
    "coordinate_noise",
    "categorical_confusion",
    "false_entities",
)
_COORDINATE_SEGMENTS = ("player_screen_pos", "nearest_enemy_offset")
_HUD_STALE_PROBABILITIES = {
    "level": "hud_xp_stale_prob",
    "weapon_category": "hud_inventory_stale_prob",
}
_COUNT_FULL_SCALE = 32
_FRAME_DURATION_MS = 1000.0 / 60.0


class PerceptionErrorWrapper(gym.Wrapper):
    """release DeployObs tensor だけを壊す状態付き Gymnasium wrapper。

    各 instance が専用 NumPy Generator と latency/HUD 履歴を所有し、
    SubprocVecEnv の worker 間で共有される module-level RNG は使いません。
    """

    def __init__(
        self,
        env: gym.Env,
        profile: PerceptionErrorProfile,
        schema: DeployObsSchema | None = None,
        *,
        seed: int | None = None,
        viewport_size: tuple[int, int] = (1920, 1080),
    ) -> None:
        """環境・profile・schema と独立 corruption state を初期化する。

        viewport は pixel 量子化を正規化座標へ変換するためだけに使い、
        world 座標や privileged state を受け取る入口は設けません。
        """
        ensure(isinstance(profile, PerceptionErrorProfile), "invalid profile")
        if schema is None:
            schema = DeployObsSchema.default_v1()
        ensure(isinstance(schema, DeployObsSchema), "invalid DeployObs schema")
        ensure(
            seed is None or (isinstance(seed, int) and not isinstance(seed, bool)),
            "seed must be int or None",
        )
        ensure(
            isinstance(viewport_size, tuple)
            and len(viewport_size) == 2
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in viewport_size
            ),
            "viewport_size must be a positive integer pair",
        )
        super().__init__(env)
        self.profile = profile
        self.schema = schema
        self.viewport_size = viewport_size
        self._profile_hash = profile.profile_hash
        self._is_clean = profile.is_clean
        self._rng = np.random.default_rng(seed)
        self._burst_active = False
        self._collapse_remaining = 0
        self._fields = {field.name: field for field in schema.fields}
        self._observable_fields = tuple(
            field
            for field in schema.fields
            if field.source_class not in {"constant", "unobservable"}
        )
        self._max_latency_frames = max(
            0,
            int(
                math.ceil(
                    profile.latency_mean_frames + 6.0 * profile.latency_std_frames
                )
            ),
        )
        self._latency_buffer: deque[np.ndarray] = deque(
            maxlen=max(1, self._max_latency_frames + 1)
        )
        self._hud_cache: dict[
            str, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        expected_shape = (schema.dim * 3,)
        ensure(
            getattr(self.observation_space, "shape", None) == expected_shape,
            f"observation_space must have shape {expected_shape}",
        )

    def _planes(
        self, observation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """flat policy tensor を value・validity・age の writable copy へ分ける。

        入力そのものを変更せず、各 corruption stage が同じ schema offset を
        使って三平面を対称に更新できるようにします。
        """
        dim = self.schema.dim
        copied = np.array(observation, dtype=np.float32, copy=True)
        return copied[:dim], copied[dim : dim * 2], copied[dim * 2 :]

    def _validated_release_tensor(self, observation: Any) -> np.ndarray:
        """入力を厳密な release DeployObsV1 tensor として検証する。

        shape・有限性・field range・欠損表現に加え、unobservable の有効化を
        oracle leakage として corruption 前に拒否します。
        """
        ensure(
            isinstance(observation, np.ndarray)
            and observation.ndim == 1
            and observation.shape == (self.schema.dim * 3,)
            and observation.dtype == np.float32,
            "observation must be a float32 DeployObsV1 tensor",
        )
        values, validity, age = self._planes(observation)
        deploy_observation = DeployObservation(
            values=values,
            validity=validity,
            age=age,
            schema_hash=self.schema.schema_hash,
            timestamp_ns=0,
            provenance="release",
        )
        deploy_observation.validate_for(self.schema)
        return np.array(observation, copy=True)

    def _field_planes(
        self,
        field: DeployObsField,
        values: np.ndarray,
        validity: np.ndarray,
        age: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """一つの named segment に対応する三平面 view を返す。

        raw array の固定 offset を別定義せず、DeployObsSchema の名前付き layout
        だけから slice を解決します。
        """
        offset, size = self.schema.layout[field.name]
        segment = slice(offset, offset + size)
        return values[segment], validity[segment], age[segment]

    def _set_missing(
        self,
        field: DeployObsField,
        values: np.ndarray,
        validity: np.ndarray,
        age: np.ndarray,
    ) -> None:
        """corrupted field を canonical missing 表現へ更新する。

        dropout/collapse 時は neutral value・validity=0・age=1 を必ず同時に
        設定し、古い値だけが policy tensor に残る状態を作りません。
        """
        field_values, field_validity, field_age = self._field_planes(
            field, values, validity, age
        )
        field_values[:] = field.neutral
        field_validity[:] = 0.0
        field_age[:] = 1.0

    def _sample_latency_frames(self) -> int:
        """profile の非負正規分布から整数 latency frame 数を得る。

        ring buffer の有限上限を超える極端な標本は上限へ clip し、
        負値や無限長の履歴を生成しません。
        """
        if self.profile.latency_std_frames == 0.0:
            sampled = self.profile.latency_mean_frames
        else:
            sampled = float(
                self._rng.normal(
                    self.profile.latency_mean_frames,
                    self.profile.latency_std_frames,
                )
            )
        return min(
            self._max_latency_frames,
            max(0, int(math.floor(sampled + 0.5))),
        )

    def _apply_latency(self, observation: np.ndarray) -> np.ndarray:
        """frame 全体を latency ring buffer から選び age を進める。

        value/validity/age の対応を崩さず同じ過去 frame を取り出し、
        observable field の age に実際に利用できた遅延分を加算します。
        """
        self._latency_buffer.append(np.array(observation, copy=True))
        requested_delay = self._sample_latency_frames()
        actual_delay = min(requested_delay, len(self._latency_buffer) - 1)
        delayed = np.array(self._latency_buffer[-1 - actual_delay], copy=True)
        if actual_delay == 0:
            return delayed
        values, validity, age = self._planes(delayed)
        for field in self._observable_fields:
            _, field_validity, field_age = self._field_planes(
                field, values, validity, age
            )
            present = field_validity > 0.0
            field_age[present] = np.minimum(
                1.0,
                field_age[present]
                + actual_delay * _FRAME_DURATION_MS / field.max_age_ms,
            )
        return np.concatenate((values, validity, age)).astype(np.float32)

    def _sample_collapse_duration(self) -> int:
        """unknown-screen collapse の正整数 duration を標本化する。

        duration mean が 0 なら collapse は開始せず、それ以外は Poisson
        標本を最低 1 frame に丸めて負の残り時間を作りません。
        """
        mean = self.profile.unknown_screen_collapse_duration_frames
        if mean <= 0.0:
            return 0
        return max(1, int(self._rng.poisson(mean)))

    def _collapse_observable_fields(
        self, values: np.ndarray, validity: np.ndarray, age: np.ndarray
    ) -> None:
        """global unknown-screen 中の全 observable field を欠損化する。

        constant と unobservable segment は対象列挙から除外済みなので、
        bias や oracle 専用 segment を corruption で書き換えません。
        """
        for field in self._observable_fields:
            self._set_missing(field, values, validity, age)

    def _apply_burst_dropout(self, observation: np.ndarray) -> np.ndarray:
        """Markov burst dropout と unknown-screen collapse を適用する。

        burst state の遷移と emission を instance 専用 RNG で行い、欠落した
        field は value/validity/age を canonical missing へ同時更新します。
        """
        values, validity, age = self._planes(observation)
        if self._burst_active:
            if self._rng.random() < self.profile.burst_exit_prob:
                self._burst_active = False
        elif self._rng.random() < self.profile.burst_enter_prob:
            self._burst_active = True

        if (
            self._burst_active
            and self._rng.random() < self.profile.burst_dropout_prob
        ):
            self._collapse_observable_fields(values, validity, age)

        if self._collapse_remaining == 0:
            if (
                self._rng.random()
                < self.profile.unknown_screen_collapse_prob
            ):
                self._collapse_remaining = self._sample_collapse_duration()
        if self._collapse_remaining > 0:
            self._collapse_observable_fields(values, validity, age)
            self._collapse_remaining -= 1
        return np.concatenate((values, validity, age)).astype(np.float32)

    def _apply_stale_field(
        self,
        field: DeployObsField,
        probability: float,
        values: np.ndarray,
        validity: np.ndarray,
        age: np.ndarray,
    ) -> None:
        """一つの HUD field を確率的に前回 cache へ戻して age を進める。

        cache 未初期化時は現在値を保存し、stale が連続した場合も age が
        単調増加して 0 未満や 1 超過にならないようにします。
        """
        field_values, field_validity, field_age = self._field_planes(
            field, values, validity, age
        )
        cached = self._hud_cache.get(field.name)
        use_stale = cached is not None and self._rng.random() < probability
        if use_stale:
            cached_values, cached_validity, cached_age = cached
            field_values[:] = cached_values
            field_validity[:] = cached_validity
            field_age[:] = np.minimum(
                1.0, cached_age + _FRAME_DURATION_MS / field.max_age_ms
            )
            self._hud_cache[field.name] = (
                field_values.copy(),
                field_validity.copy(),
                field_age.copy(),
            )
        else:
            self._hud_cache[field.name] = (
                field_values.copy(),
                field_validity.copy(),
                field_age.copy(),
            )

    def _apply_coordinate_noise(self, observation: np.ndarray) -> np.ndarray:
        """screen-space dx/dy noise・pixel 量子化・HP 誤読を適用する。

        有効な observable 値だけを変更し、各 field の schema range へ clip
        するため NaN/Inf や画面外の impossible coordinate を生成しません。
        """
        values, validity, age = self._planes(observation)
        for field_name in _COORDINATE_SEGMENTS:
            field = self._fields[field_name]
            field_values, field_validity, _ = self._field_planes(
                field, values, validity, age
            )
            present = field_validity > 0.0
            if self.profile.coord_noise_std > 0.0:
                noise = self._rng.normal(
                    0.0, self.profile.coord_noise_std, size=field.size
                )
                field_values[present] += noise[present].astype(np.float32)
            if self.profile.coord_quantization_px > 0.0:
                for index in range(field.size):
                    if not present[index]:
                        continue
                    pixels = self.viewport_size[index]
                    step = 2.0 * self.profile.coord_quantization_px / pixels
                    field_values[index] = round(field_values[index] / step) * step
            np.clip(
                field_values,
                field.minimum,
                field.maximum,
                out=field_values,
            )

        hp_field = self._fields["player_hp"]
        hp_values, hp_validity, _ = self._field_planes(
            hp_field, values, validity, age
        )
        hp_present = hp_validity > 0.0
        if self.profile.hud_hp_misread_std > 0.0:
            hp_noise = self._rng.normal(
                0.0, self.profile.hud_hp_misread_std, size=hp_field.size
            )
            hp_values[hp_present] += hp_noise[hp_present].astype(np.float32)
            np.clip(
                hp_values, hp_field.minimum, hp_field.maximum, out=hp_values
            )

        xp_field = self._fields["level"]
        self._apply_stale_field(
            xp_field,
            self.profile.hud_xp_stale_prob,
            values,
            validity,
            age,
        )
        return np.concatenate((values, validity, age)).astype(np.float32)

    def _confused_category(
        self, value: float, matrix: tuple[tuple[float, ...], ...]
    ) -> float:
        """正規化 scalar category を confusion matrix で再標本化する。

        matrix 行和の残余は元 category 維持へ割り当て、出力 index を
        必ず既知カテゴリ [0,size-1] から選ぶため invalid category を作りません。
        """
        if not matrix:
            return value
        size = len(matrix)
        source_index = min(size - 1, max(0, int(math.floor(value * (size - 1) + 0.5))))
        draw = float(self._rng.random())
        cumulative = 0.0
        for target_index, probability in enumerate(matrix[source_index]):
            cumulative += probability
            if draw < cumulative:
                return target_index / (size - 1) if size > 1 else 0.0
        return source_index / (size - 1) if size > 1 else 0.0

    def _apply_categorical_confusion(self, observation: np.ndarray) -> np.ndarray:
        """HUD inventory stale と item categorical confusion を適用する。

        DeployObsV1 に存在する named categorical segment は weapon_category
        だけなので、enemy matrix は将来 field を捏造せず profile 情報として保持します。
        """
        values, validity, age = self._planes(observation)
        weapon_field = self._fields["weapon_category"]
        self._apply_stale_field(
            weapon_field,
            self.profile.hud_inventory_stale_prob,
            values,
            validity,
            age,
        )
        weapon_values, weapon_validity, _ = self._field_planes(
            weapon_field, values, validity, age
        )
        if weapon_validity[0] > 0.0 and self.profile.item_confusion_matrix:
            weapon_values[0] = self._confused_category(
                float(weapon_values[0]), self.profile.item_confusion_matrix
            )
        return np.concatenate((values, validity, age)).astype(np.float32)

    def _apply_false_entities(self, observation: np.ndarray) -> np.ndarray:
        """visible entity count を profile の検出上限へ clip する。

        DeployObsV1 は個別 entity list を持たないため shape を増やさず、
        false/double detection の集約結果だけを count segment の範囲内へ制限します。
        """
        values, validity, age = self._planes(observation)
        count_field = self._fields["visible_enemy_count"]
        count_values, count_validity, _ = self._field_planes(
            count_field, values, validity, age
        )
        if count_validity[0] > 0.0:
            normalized_cap = min(
                1.0, self.profile.count_clip_max / _COUNT_FULL_SCALE
            )
            cap = count_field.minimum + (
                count_field.maximum - count_field.minimum
            ) * normalized_cap
            count_values[0] = min(float(count_values[0]), cap)
        return np.concatenate((values, validity, age)).astype(np.float32)

    def corrupt_observation(self, observation: np.ndarray) -> np.ndarray:
        """検証済み tensor へ固定順序の corruption pipeline を適用する。

        latency → burst dropout → coordinate noise → categorical confusion →
        false entities を直列呼び出しで固定し、外部設定から順序変更できなくします。
        """
        source = self._validated_release_tensor(observation)
        if self._is_clean:
            return source
        corrupted = self._apply_latency(source)
        corrupted = self._apply_burst_dropout(corrupted)
        corrupted = self._apply_coordinate_noise(corrupted)
        corrupted = self._apply_categorical_confusion(corrupted)
        corrupted = self._apply_false_entities(corrupted)
        return self._validated_release_tensor(corrupted)

    def _diagnostic_info(
        self, info: Mapping[str, Any], source_truth: np.ndarray
    ) -> dict[str, Any]:
        """source truth を diagnostics 専用 namespace へ防御的に追加する。

        observation/reward へ truth を返す経路は作らず、既存 diagnostics も
        shallow copy して下位環境の辞書を破壊しません。
        """
        ensure(isinstance(info, Mapping), "info must be a mapping")
        existing = info.get("diagnostics", {})
        ensure(isinstance(existing, Mapping), "info.diagnostics must be a mapping")
        result = dict(info)
        diagnostics = dict(existing)
        diagnostics["perception_error"] = {
            "source_truth": np.array(source_truth, copy=True),
            "stage_order": _STAGE_ORDER,
            "profile_hash": self._profile_hash,
        }
        result["diagnostics"] = diagnostics
        return result

    def _reset_corruption_history(self, seed: int | None) -> None:
        """episode 境界で temporal corruption state を初期状態へ戻す。

        seed 指定時だけ RNG を再生成し、指定なし reset では独立乱数系列を
        継続しながら前 episode の latency/stale 状態を持ち越しません。
        """
        if seed is not None:
            ensure(
                isinstance(seed, int) and not isinstance(seed, bool),
                "seed must be int or None",
            )
            self._rng = np.random.default_rng(seed)
        self._burst_active = False
        self._collapse_remaining = 0
        self._latency_buffer.clear()
        self._hud_cache.clear()

    def reset(self, **kwargs: Any):
        """下位環境を reset して corruption 済み observation と info を返す。

        Gymnasium の seed/options をそのまま委譲し、episode 間で burst や
        latency frame が漏れないよう先に wrapper state を初期化します。
        """
        seed = kwargs.get("seed")
        self._reset_corruption_history(seed)
        source, info = self.env.reset(**kwargs)
        source_truth = self._validated_release_tensor(source)
        return (
            self.corrupt_observation(source_truth),
            self._diagnostic_info(info, source_truth),
        )

    def step(self, action: Any):
        """下位環境の一遷移へ corruption だけを追加する。

        reward・terminated・truncated は変更せず、source observation は
        diagnostics へだけ保存して policy observation と分離します。
        """
        source, reward, terminated, truncated, info = self.env.step(action)
        source_truth = self._validated_release_tensor(source)
        return (
            self.corrupt_observation(source_truth),
            reward,
            terminated,
            truncated,
            self._diagnostic_info(info, source_truth),
        )

    def get_corruption_state(self) -> dict[str, Any]:
        """RNG と全 temporal corruption state を serializable dict で返す。

        NumPy array は list へ変換し、SubprocVecEnv の `env_method` と pickle を
        通した後でも worker ごとの系列を byte-identically 再開できます。
        """
        hud_cache = {
            name: {
                "values": cached[0].tolist(),
                "validity": cached[1].tolist(),
                "age": cached[2].tolist(),
            }
            for name, cached in self._hud_cache.items()
        }
        return {
            "state_version": _CORRUPTION_STATE_VERSION,
            "schema_hash": self.schema.schema_hash,
            "profile_hash": self._profile_hash,
            "rng_state": deepcopy(self._rng.bit_generator.state),
            "burst_active": self._burst_active,
            "collapse_remaining": self._collapse_remaining,
            "latency_buffer": [
                observation.tolist() for observation in self._latency_buffer
            ],
            "hud_cache": hud_cache,
        }

    def _validated_cached_field(
        self, name: str, payload: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """import state の一 HUD cache entry を検証して配列へ戻す。

        未知 field・shape 違い・非有限値・範囲外の cache を拒否し、
        worker state の破損を次の observation へ遅延させません。
        """
        ensure(
            name in _HUD_STALE_PROBABILITIES
            and isinstance(payload, Mapping)
            and set(payload) == _CACHE_KEYS,
            "invalid HUD cache entry",
        )
        field = self._fields[name]
        arrays = tuple(
            np.asarray(payload[key], dtype=np.float32)
            for key in ("values", "validity", "age")
        )
        ensure(
            all(array.shape == (field.size,) for array in arrays)
            and all(np.all(np.isfinite(array)) for array in arrays),
            "invalid HUD cache array",
        )
        values, validity, age = arrays
        ensure(
            np.all((values >= field.minimum) & (values <= field.maximum))
            and np.all((validity >= 0.0) & (validity <= 1.0))
            and np.all((age >= 0.0) & (age <= 1.0)),
            "HUD cache value out of range",
        )
        missing = validity == 0.0
        ensure(
            np.all(values[missing] == field.neutral)
            and np.all(age[missing] == 1.0),
            "invalid HUD cache missing representation",
        )
        return tuple(np.array(array, copy=True) for array in arrays)

    def set_corruption_state(self, state: Mapping[str, Any]) -> None:
        """export 済み state を全検証後に atomic import する。

        version/profile/schema binding、RNG、buffer、cache を一時変数へ復元し、
        一項目でも不正なら現在 wrapper state を部分更新しません。
        """
        ensure(
            isinstance(state, Mapping) and set(state) == _STATE_KEYS,
            "corruption state keys mismatch",
        )
        ensure(
            state["state_version"] == _CORRUPTION_STATE_VERSION,
            "unsupported corruption state version",
        )
        ensure(
            state["schema_hash"] == self.schema.schema_hash
            and state["profile_hash"] == self._profile_hash,
            "corruption state binding mismatch",
        )
        ensure(type(state["burst_active"]) is bool, "invalid burst state")
        ensure(
            isinstance(state["collapse_remaining"], int)
            and not isinstance(state["collapse_remaining"], bool)
            and state["collapse_remaining"] >= 0,
            "invalid collapse state",
        )

        candidate_rng = np.random.default_rng()
        try:
            candidate_rng.bit_generator.state = deepcopy(state["rng_state"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("invalid RNG state") from exc

        latency_payload = state["latency_buffer"]
        ensure(
            isinstance(latency_payload, list)
            and len(latency_payload) <= self._latency_buffer.maxlen,
            "invalid latency buffer",
        )
        candidate_latency: deque[np.ndarray] = deque(
            maxlen=self._latency_buffer.maxlen
        )
        for payload in latency_payload:
            array = np.asarray(payload, dtype=np.float32)
            candidate_latency.append(self._validated_release_tensor(array))

        cache_payload = state["hud_cache"]
        ensure(isinstance(cache_payload, Mapping), "invalid HUD cache")
        ensure(
            set(cache_payload) <= set(_HUD_STALE_PROBABILITIES),
            "unknown HUD cache field",
        )
        candidate_cache = {
            name: self._validated_cached_field(name, payload)
            for name, payload in cache_payload.items()
        }

        self._rng = candidate_rng
        self._burst_active = state["burst_active"]
        self._collapse_remaining = state["collapse_remaining"]
        self._latency_buffer = candidate_latency
        self._hud_cache = candidate_cache

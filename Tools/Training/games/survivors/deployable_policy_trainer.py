"""Survivors combat Student Policy の sequence distillation trainer。
GRU reset、KL/Huber loss、四段階 corruption curriculum、DAgger 境界、正式 dependency gate と
学習に関わる全 mutable state の checkpoint resume を一つの制御面にまとめます。
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.fidelity_verdict import FidelityVerdict, verify_current_fidelity
from reinbalance_survivors_contracts.perception_error import PerceptionErrorProfile
from games.survivors.combat_distillation_dataset import CombatDistillationDataset
CHECKPOINT_SCHEMA_VERSION = "survivors.deployable_policy_checkpoint.v1"
def _sha256(value: Any, label: str) -> str:
    """小文字 SHA-256 identity を検証する。
    formal profile や schema の取り違えを短縮値で曖昧にしません。
    """
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value
@dataclass(frozen=True)
class FormalDependencies:
    """正式 student run を解禁する current fidelity と measured profile の束縛。
    fixture/bootstrap を種別文字列で明示拒否し、profile hash は frozen config の期待値と比較します。
    """
    fidelity_verdict: FidelityVerdict | Mapping[str, Any]
    current_gating_producer_hashes: Mapping[str, str]
    perception_profile: PerceptionErrorProfile | None
    required_perception_profile_hash: str
    profile_source: str = "measured"
    def __post_init__(self) -> None:
        """formal dependency の source kind と基本型を fail-closed 検証する。
        measured と明示されない profile は内容から推測せず、この境界で拒否します。
        """
        if self.profile_source != "measured":
            raise ValueError("formal training requires a measured perception profile")
        _sha256(self.required_perception_profile_hash, "required_perception_profile_hash")
        if not isinstance(self.perception_profile, PerceptionErrorProfile):
            raise ValueError("formal training requires a measured perception profile")
        if not isinstance(self.current_gating_producer_hashes, Mapping):
            raise ValueError("current fidelity hashes are required")
    def validate(self) -> dict[str, str]:
        """current integration verdict と measured profile freshness を step 0 で検証する。
        blocking verdict、producer hash 差、空 calibration、期待 profile hash 差を全て停止します。
        development_only=True の profile は production formal training で拒否します。
        """
        profile = self.perception_profile
        assert profile is not None
        if not (hasattr(profile, "development_only") and profile.development_only is False):
            raise ValueError(
                "formal training requires a production (development_only=False) perception profile"
            )
        if profile.profile_hash != self.required_perception_profile_hash:
            raise ValueError("measured perception profile is stale")
        if not profile.calibration_session_ids:
            raise ValueError("measured perception profile requires calibration sessions")
        checked = verify_current_fidelity(
            self.fidelity_verdict, self.current_gating_producer_hashes, "integration",
        )
        if checked.blocking_reasons:
            raise ValueError("fidelity verdict contains blocking reasons")
        return {"fidelity_verdict": checked.identity_hash, "perception_profile": profile.profile_hash}
def validate_formal_dependencies(dependencies: FormalDependencies | None) -> dict[str, str]:
    """存在する正式 dependency 一式を検証して identities を返す。
    missing を development fallback にせず training step 0 の明示エラーにします。
    """
    if not isinstance(dependencies, FormalDependencies):
        raise ValueError("formal dependencies are missing")
    return dependencies.validate()
@dataclass(frozen=True)
class CurriculumStage:
    """一つの corruption curriculum 段階。
    名前、開始 update、clean から measured corruption への補間率を保持します。
    """
    name: str
    start_update: int
    corruption_scale: float
@dataclass(frozen=True)
class CurriculumConfig:
    """四段階移行と DAgger shard 追加 update を固定する config。
    unit test は短い境界へ差し替えられますが、run 中に条件を変更できない frozen 値です。
    """
    stage_start_updates: tuple[int, int, int, int] = (0, 1_000, 5_000, 10_000)
    dagger_add_updates: tuple[int, ...] = (5_000, 10_000)
    def __post_init__(self) -> None:
        """四つの単調な開始点とその一部である DAgger 境界を検証する。
        clean=0 を必須にし、途中から始まる曖昧な curriculum を作りません。
        """
        starts = self.stage_start_updates
        if len(starts) != 4 or starts[0] != 0 or any(type(x) is not int or x < 0 for x in starts) or tuple(sorted(set(starts))) != starts:
            raise ValueError("curriculum requires four increasing stage boundaries starting at zero")
        if any(type(x) is not int or x not in starts[1:] for x in self.dagger_add_updates) or len(set(self.dagger_add_updates)) != len(self.dagger_add_updates):
            raise ValueError("DAgger additions must use unique curriculum boundaries")
class CurriculumState:
    """現在 stage と追加済み DAgger shard を保持する resume state。
    update は後退できず、shard は config の厳密な追加点で一度だけ受け入れます。
    """
    def __init__(self, config: CurriculumConfig) -> None:
        """frozen config に対する clean 初期状態を作る。
        stage 定義は常に clean/light/measured/full の同じ順序です。
        """
        self.config = config
        self.stages = tuple(
            CurriculumStage(name, start, scale) for name, start, scale in zip(
                ("clean", "light", "measured", "full"), config.stage_start_updates, (0., 1 / 3, 2 / 3, 1.), strict=True,
            )
        )
        self.reset()
    def reset(self) -> None:
        """curriculum と DAgger を clean の初期状態へ戻す。
        checkpoint resume test と新規 run の両方が同じ初期化を使います。
        """
        self.completed_updates, self.stage_index, self._dagger_shards = 0, 0, []
    @property
    def current_stage(self) -> CurriculumStage:
        """現在の immutable stage 定義を返す。
        corruption wrapper の強度選択はこの値だけを参照できます。
        """
        return self.stages[self.stage_index]
    @property
    def dagger_shards(self) -> tuple[str, ...]:
        """追加順を保持した DAgger shard identities を返す。
        外部から list を変更できない tuple view として公開します。
        """
        return tuple(row["shard_id"] for row in self._dagger_shards)
    def advance(self, completed_updates: int) -> CurriculumStage:
        """固定 update 条件から到達可能な最新 stage へ進める。
        metric の後付けや update 後退を許さず resume 前後で同じ段階を選びます。
        """
        if type(completed_updates) is not int or completed_updates < self.completed_updates:
            raise ValueError("curriculum updates cannot move backwards")
        self.completed_updates = completed_updates
        self.stage_index = max(i for i, stage in enumerate(self.stages) if stage.start_update <= completed_updates)
        return self.current_stage
    def add_dagger_shard(self, shard_id: str, *, at_update: int) -> None:
        """config に固定した追加境界で新規 DAgger shard を受け入れる。
        境界外、未到達、空 identity、重複を既存 shard mutation より前に拒否します。
        """
        if at_update not in self.config.dagger_add_updates or at_update > self.completed_updates:
            raise ValueError("DAgger shard addition is outside a configured boundary")
        if not isinstance(shard_id, str) or not shard_id:
            raise ValueError("DAgger shard identity must be non-empty")
        if shard_id in self.dagger_shards:
            raise ValueError("duplicate DAgger shard")
        self._dagger_shards.append({"shard_id": shard_id, "added_at_update": at_update})
    def state_dict(self) -> dict[str, int]:
        """curriculum position を checkpoint wire へ変換する。
        config 自体は trainer 構築値へ束縛し、mutable position だけを保存します。
        """
        return {"start_updates": list(self.config.stage_start_updates), "completed_updates": self.completed_updates, "stage_index": self.stage_index}
    def dagger_state_dict(self) -> dict[str, Any]:
        """DAgger boundary config と shard identities を別 state として返す。
        curriculum state だけ復元して shard lineage を失う部分 resume を防ぎます。
        """
        return {"add_updates": list(self.config.dagger_add_updates), "shards": deepcopy(self._dagger_shards)}
    def load_state_dict(self, state: Mapping[str, Any], dagger: Mapping[str, Any]) -> None:
        """curriculum/DAgger sibling を binding 検証後に同時復元する。
        stage index は completed update から再計算し、保存値との矛盾を拒否します。
        """
        if set(state) != {"start_updates", "completed_updates", "stage_index"} or set(dagger) != {"add_updates", "shards"}:
            raise ValueError("curriculum checkpoint fields mismatch")
        if tuple(state["start_updates"]) != self.config.stage_start_updates or tuple(dagger["add_updates"]) != self.config.dagger_add_updates:
            raise ValueError("curriculum/DAgger boundary config mismatch")
        candidate = CurriculumState(self.config)
        candidate.advance(state["completed_updates"])
        if candidate.stage_index != state["stage_index"]:
            raise ValueError("curriculum stage mismatch")
        for row in dagger["shards"]:
            if not isinstance(row, Mapping) or set(row) != {"shard_id", "added_at_update"}:
                raise ValueError("invalid DAgger checkpoint state")
            candidate.add_dagger_shard(row["shard_id"], at_update=row["added_at_update"])
        self.completed_updates, self.stage_index, self._dagger_shards = candidate.completed_updates, candidate.stage_index, candidate._dagger_shards
class DeployableCombatPolicy(nn.Module):
    """DeployObs sequence を action logits と state value へ写像する GRU policy。
    episode reset mask が立つ batch 要素の hidden state を timestep 入力前にゼロへ戻します。
    """
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        """正の tensor 次元から GRUCell と二つの head を作る。
        actor/value が同じ recurrent state を共有する小さな deployable model です。
        """
        super().__init__()
        if any(type(x) is not int or x <= 0 for x in (observation_dim, action_dim, hidden_dim)):
            raise ValueError("policy dimensions must be positive integers")
        self.observation_dim, self.action_dim, self.hidden_dim = observation_dim, action_dim, hidden_dim
        self.recurrent = nn.GRUCell(observation_dim, hidden_dim)
        self.actor, self.value = nn.Linear(hidden_dim, action_dim), nn.Linear(hidden_dim, 1)
    def forward_sequence(self, observations: th.Tensor, episode_reset_mask: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """時系列順に reset を適用して actor/value 出力を返す。
        padding loss は trainer mask が除外しますが、reset semantics は model 内で明示処理します。
        """
        if observations.ndim != 3 or episode_reset_mask.shape != observations.shape[:2]:
            raise ValueError("policy sequence shape mismatch")
        hidden = observations.new_zeros((observations.shape[0], self.hidden_dim))
        logits, values = [], []
        for index in range(observations.shape[1]):
            hidden = hidden.masked_fill(episode_reset_mask[:, index, None], 0.0)
            hidden = self.recurrent(observations[:, index], hidden)
            logits.append(self.actor(hidden))
            values.append(self.value(hidden).squeeze(-1))
        return th.stack(logits, 1), th.stack(values, 1)
def sequence_distillation_loss(
    student_logits: th.Tensor, student_values: th.Tensor, dataset: CombatDistillationDataset,
    *, actor_weight: float = 1.0, value_weight: float = 1.0, huber_delta: float = 1.0,
) -> dict[str, th.Tensor]:
    """教師 logits の KL と value Huber を non-burn valid timestep で集計する。
    padding と recurrent state warm-up は actor/value の両 sibling loss から同じ mask で除外します。
    """
    device = student_logits.device
    mask = th.as_tensor(dataset.valid_mask & ~dataset.burn_in_mask, device=device).bool()
    if student_logits.shape[:2] != mask.shape or student_values.shape != mask.shape or not bool(mask.any()):
        raise ValueError("student output or distillation mask is invalid")
    teacher_logits = th.as_tensor(np.array(dataset.action_logits), device=device)
    teacher_values = th.as_tensor(np.array(dataset.teacher_values), device=device)
    actor = F.kl_div(
        F.log_softmax(student_logits[mask], dim=-1), F.softmax(teacher_logits[mask], dim=-1), reduction="batchmean",
    )
    value = F.huber_loss(student_values[mask], teacher_values[mask], delta=huber_delta)
    return {"actor_kl": actor, "value_huber": value, "total": actor * actor_weight + value * value_weight}
def _vec_state(value: Any | None) -> bytes | None:
    """VecNormalize の組込み pickle state を checkpoint payload 化する。
    SB3 が venv を除いて保存する公式 state と running statistics を同じ世代で保持します。
    """
    try:
        return None if value is None else pickle.dumps(value, protocol=5)
    except (pickle.PickleError, TypeError) as exc:
        raise ValueError(f"VecNormalize state is not serializable: {exc}") from exc
def _restore_vec(value: Any | None, state: bytes | None) -> None:
    """VecNormalize pickle の型と有無を照合して既存 wrapper へ復元する。
    現在の venv 接続は維持し、保存された normalization 属性だけを同じ class から戻します。
    """
    if (value is None) != (state is None):
        raise ValueError("VecNormalize checkpoint presence mismatch")
    if value is None or state is None:
        return
    try:
        restored = pickle.loads(state)
    except (pickle.PickleError, EOFError, TypeError) as exc:
        raise ValueError(f"invalid VecNormalize checkpoint: {exc}") from exc
    if type(restored) is not type(value) or not hasattr(restored, "obs_rms"):
        raise ValueError("VecNormalize checkpoint type mismatch")
    preserved_venv = getattr(value, "venv", None)
    value.__dict__.update(deepcopy(restored.__dict__))
    if preserved_venv is not None:
        value.venv = preserved_venv
class DeployablePolicyTrainer:
    """release-gated sequence optimizer と完全 checkpoint resume を提供する。
    dataset/formal gate を optimizer mutation 前に実行し、development run は常に非昇格 metadata を持ちます。
    """
    def __init__(
        self, model: DeployableCombatPolicy, *, formal_mode: bool = False,
        formal_dependencies: FormalDependencies | None = None,
        curriculum_config: CurriculumConfig | None = None, learning_rate: float = 3e-4,
        corrupt_fn: Any | None = None,
    ) -> None:
        """model、AdamW、frozen curriculum と formal dependency 候補を初期化する。
        corrupt_fn は (obs_flat: ndarray, scale: float) -> ndarray の callable で、
        curriculum stage の corruption_scale を学習観測へ反映するために使います。
        dependency の鮮度検証自体は dataset と同じ training step 0 boundary まで遅延します。
        """
        if not isinstance(model, DeployableCombatPolicy) or type(formal_mode) is not bool:
            raise ValueError("invalid deployable trainer configuration")
        self.model, self.formal_mode, self.formal_dependencies = model, formal_mode, formal_dependencies
        self.optimizer = th.optim.AdamW(model.parameters(), lr=learning_rate)
        self.curriculum = CurriculumState(curriculum_config or CurriculumConfig())
        self.training_steps, self._formal_identities = 0, None
        self.schema = DeployObsSchema.default_v1()
        self.corrupt_fn = corrupt_fn
    def _step_zero_gate(self, dataset: CombatDistillationDataset) -> None:
        """全 release leakage と formal dependency を最初の optimizer 呼出し前に検証する。
        development run も raw/action/split leakage は許さず、formal run だけ外部監査も要求します。
        """
        dataset.assert_release_training_ready(self.schema)
        if self.formal_mode and self._formal_identities is None:
            self._formal_identities = validate_formal_dependencies(self.formal_dependencies)
    def compute_loss(
        self,
        dataset: CombatDistillationDataset,
        *,
        corruption_scale: float = 0.0,
        dagger_datasets: Sequence[CombatDistillationDataset] = (),
    ) -> dict[str, th.Tensor]:
        """curriculum corruption と DAgger shard を反映した distillation loss を計算する。
        dagger_datasets はバッチ次元へ連結して扱い、corruption_scale は corrupt_fn に渡します。
        """
        if dagger_datasets:
            for dagger in dagger_datasets:
                if dagger.deploy_schema_hash != dataset.deploy_schema_hash:
                    raise ValueError("DAgger dataset deploy schema hash mismatch")
                # DAgger dataset も main dataset と同じ release gate を通す
                dagger.assert_release_training_ready(self.schema)
            observations = np.concatenate(
                [np.array(dataset.observations)] + [np.array(d.observations) for d in dagger_datasets]
            )
            teacher_logits_np = np.concatenate(
                [np.array(dataset.action_logits)] + [np.array(d.action_logits) for d in dagger_datasets]
            )
            teacher_values_np = np.concatenate(
                [np.array(dataset.teacher_values)] + [np.array(d.teacher_values) for d in dagger_datasets]
            )
            valid_mask = np.concatenate(
                [np.array(dataset.valid_mask)] + [np.array(d.valid_mask) for d in dagger_datasets]
            )
            burn_mask = np.concatenate(
                [np.array(dataset.burn_in_mask)] + [np.array(d.burn_in_mask) for d in dagger_datasets]
            )
            resets = np.concatenate(
                [np.array(dataset.episode_reset_mask)] + [np.array(d.episode_reset_mask) for d in dagger_datasets]
            )
        else:
            observations = np.array(dataset.observations)
            teacher_logits_np = np.array(dataset.action_logits)
            teacher_values_np = np.array(dataset.teacher_values)
            valid_mask = np.array(dataset.valid_mask)
            burn_mask = np.array(dataset.burn_in_mask)
            resets = np.array(dataset.episode_reset_mask)
        if self.corrupt_fn is not None and corruption_scale > 0.0:
            shape = observations.shape
            observations = np.asarray(
                self.corrupt_fn(observations.reshape(-1, shape[-1]), corruption_scale),
                dtype=np.float32,
            ).reshape(shape)
        device = next(self.model.parameters()).device
        obs_t = th.as_tensor(observations, device=device)
        resets_t = th.as_tensor(resets, device=device).bool()
        logits, values = self.model.forward_sequence(obs_t, resets_t)
        mask = th.as_tensor(valid_mask & ~burn_mask, device=device).bool()
        if not bool(mask.any()):
            raise ValueError("distillation mask has no valid non-burn timesteps")
        teacher_logits_t = th.as_tensor(teacher_logits_np, device=device)
        teacher_values_t = th.as_tensor(teacher_values_np, device=device)
        actor = F.kl_div(
            F.log_softmax(logits[mask], dim=-1),
            F.softmax(teacher_logits_t[mask], dim=-1),
            reduction="batchmean",
        )
        value = F.huber_loss(values[mask], teacher_values_t[mask], delta=1.0)
        return {"actor_kl": actor, "value_huber": value, "total": actor + value}
    def train_step(
        self,
        dataset: CombatDistillationDataset,
        *,
        dagger_datasets: Sequence[CombatDistillationDataset] = (),
    ) -> dict[str, float]:
        """step-0 gate 後に corruption と DAgger shard を反映した optimizer update を行う。
        gate または loss が失敗した場合は counter を進めず resume 上の成功として扱いません。
        """
        self._step_zero_gate(dataset)
        scale = self.curriculum.current_stage.corruption_scale
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.compute_loss(dataset, corruption_scale=scale, dagger_datasets=dagger_datasets)
        losses["total"].backward()
        self.optimizer.step()
        self.training_steps += 1
        self.curriculum.advance(self.training_steps)
        return {name: float(value.detach().cpu()) for name, value in losses.items()}
    def _model_config(self) -> dict[str, int]:
        """checkpoint/package reconstruction に必要な model 次元を返す。
        実装 class を推測せず固定三次元だけを serialization contract にします。
        """
        return {"observation_dim": self.model.observation_dim, "action_dim": self.model.action_dim, "hidden_dim": self.model.hidden_dim}
    def save_checkpoint(self, path: Path, *, vec_normalize: Any | None = None, error_wrappers: Sequence[Any] = ()) -> None:
        """model/optimizer/normalization/error RNG/curriculum/DAgger を一つへ保存する。
        formal metadata は dependency 検証済みの場合だけ eligible=true とし、development は常に false です。
        """
        if self.formal_mode and self._formal_identities is None:
            self._formal_identities = validate_formal_dependencies(self.formal_dependencies)
        if self.curriculum.completed_updates != self.training_steps:
            raise ValueError("curriculum/training step checkpoint mismatch")
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION, "model_config": self._model_config(),
            "model_state_dict": self.model.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(),
            "training_steps": self.training_steps, "deploy_schema_hash": self.schema.schema_hash,
            "development_only": not self.formal_mode, "formal_student_eligible": self.formal_mode,
            "formal_dependency_identities": dict(self._formal_identities or {}), "vec_normalize_state": _vec_state(vec_normalize),
            "error_wrapper_states": [deepcopy(wrapper.get_corruption_state()) for wrapper in error_wrappers],
            "curriculum_state": self.curriculum.state_dict(), "dagger_state": self.curriculum.dagger_state_dict(),
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        th.save(payload, destination)
    def load_checkpoint(self, path: Path, *, vec_normalize: Any | None = None, error_wrappers: Sequence[Any] = ()) -> None:
        """全 binding と sibling state の存在を確認して checkpoint を復元する。
        formal trainer は development checkpoint を resume できず、wrapper 数や VecNormalize 有無の差も拒否します。
        """
        payload = th.load(Path(path), map_location=next(self.model.parameters()).device, weights_only=False)
        required = {
            "schema_version", "model_config", "model_state_dict", "optimizer_state_dict", "training_steps", "deploy_schema_hash",
            "development_only", "formal_student_eligible", "formal_dependency_identities", "vec_normalize_state",
            "error_wrapper_states", "curriculum_state", "dagger_state",
        }
        if not isinstance(payload, Mapping) or set(payload) != required or payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("deployable checkpoint fields mismatch")
        if payload["model_config"] != self._model_config() or payload["deploy_schema_hash"] != self.schema.schema_hash:
            raise ValueError("deployable checkpoint binding mismatch")
        if type(payload["development_only"]) is not bool or type(payload["formal_student_eligible"]) is not bool or payload["development_only"] == payload["formal_student_eligible"]:
            raise ValueError("checkpoint eligibility metadata is inconsistent")
        if self.formal_mode and not payload["formal_student_eligible"]:
            raise ValueError("formal trainer cannot resume development checkpoint")
        states = payload["error_wrapper_states"]
        if not isinstance(states, list) or len(states) != len(error_wrappers):
            raise ValueError("error wrapper checkpoint count mismatch")
        if payload["curriculum_state"]["completed_updates"] != payload["training_steps"]:
            raise ValueError("curriculum/training step checkpoint mismatch")
        # formal resume は state mutation 前に current dependencies を検証し、
        # missing・stale・identity mismatch の場合は model/optimizer/curriculum を変更しない。
        if self.formal_mode:
            fresh = validate_formal_dependencies(self.formal_dependencies)
            checkpoint_ids = dict(payload["formal_dependency_identities"])
            if fresh != checkpoint_ids:
                raise ValueError("formal dependency identities mismatch with checkpoint")
            pending_identities: dict[str, str] | None = fresh
        else:
            pending_identities = None
        # 全 preflight 検証完了後にのみ state を変更する
        _restore_vec(vec_normalize, payload["vec_normalize_state"])
        self.curriculum.load_state_dict(payload["curriculum_state"], payload["dagger_state"])
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        for wrapper, state in zip(error_wrappers, states, strict=True):
            wrapper.set_corruption_state(deepcopy(state))
        self.training_steps = payload["training_steps"]
        self._formal_identities = pending_identities

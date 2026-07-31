"""Survivors recurrent critic feature 上の extrinsic utility overlay を定義する。

source policy 全体を freeze し、critic latent を入力とする小さな head だけを paired
outcome target で学習する。artifact は source/model/schema/context へ fail-closed に束縛する。
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from torch import nn

from reinbalance_survivors_contracts.canonical_json import (
    canonical_hash,
    canonical_json_bytes,
    sha256_hex,
)

from games.survivors.recurrent_policy_session import CriticContext
from games.survivors.teacher_validation_split import freeze_episode_split
from games.survivors.value_source_loader import LoadedValueSource

SCHEMA_VERSION = "survivors.extrinsic_value_overlay.v1"
STATE_SCHEMA_VERSION = "survivors.extrinsic_value_overlay_state.v1"
MANIFEST_FILENAME = "extrinsic_value_overlay_manifest.json"
STATE_FILENAME = "extrinsic_value_overlay.pt"
HEAD_HIDDEN_DIM = 128
HUBER_WEIGHT = 1.0
PAIRWISE_WEIGHT = 0.5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 10
_ELIGIBLE_STATUS = "observed"
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "overlay_identity",
        "source_model_hash",
        "schema_hash",
        "vecnormalize_hash",
        "actor_invariance_hash",
        "context_shape",
        "input_dim",
        "head_architecture",
        "state_dict_filename",
        "state_dict_hash",
        "paired_dataset_hash",
        "target_mean",
        "target_std",
        "loss_weights",
        "optimizer",
        "max_epochs",
        "patience",
        "epochs_run",
        "best_epoch",
        "best_val_ndcg",
        "final_test_status",
    }
)


class ExtrinsicValueOverlayError(ValueError):
    """overlay dataset・学習・artifact binding 違反を表す例外。

    source 不一致や final-test leakage を警告で継続せず、公開前の一境界で拒否する。
    """


class ExtrinsicValueHead(nn.Module):
    """critic latent から標準化 extrinsic utility を予測する head。

    architecture は Linear(input, 128) → GELU → Linear(128, 1) に固定する。
    """

    def __init__(self, input_dim: int) -> None:
        """正の critic latent 次元から固定 head を構築する。

        layer 幅を manifest と実 state dict の両方で検証できる形に限定する。
        """
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0:
            raise ExtrinsicValueOverlayError("input_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HEAD_HIDDEN_DIM, 1),
        )

    def forward(self, features: th.Tensor) -> th.Tensor:
        """二次元 critic feature batch の scalar prediction を返す。

        暗黙 flatten を行わず、最後の singleton output 軸だけを除く。
        """
        if features.ndim != 2:
            raise ExtrinsicValueOverlayError(
                "overlay features must be a two-dimensional batch"
            )
        return self.network(features).squeeze(-1)


@dataclass(frozen=True, slots=True)
class OverlayExample:
    """一 candidate branch の feature・extrinsic target・group identity。

    status/quarantine は scalar と pairwise の両 loss mask へ同じ eligibility を適用する。
    """

    example_id: str
    episode_id: str
    decision_seed: str
    decision_id: str
    choice_id: str
    features: np.ndarray
    target: float
    primary_rank: tuple[Any, ...]
    status: str = _ELIGIBLE_STATUS
    quarantined: bool = False

    def __post_init__(self) -> None:
        """identity・finite feature/target・rank を construction 時に検証する。

        caller 配列の後編集で dataset hash と学習入力が分岐しないよう read-only copy を保持する。
        """
        for field in (
            "example_id",
            "episode_id",
            "decision_seed",
            "decision_id",
            "choice_id",
            "status",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ExtrinsicValueOverlayError(f"{field} must be non-empty")
        if type(self.quarantined) is not bool:
            raise ExtrinsicValueOverlayError("quarantined must be bool")
        try:
            feature_array = np.asarray(self.features, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ExtrinsicValueOverlayError("features must be numeric") from exc
        if (
            feature_array.ndim != 1
            or feature_array.size <= 0
            or not np.all(np.isfinite(feature_array))
        ):
            raise ExtrinsicValueOverlayError(
                "features must be a finite non-empty vector"
            )
        if (
            isinstance(self.target, bool)
            or not isinstance(self.target, (int, float))
            or not math.isfinite(float(self.target))
        ):
            raise ExtrinsicValueOverlayError("target must be finite")
        if not isinstance(self.primary_rank, tuple) or not self.primary_rank:
            raise ExtrinsicValueOverlayError(
                "primary_rank must be a non-empty tuple"
            )
        for item in self.primary_rank:
            if isinstance(item, bool):
                continue
            if (
                not isinstance(item, (int, float, str))
                or (
                    isinstance(item, (int, float))
                    and not math.isfinite(float(item))
                )
            ):
                raise ExtrinsicValueOverlayError(
                    "primary_rank values must be finite scalar values"
                )
        detached = feature_array.copy()
        detached.setflags(write=False)
        object.__setattr__(self, "features", detached)
        object.__setattr__(self, "target", float(self.target))

    @property
    def loss_eligible(self) -> bool:
        """scalar/pairwise loss に使える観測済み branch かを返す。

        censored・early failure・failed・quarantine を全 partition で対称に除外する。
        """
        return self.status == _ELIGIBLE_STATUS and not self.quarantined

    def to_wire(self) -> dict[str, Any]:
        """dataset identity 用の有限 canonical object を返す。

        float32 feature は値リストとして固定し、配列バイトの独自 hash は導入しない。
        """
        return {
            "example_id": self.example_id,
            "episode_id": self.episode_id,
            "decision_seed": self.decision_seed,
            "decision_id": self.decision_id,
            "choice_id": self.choice_id,
            "features": [float(value) for value in self.features],
            "target": self.target,
            "primary_rank": list(self.primary_rank),
            "status": self.status,
            "quarantined": self.quarantined,
        }

    @classmethod
    def from_wire(cls, value: Any) -> "OverlayExample":
        """exact-schema training row を OverlayExample へ変換する。

        未知 field や rank/list 以外を受理せず dataset producer の drift を閉じる。
        """
        fields = {
            "example_id",
            "episode_id",
            "decision_seed",
            "decision_id",
            "choice_id",
            "features",
            "target",
            "primary_rank",
            "status",
            "quarantined",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ExtrinsicValueOverlayError(
                "overlay example fields mismatch"
            )
        if not isinstance(value["primary_rank"], list):
            raise ExtrinsicValueOverlayError(
                "primary_rank must be an array"
            )
        return cls(
            example_id=value["example_id"],
            episode_id=value["episode_id"],
            decision_seed=value["decision_seed"],
            decision_id=value["decision_id"],
            choice_id=value["choice_id"],
            features=np.asarray(value["features"], dtype=np.float32),
            target=value["target"],
            primary_rank=tuple(value["primary_rank"]),
            status=value["status"],
            quarantined=value["quarantined"],
        )


class OverlayPartitions:
    """episode/decision-seed 固定 split と train-only target scale を保持する。

    final test は method lock の明示なしに取得できず、fit/model selection API へ公開しない。
    """

    def __init__(
        self,
        examples: Sequence[OverlayExample],
        assignments: Mapping[str, str],
        *,
        target_mean: float,
        target_std: float,
        dataset_hash: str,
    ) -> None:
        """検証済み examples と immutable assignment を保持する。

        partition ごとの選択は内部で行い、final open 状態だけを明示的に追跡する。
        """
        self.examples = tuple(examples)
        self.assignments = dict(assignments)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.dataset_hash = dataset_hash
        self._final_test_opened = False

    def assignment_for(self, example: OverlayExample) -> str:
        """example の episode binding から partition 名を返す。

        object identity や branch 順ではなく immutable episode ID だけを参照する。
        """
        try:
            return self.assignments[example.episode_id]
        except KeyError as exc:
            raise ExtrinsicValueOverlayError(
                "example episode is not assigned"
            ) from exc

    def development_examples(
        self,
        partition: str,
    ) -> tuple[OverlayExample, ...]:
        """development train/validation の eligible rows だけを返す。

        final_test 名を渡して開封を迂回できないよう二つの development 名へ限定する。
        """
        if partition not in {
            "development_train",
            "development_validation",
        }:
            raise ExtrinsicValueOverlayError(
                "only development partitions are available for fit"
            )
        return tuple(
            example
            for example in self.examples
            if example.loss_eligible
            and self.assignment_for(example) == partition
        )

    @property
    def final_test_opened(self) -> bool:
        """untouched final test の開封状態を返す。

        fit_overlay はこの flag を変更せず、revalidation caller だけが開封できる。
        """
        return self._final_test_opened

    def open_final_test(
        self,
        *,
        method_locked: bool,
    ) -> tuple[OverlayExample, ...]:
        """method lock 後に限り final rows を一度開封する。

        calibration fit や checkpoint 選択の途中では false を渡すしかなく、必ず拒否される。
        """
        if method_locked is not True:
            raise ExtrinsicValueOverlayError(
                "final test requires an explicit method lock"
            )
        if self._final_test_opened:
            raise ExtrinsicValueOverlayError(
                "final test has already been opened"
            )
        self._final_test_opened = True
        return tuple(
            example
            for example in self.examples
            if self.assignment_for(example) == "final_test"
        )

    @property
    def scalar_example_ids(self) -> set[str]:
        """全 partition の scalar eligibility 診断集合を返す。

        split を変えずに censored/quarantine mask の対称性をテストする用途に限定する。
        """
        return {
            example.example_id
            for example in self.examples
            if example.loss_eligible
        }

    @property
    def pair_example_ids(self) -> set[str]:
        """非 tie pairwise loss に実際に参加できる example IDs を返す。

        同 decision の eligible sibling が不足する候補と primary tie を除く。
        """
        identifiers: set[str] = set()
        for partition in (
            "development_train",
            "development_validation",
            "final_test",
        ):
            examples = tuple(
                example
                for example in self.examples
                if example.loss_eligible
                and self.assignment_for(example) == partition
            )
            for left, right in _pairwise_examples(examples):
                identifiers.add(left.example_id)
                identifiers.add(right.example_id)
        return identifiers


@dataclass(frozen=True, slots=True)
class OverlayTrainingSummary:
    """best validation NDCG checkpoint と artifact metadata を保持する。

    final test 指標は持たず、development model selection の結果だけを保存対象へ渡す。
    """

    target_mean: float
    target_std: float
    best_val_ndcg: float
    best_epoch: int
    epochs_run: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    actor_invariance_hash: str


class ExtrinsicValueOverlay:
    """frozen source と独立 trainable head を組み合わせる runtime overlay。

    source policy は子 module に登録せず、state dict と optimizer は head だけを対象にする。
    """

    def __init__(
        self,
        source: LoadedValueSource,
        head: ExtrinsicValueHead,
        *,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ) -> None:
        """source binding・head・target normalization を保持する。

        construction 時の source policy hash を保存し、学習・保存前後の bitwise freeze を検査する。
        """
        self.source = source
        self.head = head
        self.input_dim = int(head.network[0].in_features)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        if not math.isfinite(self.target_mean) or not (
            math.isfinite(self.target_std) and self.target_std > 0.0
        ):
            raise ExtrinsicValueOverlayError(
                "target normalization must be finite with positive std"
            )
        for parameter in source.policy.parameters():
            parameter.requires_grad_(False)
        source.policy.set_training_mode(False)
        self._policy_hash_at_creation = _state_dict_hash(
            source.policy.state_dict()
        )
        self._actor_invariance_hash = policy_invariance_hash(source.policy)

    @classmethod
    def create(cls, source: LoadedValueSource) -> "ExtrinsicValueOverlay":
        """source critic latent 次元から新しい overlay を作る。

        source の value_net を複製せず固定 128-unit head だけを policy device に配置する。
        """
        input_dim = getattr(source.policy.mlp_extractor, "latent_dim_vf", None)
        if type(input_dim) is not int or input_dim <= 0:
            raise ExtrinsicValueOverlayError(
                "source critic latent dimension is invalid"
            )
        head = ExtrinsicValueHead(input_dim).to(source.policy.device)
        return cls(source, head)

    @property
    def actor_invariance_hash(self) -> str:
        """construction 時の actor/feature/critic-LSTM hash を返す。

        manifest はこの値へ binding され、load 時に現在 source から再計算する。
        """
        return self._actor_invariance_hash

    def make_optimizer(
        self,
        *,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
    ) -> th.optim.Adam:
        """overlay head parameter だけを持つ Adam optimizer を作る。

        requires_grad filter ではなく head の明示列挙により frozen source の登録を防ぐ。
        """
        if not (math.isfinite(learning_rate) and learning_rate > 0.0):
            raise ExtrinsicValueOverlayError(
                "learning_rate must be positive"
            )
        if not (math.isfinite(weight_decay) and weight_decay >= 0.0):
            raise ExtrinsicValueOverlayError(
                "weight_decay must be non-negative"
            )
        return th.optim.Adam(
            list(self.head.parameters()),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )

    def assert_source_frozen(self) -> None:
        """source policy 全 state dict の bitwise equality を検査する。

        actor invariance group に含まれない critic MLP/value head の accidental update も拒否する。
        """
        if _state_dict_hash(self.source.policy.state_dict()) != (
            self._policy_hash_at_creation
        ):
            raise ExtrinsicValueOverlayError(
                "source policy changed while training overlay"
            )
        if policy_invariance_hash(self.source.policy) != (
            self._actor_invariance_hash
        ):
            raise ExtrinsicValueOverlayError(
                "actor/feature/critic LSTM invariance changed"
            )

    def predict_normalized(
        self,
        normalized_obs_batch: np.ndarray,
        context: CriticContext,
    ) -> np.ndarray:
        """source critic latent から標準化 utility prediction を返す。

        source は no-grad/eval のまま使い、candidate ごとに同じ recurrent hidden-in を複製する。
        """
        features = _extract_critic_latent(
            self.source,
            normalized_obs_batch,
            context,
        )
        self.head.eval()
        with th.no_grad():
            predicted = self.head(features)
        result = predicted.detach().cpu().numpy().astype(
            np.float32,
            copy=False,
        )
        if result.ndim != 1 or not np.all(np.isfinite(result)):
            raise ExtrinsicValueOverlayError(
                "overlay returned invalid candidate values"
            )
        return result

    def unscale_target(self, normalized_value: float) -> float:
        """標準化 prediction を extrinsic utility scale へ戻す。

        raw critic の reward RMS は使わず、train partition だけで fit した mean/std を使う。
        """
        return (
            float(normalized_value) * self.target_std + self.target_mean
        )


def _state_dict_hash(state_dict: Mapping[str, th.Tensor]) -> str:
    """tensor state dict の bitwise canonical identity を返す。

    tensor bytes の SHA-256 と metadata 集合の canonical hash は共有 canonical_json 実装へ委譲する。
    """
    entries: list[dict[str, Any]] = []
    for name, tensor in sorted(state_dict.items()):
        detached = tensor.detach().cpu().contiguous()
        entries.append(
            {
                "name": name,
                "dtype": str(detached.dtype),
                "shape": list(detached.shape),
                "bytes_sha256": sha256_hex(detached.numpy().tobytes()),
            }
        )
    return canonical_hash(entries)


def _module_state(prefix: str, module: Any) -> dict[str, th.Tensor]:
    """optional torch module の state dict を prefix 付きで返す。

    shared/disabled recurrent path でも同じ hash payload 型を維持する。
    """
    if module is None:
        return {}
    return {
        f"{prefix}.{name}": tensor
        for name, tensor in module.state_dict().items()
    }


def policy_invariance_hash(policy: Any) -> str:
    """actor weights・feature extractor・critic LSTM の bitwise hash を返す。

    separate/shared/disabled critic sibling を列挙し、各構成を同じ canonical group payload にする。
    """
    actor: dict[str, th.Tensor] = {}
    for name in ("lstm_actor", "action_net"):
        actor.update(_module_state(name, getattr(policy, name, None)))
    actor.update(
        _module_state(
            "policy_net",
            getattr(policy.mlp_extractor, "policy_net", None),
        )
    )
    log_std = getattr(policy, "log_std", None)
    if isinstance(log_std, th.Tensor):
        actor["log_std"] = log_std

    features: dict[str, th.Tensor] = {}
    seen_modules: set[int] = set()
    for name in (
        "features_extractor",
        "pi_features_extractor",
        "vf_features_extractor",
    ):
        module = getattr(policy, name, None)
        if module is not None and id(module) not in seen_modules:
            seen_modules.add(id(module))
            features.update(_module_state(name, module))

    critic_lstm = getattr(policy, "lstm_critic", None)
    if critic_lstm is None and bool(getattr(policy, "shared_lstm", False)):
        critic_lstm = getattr(policy, "lstm_actor", None)
    payload = {
        "actor": _state_dict_hash(actor),
        "feature_extractor": _state_dict_hash(features),
        "critic_lstm": _state_dict_hash(
            _module_state("critic_lstm", critic_lstm)
        ),
    }
    return canonical_hash(payload)


def _source_bindings(source: LoadedValueSource) -> dict[str, Any]:
    """LoadedValueSource から overlay の immutable binding を抽出する。

    descriptor を再 hash せず、loader が検証した model/VecNormalize/schema identity を使う。
    """
    return {
        "source_model_hash": source.descriptor["artifacts"]["model"][
            "sha256"
        ],
        "schema_hash": source.policy_state_schema[
            "policy_state_schema_hash"
        ],
        "vecnormalize_hash": source.descriptor["artifacts"][
            "vecnormalize"
        ]["sha256"],
        "context_shape": list(
            source.policy_state_schema["vf_state_shape"]
        ),
    }


def _pairwise_examples(
    examples: Sequence[OverlayExample],
) -> list[tuple[OverlayExample, OverlayExample]]:
    """同 decision 内の eligible 非 tie pair を winner-first で返す。

    candidate 順や choice ID を教師順位へ使わず primary rank tuple だけを比較する。
    """
    grouped: dict[str, list[OverlayExample]] = defaultdict(list)
    for example in examples:
        if example.loss_eligible:
            grouped[example.decision_id].append(example)
    pairs: list[tuple[OverlayExample, OverlayExample]] = []
    for decision_examples in grouped.values():
        ordered = sorted(
            decision_examples,
            key=lambda item: item.example_id,
        )
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                if left.primary_rank == right.primary_rank:
                    continue
                pairs.append(
                    (left, right)
                    if left.primary_rank > right.primary_rank
                    else (right, left)
                )
    return pairs


def prepare_overlay_partitions(
    examples: Sequence[OverlayExample],
    *,
    split_seed: str,
    require_all_partitions: bool = True,
) -> OverlayPartitions:
    """examples を episode/decision-seed group で三 partition へ固定する。

    target mean/std は eligible development_train rows だけから計算し final target を参照しない。
    """
    if (
        isinstance(examples, (str, bytes))
        or not examples
        or any(not isinstance(item, OverlayExample) for item in examples)
    ):
        raise ExtrinsicValueOverlayError(
            "overlay examples are required"
        )
    identifiers = [example.example_id for example in examples]
    if len(set(identifiers)) != len(identifiers):
        raise ExtrinsicValueOverlayError(
            "overlay example IDs must be unique"
        )
    input_dims = {int(example.features.size) for example in examples}
    if len(input_dims) != 1:
        raise ExtrinsicValueOverlayError(
            "overlay feature dimensions must match"
        )
    episode_seeds: dict[str, set[str]] = defaultdict(set)
    decision_choices: set[tuple[str, str]] = set()
    for example in examples:
        episode_seeds[example.episode_id].add(example.decision_seed)
        decision_choice = (example.decision_id, example.choice_id)
        if decision_choice in decision_choices:
            raise ExtrinsicValueOverlayError(
                "choice IDs must be unique within each decision"
            )
        decision_choices.add(decision_choice)
    group_ids = {
        episode_id: (
            f"{episode_id}\x1f"
            + canonical_hash(
                {"decision_seeds": sorted(decision_seeds)}
            )
        )
        for episode_id, decision_seeds in episode_seeds.items()
    }
    if len(group_ids) < 3:
        if require_all_partitions:
            raise ExtrinsicValueOverlayError(
                "at least three episode/decision-seed groups are required"
            )
        assignments = {
            episode_id: "development_train"
            for episode_id in group_ids
        }
    else:
        split = freeze_episode_split(
            sorted(group_ids.values()),
            seed=split_seed,
        )
        assignments = {
            episode_id: split["episode_partitions"][group_id]
            for episode_id, group_id in group_ids.items()
        }
    train_targets = np.asarray(
        [
            example.target
            for example in examples
            if example.loss_eligible
            and assignments[example.episode_id] == "development_train"
        ],
        dtype=np.float64,
    )
    if train_targets.size == 0:
        raise ExtrinsicValueOverlayError(
            "development_train has no eligible targets"
        )
    if require_all_partitions:
        validation_count = sum(
            example.loss_eligible
            and assignments[example.episode_id]
            == "development_validation"
            for example in examples
        )
        if validation_count == 0:
            raise ExtrinsicValueOverlayError(
                "development_validation has no eligible targets"
            )
    target_mean = float(np.mean(train_targets))
    measured_std = float(np.std(train_targets))
    target_std = measured_std if measured_std > 1e-8 else 1.0
    dataset_hash = canonical_hash(
        [
            example.to_wire()
            for example in sorted(
                examples,
                key=lambda item: item.example_id,
            )
        ]
    )
    return OverlayPartitions(
        examples,
        assignments,
        target_mean=target_mean,
        target_std=target_std,
        dataset_hash=dataset_hash,
    )


def _feature_tensor(
    examples: Sequence[OverlayExample],
    *,
    device: Any,
) -> th.Tensor:
    """example feature rows を contiguous float32 tensor に変換する。

    source policy tensor を再利用せず overlay head の device へ直接配置する。
    """
    array = np.stack([example.features for example in examples])
    return th.as_tensor(array, dtype=th.float32, device=device)


def _target_tensor(
    examples: Sequence[OverlayExample],
    *,
    mean: float,
    std: float,
    device: Any,
) -> th.Tensor:
    """train-only mean/std で標準化した target tensor を返す。

    validation も同じ train scale を適用し、partition ごとの再 fit を行わない。
    """
    values = np.asarray(
        [(example.target - mean) / std for example in examples],
        dtype=np.float32,
    )
    return th.as_tensor(values, dtype=th.float32, device=device)


def _pair_indices(
    examples: Sequence[OverlayExample],
) -> list[tuple[int, int]]:
    """winner-first example pair を tensor row index に変換する。

    primary ties と mask 対象 branch は _pairwise_examples で除外済みとする。
    """
    index = {
        example.example_id: row
        for row, example in enumerate(examples)
    }
    return [
        (index[winner.example_id], index[loser.example_id])
        for winner, loser in _pairwise_examples(examples)
    ]


def _validation_ndcg(
    examples: Sequence[OverlayExample],
    predictions: np.ndarray,
) -> float:
    """decision ごとの NDCG を計算し validation 平均を返す。

    primary rank の dense relevance を使い、候補入力順は choice ID の tie-break 診断だけに限定する。
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example.decision_id].append(index)
    values: list[float] = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        distinct_ranks = sorted(
            {examples[index].primary_rank for index in indices},
            reverse=True,
        )
        if len(distinct_ranks) < 2:
            continue
        relevance = {
            rank: float(len(distinct_ranks) - rank_index)
            for rank_index, rank in enumerate(distinct_ranks)
        }
        predicted_order = sorted(
            indices,
            key=lambda index: (
                -float(predictions[index]),
                examples[index].choice_id,
            ),
        )
        ideal_order = sorted(
            indices,
            key=lambda index: (
                -relevance[examples[index].primary_rank],
                examples[index].choice_id,
            ),
        )

        def dcg(order: Sequence[int]) -> float:
            """candidate 順の discounted cumulative gain を返す。

            decision 内の全候補を使い、validation fixture の候補数へ依存しない。
            """
            return sum(
                (2.0 ** relevance[examples[index].primary_rank] - 1.0)
                / math.log2(position + 2.0)
                for position, index in enumerate(order)
            )

        ideal = dcg(ideal_order)
        values.append(1.0 if ideal <= 0.0 else dcg(predicted_order) / ideal)
    return 0.0 if not values else float(np.mean(values))


def fit_overlay(
    overlay: ExtrinsicValueOverlay,
    partitions: OverlayPartitions,
    *,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
) -> OverlayTrainingSummary:
    """Huber + pairwise logistic loss で head だけを学習する。

    best development-validation NDCG checkpoint を復元し、final test は一度も開封しない。
    """
    if type(max_epochs) is not int or max_epochs <= 0:
        raise ExtrinsicValueOverlayError("max_epochs must be positive")
    if type(patience) is not int or patience <= 0:
        raise ExtrinsicValueOverlayError("patience must be positive")
    if partitions.final_test_opened:
        raise ExtrinsicValueOverlayError(
            "final test must remain sealed during fit"
        )
    train_examples = partitions.development_examples(
        "development_train"
    )
    validation_examples = partitions.development_examples(
        "development_validation"
    )
    if not train_examples or not validation_examples:
        raise ExtrinsicValueOverlayError(
            "train and validation examples are required"
        )
    if any(
        example.features.size != overlay.input_dim
        for example in (*train_examples, *validation_examples)
    ):
        raise ExtrinsicValueOverlayError(
            "dataset feature dimension does not match overlay head"
        )
    overlay.target_mean = partitions.target_mean
    overlay.target_std = partitions.target_std
    device = next(overlay.head.parameters()).device
    train_features = _feature_tensor(train_examples, device=device)
    train_targets = _target_tensor(
        train_examples,
        mean=partitions.target_mean,
        std=partitions.target_std,
        device=device,
    )
    validation_features = _feature_tensor(
        validation_examples,
        device=device,
    )
    train_pairs = _pair_indices(train_examples)
    optimizer = overlay.make_optimizer(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    best_ndcg = -math.inf
    best_epoch = 0
    best_state: dict[str, th.Tensor] | None = None
    epochs_without_improvement = 0
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        overlay.head.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = overlay.head(train_features)
        huber = nn.functional.smooth_l1_loss(
            predictions,
            train_targets,
        )
        if train_pairs:
            winner_indices = th.as_tensor(
                [winner for winner, _ in train_pairs],
                dtype=th.long,
                device=device,
            )
            loser_indices = th.as_tensor(
                [loser for _, loser in train_pairs],
                dtype=th.long,
                device=device,
            )
            pairwise = nn.functional.softplus(
                -(
                    predictions[winner_indices]
                    - predictions[loser_indices]
                )
            ).mean()
        else:
            pairwise = predictions.sum() * 0.0
        loss = HUBER_WEIGHT * huber + PAIRWISE_WEIGHT * pairwise
        loss.backward()
        optimizer.step()

        overlay.head.eval()
        with th.no_grad():
            validation_predictions = (
                overlay.head(validation_features)
                .detach()
                .cpu()
                .numpy()
            )
        ndcg = _validation_ndcg(
            validation_examples,
            validation_predictions,
        )
        epochs_run = epoch
        if ndcg > best_ndcg + 1e-12:
            best_ndcg = ndcg
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in overlay.head.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise ExtrinsicValueOverlayError(
            "no validation checkpoint was selected"
        )
    overlay.head.load_state_dict(best_state, strict=True)
    overlay.assert_source_frozen()
    if partitions.final_test_opened:
        raise ExtrinsicValueOverlayError(
            "final test was opened during fit"
        )
    return OverlayTrainingSummary(
        target_mean=partitions.target_mean,
        target_std=partitions.target_std,
        best_val_ndcg=best_ndcg,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        actor_invariance_hash=overlay.actor_invariance_hash,
    )


def _manifest_identity_payload(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """自己参照 overlay_identity を除く manifest payload を返す。

    source binding・state hash・training selection metadata の全 field を identity に含める。
    """
    return {
        key: manifest[key]
        for key in sorted(_MANIFEST_FIELDS - {"overlay_identity"})
    }


def _validate_sha256(value: Any, label: str) -> str:
    """lowercase SHA-256 文字列を検証して返す。

    identity field sibling へ同じ長さ・文字集合規則を対称適用する。
    """
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExtrinsicValueOverlayError(
            f"{label} must be lowercase sha256"
        )
    return value


def _validate_manifest(
    manifest: Any,
    source: LoadedValueSource,
    *,
    expected_context_shape: Sequence[int] | None = None,
) -> None:
    """overlay manifest の exact schema・source/context/state metadata を検証する。

    model・VecNormalize・policy schema・actor invariance・context shape の全 binding を閉じる。
    """
    if not isinstance(manifest, Mapping) or set(manifest) != _MANIFEST_FIELDS:
        raise ExtrinsicValueOverlayError(
            "overlay manifest fields mismatch"
        )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ExtrinsicValueOverlayError(
            "unsupported overlay schema"
        )
    bindings = _source_bindings(source)
    binding_messages = {
        "source_model_hash": "source model hash mismatch",
        "schema_hash": "schema hash mismatch",
        "vecnormalize_hash": "vecnormalize hash mismatch",
        "context_shape": "context shape mismatch",
    }
    for field, expected in bindings.items():
        if manifest[field] != expected:
            raise ExtrinsicValueOverlayError(
                binding_messages[field]
            )
    if (
        expected_context_shape is not None
        and list(expected_context_shape) != manifest["context_shape"]
    ):
        raise ExtrinsicValueOverlayError(
            "expected context shape mismatch"
        )
    for field in (
        "overlay_identity",
        "source_model_hash",
        "schema_hash",
        "vecnormalize_hash",
        "actor_invariance_hash",
        "state_dict_hash",
        "paired_dataset_hash",
    ):
        _validate_sha256(manifest[field], field)
    if manifest["actor_invariance_hash"] != policy_invariance_hash(
        source.policy
    ):
        raise ExtrinsicValueOverlayError(
            "actor invariance hash mismatch"
        )
    input_dim = getattr(source.policy.mlp_extractor, "latent_dim_vf", None)
    if manifest["input_dim"] != input_dim:
        raise ExtrinsicValueOverlayError(
            "overlay input dimension mismatch"
        )
    if manifest["head_architecture"] != {
        "layers": [
            {"type": "Linear", "in_features": input_dim, "out_features": 128},
            {"type": "GELU"},
            {"type": "Linear", "in_features": 128, "out_features": 1},
        ]
    }:
        raise ExtrinsicValueOverlayError(
            "overlay head architecture mismatch"
        )
    if manifest["state_dict_filename"] != STATE_FILENAME:
        raise ExtrinsicValueOverlayError(
            "overlay state filename mismatch"
        )
    if manifest["loss_weights"] != {
        "huber": HUBER_WEIGHT,
        "pairwise_logistic": PAIRWISE_WEIGHT,
    }:
        raise ExtrinsicValueOverlayError(
            "overlay loss weights mismatch"
        )
    optimizer = manifest["optimizer"]
    if (
        not isinstance(optimizer, Mapping)
        or set(optimizer)
        != {"name", "learning_rate", "weight_decay"}
        or optimizer["name"] != "Adam"
    ):
        raise ExtrinsicValueOverlayError(
            "overlay optimizer metadata mismatch"
        )
    for field in (
        "target_mean",
        "target_std",
        "best_val_ndcg",
    ):
        value = manifest[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ExtrinsicValueOverlayError(
                f"{field} must be finite"
            )
    if float(manifest["target_std"]) <= 0.0:
        raise ExtrinsicValueOverlayError(
            "target_std must be positive"
        )
    if not 0.0 <= float(manifest["best_val_ndcg"]) <= 1.0 + 1e-12:
        raise ExtrinsicValueOverlayError(
            "best_val_ndcg is outside [0, 1]"
        )
    for field in (
        "max_epochs",
        "patience",
        "epochs_run",
        "best_epoch",
    ):
        value = manifest[field]
        if type(value) is not int or value <= 0:
            raise ExtrinsicValueOverlayError(
                f"{field} must be positive"
            )
    if (
        manifest["best_epoch"] > manifest["epochs_run"]
        or manifest["epochs_run"] > manifest["max_epochs"]
    ):
        raise ExtrinsicValueOverlayError(
            "overlay epoch metadata is inconsistent"
        )
    if manifest["final_test_status"] != "sealed_unopened":
        raise ExtrinsicValueOverlayError(
            "final test was not sealed at overlay release"
        )
    expected_identity = canonical_hash(
        _manifest_identity_payload(manifest)
    )
    if manifest["overlay_identity"] != expected_identity:
        raise ExtrinsicValueOverlayError(
            "overlay manifest identity mismatch"
        )


def save_extrinsic_value_overlay(
    overlay: ExtrinsicValueOverlay,
    output_dir: Path,
    *,
    summary: OverlayTrainingSummary,
    paired_dataset_hash: str,
) -> Path:
    """head state dict と source-bound canonical manifest を保存する。

    state を先に確定し manifest を commit marker として最後に置き、片方欠損を load で拒否する。
    """
    overlay.assert_source_frozen()
    if summary.actor_invariance_hash != overlay.actor_invariance_hash:
        raise ExtrinsicValueOverlayError(
            "training summary actor invariance mismatch"
        )
    _validate_sha256(paired_dataset_hash, "paired_dataset_hash")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / STATE_FILENAME
    manifest_path = destination / MANIFEST_FILENAME
    if state_path.exists() or manifest_path.exists():
        raise ExtrinsicValueOverlayError(
            "overlay artifact already exists"
        )
    state_descriptor, state_temp_name = tempfile.mkstemp(
        prefix=".extrinsic-value-overlay-",
        suffix=".pt.tmp",
        dir=destination,
    )
    os.close(state_descriptor)
    state_temp = Path(state_temp_name)
    manifest_temp = destination / f".{MANIFEST_FILENAME}.tmp"
    try:
        th.save(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "state_dict": overlay.head.state_dict(),
            },
            state_temp,
        )
        state_hash = sha256_hex(state_temp.read_bytes())
        bindings = _source_bindings(overlay.source)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "overlay_identity": "",
            **bindings,
            "actor_invariance_hash": overlay.actor_invariance_hash,
            "input_dim": overlay.input_dim,
            "head_architecture": {
                "layers": [
                    {
                        "type": "Linear",
                        "in_features": overlay.input_dim,
                        "out_features": HEAD_HIDDEN_DIM,
                    },
                    {"type": "GELU"},
                    {
                        "type": "Linear",
                        "in_features": HEAD_HIDDEN_DIM,
                        "out_features": 1,
                    },
                ]
            },
            "state_dict_filename": STATE_FILENAME,
            "state_dict_hash": state_hash,
            "paired_dataset_hash": paired_dataset_hash,
            "target_mean": summary.target_mean,
            "target_std": summary.target_std,
            "loss_weights": {
                "huber": HUBER_WEIGHT,
                "pairwise_logistic": PAIRWISE_WEIGHT,
            },
            "optimizer": {
                "name": "Adam",
                "learning_rate": summary.learning_rate,
                "weight_decay": summary.weight_decay,
            },
            "max_epochs": summary.max_epochs,
            "patience": summary.patience,
            "epochs_run": summary.epochs_run,
            "best_epoch": summary.best_epoch,
            "best_val_ndcg": summary.best_val_ndcg,
            "final_test_status": "sealed_unopened",
        }
        manifest["overlay_identity"] = canonical_hash(
            _manifest_identity_payload(manifest)
        )
        _validate_manifest(manifest, overlay.source)
        manifest_temp.write_bytes(
            canonical_json_bytes(manifest) + b"\n"
        )
        os.replace(state_temp, state_path)
        os.replace(manifest_temp, manifest_path)
    finally:
        if state_temp.exists():
            state_temp.unlink()
        if manifest_temp.exists():
            manifest_temp.unlink()
    return manifest_path


def _resolve_manifest_path(path: Path) -> Path:
    """directory・manifest・state path から固定 manifest path を解決する。

    任意 filename 探索はせず、三つの明示形式だけを受け入れる。
    """
    candidate = Path(path)
    if candidate.is_dir():
        return candidate / MANIFEST_FILENAME
    if candidate.name == STATE_FILENAME:
        return candidate.with_name(MANIFEST_FILENAME)
    return candidate


def load_extrinsic_value_overlay(
    path: Path,
    source: LoadedValueSource,
    *,
    expected_context_shape: Sequence[int] | None = None,
) -> ExtrinsicValueOverlay:
    """manifest/state の hash と全 source binding を検証して overlay をロードする。

    source model・schema・VecNormalize・actor・context shape の不一致を head 適用前に拒否する。
    """
    manifest_path = _resolve_manifest_path(path)
    if not manifest_path.is_file():
        raise ExtrinsicValueOverlayError(
            "overlay manifest is missing"
        )
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtrinsicValueOverlayError(
            "overlay manifest could not be read"
        ) from exc
    _validate_manifest(
        manifest,
        source,
        expected_context_shape=expected_context_shape,
    )
    state_path = manifest_path.parent / manifest["state_dict_filename"]
    if not state_path.is_file():
        raise ExtrinsicValueOverlayError(
            "overlay state dict is missing"
        )
    if sha256_hex(state_path.read_bytes()) != manifest["state_dict_hash"]:
        raise ExtrinsicValueOverlayError(
            "overlay state dict hash mismatch"
        )
    try:
        payload = th.load(
            state_path,
            map_location=source.policy.device,
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ExtrinsicValueOverlayError(
            "overlay state dict could not be loaded"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema_version", "state_dict"}
        or payload["schema_version"] != STATE_SCHEMA_VERSION
        or not isinstance(payload["state_dict"], Mapping)
    ):
        raise ExtrinsicValueOverlayError(
            "overlay state payload schema mismatch"
        )
    overlay = ExtrinsicValueOverlay(
        source,
        ExtrinsicValueHead(manifest["input_dim"]).to(
            source.policy.device
        ),
        target_mean=float(manifest["target_mean"]),
        target_std=float(manifest["target_std"]),
    )
    try:
        overlay.head.load_state_dict(
            payload["state_dict"],
            strict=True,
        )
    except RuntimeError as exc:
        raise ExtrinsicValueOverlayError(
            "overlay state architecture mismatch"
        ) from exc
    overlay.head.eval()
    overlay.assert_source_frozen()
    return overlay


def _extract_critic_latent(
    source: LoadedValueSource,
    normalized_obs_batch: np.ndarray,
    context: CriticContext,
) -> th.Tensor:
    """source policy の value_net 直前 critic latent を抽出する。

    RecurrentPPO の separate/shared/disabled critic と PPO の全 sibling を predict_values と同形に処理する。
    """
    batch = np.asarray(normalized_obs_batch, dtype=np.float32)
    if (
        batch.ndim != 2
        or batch.shape[0] <= 0
        or batch.shape[1] != source.observation_dim
        or not np.all(np.isfinite(batch))
    ):
        raise ExtrinsicValueOverlayError(
            "normalized observation batch shape is invalid"
        )
    policy = source.policy
    obs_tensor, _ = policy.obs_to_tensor(batch)
    with th.no_grad():
        extracted = policy.extract_features(obs_tensor)
        if isinstance(extracted, tuple):
            _, vf_features = extracted
        else:
            vf_features = extracted
        if source.algorithm == "RecurrentPPO":
            if context.vf is None:
                raise ExtrinsicValueOverlayError(
                    "recurrent overlay requires vf context"
                )
            repeated = tuple(
                np.repeat(component, len(batch), axis=1)
                for component in context.vf
            )
            state = tuple(
                th.as_tensor(
                    component,
                    dtype=th.float32,
                    device=policy.device,
                )
                for component in repeated
            )
            starts = th.full(
                (len(batch),),
                float(context.episode_start),
                dtype=th.float32,
                device=policy.device,
            )
            if policy.lstm_critic is not None:
                latent_vf, _ = policy._process_sequence(
                    vf_features,
                    state,
                    starts,
                    policy.lstm_critic,
                )
            elif policy.shared_lstm:
                latent_vf, _ = policy._process_sequence(
                    vf_features,
                    state,
                    starts,
                    policy.lstm_actor,
                )
                latent_vf = latent_vf.detach()
            else:
                latent_vf = policy.critic(vf_features)
        else:
            latent_vf = vf_features
        latent_vf = policy.mlp_extractor.forward_critic(latent_vf)
    return latent_vf.detach()

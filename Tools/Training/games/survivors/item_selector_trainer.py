"""Survivors ItemSelector の feature loader、class balance、trainer を定義する。

共有 ``ItemDecisionFeatures`` wire と release 済み soft target/mask/reliability だけを読み、
development train/validation で AdamW 学習と validation NDCG checkpoint 選択を行う。
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import torch as th
from torch.utils.data import DataLoader, Dataset

from reinbalance_survivors_contracts.item_decision import ItemDecisionFeatures

from games.survivors.item_selector_loss import item_selector_loss
from games.survivors.item_selector_model import ItemSelector

CHECKPOINT_SCHEMA_VERSION = "survivors.item_selector_checkpoint.v1"
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 512
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 12
_PADDING_ITEM_ID = "__padding__"
_PADDING_KIND = "padding"
_MODEL_PARTITIONS = frozenset({"train", "validation"})


class CheckpointBindingError(ValueError):
    """checkpoint の target capability または shape binding 違反を表す。

    state/optimizer mutation より前に送出し、別 target 用 model の暗黙 resume を防ぐ。
    """


class DevelopmentPartitionSource(Protocol):
    """development loader が必要とする partition reader の最小 interface。

    test reader を interface に含めず、trainer から final partition へ到達できない形にする。
    """

    def read_partition(self, partition: str) -> Sequence[Mapping[str, Any]]:
        """要求された development partition の rows を返す。

        実装側は ``train`` と ``validation`` だけを受け取る。
        """
        ...


@dataclass(frozen=True, slots=True)
class EncodedItemSelectorRow:
    """一 decision の model tensors と label metadata を分離して保持する。

    tensor は未加重の feature/soft target で、reliability は scalar のまま loss 境界へ渡す。
    """

    decision_id: str
    partition: str
    context_features: th.Tensor
    candidate_features: th.Tensor
    candidate_mask: th.Tensor
    teacher_soft_target: th.Tensor
    reliability_weight: float
    candidate_item_ids: tuple[str, ...]
    candidate_kinds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrozenClassWeights:
    """train partition だけで fit した item/kind class balance を保持する。

    kind exposure count を teacher item mass で割った比を row soft target で平均し、train row
    平均が1になる scale を固定する。validation では同じ表を参照するだけで再 fit しない。
    """

    item_weights: Mapping[str, float]
    item_coefficients: Mapping[str, float]
    item_kinds: Mapping[str, str]
    kind_count: Mapping[str, int]
    item_mass: Mapping[str, float]
    normalization_scale: float
    train_mean_before_clip: float

    def weight_for(self, row: EncodedItemSelectorRow) -> float:
        """frozen item weight の teacher-soft expectation を ``[0.5,4]`` で返す。

        未知 item/kind を validation から補完せず、target capability の不一致として拒否する。
        """
        probabilities = _valid_target_probabilities(row)
        result = 0.0
        for index, probability in probabilities:
            item_id = row.candidate_item_ids[index]
            kind = row.candidate_kinds[index]
            if item_id not in self.item_coefficients:
                raise ValueError(f"unknown item {item_id!r} in frozen class weights")
            if self.item_kinds[item_id] != kind:
                raise ValueError(f"item kind changed for {item_id!r}")
            result += probability * float(self.item_coefficients[item_id])
        return min(max(result, 0.5), 4.0)


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """ItemSelector の固定 optimizer、batch、early-stop 設定。

    default は phase 契約値で、短い unit test では明示 override できる。
    """

    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    patience: int = DEFAULT_PATIENCE

    def __post_init__(self) -> None:
        """optimizer scalar と正整数の epoch/batch 設定を検証する。

        NaN や空 epoch を暗黙補正せず construction 時に拒否する。
        """
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative and finite")
        for name in ("batch_size", "max_epochs", "patience"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ItemSelectorTrainingSummary:
    """best validation NDCG と early-stop 結果を保持する。

    test metric は意図的に持たず、development model selection の情報だけを公開する。
    """

    best_val_ndcg: float
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    class_weights: FrozenClassWeights


def _stable_string_feature(value: str) -> float:
    """categorical wire 文字列を deterministic な有限 scalar へ写像する。

    Python の process-randomized ``hash`` を使わず SHA-256 先頭64 bit を ``[-1,1]`` にする。
    """
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (integer / float((1 << 64) - 1)) * 2.0 - 1.0


def _flatten_feature(value: Any, *, label: str) -> list[float]:
    """共有 wire の named/list feature を schema 定義順の float vector にする。

    ``from_wire`` 後の ``to_wire`` が再生成した key 順を使う。bool、数値、文字列、nested
    sequence/mapping だけを受理し、教師 object の自由形式入力面を encoder に作らない。
    """
    if isinstance(value, bool):
        return [1.0 if value else 0.0]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must contain finite values")
        return [number]
    if isinstance(value, str):
        if not value:
            raise ValueError(f"{label} strings must be non-empty")
        return [_stable_string_feature(value)]
    if isinstance(value, Mapping):
        result: list[float] = []
        for key in value:
            if not isinstance(key, str):
                raise ValueError(f"{label} mapping keys must be strings")
            result.extend(_flatten_feature(value[key], label=f"{label}.{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            result.extend(_flatten_feature(item, label=f"{label}[{index}]"))
        return result
    raise ValueError(f"{label} contains unsupported feature type")


def _decision_wire(value: Any) -> Mapping[str, Any]:
    """``ItemDecisionFeatures`` を round-trip して canonical feature wire を返す。

    caller row の teacher/privileged sibling field は渡さず、共有契約の ``to_wire`` 出力だけを
    model encoder の入力にする。
    """
    if isinstance(value, ItemDecisionFeatures):
        decision = value
    elif isinstance(value, Mapping):
        decision = ItemDecisionFeatures.from_wire(value)
    else:
        raise ValueError("item_decision_features must be shared wire data")
    wire = decision.to_wire()
    if not isinstance(wire, Mapping):
        raise ValueError("ItemDecisionFeatures.to_wire() must return an object")
    return wire


def _candidate_vector(candidate: Mapping[str, Any], *, index: int) -> list[float]:
    """candidate wire の allow-listed feature payload を数値化する。

    scaffold 契約では ``features``、正式 named 契約では schema marker 以外の公開 field を使う。
    """
    if "features" in candidate:
        return _flatten_feature(candidate["features"], label=f"candidate[{index}].features")
    public = {
        key: value
        for key, value in candidate.items()
        if key != "schema_version"
    }
    return _flatten_feature(public, label=f"candidate[{index}]")


def _wire_candidate_identity(candidate: Mapping[str, Any]) -> tuple[str, str | None]:
    """scaffold/formal candidate wire から item ID と任意 kind を返す。

    position index を identity の代用にせず、共有 wire に記録された文字列だけを使う。
    """
    item_id = candidate.get("item_id", candidate.get("candidate_id"))
    kind = candidate.get("kind")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("candidate wire item identity is missing")
    if kind is not None and (not isinstance(kind, str) or not kind):
        raise ValueError("candidate wire kind is invalid")
    return item_id, kind


def encode_item_selector_row(
    row: Mapping[str, Any],
    *,
    nmax: int,
) -> EncodedItemSelectorRow:
    """release row を未加重 fixed-shape model/label tensors へ変換する。

    ``teacher_soft_target``、``candidate_mask``、``reliability_weight`` は raw score から再生成せず、
    dataset 値をそのまま tensor/scalar 化して padding だけを追加する。
    """
    if not isinstance(row, Mapping):
        raise ValueError("item selector row must be an object")
    if type(nmax) is not int or nmax <= 0:
        raise ValueError("Nmax must be a positive integer")
    if row.get("label_action_kind") != "choose_card":
        raise ValueError("ItemSelector rows require explicit choose_card action kind")
    partition = row.get("split", row.get("partition"))
    if partition not in _MODEL_PARTITIONS:
        raise ValueError("ItemSelector loader accepts train/validation rows only")
    decision_id = row.get("decision_id")
    if not isinstance(decision_id, str) or not decision_id:
        raise ValueError("decision_id must be non-empty")

    wire = _decision_wire(row.get("item_decision_features"))
    if wire.get("decision_id") != decision_id:
        raise ValueError("decision ID does not match feature wire")
    declared_nmax = wire.get("max_item_cards")
    if declared_nmax is not None and declared_nmax != nmax:
        raise ValueError("feature wire Nmax does not match target Nmax")
    raw_context = wire.get("context_features")
    raw_candidates = wire.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("feature wire candidates must be a non-empty array")
    if any(not isinstance(candidate, Mapping) for candidate in raw_candidates):
        raise ValueError("feature wire candidates must be objects")

    raw_mask = row.get("candidate_mask")
    raw_target_value = row.get("teacher_soft_target")
    raw_target = (
        raw_target_value.get("probabilities")
        if isinstance(raw_target_value, Mapping)
        else raw_target_value
    )
    if (
        not isinstance(raw_mask, (list, tuple))
        or not isinstance(raw_target, (list, tuple))
        or len(raw_mask) != len(raw_candidates)
        or len(raw_target) != len(raw_mask)
        or len(raw_mask) > nmax
        or any(type(value) is not bool for value in raw_mask)
    ):
        raise ValueError("dataset target/mask/candidate lengths are invalid")
    if (
        isinstance(raw_context, Mapping)
        and "card_mask" in raw_context
        and list(raw_context["card_mask"]) != list(raw_mask)
    ):
        raise ValueError("dataset candidate_mask does not match feature wire card_mask")
    target_values: list[float] = []
    for value in raw_target:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("teacher_soft_target must contain numbers")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError("teacher_soft_target must be finite and non-negative")
        target_values.append(number)
    if sum(value for value, valid in zip(target_values, raw_mask, strict=True) if valid) <= 0.0:
        raise ValueError("teacher_soft_target has no valid mass")
    reliability = row.get("reliability_weight")
    if (
        isinstance(reliability, bool)
        or not isinstance(reliability, (int, float))
        or not math.isfinite(float(reliability))
        or not 0.0 <= float(reliability) <= 1.0
    ):
        raise ValueError("reliability_weight must be in [0, 1]")
    if (
        isinstance(raw_target_value, Mapping)
        and "reliability_weight" in raw_target_value
        and float(raw_target_value["reliability_weight"]) != float(reliability)
    ):
        raise ValueError("teacher target reliability binding mismatch")

    wire_identities = [_wire_candidate_identity(candidate) for candidate in raw_candidates]
    supplied_ids = row.get("candidate_item_ids")
    if supplied_ids is None:
        item_ids = [item_id for item_id, _ in wire_identities]
    elif (
        not isinstance(supplied_ids, (list, tuple))
        or list(supplied_ids) != [item_id for item_id, _ in wire_identities]
    ):
        raise ValueError("candidate_item_ids do not match feature wire")
    else:
        item_ids = list(supplied_ids)
    if (
        isinstance(raw_target_value, Mapping)
        and "candidate_item_ids" in raw_target_value
        and list(raw_target_value["candidate_item_ids"]) != item_ids
    ):
        raise ValueError("teacher target candidate identities do not match feature wire")
    supplied_kinds = row.get("candidate_kinds")
    if supplied_kinds is None:
        if any(kind is None for _, kind in wire_identities):
            raise ValueError("candidate kinds are required for class weighting")
        kinds = [str(kind) for _, kind in wire_identities]
    elif (
        not isinstance(supplied_kinds, (list, tuple))
        or len(supplied_kinds) != len(raw_candidates)
        or any(not isinstance(kind, str) or not kind for kind in supplied_kinds)
    ):
        raise ValueError("candidate_kinds are invalid")
    else:
        kinds = list(supplied_kinds)
        for supplied, (_, wire_kind) in zip(kinds, wire_identities, strict=True):
            if wire_kind is not None and supplied != wire_kind:
                raise ValueError("candidate_kinds do not match feature wire")
    valid_ids = [item_id for item_id, valid in zip(item_ids, raw_mask, strict=True) if valid]
    if len(set(valid_ids)) != len(valid_ids):
        raise ValueError("valid candidate item identities must be unique")

    context_vector = _flatten_feature(raw_context, label="context_features")
    candidate_vectors = [
        _candidate_vector(candidate, index=index)
        for index, candidate in enumerate(raw_candidates)
    ]
    candidate_dim = len(candidate_vectors[0])
    if not context_vector or candidate_dim <= 0 or any(
        len(vector) != candidate_dim for vector in candidate_vectors
    ):
        raise ValueError("feature vectors must have fixed positive dimensions")
    padding_count = nmax - len(raw_candidates)
    candidate_vectors.extend([[0.0] * candidate_dim for _ in range(padding_count)])
    mask = list(raw_mask) + [False] * padding_count
    target_values.extend([0.0] * padding_count)
    item_ids.extend([_PADDING_ITEM_ID] * padding_count)
    kinds.extend([_PADDING_KIND] * padding_count)
    return EncodedItemSelectorRow(
        decision_id=decision_id,
        partition=str(partition),
        context_features=th.tensor(context_vector, dtype=th.float32),
        candidate_features=th.tensor(candidate_vectors, dtype=th.float32),
        candidate_mask=th.tensor(mask, dtype=th.bool),
        teacher_soft_target=th.tensor(target_values, dtype=th.float32),
        reliability_weight=float(reliability),
        candidate_item_ids=tuple(item_ids),
        candidate_kinds=tuple(kinds),
    )


def load_development_rows(
    source: DevelopmentPartitionSource,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """source から train、validation の順に読み test API を呼ばない。

    各 row の partition 宣言を要求名と照合し、partition substitution も loader 境界で拒否する。
    """
    train = tuple(source.read_partition("train"))
    validation = tuple(source.read_partition("validation"))
    for expected, rows in (("train", train), ("validation", validation)):
        for row in rows:
            if row.get("split", row.get("partition")) != expected:
                raise ValueError(f"{expected} reader returned a different partition")
    return train, validation


def _valid_target_probabilities(
    row: EncodedItemSelectorRow,
) -> list[tuple[int, float]]:
    """row の valid teacher target を確率和1へ正規化して返す。

    class balance も loss と同じ masked soft target semantics を使い、hard argmax に潰さない。
    """
    values = [
        (index, float(row.teacher_soft_target[index]))
        for index, valid in enumerate(row.candidate_mask.tolist())
        if valid
    ]
    mass = sum(value for _, value in values)
    if mass <= 0.0:
        raise ValueError("row has no valid teacher target mass")
    return [(index, value / mass) for index, value in values]


def fit_class_weights(
    train_rows: Sequence[EncodedItemSelectorRow],
) -> FrozenClassWeights:
    """train rows の ``kind_count/item_mass`` から frozen row class weights を fit する。

    raw row expectation の train 平均を1へ正規化し、その後 item/row weight を ``[0.5,4]`` に
    clip する。validation/test/batch を統計入力として受け取る API は持たない。
    """
    if not train_rows:
        raise ValueError("train rows are required to fit class weights")
    if any(row.partition != "train" for row in train_rows):
        raise ValueError("class weights can be fit from train partition only")
    kind_count: Counter[str] = Counter()
    item_mass: defaultdict[str, float] = defaultdict(float)
    item_kinds: dict[str, str] = {}
    for row in train_rows:
        probabilities = dict(_valid_target_probabilities(row))
        for index, valid in enumerate(row.candidate_mask.tolist()):
            if not valid:
                continue
            item_id = row.candidate_item_ids[index]
            kind = row.candidate_kinds[index]
            previous = item_kinds.setdefault(item_id, kind)
            if previous != kind:
                raise ValueError(f"item {item_id!r} has multiple kinds")
            kind_count[kind] += 1
            item_mass[item_id] += probabilities[index]
    if any(mass <= 0.0 for mass in item_mass.values()):
        raise ValueError("every train item must have positive teacher mass")
    raw_item_weights = {
        item_id: float(kind_count[kind]) / item_mass[item_id]
        for item_id, kind in item_kinds.items()
    }

    def raw_row_weight(row: EncodedItemSelectorRow) -> float:
        """未正規化 item ratio の teacher-soft expectation を返す。

        normalize scale fitting と frozen table 作成で同じ式を共有する。
        """
        return sum(
            probability * raw_item_weights[row.candidate_item_ids[index]]
            for index, probability in _valid_target_probabilities(row)
        )

    raw_rows = [raw_row_weight(row) for row in train_rows]
    raw_mean = sum(raw_rows) / len(raw_rows)
    if not math.isfinite(raw_mean) or raw_mean <= 0.0:
        raise ValueError("class weight mean is invalid")
    scale = 1.0 / raw_mean
    item_weights = {
        item_id: min(max(raw * scale, 0.5), 4.0)
        for item_id, raw in raw_item_weights.items()
    }
    item_coefficients = {
        item_id: raw * scale
        for item_id, raw in raw_item_weights.items()
    }
    normalized_mean = sum(value * scale for value in raw_rows) / len(raw_rows)
    return FrozenClassWeights(
        item_weights=MappingProxyType(dict(sorted(item_weights.items()))),
        item_coefficients=MappingProxyType(dict(sorted(item_coefficients.items()))),
        item_kinds=MappingProxyType(dict(sorted(item_kinds.items()))),
        kind_count=MappingProxyType(dict(sorted(kind_count.items()))),
        item_mass=MappingProxyType(dict(sorted(item_mass.items()))),
        normalization_scale=scale,
        train_mean_before_clip=normalized_mean,
    )


class _EncodedDataset(Dataset[EncodedItemSelectorRow]):
    """encoded rows と frozen class weights を PyTorch Dataset として公開する。

    reliability/target/logit を加工せず、row class weight だけを ``__getitem__`` 時に lookup する。
    """

    def __init__(
        self,
        rows: Sequence[EncodedItemSelectorRow],
        class_weights: FrozenClassWeights,
    ) -> None:
        """非空 row tuple と train-fitted weight table を保持する。

        caller 側 sequence の後続変更が epoch 内容を変えないよう tuple 化する。
        """
        if not rows:
            raise ValueError("encoded dataset rows are required")
        self._rows = tuple(rows)
        self._class_weights = class_weights

    def __len__(self) -> int:
        """dataset の decision row 数を返す。

        padding candidate 数ではなく optimizer sample 単位を数える。
        """
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """未加重 tensors と独立 row scalar weights を返す。

        reliability は feature/target に掛けず batch loss まで保持する。
        """
        row = self._rows[index]
        return {
            "context_features": row.context_features,
            "candidate_features": row.candidate_features,
            "candidate_mask": row.candidate_mask,
            "teacher_soft_target": row.teacher_soft_target,
            "reliability_weight": th.tensor(row.reliability_weight, dtype=th.float32),
            "class_weight": th.tensor(self._class_weights.weight_for(row), dtype=th.float32),
        }


def validation_ndcg(
    logits: th.Tensor,
    teacher_soft_target: th.Tensor,
    candidate_mask: th.Tensor,
) -> float:
    """teacher soft relevance に対する decision 平均 NDCG を返す。

    valid candidate のみを使い、checkpoint criterion と training loss を独立に評価する。
    """
    if logits.shape != teacher_soft_target.shape or logits.shape != candidate_mask.shape:
        raise ValueError("NDCG tensors must share [B, Nmax] shape")
    values: list[float] = []
    for row in range(logits.shape[0]):
        valid = candidate_mask[row].nonzero(as_tuple=False).flatten().tolist()
        if not valid:
            raise ValueError("NDCG row has no valid candidates")
        predicted = sorted(valid, key=lambda index: (-float(logits[row, index]), index))
        ideal = sorted(
            valid,
            key=lambda index: (-float(teacher_soft_target[row, index]), index),
        )

        def dcg(order: Sequence[int]) -> float:
            """指定 candidate 順の soft-relevance discounted gain を返す。

            gain は hard outcome でなく teacher probability を直接 graded relevance にする。
            """
            return sum(
                (2.0 ** float(teacher_soft_target[row, index]) - 1.0)
                / math.log2(position + 2.0)
                for position, index in enumerate(order)
            )

        ideal_value = dcg(ideal)
        values.append(1.0 if ideal_value <= 0.0 else dcg(predicted) / ideal_value)
    return float(sum(values) / len(values))


class ItemSelectorTrainer:
    """development-only ItemSelector 学習と best-NDCG checkpoint を所有する。

    optimizer は AdamW に固定し、checkpoint/resume は target capability hash と Nmax の両方へ
    fail-closed に束縛する。
    """

    def __init__(
        self,
        model: ItemSelector,
        *,
        target_capability_hash: str,
        nmax: int,
        config: TrainerConfig | None = None,
        device: th.device | str = "cpu",
    ) -> None:
        """model、target binding、固定 hyperparameters、device を保持する。

        capability は canonical SHA-256、Nmax は正整数だけを受理する。
        """
        if not isinstance(model, ItemSelector):
            raise TypeError("model must be an ItemSelector")
        if (
            not isinstance(target_capability_hash, str)
            or len(target_capability_hash) != 64
            or any(character not in "0123456789abcdef" for character in target_capability_hash)
        ):
            raise ValueError("target_capability_hash must be lowercase sha256")
        if type(nmax) is not int or nmax <= 0:
            raise ValueError("Nmax must be a positive integer")
        self.model = model.to(device)
        self.target_capability_hash = target_capability_hash
        self.nmax = nmax
        self.config = config or TrainerConfig()
        self.device = th.device(device)

    def make_optimizer(self) -> th.optim.AdamW:
        """全 ItemSelector parameter を持つ固定 AdamW optimizer を返す。

        lr/weight decay は config から一度だけ設定し、validation ごとに作り直さない。
        """
        return th.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _checkpoint_payload(
        self,
        *,
        optimizer: th.optim.Optimizer | None,
        epoch: int,
        best_val_ndcg: float,
    ) -> dict[str, Any]:
        """model state と immutable target binding を一つの payload にする。

        final test metric や test row identity は保存せず、development criterion だけを記録する。
        """
        if type(epoch) is not int or epoch < 0:
            raise ValueError("checkpoint epoch must be non-negative")
        if not math.isfinite(best_val_ndcg):
            raise ValueError("best validation NDCG must be finite")
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "target_capability_hash": self.target_capability_hash,
            "nmax": self.nmax,
            "context_dim": self.model.context_dim,
            "candidate_dim": self.model.candidate_dim,
            "epoch": epoch,
            "best_val_ndcg": float(best_val_ndcg),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        }

    def save_checkpoint(
        self,
        path: Path,
        *,
        optimizer: th.optim.Optimizer | None,
        epoch: int,
        best_val_ndcg: float,
    ) -> None:
        """binding 済み checkpoint を同一 directory の一時 file 経由で置換保存する。

        不完全な torch archive を resume path として公開しない。
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            th.save(
                self._checkpoint_payload(
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_ndcg=best_val_ndcg,
                ),
                temporary,
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def load_checkpoint(
        self,
        path: Path,
        *,
        optimizer: th.optim.Optimizer | None = None,
    ) -> Mapping[str, Any]:
        """binding 検証後にだけ model と任意 optimizer state を復元する。

        capability、Nmax、feature 次元の全 sibling mismatch を state mutation 前に拒否する。
        """
        try:
            payload = th.load(Path(path), map_location=self.device, weights_only=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise CheckpointBindingError(f"cannot load checkpoint: {exc}") from exc
        if not isinstance(payload, Mapping) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointBindingError("unsupported checkpoint schema")
        if payload.get("target_capability_hash") != self.target_capability_hash:
            raise CheckpointBindingError("target capability hash mismatch")
        if payload.get("nmax") != self.nmax:
            raise CheckpointBindingError("checkpoint Nmax mismatch")
        if payload.get("context_dim") != self.model.context_dim:
            raise CheckpointBindingError("checkpoint context feature dimension mismatch")
        if payload.get("candidate_dim") != self.model.candidate_dim:
            raise CheckpointBindingError("checkpoint candidate feature dimension mismatch")
        model_state = payload.get("model_state_dict")
        optimizer_state = payload.get("optimizer_state_dict")
        if not isinstance(model_state, Mapping):
            raise CheckpointBindingError("checkpoint model state is missing")
        if optimizer is not None and optimizer_state is not None and not isinstance(optimizer_state, Mapping):
            raise CheckpointBindingError("checkpoint optimizer state is invalid")
        self.model.load_state_dict(model_state, strict=True)
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        return payload

    def _encode_partitions(
        self,
        train_rows: Sequence[Mapping[str, Any]],
        validation_rows: Sequence[Mapping[str, Any]],
    ) -> tuple[list[EncodedItemSelectorRow], list[EncodedItemSelectorRow]]:
        """明示 train/validation sequences を fixed Nmax rows へ encode する。

        combined rows や test filter API を受け取らず、caller が渡した development view だけを読む。
        """
        train = [encode_item_selector_row(row, nmax=self.nmax) for row in train_rows]
        validation = [
            encode_item_selector_row(row, nmax=self.nmax)
            for row in validation_rows
        ]
        if not train or not validation:
            raise ValueError("train and validation rows are required")
        all_rows = (*train, *validation)
        if any(row.partition != "train" for row in train):
            raise ValueError("train_rows contain another partition")
        if any(row.partition != "validation" for row in validation):
            raise ValueError("validation_rows contain another partition")
        if any(row.context_features.numel() != self.model.context_dim for row in all_rows):
            raise ValueError("dataset context dimension does not match model")
        if any(row.candidate_features.shape != (self.nmax, self.model.candidate_dim) for row in all_rows):
            raise ValueError("dataset candidate shape does not match model/Nmax")
        return train, validation

    def fit(
        self,
        train_rows: Sequence[Mapping[str, Any]],
        validation_rows: Sequence[Mapping[str, Any]],
        *,
        checkpoint_path: Path | None = None,
        resume_from: Path | None = None,
    ) -> ItemSelectorTrainingSummary:
        """複合 loss で学習し best validation NDCG model を復元する。

        patience は NDCG 非改善 epoch 数で数え、test partition は入力/API のどちらにも持たない。
        """
        train, validation = self._encode_partitions(train_rows, validation_rows)
        class_weights = fit_class_weights(train)
        training_dataset = _EncodedDataset(train, class_weights)
        validation_dataset = _EncodedDataset(validation, class_weights)
        generator = th.Generator().manual_seed(0)
        training_loader = DataLoader(
            training_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        optimizer = self.make_optimizer()
        start_epoch = 1
        best_ndcg = -math.inf
        best_epoch = 0
        if resume_from is not None:
            payload = self.load_checkpoint(resume_from, optimizer=optimizer)
            start_epoch = int(payload["epoch"]) + 1
            best_ndcg = float(payload["best_val_ndcg"])
            best_epoch = int(payload["epoch"])
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_without_improvement = 0
        epochs_run = start_epoch - 1
        for epoch in range(start_epoch, self.config.max_epochs + 1):
            self.model.train()
            for batch in training_loader:
                context = batch["context_features"].to(self.device)
                candidates = batch["candidate_features"].to(self.device)
                mask = batch["candidate_mask"].to(self.device)
                target = batch["teacher_soft_target"].to(self.device)
                reliability = batch["reliability_weight"].to(self.device)
                class_weight = batch["class_weight"].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(context, candidates, mask)
                loss = item_selector_loss(
                    logits,
                    target,
                    mask,
                    reliability_weight=reliability,
                    class_weight=class_weight,
                )
                loss.backward()
                optimizer.step()

            self.model.eval()
            validation_ndcg_sum = 0.0
            validation_row_count = 0
            with th.no_grad():
                for batch in validation_loader:
                    context = batch["context_features"].to(self.device)
                    candidates = batch["candidate_features"].to(self.device)
                    mask = batch["candidate_mask"].to(self.device)
                    target = batch["teacher_soft_target"].to(self.device)
                    logits = self.model(context, candidates, mask)
                    batch_size = int(logits.shape[0])
                    validation_ndcg_sum += validation_ndcg(logits, target, mask) * batch_size
                    validation_row_count += batch_size
            current_ndcg = float(validation_ndcg_sum / validation_row_count)
            epochs_run = epoch
            if current_ndcg > best_ndcg + 1e-12:
                best_ndcg = current_ndcg
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
                if checkpoint_path is not None:
                    self.save_checkpoint(
                        checkpoint_path,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_val_ndcg=best_ndcg,
                    )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    break
        self.model.load_state_dict(best_state, strict=True)
        return ItemSelectorTrainingSummary(
            best_val_ndcg=best_ndcg,
            best_epoch=best_epoch,
            epochs_run=epochs_run,
            stopped_early=epochs_run < self.config.max_epochs,
            class_weights=class_weights,
        )

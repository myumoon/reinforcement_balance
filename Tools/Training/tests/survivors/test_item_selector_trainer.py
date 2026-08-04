"""ItemSelector loader、class balance、trainer、checkpoint binding を検証する。

development partition だけで前処理と checkpoint 選択を行い、test および教師専用 raw score を
モデル入力へ到達させない。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as th

from reinbalance_survivors_contracts.item_decision import (
    CandidateFeatures,
    ItemDecisionFeatures,
)

from games.survivors.item_selector_model import ItemSelector
from games.survivors.item_selector_trainer import (
    CheckpointBindingError,
    DevelopmentPartitionSource,
    ItemSelectorTrainer,
    TrainerConfig,
    encode_item_selector_row,
    fit_class_weights,
    load_development_rows,
)

CAPABILITY = "a" * 64


def _candidate(item_id: str) -> CandidateFeatures:
    """テスト用の weapon カード候補を返す。

    kind/owned などは最小有効値で固定し、item_id だけで識別する。
    """
    return CandidateFeatures(
        kind="weapon",
        item_id=item_id,
        new_level=1,
        owned=False,
        is_new=True,
        is_evolve=False,
        is_union=False,
        has_prerequisite=False,
        slot_capacity=1,
    )


def _row(
    decision: str,
    target: tuple[float, float],
    *,
    reliability: float = 0.75,
    split: str = "train",
) -> dict:
    """context_only_v1 の 2 候補 development row を返す。

    raw teacher score は入力に置くが、encoder の feature vector へ含まれないことを別途検証する。
    weapon_slots=(1,0,0,0,0,0) + passive_slots=(0,0,0,0,0,0) → empty_slot_count=11。
    """
    features = ItemDecisionFeatures(
        decision_id=decision,
        feature_schema="context_only_v1",
        elapsed_time=60.0,
        level=2,
        hp_ratio=1.0,
        xp_ratio=0.5,
        weapon_slots=(1, 0, 0, 0, 0, 0),
        passive_slots=(0, 0, 0, 0, 0, 0),
        empty_slot_count=11,
        evolution_readiness=0.0,
        choice_count=2,
        card_mask=(True, True),
        fallback_kind="none",
        ui_state_validity=1.0,
        ui_state_age=0.1,
        candidates=(_candidate("wand"), _candidate("knife")),
        max_item_cards=2,
    )
    return {
        "decision_id": decision,
        "split": split,
        "label_action_kind": "choose_card",
        "item_decision_features": features.to_wire(),
        "candidate_mask": [True, True],
        "candidate_item_ids": ["wand", "knife"],
        "candidate_kinds": ["weapon", "weapon"],
        "teacher_soft_target": list(target),
        "reliability_weight": reliability,
        "teacher_scores": [9999.0, -9999.0],
    }


def test_encoder_consumes_released_fields_without_raw_score_or_preweighting() -> None:
    """共有 wire の context/candidate と dataset target/mask/reliability をそのまま分離する。

    raw score を変えても feature は同じで、target/logit へ reliability を事前乗算しない。
    """
    first = _row("d-1", (0.8, 0.2), reliability=0.25)
    second = {**first, "teacher_scores": [-1e30, 1e30]}
    encoded = encode_item_selector_row(first, nmax=2)
    changed = encode_item_selector_row(second, nmax=2)

    th.testing.assert_close(encoded.context_features, changed.context_features)
    th.testing.assert_close(encoded.candidate_features, changed.candidate_features)
    th.testing.assert_close(encoded.teacher_soft_target, th.tensor([0.8, 0.2]))
    assert encoded.candidate_mask.tolist() == [True, True]
    assert encoded.reliability_weight == 0.25


def test_non_choose_card_rows_are_rejected() -> None:
    """fallback/meta/ack action を selector candidate 集合へ混入させない。

    action ownership が明示されない行も choose_card と推測しない。
    """
    for action in (None, "fallback", "reroll", "ack_chest"):
        row = _row("bad", (0.5, 0.5))
        if action is None:
            row.pop("label_action_kind")
        else:
            row["label_action_kind"] = action
        with pytest.raises(ValueError, match="choose_card"):
            encode_item_selector_row(row, nmax=2)


def test_class_weights_fit_train_only_then_freeze() -> None:
    """class statistics と mean normalization を train rows だけから確定する。

    validation target の極端な変更で frozen weight が再 fit されないことを確認する。
    """
    train = [
        encode_item_selector_row(_row("t-1", (0.9, 0.1)), nmax=2),
        encode_item_selector_row(_row("t-2", (0.8, 0.2)), nmax=2),
        encode_item_selector_row(_row("t-3", (0.7, 0.3)), nmax=2),
    ]
    frozen = fit_class_weights(train)
    validation_a = encode_item_selector_row(_row("v", (1.0, 0.0), split="validation"), nmax=2)
    validation_b = encode_item_selector_row(_row("v", (0.0, 1.0), split="validation"), nmax=2)

    assert frozen.train_mean_before_clip == pytest.approx(1.0)
    assert dict(frozen.kind_count) == {"weapon": 6}
    assert dict(frozen.item_mass) == pytest.approx({"wand": 2.4, "knife": 0.6})
    assert dict(frozen.item_weights) == pytest.approx({"wand": 0.625, "knife": 2.5})
    assert frozen.weight_for(validation_a) != frozen.weight_for(validation_b)
    assert frozen.item_weights == fit_class_weights(train).item_weights


class _AuditedSource(DevelopmentPartitionSource):
    """partition request 名を記録する in-memory development source。

    test request が来た場合は即座に失敗し、loader leakage を観測可能にする。
    """

    def __init__(self) -> None:
        """空の partition request 監査履歴を初期化する。

        rows 自体は固定 fixture として ``read_partition`` 呼出し時に生成する。
        """
        self.requests: list[str] = []

    def read_partition(self, partition: str):
        """要求 partition を記録し development fixture を一行返す。

        test が要求された場合は leakage として即座に AssertionError を送出する。
        """
        self.requests.append(partition)
        if partition == "test":
            raise AssertionError("test partition was opened")
        return [_row(f"{partition}-1", (0.6, 0.4), split=partition)]


def test_development_loader_never_requests_test_partition() -> None:
    """trainer 用 loader が train/validation 以外の reader を呼ばない。

    final evaluation 用 API を同じ source に持たせても development path からは到達しない。
    """
    source = _AuditedSource()
    train, validation = load_development_rows(source)

    assert len(train) == len(validation) == 1
    assert source.requests == ["train", "validation"]


def test_trainer_defaults_use_adamw_and_checkpoint_rejects_binding_mismatch(
    tmp_path: Path,
) -> None:
    """固定 optimizer/hyperparameter と capability/Nmax resume gate を確認する。

    binding 検査は state dict を変更する前に実行される。
    """
    config = TrainerConfig()
    model = ItemSelector(context_dim=3, candidate_dim=3)
    trainer = ItemSelectorTrainer(model, target_capability_hash=CAPABILITY, nmax=2)
    optimizer = trainer.make_optimizer()

    assert isinstance(optimizer, th.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-4)
    assert (config.batch_size, config.max_epochs, config.patience) == (512, 100, 12)

    checkpoint = tmp_path / "selector.pt"
    trainer.save_checkpoint(checkpoint, optimizer=optimizer, epoch=3, best_val_ndcg=0.7)
    with pytest.raises(CheckpointBindingError, match="capability"):
        ItemSelectorTrainer(model, target_capability_hash="b" * 64, nmax=2).load_checkpoint(checkpoint)
    with pytest.raises(CheckpointBindingError, match="Nmax"):
        ItemSelectorTrainer(model, target_capability_hash=CAPABILITY, nmax=3).load_checkpoint(checkpoint)


def test_fit_checkpoints_and_restores_best_validation_ndcg(tmp_path: Path) -> None:
    """短い学習でも validation NDCG を唯一の best criterion として保存する。

    fit 後の model state は checkpoint の best epoch と一致し、最終 epoch state を返さない。
    """
    th.manual_seed(3)
    # context_only_v1 + max_item_cards=2: context_dim=24, candidate_dim=9
    model = ItemSelector(context_dim=24, candidate_dim=9)
    trainer = ItemSelectorTrainer(
        model,
        target_capability_hash=CAPABILITY,
        nmax=2,
        config=TrainerConfig(batch_size=2, max_epochs=3, patience=1),
    )
    train = [
        _row("train-1", (0.9, 0.1)),
        _row("train-2", (0.1, 0.9)),
    ]
    validation = [
        _row("validation-1", (0.8, 0.2), split="validation"),
        _row("validation-2", (0.2, 0.8), split="validation"),
    ]
    checkpoint = tmp_path / "best.pt"

    summary = trainer.fit(train, validation, checkpoint_path=checkpoint)
    payload = th.load(checkpoint, map_location="cpu", weights_only=False)

    assert payload["best_val_ndcg"] == pytest.approx(summary.best_val_ndcg)
    assert payload["epoch"] == summary.best_epoch
    for name, value in trainer.model.state_dict().items():
        th.testing.assert_close(value, payload["model_state_dict"][name])

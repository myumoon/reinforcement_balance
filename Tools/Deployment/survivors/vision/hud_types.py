"""HUD と choice が共有するカード・ボタンのデータ型。

parser 同士の循環 import を避けるため、画素解析を持たない型と候補 hash をまとめます。
"""

from dataclasses import dataclass

from reinbalance_survivors_contracts.canonical_json import canonical_hash


@dataclass(frozen=True, slots=True)
class ParsedCard:
    """レベルアップカードスロットの解析結果。

    カードの位置、アイテム名、種別、レベルと読み取りの信頼度をまとめます。
    アイテム名やレベルが読めなかった場合は None のまま残します。
    """

    slot_index: int
    item_id: str | None
    kind: str
    level: int | None
    confidence: float
    reason: str
    roi_xyxy: tuple[int, int, int, int] | None


@dataclass(frozen=True, slots=True)
class ParsedButton:
    """UI ボタンの種類と読み取り結果。

    reroll・skip・banish・ack_chest・confirm の区別に、位置と判定理由を添えます。
    """

    button_type: str
    confidence: float
    reason: str
    roi_xyxy: tuple[int, int, int, int] | None


def _compute_candidate_set_hash(screen_state: str, cards: tuple[ParsedCard, ...]) -> str:
    """画面状態とカード ID セットの canonical hash を計算する。

    カード名を並べ替えて画面状態と組み合わせるので、同じ候補なら表示順に左右されません。
    読めないカード名は unknown として区別します。
    """
    card_ids = sorted(c.item_id or "unknown" for c in cards)
    return canonical_hash({"screen_state": screen_state, "card_ids": card_ids})

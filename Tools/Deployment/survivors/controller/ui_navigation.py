"""UiIntentV1 を PerceptionSnapshot 上の typed target へ解決し、input effect へ変換する。

やさしい説明:
このファイルは「決定(何をしたいか)」と「実際にクリックする場所」を橋渡しします。
05-01 が作った `UiIntentV1`(例: 「このカードを選ぶ」)を受け取り、
`PerceptionSnapshot.ui_presentation` の中から、その意図と一致する ROI(操作矩形)を
1つだけ探します。0 件・複数件・無効・信頼度不足のときは何もクリックしません
(黙って諦める = fail-closed)。見つかった ROI の中心座標だけを
`InputLeaseController` へ渡し、固定座標や別経路の座標は一切使いません。

このモジュールは `UiIntentV1` を生成・再分類しません(受け取ったものをそのまま
解決するだけ)。また、画面解析結果の内部表現や非モデル UI 決定ロジック側の
モジュールは import しません(所有権の境界を守るための自己制約。境界の詳細は
`docs/deployment/controller_states.md` を参照)。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Union

import yaml

from reinbalance_survivors_contracts.canonical_json import canonical_hash
from reinbalance_survivors_contracts.ui_intent import UiIntentKind, UiIntentV1
from survivors.input.controller import InputLeaseController
from survivors.perception_snapshot import (
    NormalizedRoi,
    PerceptionSnapshot,
    UiButtonTargetV1,
    UiCandidateTargetV1,
    UiPresentationSnapshotV1,
    is_equivalent_ui_target,
)

__all__ = [
    "NAVIGATION_PROFILE_SCHEMA_VERSION",
    "NavigationProfile",
    "load_navigation_profile",
    "Effect",
    "ExecutionOutcome",
    "resolve_ui_target",
    "choose_card",
    "choose_fallback",
    "choose_button",
    "execute_effect",
    "build_ui_action_telemetry",
]

# resolver が受理する typed UI target の型(候補カード or ボタン)。
UiTarget = Union[UiCandidateTargetV1, UiButtonTargetV1]

# ROI の validity が True でも、この信頼度未満なら click 対象として採用しない
# (04-09 parser 自体の閾値とは独立な、05-03 resolver 独自の安全マージン)。
_MIN_TARGET_CONFIDENCE = 0.5

NAVIGATION_PROFILE_SCHEMA_VERSION = "survivors.ui_navigation_profile.v1"

# タイムアウト/リトライ回数を state_machine.py の enum と疎結合にするための文字列 key。
# plan 本文タスク2: level_up=2s / chest=5s / unknown=1s / run_setup=15s を固定する。
# target_reached だけは「30:00 pending transition」専用の別 timeout(profile 由来)。
# RECOVER には専用 timeout を持たせない(debounce_frames 分だけ待ち、確定しなければ
# 即座に PAUSED/UNKNOWN 側へ戻して、それぞれの timeout 管理に委ねるため)。
TimeoutKey = Literal["level_up", "chest", "unknown", "run_setup", "target_reached"]

_PROFILE_WIRE_KEYS = frozenset(
    {
        "schema_version",
        "debounce_frames",
        "retry_budget",
        "retry_after_ms",
        "min_target_confidence",
        "timeouts_ms",
        "labels_ja",
    }
)

# retry_after_ns の既定値(200ms)。capture/perception の遅延が1フレーム(16ms)を
# 超える実機でも、初回 click をゲームがまだ処理し切っていないうちに retry して
# 二重clickにならないための安全マージン(M4 fix)。
_DEFAULT_RETRY_AFTER_NS = 200_000_000


@dataclass(frozen=True)
class NavigationProfile:
    """1080p 日本語 UI クライアント向けの timeout/retry/confidence プロファイル。

    やさしい説明: 「何フレーム連続で確認したら信じるか」「何秒待って
    ダメなら諦めるか」といったチューニング値をまとめた設定です。
    実際にクリックする座標はここには一切含みません(座標は毎回
    `PerceptionSnapshot.ui_presentation` から取得します)。座標を固定値として
    ここに書くと、画面レイアウトが変わったときに誤クリックする危険があるため、
    意図的に持たせていません。
    """

    schema_version: str = NAVIGATION_PROFILE_SCHEMA_VERSION
    debounce_frames: int = 3
    retry_budget: int = 1
    retry_after_ns: int = _DEFAULT_RETRY_AFTER_NS
    min_target_confidence: float = _MIN_TARGET_CONFIDENCE
    level_up_timeout_ns: int = 2_000_000_000
    chest_timeout_ns: int = 5_000_000_000
    unknown_timeout_ns: int = 1_000_000_000
    run_setup_timeout_ns: int = 15_000_000_000
    target_reached_timeout_ns: int = 5_000_000_000

    def __post_init__(self) -> None:
        """schema_version と数値レンジを検証する。

        やさしい説明: 想定外の設定ファイルを読み込んで暴走しないよう、
        起動時点で値の形をチェックします。
        """
        if self.schema_version != NAVIGATION_PROFILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {self.schema_version!r}")
        if self.debounce_frames < 1:
            raise ValueError("debounce_frames must be >= 1")
        if self.retry_budget < 0:
            raise ValueError("retry_budget must be >= 0")
        if self.retry_after_ns < 0:
            raise ValueError("retry_after_ns must be >= 0")
        if not (0.0 <= self.min_target_confidence <= 1.0):
            raise ValueError("min_target_confidence must be in [0, 1]")
        for name in (
            "level_up_timeout_ns",
            "chest_timeout_ns",
            "unknown_timeout_ns",
            "run_setup_timeout_ns",
            "target_reached_timeout_ns",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def timeout_ns_for(self, key: TimeoutKey) -> int:
        """`state_machine.py` の状態名(文字列 key)から timeout(ns)を返す。

        やさしい説明: 状態ごとに待って良い時間が違うので、その対応表です。
        """
        table: dict[str, int] = {
            "level_up": self.level_up_timeout_ns,
            "chest": self.chest_timeout_ns,
            "unknown": self.unknown_timeout_ns,
            "run_setup": self.run_setup_timeout_ns,
            "target_reached": self.target_reached_timeout_ns,
        }
        return table[key]

    @classmethod
    def default_v1(cls) -> "NavigationProfile":
        """コード内蔵の既定プロファイルを返す(YAML が無くても動く既定値)。"""
        return cls()

    @classmethod
    def from_wire(cls, data: Mapping[str, Any]) -> "NavigationProfile":
        """YAML から読み込んだ dict を検証して `NavigationProfile` を作る。"""
        unknown = set(data) - _PROFILE_WIRE_KEYS
        if unknown:
            raise ValueError(f"unknown NavigationProfile fields: {sorted(unknown)}")
        timeouts_ms = data.get("timeouts_ms", {})
        if not isinstance(timeouts_ms, Mapping):
            raise ValueError("timeouts_ms must be a mapping")

        def _ms_to_ns(key: str, default_ns: int) -> int:
            if key not in timeouts_ms:
                return default_ns
            return int(timeouts_ms[key]) * 1_000_000

        defaults = cls()
        retry_after_ms = data.get("retry_after_ms")
        retry_after_ns = (
            int(retry_after_ms) * 1_000_000 if retry_after_ms is not None else defaults.retry_after_ns
        )
        return cls(
            schema_version=data.get("schema_version", NAVIGATION_PROFILE_SCHEMA_VERSION),
            debounce_frames=int(data.get("debounce_frames", defaults.debounce_frames)),
            retry_budget=int(data.get("retry_budget", defaults.retry_budget)),
            retry_after_ns=retry_after_ns,
            min_target_confidence=float(
                data.get("min_target_confidence", defaults.min_target_confidence)
            ),
            level_up_timeout_ns=_ms_to_ns("level_up", defaults.level_up_timeout_ns),
            chest_timeout_ns=_ms_to_ns("chest", defaults.chest_timeout_ns),
            unknown_timeout_ns=_ms_to_ns("unknown", defaults.unknown_timeout_ns),
            run_setup_timeout_ns=_ms_to_ns("run_setup", defaults.run_setup_timeout_ns),
            target_reached_timeout_ns=_ms_to_ns(
                "target_reached", defaults.target_reached_timeout_ns
            ),
        )


def load_navigation_profile(path: str | Path) -> NavigationProfile:
    """``ui_navigation_1080p_ja_v1.yaml`` のような設定ファイルを読み込む。

    やさしい説明: ファイルから設定を読み込むだけの小さな関数です。
    座標は含まれないため、ここでの読み込みが誤クリックにつながることはありません。
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("navigation profile YAML must be a mapping")
    return NavigationProfile.from_wire(data)


@dataclass(frozen=True)
class Effect:
    """state machine が1 tick で発行する、唯一の入力/制御 effect。

    やさしい説明: 「今このタイミングで何をするか」を表す小さな指示書です。
    ``kind="move"`` なら移動、``kind="ui_click"``/``"ui_key"`` なら resolver が
    決めた target への click/Enter/Escape、``kind="release_all"`` は
    movement/UI の強制解放、``kind="combat_reset"`` は死亡/結果画面での
    combat LSTM state 破棄通知、``kind="controller_stop"``/
    ``"process_terminate"`` は formal terminal 確定直後だけ発行される
    制御用の指示です。``target``/``intent``/``mode`` は UI click effect の
    ときだけ埋まり、再検証や監査 telemetry の材料になります。
    """

    kind: Literal[
        "move",
        "ui_click",
        "ui_key",
        "release_all",
        "combat_reset",
        "controller_stop",
        "process_terminate",
    ]
    action_index: int | None = None
    target: UiTarget | None = None
    key: Literal["ENTER", "ESCAPE"] | None = None
    intent: UiIntentV1 | None = None
    mode: Literal["initial", "retry"] | None = None
    reason: str = ""


@dataclass(frozen=True)
class ExecutionOutcome:
    """input effect 実行の結果。

    やさしい説明: ``ack`` は「OS へ送るところまでは受理された」という意味だけで、
    ゲーム画面が実際に変化したかどうかは保証しません。適用できたかどうかは、
    次に観測する `PerceptionSnapshot` の `ui_state_key`/`screen_state` の変化で
    別途確認してください(ack と適用成功を混同しないことが本モジュールの
    安全上の要点です)。

    ``combat_reset``/``controller_stop``/``process_terminate`` の3種類は
    そもそも OS への入力ではないため、``ack=True`` は「OS が受理した」ではなく
    「この effect の実行責任は自分(``execute_effect``)ではなく呼び出し側に
    委譲する、という通知を確かに発行した」という意味でしかありません。
    ``combat_reset`` はモデル memory 破棄の合図、``controller_stop``/
    ``process_terminate`` は controller の停止/ゲームプロセス終了要求であり、
    実際の停止/終了処理を行う責任は呼び出し側(将来の 05-04 launcher)に
    あります。``execute_effect`` 自身はこれらを何も実行しません。
    """

    effect: Effect
    ack: bool


def _roi_center(roi: NormalizedRoi) -> tuple[float, float]:
    """ROI の中心座標(0.0〜1.0 正規化)を返す。"""
    return ((roi.left + roi.right) / 2.0, (roi.top + roi.bottom) / 2.0)


def _lookup_target(
    intent: UiIntentV1,
    presentation: UiPresentationSnapshotV1,
    *,
    min_confidence: float = _MIN_TARGET_CONFIDENCE,
) -> UiTarget | None:
    """intent の semantic identity に一致する typed target を1件だけ探す。

    やさしい説明: 「このindexのカード」「このIDのfallback」または
    「この名前のボタン」を UI 一覧から探します。0件・複数件・
    無効(validity=False)・信頼度不足・(ボタンなら)capability=False の
    ときは None を返し、クリックを諦めます。信頼度の閾値は
    ``min_confidence``(既定はモジュール定数だが、呼び出し側から
    ``NavigationProfile.min_target_confidence`` を渡すことを想定する)。

    ``choose_card`` は契約上 ``target_id`` を持たず(``candidate_set_hash`` +
    ``target_index`` が identity)、``choose_fallback`` は逆に ``target_id`` +
    ``target_index`` を持ち ``candidate_set_hash`` を持ちません
    (:mod:`reinbalance_survivors_contracts.ui_intent` の one-of ルール)。
    """
    if intent.kind is UiIntentKind.CHOOSE_CARD:
        if intent.candidate_set_hash != presentation.candidate_set_hash:
            # reroll 等で候補集合が入れ替わっていたら、同じ index でも別カードなので拒否する。
            return None
        matches = [
            candidate
            for candidate in presentation.candidates
            if candidate.choice_index == intent.target_index
            and candidate.semantic_kind == "item_card"
        ]
    elif intent.kind is UiIntentKind.CHOOSE_FALLBACK:
        matches = [
            candidate
            for candidate in presentation.candidates
            if candidate.choice_id == intent.target_id
            and candidate.choice_index == intent.target_index
            and candidate.semantic_kind == "fallback_reward"
        ]
    elif intent.kind in (UiIntentKind.NO_OP, UiIntentKind.STOP):
        return None
    else:
        matches = [
            button
            for button in presentation.buttons
            if button.semantic_action == intent.semantic_action
        ]
    if len(matches) != 1:
        return None
    target = matches[0]
    if not target.validity or target.confidence < min_confidence:
        return None
    if isinstance(target, UiButtonTargetV1) and not target.capability:
        return None
    return target


def choose_card(
    intent: UiIntentV1,
    presentation: UiPresentationSnapshotV1,
    *,
    min_confidence: float = _MIN_TARGET_CONFIDENCE,
) -> UiCandidateTargetV1 | None:
    """``choose_card`` intent を、target index と semantic_kind/candidate_set_hash が一致する候補へ解決する。"""
    if intent.kind is not UiIntentKind.CHOOSE_CARD:
        return None
    target = _lookup_target(intent, presentation, min_confidence=min_confidence)
    return target if isinstance(target, UiCandidateTargetV1) else None


def choose_fallback(
    intent: UiIntentV1,
    presentation: UiPresentationSnapshotV1,
    *,
    min_confidence: float = _MIN_TARGET_CONFIDENCE,
) -> UiCandidateTargetV1 | None:
    """``choose_fallback`` intent を、target id/index と semantic_kind が一致する候補へ解決する。"""
    if intent.kind is not UiIntentKind.CHOOSE_FALLBACK:
        return None
    target = _lookup_target(intent, presentation, min_confidence=min_confidence)
    return target if isinstance(target, UiCandidateTargetV1) else None


def choose_button(
    intent: UiIntentV1,
    presentation: UiPresentationSnapshotV1,
    *,
    min_confidence: float = _MIN_TARGET_CONFIDENCE,
) -> UiButtonTargetV1 | None:
    """reroll/skip/banish/ack_chest/confirm intent を semantic/capability が一致するボタンへ解決する。"""
    if intent.kind not in (
        UiIntentKind.REROLL,
        UiIntentKind.SKIP,
        UiIntentKind.BANISH,
        UiIntentKind.ACK_CHEST,
        UiIntentKind.CONFIRM,
    ):
        return None
    target = _lookup_target(intent, presentation, min_confidence=min_confidence)
    return target if isinstance(target, UiButtonTargetV1) else None


def resolve_ui_target(
    intent: UiIntentV1,
    snapshot: PerceptionSnapshot,
    mode: Literal["initial", "retry"],
    *,
    original_snapshot: PerceptionSnapshot | None = None,
    min_confidence: float = _MIN_TARGET_CONFIDENCE,
) -> UiTarget | None:
    """UiIntentV1 を、指定モードの照合規則で typed UI target へ解決する。

    やさしい説明: ``mode="initial"`` は「意図が生まれた、まさにその瞬間の
    画面」から探します(`AgentDecision` が既に snapshot/frame/content の
    完全一致を保証しているので、ここでは同じ snapshot を渡してもらうだけで
    十分です)。``mode="retry"`` は「1回だけ、もう一度送ってよいか」を判定する
    経路で、``original_snapshot``(最初に送った瞬間の snapshot)を必須にし、
    新しい snapshot が時系列で後であること、同じ ``ui_state_key`` であること、
    そして ``is_equivalent_ui_target`` が定める全 gate(semantic 一致・
    validity・信頼度・IoU・中心移動量)を満たすことを要求します。
    1つでも欠けたら None を返し、二重送信を防ぎます。``min_confidence`` は
    採用する最低信頼度で、呼び出し側(state machine)が
    ``NavigationProfile.min_target_confidence`` を渡すことを想定します
    (省略時はモジュール既定値)。
    """
    if intent.kind in (UiIntentKind.NO_OP, UiIntentKind.STOP):
        return None

    if mode == "initial":
        if (
            intent.source_snapshot_hash != snapshot.snapshot_id
            or intent.source_frame_hash != snapshot.frame_id
            or intent.source_content_hash != snapshot.source_content_hash
        ):
            return None
        return _lookup_target(intent, snapshot.ui_presentation, min_confidence=min_confidence)

    if mode == "retry":
        if original_snapshot is None:
            return None
        if snapshot.snapshot_id == original_snapshot.snapshot_id:
            # 同一 snapshot の使い回しは retry とみなさない(stale reuse を拒否)。
            return None
        if snapshot.ui_state_key != original_snapshot.ui_state_key:
            return None
        if intent.ui_state_key != original_snapshot.ui_state_key:
            return None
        old_target = _lookup_target(intent, original_snapshot.ui_presentation, min_confidence=min_confidence)
        new_target = _lookup_target(intent, snapshot.ui_presentation, min_confidence=min_confidence)
        if old_target is None or new_target is None:
            return None
        if not is_equivalent_ui_target(
            old_target,
            new_target,
            old_captured_ns=original_snapshot.captured_ns,
            new_captured_ns=snapshot.captured_ns,
            old_ui_state_key=original_snapshot.ui_state_key,
            new_ui_state_key=snapshot.ui_state_key,
        ):
            return None
        return new_target

    raise ValueError(f"unknown resolve_ui_target mode: {mode!r}")


def execute_effect(effect: Effect, controller: InputLeaseController) -> ExecutionOutcome:
    """1つの ``Effect`` を ``InputLeaseController`` の公開 API 経由で実行する。

    やさしい説明: ここが実際に OS へ入力を送る唯一の場所です。
    ``combat_reset`` はモデル memory 破棄の合図であり OS 入力ではないため、
    ``controller_stop``/``process_terminate`` と同様に呼び出し側(launcher)への
    通知として ``ack=True`` を返すだけで、controller の API は呼びません。
    この ``ack=True`` は「OS への送信を受理された」という意味ではなく、
    「実行責任を呼び出し側へ委譲する通知を発行した」という意味だけです
    (実際の controller 停止/プロセス終了処理は 05-04 launcher の責務であり、
    このモジュールは一切実行しません。``ExecutionOutcome`` の docstring も
    参照してください)。
    """
    if effect.kind == "move":
        assert effect.action_index is not None
        ack = controller.send_action(effect.action_index)
    elif effect.kind == "ui_click":
        assert effect.target is not None
        normalized_x, normalized_y = _roi_center(effect.target.roi)
        ack = controller.send_ui_click(normalized_x, normalized_y)
    elif effect.kind == "ui_key":
        assert effect.key is not None
        ack = controller.send_ui_key(effect.key)
    elif effect.kind == "release_all":
        ack = controller.emergency_release()
    elif effect.kind in ("combat_reset", "controller_stop", "process_terminate"):
        ack = True
    else:
        raise ValueError(f"unknown effect kind: {effect.kind!r}")
    return ExecutionOutcome(effect=effect, ack=ack)


def build_ui_action_telemetry(
    intent: UiIntentV1,
    target: UiTarget,
    mode: Literal["initial", "retry"],
    snapshot: PerceptionSnapshot,
) -> Mapping[str, Any]:
    """UI effect 1件分の監査用 telemetry payload を組み立てる。

    やさしい説明: 「何を」「どの候補/ボタンへ」「初回かretryか」「どの
    snapshot/ui_state_key を根拠にしたか」を後から追跡できるよう記録します。
    これは ``send_ui_click``/``send_ui_key`` の ack(OS受理)とは別物であり、
    実際にゲーム側で適用されたかどうかは、次の snapshot の
    ``ui_state_key``/``screen_state`` 変化で別途確認してください。
    座標そのものではなく hash だけを残し、座標の生ログ流出を防ぎます。
    """
    return {
        "kind": intent.kind.value,
        "semantic_action": intent.semantic_action,
        "target_id": intent.target_id,
        "target_index": intent.target_index,
        "source_snapshot_id": snapshot.snapshot_id,
        "ui_state_key": snapshot.ui_state_key,
        "roi_hash": canonical_hash(
            {"roi": [target.roi.left, target.roi.top, target.roi.right, target.roi.bottom]}
        ),
        "mode": mode,
    }

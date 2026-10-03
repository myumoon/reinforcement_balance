"""EntityTracker — フレーム間エンティティ追跡。

normalized center distance + IoU + class penalty の deterministic greedy matching で
DetectionResult とトラックを対応付ける。
大分類（class map の coarse_category）が違う組は対応付けず、大分類 weapon では
細分類（class）が一致するときだけ対応付ける。

各トラックは velocity EMA / confidence decay / age / last-seen timestamp / 初観測時刻を保持する。
player_anchor (class_id=1) が未検出のときは viewport 中央を low confidence で返す。
"""
from __future__ import annotations

import dataclasses
import itertools
import pathlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterator, Mapping

import numpy as np

from survivors.vision.world_dataset import WorldClassMap, load_class_map
from survivors.vision.world_detector import DetectionResult


# ---- class map ----

DEFAULT_CLASS_MAP_PATH = pathlib.Path(__file__).resolve().parents[2] / "configs" / "world_class_map_v2.yaml"


@lru_cache(maxsize=None)
def _cached_class_map(path: pathlib.Path) -> WorldClassMap:
    """class map をパスごとに1回だけ読み込んで共有する。

    from_state はフレームごとに呼ばれるので、毎回 YAML を読み直さないためのキャッシュ。
    """
    return load_class_map(path)


def default_class_map() -> WorldClassMap:
    """既定の class map（world_class_map_v2.yaml）を返す。

    tracker の大分類表や from_state の既定値はここから作り、クラス表を直書きしない。
    """
    return _cached_class_map(DEFAULT_CLASS_MAP_PATH)


# ---- track ----

_ID_COUNTER: Iterator[int] = itertools.count(1)


@dataclass
class Track:
    """1 エンティティのトラック状態。

    フレームをまたいで ID を維持し、velocity EMA と confidence decay を適用する。
    box_xyxy は平滑化せず最新の検出矩形で置き換え、class_id は生成時から変えない。
    """

    track_id: int
    class_id: int
    box_xyxy: np.ndarray        # (4,) float32 現在 box
    confidence: float
    first_seen_timestamp_ns: int  # track を生成したフレームの時刻
    velocity_x: float = 0.0    # pixel/frame EMA
    velocity_y: float = 0.0
    age: int = 0
    last_seen_frame_index: int = -1
    _missed_frames: int = field(default=0, repr=False)

    def update(
        self,
        box_xyxy: np.ndarray,
        confidence: float,
        frame_index: int,
        *,
        velocity_ema_alpha: float,
    ) -> None:
        """検出と一致したときにトラックを更新する。"""
        old_cx = (self.box_xyxy[0] + self.box_xyxy[2]) / 2.0
        old_cy = (self.box_xyxy[1] + self.box_xyxy[3]) / 2.0
        new_cx = (box_xyxy[0] + box_xyxy[2]) / 2.0
        new_cy = (box_xyxy[1] + box_xyxy[3]) / 2.0
        raw_vx = new_cx - old_cx
        raw_vy = new_cy - old_cy
        # EMA: 初フレームは raw = EMA
        self.velocity_x = velocity_ema_alpha * raw_vx + (1 - velocity_ema_alpha) * self.velocity_x
        self.velocity_y = velocity_ema_alpha * raw_vy + (1 - velocity_ema_alpha) * self.velocity_y
        self.box_xyxy = box_xyxy
        self.confidence = confidence
        self.last_seen_frame_index = frame_index
        self._missed_frames = 0
        self.age += 1

    def miss(self, *, confidence_decay: float) -> None:
        """検出なしフレームで age を増やし confidence を decay する。"""
        self.confidence *= confidence_decay
        self._missed_frames += 1
        self.age += 1


# ---- output schema ----

@dataclass(frozen=True)
class PlayerAnchorState:
    """player_anchor の位置（または fallback）。

    TrackedWorldStateV2 が player-relative 座標を計算するために使う。
    """

    normalized_cx: float
    normalized_cy: float
    confidence: float
    is_fallback: bool  # player_anchor 未検出のとき True


@dataclass(frozen=True)
class TrackedWorldState:
    """1 フレームの全トラック状態（tracker 内部で使用）。

    TrackedWorldStateV2.from_state() で外部 schema へ変換する。
    """

    frame_index: int
    timestamp_ns: int
    tracks: list[Track]
    player_anchor: PlayerAnchorState
    image_width: int
    image_height: int


# ---- V2 schema ----

@dataclass(frozen=True)
class TrackedEntityV2:
    """04-09 / 04-13 が消費する個別トラックの v2 schema。フィールドは golden fixture で固定。

    v1 の 15 field の末尾に、最新検出矩形の正規化幅・高さ（平滑化なし）と
    track 生成フレームの時刻 first_seen_timestamp_ns を足したもの。
    track が途切れて作り直されると first_seen_timestamp_ns も新しい時刻になる。
    """

    track_id: int
    class_id: int
    class_name: str
    coarse_class: str
    confidence: float
    age: int
    last_seen_frame_index: int
    normalized_cx: float
    normalized_cy: float
    player_relative_x: float
    player_relative_y: float
    velocity_x: float
    velocity_y: float
    on_screen: bool
    clipped: bool
    normalized_width: float
    normalized_height: float
    first_seen_timestamp_ns: int


@dataclass(frozen=True)
class TrackedWorldStateV2:
    """TrackedWorldState の v2 export schema。04-09 (real_obs_assembler) が参照する。

    timestamp / confidence / age / on_screen / clipped / coarse_class / 矩形サイズ / 初観測時刻を含む。
    v1 からの変更は TrackedEntityV2 の 3 field 追加だけ。
    """

    frame_index: int
    timestamp_ns: int
    tracks: list[TrackedEntityV2]
    player_anchor: PlayerAnchorState | None

    @classmethod
    def from_state(
        cls,
        state: TrackedWorldState,
        frame_index: int,
        timestamp_ns: int,
        class_map_path: "pathlib.Path | None" = None,
    ) -> "TrackedWorldStateV2":
        """TrackedWorldState → V2 schema へ変換する。

        class_map_path が指定されない場合は既定の world_class_map_v2.yaml を使用する。
        package から restore する場合は package 内の class_map を渡すこと。
        大分類（coarse_class）は class map の coarse_category から引く。
        """
        cm = _cached_class_map(pathlib.Path(class_map_path or DEFAULT_CLASS_MAP_PATH))

        anchor_cx = state.player_anchor.normalized_cx
        anchor_cy = state.player_anchor.normalized_cy

        entities: list[TrackedEntityV2] = []
        for t in state.tracks:
            cx = (t.box_xyxy[0] + t.box_xyxy[2]) / 2.0 / state.image_width
            cy = (t.box_xyxy[1] + t.box_xyxy[3]) / 2.0 / state.image_height
            # 矩形交差判定: 左上だけでなく「任意の部分が画面と重なるか」で on_screen を決める
            # numpy 比較は numpy.bool_ を返すので Python bool へキャストする
            on_screen = bool(
                t.box_xyxy[2] > 0  # x2 > 0
                and t.box_xyxy[3] > 0  # y2 > 0
                and t.box_xyxy[0] < state.image_width
                and t.box_xyxy[1] < state.image_height
            )
            clipped = bool(
                t.box_xyxy[0] < 0 or t.box_xyxy[1] < 0
                or t.box_xyxy[2] > state.image_width or t.box_xyxy[3] > state.image_height
            )
            try:
                name = cm.id_to_name(t.class_id)
            except KeyError:
                name = "unknown"
            try:
                coarse = cm.coarse_for(name)
            except KeyError:
                coarse = "unknown"

            entities.append(
                TrackedEntityV2(
                    track_id=t.track_id,
                    class_id=t.class_id,
                    class_name=name,
                    coarse_class=coarse,
                    confidence=t.confidence,
                    age=t.age,
                    last_seen_frame_index=t.last_seen_frame_index,
                    normalized_cx=cx,
                    normalized_cy=cy,
                    player_relative_x=cx - anchor_cx,
                    player_relative_y=cy - anchor_cy,
                    velocity_x=t.velocity_x,
                    velocity_y=t.velocity_y,
                    on_screen=on_screen,
                    clipped=clipped,
                    normalized_width=(t.box_xyxy[2] - t.box_xyxy[0]) / state.image_width,
                    normalized_height=(t.box_xyxy[3] - t.box_xyxy[1]) / state.image_height,
                    first_seen_timestamp_ns=t.first_seen_timestamp_ns,
                )
            )

        return cls(
            frame_index=frame_index,
            timestamp_ns=timestamp_ns,
            tracks=entities,
            player_anchor=state.player_anchor,
        )

    @staticmethod
    def track_field_names() -> list[str]:
        """TrackedEntityV2 のフィールド名リストを返す（golden fixture 固定用）。"""
        return [f.name for f in dataclasses.fields(TrackedEntityV2)]


# ---- tracker ----

_PLAYER_ANCHOR_CLASS_ID = 1
_PLAYER_FALLBACK_CONFIDENCE = 0.1


class EntityTracker:
    """greedy matching で DetectionResult をトラックに紐付ける tracker。

    class-specific max_age で未検出トラックを削除し、
    player_anchor 未検出時は viewport 中央 (0.5, 0.5) を fallback で返す。
    coarse_by_class_id（class_id → class map の大分類）は必須で、大分類が違う組と、
    大分類 weapon で細分類が違う組はマッチさせない。
    """

    def __init__(
        self,
        max_age_by_class: dict[int, int],
        max_match_cost: float,
        velocity_ema_alpha: float,
        confidence_decay_per_frame: float,
        *,
        coarse_by_class_id: Mapping[int, str],
    ) -> None:
        self._coarse_by_class_id = dict(coarse_by_class_id)
        self._max_age_by_class = max_age_by_class
        self._max_match_cost = max_match_cost
        self._ema_alpha = velocity_ema_alpha
        self._conf_decay = confidence_decay_per_frame
        self._tracks: list[Track] = []

    def update(
        self,
        detection: DetectionResult,
        frame_index: int,
        timestamp_ns: int,
    ) -> TrackedWorldState:
        """新規 DetectionResult でトラックを更新し TrackedWorldState を返す。"""
        matched_track_ids, matched_det_ids = self._match(detection)

        matched_tracks = {t.track_id for t in self._tracks if t.track_id in matched_track_ids}

        # マッチしたトラックを更新
        for t, det_idx in zip(
            [t for t in self._tracks if t.track_id in matched_track_ids],
            [matched_det_ids[matched_track_ids.index(t.track_id)] for t in self._tracks if t.track_id in matched_track_ids],
        ):
            t.update(
                detection.boxes_xyxy[det_idx],
                float(detection.scores[det_idx]),
                frame_index,
                velocity_ema_alpha=self._ema_alpha,
            )

        # マッチしなかったトラックを miss
        for t in self._tracks:
            if t.track_id not in matched_track_ids:
                t.miss(confidence_decay=self._conf_decay)

        # マッチしなかった検出を新トラックとして追加
        matched_det_set = set(matched_det_ids)
        for i in range(len(detection)):
            if i not in matched_det_set:
                new_track = Track(
                    track_id=next(_ID_COUNTER),
                    class_id=int(detection.class_ids[i]),
                    box_xyxy=detection.boxes_xyxy[i].copy(),
                    confidence=float(detection.scores[i]),
                    first_seen_timestamp_ns=timestamp_ns,
                    last_seen_frame_index=frame_index,
                )
                self._tracks.append(new_track)

        # 期限切れトラックを削除
        self._tracks = [
            t for t in self._tracks
            if t._missed_frames <= self._max_age_by_class.get(t.class_id, 5)
        ]

        # player anchor fallback
        player_tracks = [t for t in self._tracks if t.class_id == _PLAYER_ANCHOR_CLASS_ID]
        if player_tracks:
            pt = player_tracks[0]
            cx = (pt.box_xyxy[0] + pt.box_xyxy[2]) / 2.0 / detection.image_width
            cy = (pt.box_xyxy[1] + pt.box_xyxy[3]) / 2.0 / detection.image_height
            player_anchor = PlayerAnchorState(
                normalized_cx=cx,
                normalized_cy=cy,
                confidence=pt.confidence,
                is_fallback=False,
            )
        else:
            player_anchor = PlayerAnchorState(
                normalized_cx=0.5,
                normalized_cy=0.5,
                confidence=_PLAYER_FALLBACK_CONFIDENCE,
                is_fallback=True,
            )

        return TrackedWorldState(
            frame_index=frame_index,
            timestamp_ns=timestamp_ns,
            tracks=list(self._tracks),
            player_anchor=player_anchor,
            image_width=detection.image_width,
            image_height=detection.image_height,
        )

    # ---- matching ----

    def _match(
        self, detection: DetectionResult
    ) -> tuple[list[int], list[int]]:
        """normalized center distance + IoU + class penalty の greedy matching。

        大分類が違う組、または大分類 weapon で class が違う組の cost は無限大にし、
        max_match_cost を必ず超えるので対応付けない（新しい track になる）。
        Returns: (matched_track_ids, matched_det_indices)
        """
        if not self._tracks or len(detection) == 0:
            return [], []

        n_tracks = len(self._tracks)
        n_dets = len(detection)

        # cost matrix (n_tracks, n_dets)
        cost = np.full((n_tracks, n_dets), 1.0, dtype=np.float32)

        for ti, track in enumerate(self._tracks):
            tx1, ty1, tx2, ty2 = track.box_xyxy
            tcx = (tx1 + tx2) / 2.0 / detection.image_width
            tcy = (ty1 + ty2) / 2.0 / detection.image_height

            for di in range(n_dets):
                dx1, dy1, dx2, dy2 = detection.boxes_xyxy[di]
                dcx = (dx1 + dx2) / 2.0 / detection.image_width
                dcy = (dy1 + dy2) / 2.0 / detection.image_height

                det_class_id = int(detection.class_ids[di])
                same_class = track.class_id == det_class_id
                track_coarse = self._coarse_by_class_id.get(track.class_id)
                if track_coarse != self._coarse_by_class_id.get(det_class_id) or (
                    track_coarse == "weapon" and not same_class
                ):
                    cost[ti, di] = np.inf
                    continue

                dist = np.sqrt((tcx - dcx) ** 2 + (tcy - dcy) ** 2)
                iou = _iou(track.box_xyxy, detection.boxes_xyxy[di])
                class_penalty = 0.0 if same_class else 0.4

                # ponytail: O(n_tracks * n_dets) で密度が低いゲームには十分
                cost[ti, di] = 0.4 * dist + 0.4 * (1.0 - iou) + 0.2 * class_penalty

        matched_tracks: list[int] = []
        matched_dets: list[int] = []
        used_det: set[int] = set()

        # greedy: cost が小さい順にマッチング
        flat_order = np.argsort(cost.ravel())
        for idx in flat_order:
            ti, di = divmod(int(idx), n_dets)
            if ti in {self._tracks.index(t) for t in self._tracks if t.track_id in matched_tracks}:
                continue
            if di in used_det:
                continue
            if cost[ti, di] > self._max_match_cost:
                break
            matched_tracks.append(self._tracks[ti].track_id)
            matched_dets.append(di)
            used_det.add(di)

        return matched_tracks, matched_dets


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """2 つの [x1,y1,x2,y2] box の IoU を返す。"""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

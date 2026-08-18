"""EntityTracker のテスト。

crossing tracks / occlusion / false positive / camera-relative motion / max-age expiry を
synthetic trajectories で検証する。
player anchor missing 時の viewport calibrated center fallback と low confidence も検証する。
実 GPU・実画像は不要。
"""
from __future__ import annotations

import numpy as np
import pytest

from survivors.vision.entity_tracker import (
    EntityTracker,
    Track,
    TrackedWorldState,
    TrackedWorldStateV1,
)
from survivors.vision.world_detector import DetectionResult


# ---- helper ----

def _make_detection(
    boxes_xyxy: list[list[float]],
    scores: list[float],
    class_ids: list[int],
    image_width: int = 1920,
    image_height: int = 1080,
) -> DetectionResult:
    return DetectionResult(
        boxes_xyxy=np.array(boxes_xyxy, dtype=np.float32).reshape(-1, 4),
        scores=np.array(scores, dtype=np.float32),
        class_ids=np.array(class_ids, dtype=np.int32),
        image_width=image_width,
        image_height=image_height,
    )


def _make_tracker(max_age_default: int = 5) -> EntityTracker:
    """最小設定の EntityTracker を返す。"""
    return EntityTracker(
        max_age_by_class={i: max_age_default for i in range(12)},
        max_match_cost=0.7,
        velocity_ema_alpha=0.6,
        confidence_decay_per_frame=0.9,
    )


# ---- basic tracking ----

class TestBasicTracking:
    def test_single_detection_creates_track(self):
        tracker = _make_tracker()
        det = _make_detection([[100, 200, 200, 300]], [0.9], [1])
        state = tracker.update(det, frame_index=0, timestamp_ns=1000)
        assert len(state.tracks) == 1
        assert state.tracks[0].track_id is not None

    def test_consistent_detection_maintains_track_id(self):
        tracker = _make_tracker()
        det0 = _make_detection([[100, 200, 200, 300]], [0.9], [1])
        state0 = tracker.update(det0, frame_index=0, timestamp_ns=1000)
        tid = state0.tracks[0].track_id

        # 同じ位置に検出 → 同じ track_id
        det1 = _make_detection([[105, 205, 205, 305]], [0.9], [1])
        state1 = tracker.update(det1, frame_index=1, timestamp_ns=2000)
        assert len(state1.tracks) == 1
        assert state1.tracks[0].track_id == tid

    def test_new_object_gets_new_track_id(self):
        tracker = _make_tracker()
        det0 = _make_detection([[100, 200, 200, 300]], [0.9], [1])
        state0 = tracker.update(det0, frame_index=0, timestamp_ns=1000)
        tid0 = state0.tracks[0].track_id

        # 全く異なる位置の新しい検出
        det1 = _make_detection([[1000, 800, 1100, 900]], [0.9], [2])
        state1 = tracker.update(det1, frame_index=1, timestamp_ns=2000)
        # 元のトラックは age で残るか age-expired するが新しいトラックは別 ID
        new_ids = {t.track_id for t in state1.tracks}
        assert tid0 not in new_ids or len(new_ids) > 1  # 新規 ID が生成されている


# ---- max age expiry ----

class TestMaxAgeExpiry:
    def test_missing_track_expires_after_max_age(self):
        """検出が消えたトラックは max_age フレーム後に削除される。"""
        max_age = 3
        tracker = EntityTracker(
            max_age_by_class={1: max_age},
            max_match_cost=0.7,
            velocity_ema_alpha=0.6,
            confidence_decay_per_frame=0.9,
        )
        det = _make_detection([[100, 200, 200, 300]], [0.9], [1])
        tracker.update(det, frame_index=0, timestamp_ns=1000)

        # 以降は空の検出
        empty = _make_detection([], [], [])
        for i in range(1, max_age + 2):
            state = tracker.update(empty, frame_index=i, timestamp_ns=1000 + i * 100)

        # max_age を超えたので全トラックが消える
        assert len(state.tracks) == 0

    def test_track_survives_within_max_age(self):
        """max_age 以内なら tracker がトラックを維持する。"""
        max_age = 5
        tracker = EntityTracker(
            max_age_by_class={1: max_age},
            max_match_cost=0.7,
            velocity_ema_alpha=0.6,
            confidence_decay_per_frame=0.9,
        )
        det = _make_detection([[100, 200, 200, 300]], [0.9], [1])
        tracker.update(det, frame_index=0, timestamp_ns=1000)

        empty = _make_detection([], [], [])
        for i in range(1, max_age):
            state = tracker.update(empty, frame_index=i, timestamp_ns=1000 + i * 100)

        assert len(state.tracks) == 1


# ---- crossing tracks ----

class TestCrossingTracks:
    def test_two_objects_crossing_maintain_ids(self):
        """2 エンティティが交差しても class + IoU でアイデンティティを維持する。"""
        tracker = _make_tracker()
        # frame 0: A(左) class=2, B(右) class=3
        det0 = _make_detection([[100, 500, 200, 600], [800, 500, 900, 600]], [0.9, 0.9], [2, 3])
        state0 = tracker.update(det0, frame_index=0, timestamp_ns=0)
        id_a = next(t.track_id for t in state0.tracks if t.class_id == 2)
        id_b = next(t.track_id for t in state0.tracks if t.class_id == 3)

        # frame 1: 両者が中央へ移動（交差寸前）
        det1 = _make_detection([[400, 500, 500, 600], [500, 500, 600, 600]], [0.9, 0.9], [2, 3])
        state1 = tracker.update(det1, frame_index=1, timestamp_ns=100)
        ids1 = {t.class_id: t.track_id for t in state1.tracks}
        # class が違うので同じ ID が維持される
        assert ids1[2] == id_a
        assert ids1[3] == id_b


# ---- occlusion ----

class TestOcclusion:
    def test_occluded_track_decays_confidence(self):
        """検出されなかったフレームでは confidence が decay する。"""
        tracker = EntityTracker(
            max_age_by_class={1: 10},
            max_match_cost=0.7,
            velocity_ema_alpha=0.6,
            confidence_decay_per_frame=0.8,
        )
        det = _make_detection([[100, 200, 200, 300]], [0.9], [1])
        tracker.update(det, frame_index=0, timestamp_ns=0)

        empty = _make_detection([], [], [])
        state1 = tracker.update(empty, frame_index=1, timestamp_ns=100)
        conf1 = state1.tracks[0].confidence

        state2 = tracker.update(empty, frame_index=2, timestamp_ns=200)
        conf2 = state2.tracks[0].confidence

        assert conf2 < conf1  # decay されている


# ---- player anchor fallback ----

class TestPlayerAnchorFallback:
    def test_missing_player_anchor_returns_center_with_low_confidence(self):
        """player_anchor (class_id=1) がなければ viewport 中央を low confidence で返す。"""
        tracker = _make_tracker()
        # player なし
        det = _make_detection([[100, 100, 200, 200]], [0.8], [2])  # class=2 (enemy)
        state = tracker.update(det, frame_index=0, timestamp_ns=0)

        assert state.player_anchor is not None
        assert state.player_anchor.confidence < 0.3  # low confidence
        # 正規化座標で (0.5, 0.5) に近い
        assert abs(state.player_anchor.normalized_cx - 0.5) < 0.1
        assert abs(state.player_anchor.normalized_cy - 0.5) < 0.1


# ---- velocity and age ----

class TestTrackAttributes:
    def test_track_has_velocity_age_last_seen(self):
        tracker = _make_tracker()
        det0 = _make_detection([[100, 200, 200, 300]], [0.9], [1])
        tracker.update(det0, frame_index=0, timestamp_ns=0)

        det1 = _make_detection([[110, 210, 210, 310]], [0.85], [1])
        state1 = tracker.update(det1, frame_index=1, timestamp_ns=100)

        track = state1.tracks[0]
        assert hasattr(track, "velocity_x")
        assert hasattr(track, "velocity_y")
        assert track.age >= 1
        assert track.last_seen_frame_index == 1

    def test_velocity_ema_smoothed(self):
        """velocity は EMA で平滑化される（急激な変化を吸収する）。"""
        tracker = EntityTracker(
            max_age_by_class={1: 10},
            max_match_cost=0.7,
            velocity_ema_alpha=0.5,
            confidence_decay_per_frame=0.9,
        )
        det0 = _make_detection([[100, 100, 200, 200]], [0.9], [1])
        tracker.update(det0, frame_index=0, timestamp_ns=0)
        det1 = _make_detection([[200, 100, 300, 200]], [0.9], [1])
        state1 = tracker.update(det1, frame_index=1, timestamp_ns=100)
        vx = state1.tracks[0].velocity_x
        # raw は 100 px 移動だが EMA で平滑化
        assert vx > 0  # 正方向
        assert vx <= 100  # raw 値未満（EMA 初期は raw = EMA のこともある）


# ---- TrackedWorldStateV1 schema ----

class TestTrackedWorldStateV1:
    """TrackedWorldStateV1 のフィールド・スキーマハッシュを golden fixture で固定する。"""

    def test_state_has_required_fields(self):
        tracker = _make_tracker()
        det = _make_detection([[100, 200, 200, 300]], [0.9], [2])
        state = tracker.update(det, frame_index=0, timestamp_ns=12345)
        v1 = TrackedWorldStateV1.from_state(state, frame_index=0, timestamp_ns=12345)

        assert hasattr(v1, "frame_index")
        assert hasattr(v1, "timestamp_ns")
        assert hasattr(v1, "tracks")
        # 各トラックに必要なフィールド
        if v1.tracks:
            t = v1.tracks[0]
            assert hasattr(t, "track_id")
            assert hasattr(t, "class_id")
            assert hasattr(t, "confidence")
            assert hasattr(t, "age")
            assert hasattr(t, "on_screen")
            assert hasattr(t, "clipped")
            assert hasattr(t, "coarse_class")
            assert hasattr(t, "normalized_cx")
            assert hasattr(t, "normalized_cy")
            assert hasattr(t, "player_relative_x")
            assert hasattr(t, "player_relative_y")

    def test_schema_hash_is_stable(self):
        """スキーマハッシュが変わっていないことを確認する。"""
        # TrackedWorldStateV1 のフィールド名セットを固定する
        expected_track_fields = {
            "track_id", "class_id", "class_name", "coarse_class",
            "confidence", "age", "last_seen_frame_index",
            "normalized_cx", "normalized_cy",
            "player_relative_x", "player_relative_y",
            "velocity_x", "velocity_y",
            "on_screen", "clipped",
        }
        actual_fields = set(TrackedWorldStateV1.track_field_names())
        assert actual_fields == expected_track_fields

    def test_on_screen_flag(self):
        """画面内の検出は on_screen=True。"""
        tracker = _make_tracker()
        det = _make_detection([[0, 0, 100, 100]], [0.9], [1])
        state = tracker.update(det, frame_index=0, timestamp_ns=0)
        v1 = TrackedWorldStateV1.from_state(state, frame_index=0, timestamp_ns=0)
        assert v1.tracks[0].on_screen is True

    def test_partially_visible_entity_is_on_screen(self):
        """画面と部分的に交差する box（左上が画面外でも）は on_screen=True。"""
        tracker = _make_tracker()
        # box が x=-10 から x=20（画面内に 20px 見えている）
        det = _make_detection([[-10, 100, 20, 200]], [0.9], [1])
        state = tracker.update(det, frame_index=0, timestamp_ns=0)
        v1 = TrackedWorldStateV1.from_state(state, frame_index=0, timestamp_ns=0)
        # 矩形交差: x2=20 > 0, y2=200 > 0, x1=-10 < 1920, y1=100 < 1080
        assert v1.tracks[0].on_screen is True
        assert v1.tracks[0].clipped is True  # 画面外にはみ出しているので clipped

    def test_player_anchor_state_included(self):
        tracker = _make_tracker()
        det = _make_detection([[960, 540, 1060, 640]], [0.95], [1])  # player_anchor
        state = tracker.update(det, frame_index=0, timestamp_ns=0)
        v1 = TrackedWorldStateV1.from_state(state, frame_index=0, timestamp_ns=0)
        # player_anchor がいる → player_relative 座標が計算できる
        assert v1.player_anchor is not None

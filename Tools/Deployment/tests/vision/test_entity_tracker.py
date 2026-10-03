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
    TrackedWorldStateV2,
    default_class_map,
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


_CLASS_MAP = default_class_map()
COARSE_BY_ID = _CLASS_MAP.coarse_by_class_id()


def _make_tracker(max_age_default: int = 5) -> EntityTracker:
    """最小設定の EntityTracker を返す（大分類は既定 class map v2 から）。"""
    return EntityTracker(
        max_age_by_class={i: max_age_default for i in range(_CLASS_MAP.num_classes)},
        max_match_cost=0.7,
        velocity_ema_alpha=0.6,
        confidence_decay_per_frame=0.9,
        coarse_by_class_id=COARSE_BY_ID,
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
            coarse_by_class_id=COARSE_BY_ID,
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
            coarse_by_class_id=COARSE_BY_ID,
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
            coarse_by_class_id=COARSE_BY_ID,
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
            coarse_by_class_id=COARSE_BY_ID,
        )
        det0 = _make_detection([[100, 100, 200, 200]], [0.9], [1])
        tracker.update(det0, frame_index=0, timestamp_ns=0)
        det1 = _make_detection([[200, 100, 300, 200]], [0.9], [1])
        state1 = tracker.update(det1, frame_index=1, timestamp_ns=100)
        vx = state1.tracks[0].velocity_x
        # raw は 100 px 移動だが EMA で平滑化
        assert vx > 0  # 正方向
        assert vx <= 100  # raw 値未満（EMA 初期は raw = EMA のこともある）


# ---- TrackedWorldStateV2 schema ----

class TestTrackedWorldStateV2:
    """TrackedWorldStateV2 のフィールド・スキーマハッシュを golden fixture で固定する。"""

    def test_state_has_required_fields(self):
        tracker = _make_tracker()
        det = _make_detection([[100, 200, 200, 300]], [0.9], [2])
        state = tracker.update(det, frame_index=0, timestamp_ns=12345)
        v1 = TrackedWorldStateV2.from_state(state, frame_index=0, timestamp_ns=12345)

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
        # TrackedWorldStateV2 のフィールド名セットを固定する
        expected_track_fields = {
            "track_id", "class_id", "class_name", "coarse_class",
            "confidence", "age", "last_seen_frame_index",
            "normalized_cx", "normalized_cy",
            "player_relative_x", "player_relative_y",
            "velocity_x", "velocity_y",
            "on_screen", "clipped",
            "normalized_width", "normalized_height", "first_seen_timestamp_ns",
        }
        actual_fields = set(TrackedWorldStateV2.track_field_names())
        assert actual_fields == expected_track_fields
        # v1 の 15 field の後ろに 3 field を足しただけで、並びは変えない。
        assert TrackedWorldStateV2.track_field_names()[-3:] == [
            "normalized_width", "normalized_height", "first_seen_timestamp_ns",
        ]

    def test_size_is_latest_detection_box(self):
        """normalized_width / height は平滑化しない最新の検出矩形の大きさ。"""
        tracker = _make_tracker()
        tracker.update(_make_detection([[100, 200, 200, 300]], [0.9], [2]), frame_index=0, timestamp_ns=0)
        state = tracker.update(_make_detection([[110, 205, 302, 413]], [0.9], [2]), frame_index=1, timestamp_ns=100)
        v2 = TrackedWorldStateV2.from_state(state, frame_index=1, timestamp_ns=100)
        assert len(v2.tracks) == 1
        assert v2.tracks[0].normalized_width == pytest.approx(192 / 1920)
        assert v2.tracks[0].normalized_height == pytest.approx(208 / 1080)

    def test_first_seen_timestamp_kept_while_matched_and_reset_on_recreate(self):
        """初観測時刻は追跡中は保たれ、track が消えて作り直されると新しい時刻になる。"""
        tracker = EntityTracker(
            max_age_by_class={2: 1},
            max_match_cost=0.7,
            velocity_ema_alpha=0.6,
            confidence_decay_per_frame=0.9,
            coarse_by_class_id=COARSE_BY_ID,
        )
        box = [[100, 200, 200, 300]]
        tracker.update(_make_detection(box, [0.9], [2]), frame_index=0, timestamp_ns=1000)
        state = tracker.update(_make_detection(box, [0.9], [2]), frame_index=1, timestamp_ns=2000)
        first = TrackedWorldStateV2.from_state(state, frame_index=1, timestamp_ns=2000).tracks[0]
        assert first.first_seen_timestamp_ns == 1000
        empty = _make_detection([], [], [])
        tracker.update(empty, frame_index=2, timestamp_ns=3000)
        assert tracker.update(empty, frame_index=3, timestamp_ns=4000).tracks == []
        state = tracker.update(_make_detection(box, [0.9], [2]), frame_index=4, timestamp_ns=5000)
        again = TrackedWorldStateV2.from_state(state, frame_index=4, timestamp_ns=5000).tracks[0]
        assert again.track_id != first.track_id
        assert again.first_seen_timestamp_ns == 5000

    def test_on_screen_flag(self):
        """画面内の検出は on_screen=True。"""
        tracker = _make_tracker()
        det = _make_detection([[0, 0, 100, 100]], [0.9], [1])
        state = tracker.update(det, frame_index=0, timestamp_ns=0)
        v1 = TrackedWorldStateV2.from_state(state, frame_index=0, timestamp_ns=0)
        assert v1.tracks[0].on_screen is True

    def test_partially_visible_entity_is_on_screen(self):
        """画面と部分的に交差する box（左上が画面外でも）は on_screen=True。"""
        tracker = _make_tracker()
        # box が x=-10 から x=20（画面内に 20px 見えている）
        det = _make_detection([[-10, 100, 20, 200]], [0.9], [1])
        state = tracker.update(det, frame_index=0, timestamp_ns=0)
        v1 = TrackedWorldStateV2.from_state(state, frame_index=0, timestamp_ns=0)
        # 矩形交差: x2=20 > 0, y2=200 > 0, x1=-10 < 1920, y1=100 < 1080
        assert v1.tracks[0].on_screen is True
        assert v1.tracks[0].clipped is True  # 画面外にはみ出しているので clipped

    def test_player_anchor_state_included(self):
        tracker = _make_tracker()
        det = _make_detection([[960, 540, 1060, 640]], [0.95], [1])  # player_anchor
        state = tracker.update(det, frame_index=0, timestamp_ns=0)
        v1 = TrackedWorldStateV2.from_state(state, frame_index=0, timestamp_ns=0)
        # player_anchor がいる → player_relative 座標が計算できる
        assert v1.player_anchor is not None


# ---- coarse category matching (class map v2) ----

_ANCHOR = _CLASS_MAP.name_to_id("player_anchor")
_PROJECTILE = _CLASS_MAP.name_to_id("weapon_projectile")
_ZONE = _CLASS_MAP.name_to_id("weapon_zone")
_AURA = _CLASS_MAP.name_to_id("weapon_aura")


class TestCoarseCategoryMatching:
    """大分類をまたぐマッチと、weapon 内で細分類が違うマッチを禁止する規則を検証する。"""

    def test_concentric_aura_and_anchor_do_not_swap(self):
        """プレイヤーと同心の大きい weapon_aura が player_anchor の track を奪わない。"""
        tracker = _make_tracker()
        anchor_box = [940, 520, 980, 560]
        aura_box = [860, 440, 1060, 640]
        state0 = tracker.update(_make_detection([anchor_box], [0.9], [_ANCHOR]), frame_index=0, timestamp_ns=0)
        anchor_id = state0.tracks[0].track_id
        # anchor が見えず aura だけが同じ中心に出たフレーム: aura は anchor track に割り当てない。
        state1 = tracker.update(_make_detection([aura_box], [0.9], [_AURA]), frame_index=1, timestamp_ns=100)
        by_id = {t.track_id: t for t in state1.tracks}
        assert by_id[anchor_id].class_id == _ANCHOR
        assert list(by_id[anchor_id].box_xyxy) == anchor_box  # anchor の矩形が膨らまない
        aura_tracks = [t for t in state1.tracks if t.class_id == _AURA]
        assert len(aura_tracks) == 1 and aura_tracks[0].track_id != anchor_id
        # 両方が出たフレームでもそれぞれの track に戻る。
        state2 = tracker.update(
            _make_detection([aura_box, anchor_box], [0.9, 0.9], [_AURA, _ANCHOR]), frame_index=2, timestamp_ns=200
        )
        ids2 = {t.class_id: t.track_id for t in state2.tracks}
        assert ids2 == {_ANCHOR: anchor_id, _AURA: aura_tracks[0].track_id}

    def test_cross_coarse_cost_is_infinite(self):
        """大分類が違う組は class penalty ではなく cost 無限大でマッチしない。"""
        tracker = _make_tracker()
        box = [100, 200, 200, 300]
        tracker.update(_make_detection([box], [0.9], [_ANCHOR]), frame_index=0, timestamp_ns=0)
        assert tracker._match(_make_detection([box], [0.9], [_AURA])) == ([], [])
        # 同じ大分類（enemy_normal → enemy_elite）は従来どおり class penalty 付きでマッチできる。
        tracker2 = _make_tracker()
        tracker2.update(_make_detection([box], [0.9], [2]), frame_index=0, timestamp_ns=0)
        assert len(tracker2._match(_make_detection([box], [0.9], [3]))[0]) == 1

    def test_fireball_to_explosion_creates_new_track(self):
        """weapon_projectile → weapon_zone（火球→爆発）は同じ位置でも新しい track になる。"""
        tracker = _make_tracker()
        box = [500, 500, 540, 540]
        state0 = tracker.update(_make_detection([box], [0.9], [_PROJECTILE]), frame_index=0, timestamp_ns=1000)
        fireball_id = state0.tracks[0].track_id
        state1 = tracker.update(_make_detection([box], [0.9], [_ZONE]), frame_index=1, timestamp_ns=2000)
        zone = [t for t in state1.tracks if t.class_id == _ZONE]
        assert len(zone) == 1 and zone[0].track_id != fireball_id
        assert zone[0].first_seen_timestamp_ns == 2000
        # 火球 track は class を変えずに miss 扱いで残る。
        assert next(t for t in state1.tracks if t.track_id == fireball_id).class_id == _PROJECTILE

    def test_weapon_track_coarse_class_is_weapon(self):
        """weapon の track は TrackedEntityV2.coarse_class == "weapon" で出力される。"""
        tracker = _make_tracker()
        ids = [_CLASS_MAP.name_to_id(n) for n in ("weapon_projectile", "weapon_zone", "weapon_orbit", "weapon_aura")]
        boxes = [[100 + 200 * i, 100, 150 + 200 * i, 150] for i in range(4)]
        state = tracker.update(_make_detection(boxes, [0.9] * 4, ids), frame_index=0, timestamp_ns=0)
        v2 = TrackedWorldStateV2.from_state(state, frame_index=0, timestamp_ns=0)
        assert sorted((t.class_name, t.coarse_class) for t in v2.tracks) == sorted(
            (n, "weapon") for n in ("weapon_projectile", "weapon_zone", "weapon_orbit", "weapon_aura")
        )

    def test_coarse_by_class_id_is_required(self):
        """大分類表は必須 keyword 引数で、省略すると構築できない。"""
        with pytest.raises(TypeError):
            EntityTracker({1: 5}, 0.7, 0.6, 0.9)  # type: ignore[call-arg]

"""WorldDetector 用 COCO dataset ローダーと preflight チェック。

本家 Vampire Survivors のアノテーション済み COCO JSON を読み込み、
session 単位での split 管理と preflight 品質チェックを提供する。
学習ハーネスと評価ハーネスの両方からインポートされる共通 dataset 層。
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml


# ---- 例外 ----

class DatasetPreflightError(ValueError):
    """preflight の最小要件を満たさないときに送出される。"""


# ---- class map ----

@dataclass(frozen=True)
class WorldClassMap:
    """background 0 + foreground 11 の合計 12 クラス固定マップ。

    ID ↔ 名前の変換を提供し、未知ラベルへのアクセスは KeyError を返す。
    04-07 / 04-08 / 04-09 が同一 hash で参照する。
    """

    schema_version: str
    class_map_version: int
    background_label: int
    background_name: str
    foreground_classes: tuple[dict, ...]
    num_classes: int

    # 内部 lookup table（frozen dataclass なので __post_init__ で構築）
    _id_to_name: dict[int, str] = field(default_factory=dict, compare=False, hash=False, repr=False)
    _name_to_id: dict[str, int] = field(default_factory=dict, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        # frozen なので object.__setattr__ で初期化する
        id_map: dict[int, str] = {self.background_label: self.background_name}
        name_map: dict[str, int] = {self.background_name: self.background_label}
        for fc in self.foreground_classes:
            id_map[fc["id"]] = fc["name"]
            name_map[fc["name"]] = fc["id"]
        object.__setattr__(self, "_id_to_name", id_map)
        object.__setattr__(self, "_name_to_id", name_map)

    def id_to_name(self, class_id: int) -> str:
        """class_id → 名前。未知 ID は KeyError。"""
        if class_id not in self._id_to_name:
            raise KeyError(f"未知 class_id: {class_id}")
        return self._id_to_name[class_id]

    def name_to_id(self, name: str) -> int:
        """名前 → class_id。未知名は KeyError。"""
        if name not in self._name_to_id:
            raise KeyError(f"未知 class name: {name}")
        return self._name_to_id[name]

    @property
    def all_class_names(self) -> list[str]:
        """background を含む全クラス名を ID 順に返す。"""
        return [self._id_to_name[i] for i in range(self.num_classes)]


def load_class_map(path: pathlib.Path | str) -> WorldClassMap:
    """YAML ファイルから WorldClassMap を読み込む。

    スキーマバージョンが一致しない場合は ValueError。
    """
    path = pathlib.Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data.get("schema_version") != "world_class_map.v1":
        raise ValueError(f"未知の schema_version: {data.get('schema_version')}")

    return WorldClassMap(
        schema_version=data["schema_version"],
        class_map_version=data["class_map_version"],
        background_label=data["background_label"],
        background_name=data["background_name"],
        foreground_classes=tuple(data["foreground_classes"]),
        num_classes=data["num_classes"],
    )


# ---- annotation contract ----

@dataclass(frozen=True)
class COCOAnnotation:
    """1 エンティティのアノテーション。COCO bbox [x, y, w, h] 形式。

    category_id は WorldClassMap の foreground ID (1..11)。
    """

    annotation_id: int
    image_id: int
    category_id: int
    bbox_xywh: tuple[float, float, float, float]
    session_id: str


@dataclass
class DatasetSample:
    """1 フレーム分のサンプル（アノテーションリスト付き）。

    image_id と file_name を保持し、loader が PNG を開けるよう準備する。
    session_id は画像レベルの session 識別子で、アノテーション無し（負例）フレームの
    split 割り当てに使用する。
    """

    image_id: int
    file_name: str
    image_width: int
    image_height: int
    annotations: list[COCOAnnotation]
    session_id: str = ""


# ---- split ----

@dataclass(frozen=True)
class SessionSplit:
    """session 単位の用途別 split。

    一度構築した後は変更不可（frozen）。
    同一 session が複数用途に割り当てられると ValueError。
    """

    train: tuple[str, ...]
    validation: tuple[str, ...]
    error_calibration: tuple[str, ...]
    final_e2e_test: tuple[str, ...]

    def __init__(
        self,
        train: list[str],
        validation: list[str],
        error_calibration: list[str],
        final_e2e_test: list[str],
    ) -> None:
        # frozen なので object.__setattr__ で代入
        object.__setattr__(self, "train", tuple(train))
        object.__setattr__(self, "validation", tuple(validation))
        object.__setattr__(self, "error_calibration", tuple(error_calibration))
        object.__setattr__(self, "final_e2e_test", tuple(final_e2e_test))
        self._validate_no_overlap()

    def _validate_no_overlap(self) -> None:
        all_lists = [self.train, self.validation, self.error_calibration, self.final_e2e_test]
        seen: set[str] = set()
        for lst in all_lists:
            for s in lst:
                if s in seen:
                    raise ValueError(f"session '{s}' が複数用途に割り当てられています（overlap）。")
                seen.add(s)

    def canonical_hash(self) -> str:
        """split 内容を確定的に SHA-256 ハッシュ化する。"""
        payload = json.dumps(
            {
                "train": sorted(self.train),
                "validation": sorted(self.validation),
                "error_calibration": sorted(self.error_calibration),
                "final_e2e_test": sorted(self.final_e2e_test),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


# ---- dataset ----

class WorldDataset:
    """COCO JSON を読み込み、フレームごとにサンプルを提供する dataset。

    class map 整合性と bbox bounds を検証し、不正なアノテーションを拒否する。
    session_id フィールドを持たないアノテーションは session_id="" として扱う。
    rejected_sessions に含まれる session_id のフレームは DatasetPreflightError を送出する。
    """

    def __init__(
        self,
        annotation_path: pathlib.Path | str,
        class_map_path: pathlib.Path | str,
        *,
        validate_bounds: bool = False,
        rejected_sessions: set[str] | None = None,
    ) -> None:
        self._class_map = load_class_map(class_map_path)
        self._rejected_sessions: set[str] = rejected_sessions or set()
        self._samples = self._load(pathlib.Path(annotation_path), validate_bounds=validate_bounds)

    # ---- public API ----

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> DatasetSample:
        return self._samples[idx]

    def run_preflight(
        self,
        min_frames: int,
        min_entities: int,
        min_classes: int,
        min_time_bands: int,
    ) -> None:
        """学習開始前の品質チェック。条件不足なら DatasetPreflightError を送出する。

        不足があっても training step 0 のまま停止する（学習ループには入らない）。
        duplicate annotation 検査・class agreement 検査は 04-07 で実装予定（現在は未対応）。
        """
        n_frames = len(self._samples)
        if n_frames < min_frames:
            raise DatasetPreflightError(
                f"min_frames 不足: {n_frames} < {min_frames}"
            )

        all_annotations = [a for s in self._samples for a in s.annotations]
        if len(all_annotations) < min_entities:
            raise DatasetPreflightError(
                f"min_entities 不足: {len(all_annotations)} < {min_entities}"
            )

        n_classes = len({a.category_id for a in all_annotations})
        if n_classes < min_classes:
            raise DatasetPreflightError(
                f"min_classes 不足: {n_classes} < {min_classes}"
            )

        # annotation.session_id は空の場合があるため、画像レベルの DatasetSample.session_id を正本として使用する
        n_bands = len({s.session_id for s in self._samples if s.session_id})
        if n_bands < min_time_bands:
            raise DatasetPreflightError(
                f"min_time_bands 不足: {n_bands} < {min_time_bands}"
            )

    def run_split_preflight(
        self,
        split: "SessionSplit",
        *,
        min_frames_per_split: int = 50,
        min_entities_per_split: int = 100,
        min_classes_per_split: int = 4,
    ) -> None:
        """SessionSplit 整合性と split ごとの最小要件チェック。

        train / validation 各 split が最小 frame / entity / class を満たすか確認する。
        0 件 slice は DatasetPreflightError で失敗する。
        error_calibration / final_e2e_test の session ID がサンプル側に存在しないことも確認する。
        """
        # DatasetSample.session_id を正本として使用する。
        missing_session = [s for s in self._samples if not s.session_id]
        if missing_session:
            image_ids = [s.image_id for s in missing_session[:5]]
            raise DatasetPreflightError(
                f"session_id が未設定の画像が {len(missing_session)} 件あります"
                f" (image_ids={image_ids}...)。session 欠損サンプルは使用できません。"
            )

        forbidden = set(split.error_calibration) | set(split.final_e2e_test)
        seen_in_data = {s.session_id for s in self._samples}
        overlap = forbidden & seen_in_data
        if overlap:
            raise DatasetPreflightError(
                f"error_calibration/final_e2e_test セッション {sorted(overlap)} がデータセットに含まれています。"
            )

        split_ids: dict[str, tuple[str, ...]] = {
            "train": split.train,
            "validation": split.validation,
        }
        for split_name, session_ids in split_ids.items():
            session_set = set(session_ids)
            if not session_set:
                raise DatasetPreflightError(f"split '{split_name}' のセッションが 0 件です。")
            samples_in_split = [
                s for s in self._samples
                if s.session_id in session_set
            ]
            n_frames = len(samples_in_split)
            if n_frames < min_frames_per_split:
                raise DatasetPreflightError(
                    f"split '{split_name}' の min_frames 不足: {n_frames} < {min_frames_per_split}"
                )
            anns = [a for s in samples_in_split for a in s.annotations]
            if len(anns) < min_entities_per_split:
                raise DatasetPreflightError(
                    f"split '{split_name}' の min_entities 不足: {len(anns)} < {min_entities_per_split}"
                )
            n_classes = len({a.category_id for a in anns})
            if n_classes < min_classes_per_split:
                raise DatasetPreflightError(
                    f"split '{split_name}' の min_classes 不足: {n_classes} < {min_classes_per_split}"
                )

    # ---- internal ----

    def _load(
        self, path: pathlib.Path, *, validate_bounds: bool
    ) -> list[DatasetSample]:
        data = json.loads(path.read_text(encoding="utf-8"))

        # image index
        images: dict[int, dict] = {img["id"]: img for img in data.get("images", [])}

        # annotation → sample
        ann_by_image: dict[int, list[COCOAnnotation]] = {iid: [] for iid in images}
        for raw in data.get("annotations", []):
            cat_id = raw["category_id"]
            if cat_id == self._class_map.background_label:
                raise ValueError(
                    f"background (id={cat_id}) はアノテーションに使用できません。"
                )
            try:
                self._class_map.id_to_name(cat_id)
            except KeyError:
                raise ValueError(
                    f"未知 category_id={cat_id} がアノテーションに含まれています。"
                )

            bbox = tuple(float(v) for v in raw["bbox"])
            x, y, w, h = bbox
            if not all(math.isfinite(v) for v in bbox):
                raise ValueError(f"bbox {bbox} に非有限値が含まれています。")
            if w <= 0 or h <= 0:
                raise ValueError(f"bbox {bbox} の幅・高さは正でなければなりません。")
            if validate_bounds:
                img = images[raw["image_id"]]
                if x < 0 or y < 0 or x + w > img["width"] or y + h > img["height"]:
                    raise ValueError(
                        f"bbox {bbox} が画像 ({img['width']}x{img['height']}) の外へはみ出しています。"
                    )

            ann_session_id = raw.get("session_id", "")
            img_session_id = images[raw["image_id"]].get("session_id", "")

            # annotation.session_id と image.session_id の一致検証（両方に値がある場合のみ）
            if ann_session_id and img_session_id and ann_session_id != img_session_id:
                raise DatasetPreflightError(
                    f"annotation.session_id={ann_session_id!r} と"
                    f" image.session_id={img_session_id!r} が一致しません"
                    f" (image_id={raw['image_id']})。"
                )

            # annotation レベルの拒否チェック
            check_session = ann_session_id or img_session_id
            if check_session and check_session in self._rejected_sessions:
                raise DatasetPreflightError(
                    f"拒否セッション '{check_session}' のフレーム (image_id={raw['image_id']}) が"
                    " dataset に含まれています。error_calibration / final_e2e_test を分離してください。"
                )
            ann = COCOAnnotation(
                annotation_id=raw.get("id", 0),
                image_id=raw["image_id"],
                category_id=cat_id,
                bbox_xywh=bbox,  # type: ignore[arg-type]
                session_id=ann_session_id,
            )
            if ann.image_id in ann_by_image:
                ann_by_image[ann.image_id].append(ann)

        ann_dir = path.parent
        samples_raw = [
            (iid, img)
            for iid, img in sorted(images.items())
        ]

        # 画像レベルの rejected_sessions チェック（annotation なし負例も対象）
        for iid, img in samples_raw:
            img_session_id = img.get("session_id", "")
            if img_session_id and img_session_id in self._rejected_sessions:
                raise DatasetPreflightError(
                    f"拒否セッション '{img_session_id}' の画像 (image_id={iid}) が"
                    " dataset に含まれています（annotation なし負例）。"
                    " error_calibration / final_e2e_test を分離してください。"
                )

        samples = [
            DatasetSample(
                image_id=iid,
                file_name=str(
                    ann_dir / img["file_name"]
                    if not pathlib.Path(img["file_name"]).is_absolute()
                    else img["file_name"]
                ),
                image_width=img["width"],
                image_height=img["height"],
                annotations=ann_by_image.get(iid, []),
                session_id=img.get("session_id", ""),
            )
            for iid, img in samples_raw
        ]
        return samples

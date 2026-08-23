"""WorldDetector — 本家 Survivors 画面のエンティティ検出器。

torchvision の SSDLite320_MobileNet_V3_Large を骨格とし、head の出力クラス数を
12（background 0 + foreground 11）へ置換する。
feasibility config で列挙されている architecture 以外を指定した場合は
silent fallback せずに UnknownArchitectureError を送出する。

development checkpoint は formal_detector_eligible=false を持ち、
formal loader（04-10）がロード時に拒否する。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

_SHA256_RE = re.compile(r"[0-9a-f]{64}")

try:
    import torch
    import torchvision
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover — CI 環境で torch が入っていない場合
    _TORCH_AVAILABLE = False


# ---- 例外 ----

class FormalDetectorRejectedError(ValueError):
    """development checkpoint を formal loader がロードしようとしたときに送出される。"""


class UnknownArchitectureError(ValueError):
    """feasibility config に存在しない architecture を指定したときに送出される。"""


# ---- 既知 architecture リスト（perception_feasibility_v1.yaml と一致） ----

_KNOWN_ARCHITECTURES = frozenset(["ssdlite320", "ssdlite640_multiscale", "tile_2x2", "coarse_density"])


# ---- config ----

def load_detector_config(path: pathlib.Path | str) -> dict[str, Any]:
    """YAML から detector config を読み込む。

    schema_version が "world_detector.v1" でなければ ValueError。
    """
    path = pathlib.Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data.get("schema_version") != "world_detector.v1":
        raise ValueError(f"未知の schema_version: {data.get('schema_version')}")
    return data


# ---- detection result ----

@dataclass
class DetectionResult:
    """1 フレームの検出結果。viewport 正規化座標を提供する。

    boxes_xyxy は画像ピクセル座標 [x1, y1, x2, y2] の shape (N, 4) 配列。
    scores と class_ids は shape (N,)。
    """

    boxes_xyxy: np.ndarray  # float32 (N, 4)
    scores: np.ndarray      # float32 (N,)
    class_ids: np.ndarray   # int32 (N,)
    image_width: int
    image_height: int

    def __len__(self) -> int:
        return len(self.scores)

    @property
    def normalized_centers(self) -> np.ndarray:
        """各 box の中心を [0,1] に正規化した shape (N, 2) 配列を返す。"""
        if len(self) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        cx = (self.boxes_xyxy[:, 0] + self.boxes_xyxy[:, 2]) / 2.0 / self.image_width
        cy = (self.boxes_xyxy[:, 1] + self.boxes_xyxy[:, 3]) / 2.0 / self.image_height
        return np.stack([cx, cy], axis=1).astype(np.float32)


# ---- checkpoint manifest ----

@dataclass
class CheckpointManifest:
    """model / data / config / build / class-map の SHA-256 ハッシュを保持する manifest。

    formal_detector_eligible=false の manifest は assert_formal_eligible() で拒否される。
    04-07 / 04-08 が weight と threshold を差し替えて formal=true に更新する。
    """

    model_hash: str
    data_hash: str
    config_hash: str
    build_hash: str
    class_map_hash: str
    formal_detector_eligible: bool
    split_hash: str = ""           # split JSON の SHA-256。空文字は旧 manifest との互換用。
    resolved_config_hash: str = "" # CLI override 適用後の config dict SHA-256。

    _HASH_FIELDS = ("model_hash", "data_hash", "config_hash", "build_hash", "class_map_hash")

    def assert_formal_eligible(self) -> None:
        """formal 昇格していない checkpoint をロードしようとすると拒否する。

        formal_detector_eligible が bool の True でない場合（型不正・False 含む）は拒否する。
        SHA-256 形式不正も拒否する。
        """
        if not (type(self.formal_detector_eligible) is bool and self.formal_detector_eligible is True):
            raise FormalDetectorRejectedError(
                "development checkpoint は formal detector として使用できません。"
                " formal_detector_eligible=true の checkpoint を 04-08 で作成してください。"
            )
        # formal 昇格済みでもハッシュ形式を検証する
        self._validate_hash_format()

    def _validate_hash_format(self) -> None:
        for fname in self._HASH_FIELDS:
            h = getattr(self, fname)
            if not _SHA256_RE.fullmatch(h):
                raise ValueError(
                    f"{fname} は SHA-256 hex 64 文字が必要です: {h!r}"
                )

    def verify_identity(self, expected: dict[str, str]) -> None:
        """期待ハッシュと manifest の値が完全一致しない場合に ValueError を送出する。

        expected のキーは _HASH_FIELDS の部分集合。
        全フィールドを渡して identity を完全に確認することを推奨する。
        """
        for field, exp in expected.items():
            if field not in self._HASH_FIELDS:
                raise ValueError(f"未知のフィールド: {field!r}")
            actual = getattr(self, field)
            if actual != exp:
                raise ValueError(
                    f"identity 不一致: {field}: manifest={actual!r}, expected={exp!r}"
                )

    def save(self, path: pathlib.Path | str) -> None:
        """JSON として保存する。"""
        path = pathlib.Path(path)
        payload = {
            "model_hash": self.model_hash,
            "data_hash": self.data_hash,
            "config_hash": self.config_hash,
            "build_hash": self.build_hash,
            "class_map_hash": self.class_map_hash,
            "formal_detector_eligible": self.formal_detector_eligible,
            "split_hash": self.split_hash,
            "resolved_config_hash": self.resolved_config_hash,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "CheckpointManifest":
        """JSON から読み込む。ハッシュ形式・型が不正な場合は ValueError。"""
        path = pathlib.Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        elig = payload["formal_detector_eligible"]
        if not isinstance(elig, bool):
            raise ValueError(
                f"formal_detector_eligible は bool 型が必要です。got: {type(elig).__name__!r}"
            )
        manifest = cls(
            model_hash=payload["model_hash"],
            data_hash=payload["data_hash"],
            config_hash=payload["config_hash"],
            build_hash=payload["build_hash"],
            class_map_hash=payload["class_map_hash"],
            formal_detector_eligible=elig,
            split_hash=payload.get("split_hash", ""),
            resolved_config_hash=payload.get("resolved_config_hash", ""),
        )
        manifest._validate_hash_format()
        return manifest


# ---- detector ----

class WorldDetector:
    """SSDLite320_MobileNet_V3_Large ベースの world entity 検出器。

    head の num_classes を 12 へ置換し、foreground 11 クラスを検出する。
    weight なしでも from_config() でインスタンスを生成でき、infer() を呼べる（推論精度は無意味）。
    """

    def __init__(self, model: Any, num_classes: int, image_width: int, image_height: int) -> None:
        self._model = model
        self._num_classes = num_classes
        self._image_width = image_width
        self._image_height = image_height

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def head_num_classes(self) -> int:
        """model head の出力クラス数（num_classes と一致する）。"""
        return self._num_classes

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        class_map_path: pathlib.Path | str,
    ) -> "WorldDetector":
        """config dict から detector を組み立てる。

        ssdlite320 のみ実装済み。それ以外は（feasibility 既知候補でも）UnknownArchitectureError。
        class map の num_classes と config の num_classes が一致しない場合も拒否する。
        """
        from survivors.vision.world_dataset import load_class_map

        arch = config["model"]["architecture"]
        num_classes = config["model"]["num_classes"]

        # class map との一致を検証する
        cm = load_class_map(pathlib.Path(class_map_path))
        if cm.num_classes != num_classes:
            raise ValueError(
                f"config num_classes={num_classes} が class_map num_classes={cm.num_classes} と不一致。"
            )

        # 明示的な architecture dispatch。未実装は silent fallback せず拒否する。
        if arch == "ssdlite320":
            w, h = config["model"].get("input_size", [320, 320])
            model = _build_ssdlite320(num_classes=num_classes)
            return cls(model=model, num_classes=num_classes, image_width=w, image_height=h)

        # feasibility 既知の候補も含め、実装されていなければ拒否する
        raise UnknownArchitectureError(
            f"architecture '{arch}' は未実装です。silent fallback は禁止されています。"
            f" 実装済み: ['ssdlite320']"
            + (
                f" (feasibility 既知だが未実装: {sorted(_KNOWN_ARCHITECTURES - {'ssdlite320'})})"
                if arch in _KNOWN_ARCHITECTURES else ""
            )
        )

    def infer(
        self,
        frame_bgr: np.ndarray,
        *,
        score_threshold: float = 0.5,
    ) -> DetectionResult:
        """BGR uint8 画像から DetectionResult を返す。

        weight なしでも呼べるが検出精度は無意味。
        入力は (H, W, 3) uint8 のみ受け付ける。
        """
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                f"frame_bgr は (H, W, 3) uint8 が必要です。shape={frame_bgr.shape}"
            )
        h, w = frame_bgr.shape[:2]

        if not _TORCH_AVAILABLE:  # pragma: no cover
            return DetectionResult(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros(0, dtype=np.float32),
                class_ids=np.zeros(0, dtype=np.int32),
                image_width=w,
                image_height=h,
            )

        import torch
        import torchvision.transforms.functional as TF

        self._model.eval()
        rgb = frame_bgr[..., ::-1].copy()
        tensor = TF.to_tensor(rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self._model(tensor)

        pred = outputs[0]
        boxes = pred["boxes"].cpu().numpy()  # (N, 4)
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()

        mask = scores >= score_threshold
        return DetectionResult(
            boxes_xyxy=boxes[mask].astype(np.float32),
            scores=scores[mask].astype(np.float32),
            class_ids=labels[mask].astype(np.int32),
            image_width=w,
            image_height=h,
        )


# ---- internal builder ----

def _build_ssdlite320(num_classes: int) -> Any:
    """SSDLite320 を num_classes=12 で構築する。

    torch が利用できない環境では stub を返す。
    """
    if not _TORCH_AVAILABLE:
        return _StubModel()

    from torchvision.models.detection import ssdlite320_mobilenet_v3_large

    # pretrained=False でランダム初期化。04-08 が formal weight をロードする。
    # ponytail: torchvision の num_classes パラメータで head を直接置換する。
    model = ssdlite320_mobilenet_v3_large(
        num_classes=num_classes,
        weights=None,
        weights_backbone=None,
    )
    return model


class _StubModel:
    """torch 不在環境用のスタブ（テスト用途のみ）。"""

    def eval(self) -> "_StubModel":
        return self

    def __call__(self, x: Any) -> list[dict[str, Any]]:
        import numpy as np
        return [{"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, dtype=int)}]

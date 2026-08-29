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
import math
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import yaml

_SHA256_RE = re.compile(r"[0-9a-f]{64}")

# 04-06 world_class_map_v1.yaml の foreground クラス名（background 除く）
_KNOWN_CLASS_NAMES = frozenset([
    "player_anchor", "enemy_normal", "enemy_elite", "enemy_boss",
    "gem_blue", "gem_green", "gem_red",
    "pickup_heal", "pickup_special",
    "hazard_projectile", "hazard_area",
])

_ALLOWED_TOP_LEVEL_KEYS = frozenset([
    "schema_version", "formal_detector_eligible",
    "model", "input", "training", "checkpoint_selection",
    "tracker", "dev_diagnostics",
])

try:
    import torch
    import torchvision
    _TORCH_AVAILABLE = True
except ModuleNotFoundError as e:  # pragma: no cover — CI 環境で torch が入っていない場合
    if e.name not in {"torch", "torchvision"}:
        raise
    _TORCH_AVAILABLE = False


# ---- 例外 ----

class FormalDetectorRejectedError(ValueError):
    """development checkpoint を formal loader がロードしようとしたときに送出される。"""


class UnknownArchitectureError(ValueError):
    """feasibility config に存在しない architecture を指定したときに送出される。"""


# ---- 既知 architecture リスト（perception_feasibility_v1.yaml と一致） ----

_KNOWN_ARCHITECTURES = frozenset(["ssdlite320", "ssdlite640_multiscale", "tile_2x2", "coarse_density"])


# ---- shared config validator ----

def _vnum(
    val: Any,
    name: str,
    lo: float | None = None,
    hi: float | None = None,
    require_int: bool = False,
) -> None:
    """有限数値バリデーション。bool拒否・型チェック・有限性・範囲チェック (inclusive)。

    validate_detector_config 内から呼ぶ内部ヘルパー。
    NaN / Infinity / bool / 文字列 / None を拒否する。
    """
    if isinstance(val, bool):
        raise ValueError(
            f"config '{name}': bool は不可。明示的な数値を指定してください。got: {val!r}"
        )
    if require_int:
        if not isinstance(val, int):
            raise ValueError(f"config '{name}': int が必要。got: {type(val).__name__!r}")
    else:
        if not isinstance(val, (int, float)):
            raise ValueError(f"config '{name}': 数値が必要。got: {type(val).__name__!r}")
    if not math.isfinite(float(val)):
        raise ValueError(f"config '{name}': NaN / Infinity は不可。got: {val!r}")
    if lo is not None and val < lo:
        raise ValueError(f"config '{name}': {val} < {lo} (下限違反)。")
    if hi is not None and val > hi:
        raise ValueError(f"config '{name}': {val} > {hi} (上限違反)。")


def validate_detector_config(config: Mapping[str, Any]) -> None:
    """world_detector.v1 config の共有バリデーション境界。

    全 public 入口から必ず呼ぶ。未知キー・型違い・非有限値・未実装値を
    model / tracker 構築・推論・publish より前に拒否する。
    validation 規則はここだけに定義し、caller へ複製しない。
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"config は mapping (dict) が必要です。got: {type(config).__name__!r}"
        )

    # --- top-level keys ---
    unknown = set(config.keys()) - _ALLOWED_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"config に未知の top-level key があります: {sorted(unknown)}")
    missing = _ALLOWED_TOP_LEVEL_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"config に必須 key が欠落しています: {sorted(missing)}")

    if config["schema_version"] != "world_detector.v1":
        raise ValueError(
            f"schema_version は 'world_detector.v1' が必要。got: {config['schema_version']!r}"
        )

    fde = config["formal_detector_eligible"]
    if not isinstance(fde, bool) or fde is not False:
        raise ValueError(
            f"formal_detector_eligible は bool の false のみ許可。"
            f" int 1 や文字列 'false' は不可。got: {fde!r}"
        )

    # --- model ---
    model = config["model"]
    if not isinstance(model, dict):
        raise ValueError("config 'model' は dict が必要。")
    _allowed_model = frozenset(["architecture", "backbone", "num_classes", "input_size", "pretrained_backbone"])
    unknown_model = set(model.keys()) - _allowed_model
    if unknown_model:
        raise ValueError(f"model に未知の key があります: {sorted(unknown_model)}")

    if model.get("architecture") != "ssdlite320":
        raise UnknownArchitectureError(
            f"model.architecture は 'ssdlite320' のみ実装済み。got: {model.get('architecture')!r}"
        )
    if model.get("backbone") != "mobilenet_v3_large":
        raise ValueError(
            f"model.backbone は 'mobilenet_v3_large' のみ実装済み。got: {model.get('backbone')!r}"
        )
    ptb = model.get("pretrained_backbone")
    if not isinstance(ptb, bool) or ptb is not False:
        raise ValueError(
            f"model.pretrained_backbone は bool の false のみ許可。"
            f" int 1 や文字列 'false' は不可。got: {ptb!r}"
        )
    nc = model.get("num_classes")
    if isinstance(nc, bool) or not isinstance(nc, int) or nc != 12:
        raise ValueError(f"model.num_classes は int 12 のみ。got: {nc!r}")
    inp_size = model.get("input_size")
    if not isinstance(inp_size, list) or len(inp_size) != 2:
        raise ValueError(f"model.input_size は 2 要素 list が必要。got: {inp_size!r}")
    for _i, _v in enumerate(inp_size):
        if isinstance(_v, bool) or not isinstance(_v, int):
            raise ValueError(f"model.input_size[{_i}] は int が必要。got: {_v!r}")
    if inp_size != [320, 320]:
        raise ValueError(f"model.input_size は [320, 320] のみ実装済み。got: {inp_size!r}")

    # --- input ---
    inp = config["input"]
    if not isinstance(inp, dict):
        raise ValueError("config 'input' は dict が必要。")
    unknown_input = set(inp.keys()) - frozenset(["normalize"])
    if unknown_input:
        raise ValueError(
            f"input に未知/未実装の key があります (augmentation 等は未実装): {sorted(unknown_input)}"
        )
    normalize = inp.get("normalize")
    if not isinstance(normalize, dict):
        raise ValueError("input.normalize は dict が必要。")
    unknown_norm = set(normalize.keys()) - frozenset(["mean", "std"])
    if unknown_norm:
        raise ValueError(f"input.normalize に未知の key があります: {sorted(unknown_norm)}")
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]
    if normalize.get("mean") != _IMAGENET_MEAN:
        raise ValueError(
            f"input.normalize.mean は {_IMAGENET_MEAN} のみ。got: {normalize.get('mean')!r}"
        )
    if normalize.get("std") != _IMAGENET_STD:
        raise ValueError(
            f"input.normalize.std は {_IMAGENET_STD} のみ。got: {normalize.get('std')!r}"
        )

    # --- training ---
    training = config["training"]
    if not isinstance(training, dict):
        raise ValueError("config 'training' は dict が必要。")
    _allowed_training = frozenset(["batch_size", "optimizer", "max_epochs", "preflight"])
    unknown_training = set(training.keys()) - _allowed_training
    if unknown_training:
        raise ValueError(
            f"training に未知/未実装の key があります"
            f" (lr_scheduler, class_weights, near_duplicate_suppression は未実装):"
            f" {sorted(unknown_training)}"
        )

    _vnum(training.get("batch_size"), "training.batch_size", lo=3, require_int=True)
    _vnum(training.get("max_epochs"), "training.max_epochs", lo=1, require_int=True)

    opt = training.get("optimizer")
    if not isinstance(opt, dict):
        raise ValueError(f"training.optimizer は dict が必要。got: {type(opt).__name__!r}")
    if set(opt.keys()) != {"sgd"}:
        raise ValueError(
            f"training.optimizer には 'sgd' キーのみ許可 (adam 等は未実装)。"
            f" got: {sorted(opt.keys())!r}"
        )
    sgd = opt["sgd"]
    if not isinstance(sgd, dict):
        raise ValueError(f"training.optimizer.sgd は dict が必要。got: {type(sgd).__name__!r}")
    unknown_sgd = set(sgd.keys()) - frozenset(["lr", "momentum", "weight_decay"])
    if unknown_sgd:
        raise ValueError(f"training.optimizer.sgd に未知の key があります: {sorted(unknown_sgd)}")
    lr = sgd.get("lr")
    _vnum(lr, "training.optimizer.sgd.lr")
    if lr <= 0:
        raise ValueError(f"training.optimizer.sgd.lr は 0 より大きい値が必要。got: {lr!r}")
    momentum = sgd.get("momentum")
    _vnum(momentum, "training.optimizer.sgd.momentum", lo=0)
    if momentum >= 1:
        raise ValueError(
            f"training.optimizer.sgd.momentum は 0 以上 1 未満が必要。got: {momentum!r}"
        )
    _vnum(sgd.get("weight_decay"), "training.optimizer.sgd.weight_decay", lo=0)

    preflight = training.get("preflight", {})
    if not isinstance(preflight, dict):
        raise ValueError("training.preflight は dict が必要。")
    _allowed_preflight = frozenset([
        "min_frames", "min_entities", "min_classes", "min_time_bands",
        "min_frames_per_split", "min_entities_per_split", "min_classes_per_split",
    ])
    unknown_preflight = set(preflight.keys()) - _allowed_preflight
    if unknown_preflight:
        raise ValueError(f"training.preflight に未知の key があります: {sorted(unknown_preflight)}")
    missing_preflight = _allowed_preflight - set(preflight.keys())
    if missing_preflight:
        raise ValueError(f"training.preflight に必須 key がありません: {sorted(missing_preflight)}")
    for _pk in _allowed_preflight:
        _vnum(preflight[_pk], f"training.preflight.{_pk}", lo=1, require_int=True)

    # --- checkpoint_selection ---
    ckpt = config["checkpoint_selection"]
    if not isinstance(ckpt, dict):
        raise ValueError("config 'checkpoint_selection' は dict が必要。")
    unknown_ckpt = set(ckpt.keys()) - frozenset(["metric", "keep_top_k", "save_every_n_epochs"])
    if unknown_ckpt:
        raise ValueError(f"checkpoint_selection に未知の key があります: {sorted(unknown_ckpt)}")
    if ckpt.get("metric") != "val_map50_95":
        raise ValueError(
            f"checkpoint_selection.metric は 'val_map50_95' のみ実装済み。"
            f" got: {ckpt.get('metric')!r}"
        )
    _vnum(ckpt.get("keep_top_k"), "checkpoint_selection.keep_top_k", lo=1, require_int=True)
    _vnum(ckpt.get("save_every_n_epochs"), "checkpoint_selection.save_every_n_epochs", lo=1, require_int=True)

    # --- tracker ---
    tracker = config["tracker"]
    if not isinstance(tracker, dict):
        raise ValueError("config 'tracker' は dict が必要。")
    _allowed_tracker = frozenset([
        "max_age_by_class", "max_match_cost", "velocity_ema_alpha", "confidence_decay_per_frame"
    ])
    unknown_tracker = set(tracker.keys()) - _allowed_tracker
    if unknown_tracker:
        raise ValueError(f"tracker に未知の key があります: {sorted(unknown_tracker)}")
    max_age = tracker.get("max_age_by_class", {})
    if not isinstance(max_age, dict):
        raise ValueError("tracker.max_age_by_class は dict が必要。")
    for _cls_name, _age in max_age.items():
        if _cls_name not in _KNOWN_CLASS_NAMES:
            raise ValueError(
                f"tracker.max_age_by_class に未知の class 名があります: {_cls_name!r}"
                f" (04-06 class map 外)"
            )
        _vnum(_age, f"tracker.max_age_by_class.{_cls_name}", lo=0, require_int=True)
    _vnum(tracker.get("max_match_cost"), "tracker.max_match_cost", lo=0)
    _vnum(tracker.get("velocity_ema_alpha"), "tracker.velocity_ema_alpha", lo=0, hi=1)
    _vnum(tracker.get("confidence_decay_per_frame"), "tracker.confidence_decay_per_frame", lo=0, hi=1)

    # --- dev_diagnostics ---
    dd = config["dev_diagnostics"]
    if not isinstance(dd, dict):
        raise ValueError("config 'dev_diagnostics' は dict が必要。")
    _allowed_dd = frozenset([
        "map50_95_min", "class_recall_min", "density_correlation_min",
        "nearest_distance_median_max", "slice_gate",
    ])
    unknown_dd = set(dd.keys()) - _allowed_dd
    if unknown_dd:
        raise ValueError(f"dev_diagnostics に未知の key があります: {sorted(unknown_dd)}")
    _vnum(dd.get("map50_95_min"), "dev_diagnostics.map50_95_min", lo=0, hi=1)
    _vnum(dd.get("density_correlation_min"), "dev_diagnostics.density_correlation_min", lo=-1, hi=1)
    _vnum(dd.get("nearest_distance_median_max"), "dev_diagnostics.nearest_distance_median_max", lo=0, hi=1)
    class_recall_min = dd.get("class_recall_min", {})
    if not isinstance(class_recall_min, dict):
        raise ValueError("dev_diagnostics.class_recall_min は dict が必要。")
    for _cls_name, _threshold in class_recall_min.items():
        if _cls_name not in _KNOWN_CLASS_NAMES:
            raise ValueError(
                f"dev_diagnostics.class_recall_min に未知の class 名があります: {_cls_name!r}"
            )
        _vnum(_threshold, f"dev_diagnostics.class_recall_min.{_cls_name}", lo=0, hi=1)
    slice_gate = dd.get("slice_gate", {})
    if not isinstance(slice_gate, dict):
        raise ValueError("dev_diagnostics.slice_gate は dict が必要。")
    _allowed_slice = frozenset(["recall_min", "min_instances", "min_sessions"])
    for _sname, _scfg in slice_gate.items():
        if not isinstance(_scfg, dict):
            raise ValueError(f"dev_diagnostics.slice_gate.{_sname!r} は dict が必要。")
        unknown_slice = set(_scfg.keys()) - _allowed_slice
        if unknown_slice:
            raise ValueError(
                f"dev_diagnostics.slice_gate.{_sname!r} に未知の key があります: {sorted(unknown_slice)}"
            )
        _vnum(_scfg.get("recall_min"), f"dev_diagnostics.slice_gate.{_sname}.recall_min", lo=0, hi=1)
        _vnum(_scfg.get("min_instances"), f"dev_diagnostics.slice_gate.{_sname}.min_instances", lo=1, require_int=True)
        _vnum(_scfg.get("min_sessions"), f"dev_diagnostics.slice_gate.{_sname}.min_sessions", lo=1, require_int=True)


# ---- config ----

def load_detector_config(path: pathlib.Path | str) -> dict[str, Any]:
    """YAML から detector config を読み込み、共有 validator を通す。

    YAML root が mapping でなければ ValueError。
    schema_version / 型 / 値 / 未知 key のチェックはすべて validate_detector_config へ委譲する。
    """
    path = pathlib.Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"config YAML の root は mapping が必要です。got: {type(data).__name__!r}"
        )
    validate_detector_config(data)
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

    04-07 は development-only artifact のみ生成する。formal_detector_eligible は常に False。
    assert_formal_eligible() は caller 指定フラグに関わらず常に FormalDetectorRejectedError を送出する。
    formal 昇格は 04-08 の validated verdict によってのみ行われる。
    """

    model_hash: str
    data_hash: str
    config_hash: str
    build_hash: str
    class_map_hash: str
    formal_detector_eligible: bool
    split_hash: str = ""           # split JSON の SHA-256。空文字は旧 manifest との互換用。
    resolved_config_hash: str = "" # CLI override 適用後の config dict SHA-256。
    development_only: bool = True  # 常に True。04-08 以外で False にはならない。
    training_mode: str = "development"  # "smoke" | "development"

    _HASH_FIELDS = ("model_hash", "data_hash", "config_hash", "build_hash", "class_map_hash")

    @staticmethod
    def _check_optional_hash_field(fname: str, h: Any) -> None:
        """split_hash / resolved_config_hash を検証する共有ヘルパー。

        save() と load() の両方から呼ぶ。
        空文字（後方互換）または lowercase SHA-256 hex 64 文字のみ許可する。
        非文字列・大文字・不正長さはすべて拒否する。
        """
        if not isinstance(h, str):
            raise ValueError(
                f"CheckpointManifest: {fname} は str が必要。got: {type(h).__name__!r}"
            )
        if h and not _SHA256_RE.fullmatch(h):
            raise ValueError(
                f"CheckpointManifest: {fname} は空文字または lowercase SHA-256 hex 64 文字が必要。"
                f" got: {h!r}"
            )

    def assert_formal_eligible(self) -> None:
        """04-07 の checkpoint は caller 指定フラグに関わらず常に formal 拒否する。

        formal_detector_eligible の値に関係なく例外を送出する。
        formal 昇格は 04-08 の validated verdict によってのみ可能。
        """
        raise FormalDetectorRejectedError(
            "04-07 checkpoint は formal detector として使用できません。"
            " formal 昇格は 04-08 の検証済み verdict によってのみ行われます。"
        )

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
        """JSON として保存する。04-07 境界を書き込み時に強制する。

        formal_detector_eligible は False、development_only は True、
        training_mode は "smoke" | "development" のみ許容する。
        必須 hash は SHA-256 hex 64 文字であることを保証する。
        """
        if self.formal_detector_eligible is not False:
            raise ValueError("CheckpointManifest.save(): formal_detector_eligible は False でなければなりません。")
        if self.development_only is not True:
            raise ValueError("CheckpointManifest.save(): development_only は True でなければなりません。")
        if self.training_mode not in ("smoke", "development"):
            raise ValueError(
                f"CheckpointManifest.save(): training_mode の許容値は 'smoke' | 'development' です。"
                f" got: {self.training_mode!r}"
            )
        self._validate_hash_format()
        self._check_optional_hash_field("split_hash", self.split_hash)
        self._check_optional_hash_field("resolved_config_hash", self.resolved_config_hash)
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
            "development_only": self.development_only,
            "training_mode": self.training_mode,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: pathlib.Path | str) -> "CheckpointManifest":
        """JSON から読み込む。ハッシュ形式・型が不正な場合は ValueError。

        04-07 境界: formal_detector_eligible は必ず False へ上書き、development_only は True、
        training_mode は "smoke" | "development" のみ許容する。
        """
        path = pathlib.Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        elig = payload["formal_detector_eligible"]
        if not isinstance(elig, bool):
            raise ValueError(
                f"formal_detector_eligible は bool 型が必要です。got: {type(elig).__name__!r}"
            )
        raw_mode = payload.get("training_mode", "development")
        if raw_mode not in ("smoke", "development"):
            raise ValueError(
                f"training_mode の許容値は 'smoke' | 'development' です。got: {raw_mode!r}"
            )
        manifest = cls(
            model_hash=payload["model_hash"],
            data_hash=payload["data_hash"],
            config_hash=payload["config_hash"],
            build_hash=payload["build_hash"],
            class_map_hash=payload["class_map_hash"],
            formal_detector_eligible=False,   # 04-07 は常に False
            split_hash=payload.get("split_hash", ""),
            resolved_config_hash=payload.get("resolved_config_hash", ""),
            development_only=True,             # 04-07 は常に True
            training_mode=raw_mode,
        )
        manifest._validate_hash_format()
        # save() と同一ヘルパーで optional hash を対称に検証する
        manifest._check_optional_hash_field("split_hash", manifest.split_hash)
        manifest._check_optional_hash_field("resolved_config_hash", manifest.resolved_config_hash)
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

        caller が直接 dict を渡す経路を守るため、必ず共有 validator を呼ぶ。
        ssdlite320 のみ実装済み。それ以外は（feasibility 既知候補でも）UnknownArchitectureError。
        class map の num_classes と config の num_classes が一致しない場合も拒否する。
        """
        validate_detector_config(config)

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

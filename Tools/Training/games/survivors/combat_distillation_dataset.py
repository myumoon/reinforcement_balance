"""Deployable combat policy 用の固定長 sequence dataset 契約。
episode reset・burn-in・padding を明示 mask として保持し、教師の action logits と
value target を DeployObs schema hash へ束縛して NPZ/manifest 間で再現します。
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes, sha256_hex
from reinbalance_survivors_contracts.deploy_obs import (
    DeployObservation, DeployObsSchema, SOURCE_CLASSES,
)
DATASET_SCHEMA_VERSION = "survivors.combat_distillation_dataset.v1"
_ARRAY_NAMES = frozenset({
    "observations", "action_logits", "teacher_values", "valid_mask",
    "burn_in_mask", "episode_reset_mask",
})
_MANIFEST_KEYS = frozenset({
    "schema_version", "deploy_schema_hash", "burn_in_steps", "episode_ids", "splits",
    "observation_source_classes", "teacher_actions_present", "data_sha256",
})
def _sha256(value: Any, label: str) -> str:
    """小文字 SHA-256 identity を検証して返す。
    schema path や短縮 hash を dataset binding として受理しません。
    """
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value
@dataclass(frozen=True)
class CombatDistillationDataset:
    """同じ長さへ padding した combat episode sequence 群。
    mask は `[batch,time]`、観測/logits はそれぞれ末尾に feature/action 次元を持ちます。
    """
    observations: np.ndarray
    action_logits: np.ndarray
    teacher_values: np.ndarray
    valid_mask: np.ndarray
    burn_in_mask: np.ndarray
    episode_reset_mask: np.ndarray
    episode_ids: tuple[str, ...]
    splits: tuple[str, ...]
    deploy_schema_hash: str
    burn_in_steps: int
    observation_source_classes: tuple[str, ...]
    teacher_actions: np.ndarray | None = None
    def __post_init__(self) -> None:
        """全 shape・mask・padding・episode 境界を construction 時に検証する。
        保存経路以外の直接構築でも同じ sequence 不変条件を回避できないようにします。
        """
        _sha256(self.deploy_schema_hash, "deploy_schema_hash")
        if type(self.burn_in_steps) is not int or self.burn_in_steps < 0:
            raise ValueError("burn_in_steps must be a non-negative integer")
        arrays = {
            "observations": np.asarray(self.observations),
            "action_logits": np.asarray(self.action_logits),
            "teacher_values": np.asarray(self.teacher_values),
            "valid_mask": np.asarray(self.valid_mask),
            "burn_in_mask": np.asarray(self.burn_in_mask),
            "episode_reset_mask": np.asarray(self.episode_reset_mask),
        }
        observations, logits, values = arrays["observations"], arrays["action_logits"], arrays["teacher_values"]
        if observations.ndim != 3 or logits.ndim != 3 or values.ndim != 2:
            raise ValueError("dataset tensors must be batched sequences")
        shape = values.shape
        if shape[0] <= 0 or shape[1] <= 0 or observations.shape[:2] != shape or logits.shape[:2] != shape:
            raise ValueError("dataset sequence shapes mismatch")
        for name in ("valid_mask", "burn_in_mask", "episode_reset_mask"):
            if arrays[name].shape != shape or arrays[name].dtype != np.bool_:
                raise ValueError(f"{name} must be a bool sequence mask")
        if observations.dtype != np.float32 or logits.dtype != np.float32 or values.dtype != np.float32:
            raise ValueError("model tensors must be float32")
        if not all(np.all(np.isfinite(array)) for array in (observations, logits, values)):
            raise ValueError("model tensors must be finite")
        valid, burn, resets = arrays["valid_mask"], arrays["burn_in_mask"], arrays["episode_reset_mask"]
        for row in range(shape[0]):
            count = int(valid[row].sum())
            if count == 0 or not np.all(valid[row, :count]) or np.any(valid[row, count:]):
                raise ValueError("valid_mask padding must be a non-empty suffix")
            if not resets[row, 0] or np.any(resets[row] & ~valid[row]):
                raise ValueError("episode reset mask must mark valid boundaries")
            expected_burn = np.zeros(shape[1], dtype=np.bool_)
            for boundary in np.flatnonzero(resets[row]):
                expected_burn[boundary:min(boundary + self.burn_in_steps, count)] = True
            if not np.array_equal(burn[row], expected_burn):
                raise ValueError("burn_in_mask does not match reset boundaries")
        if np.any(logits[~valid] != 0) or np.any(values[~valid] != 0):
            raise ValueError("padding teacher targets must be zero")
        if len(self.episode_ids) != shape[0] or any(type(x) is not str or not x for x in self.episode_ids):
            raise ValueError("episode_ids must identify every sequence")
        if len(self.splits) != shape[0] or any(x not in {"train", "validation", "test"} for x in self.splits):
            raise ValueError("splits must identify every sequence")
        if not self.observation_source_classes or len(set(self.observation_source_classes)) != len(self.observation_source_classes):
            raise ValueError("observation source classes must be unique and non-empty")
        if any(source not in SOURCE_CLASSES for source in self.observation_source_classes):
            raise ValueError("unknown observation source class")
        if self.teacher_actions is not None:
            actions = np.asarray(self.teacher_actions)
            if actions.shape != shape or actions.dtype.kind not in "iu":
                raise ValueError("teacher_actions must be an integer sequence")
            object.__setattr__(self, "teacher_actions", np.array(actions, copy=True))
        for name, array in arrays.items():
            copied = np.array(array, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
    def assert_release_training_ready(self, schema: DeployObsSchema) -> None:
        """release optimizer が利用できる dataset だけを step 0 で許可する。
        unobservable source、hard teacher actions、train 外 split、別 schema/oracle tensor を対称に拒否します。
        """
        if not isinstance(schema, DeployObsSchema) or self.deploy_schema_hash != schema.schema_hash:
            raise ValueError("deploy schema hash mismatch")
        if "unobservable" in self.observation_source_classes:
            raise ValueError("unobservable privileged raw observations are forbidden")
        if self.teacher_actions is not None:
            raise ValueError("teacher actions are forbidden in release training")
        if any(split != "train" for split in self.splits):
            raise ValueError("split leakage: release training accepts train sequences only")
        if self.observations.shape[2] != schema.dim * 3:
            raise ValueError("DeployObs tensor dimension mismatch")
        for batch, time in zip(*np.nonzero(self.valid_mask), strict=True):
            tensor = self.observations[batch, time]
            DeployObservation(
                tensor[:schema.dim], tensor[schema.dim:schema.dim * 2], tensor[schema.dim * 2:],
                schema.schema_hash, 0, "release",
            ).validate_for(schema)
    def save(self, path: Path) -> None:
        """dataset を hash-bound manifest と pickle 無効 NPZ へ保存する。
        既存 directory への追記を拒否し、別世代の shard と manifest を混合しません。
        """
        root = Path(path)
        if root.exists():
            raise ValueError("dataset output already exists")
        root.mkdir(parents=True)
        payload = {name: getattr(self, name) for name in _ARRAY_NAMES}
        if self.teacher_actions is not None:
            payload["teacher_actions"] = self.teacher_actions
        np.savez_compressed(root / "data.npz", **payload)
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION, "deploy_schema_hash": self.deploy_schema_hash,
            "burn_in_steps": self.burn_in_steps, "episode_ids": list(self.episode_ids), "splits": list(self.splits),
            "observation_source_classes": list(self.observation_source_classes),
            "teacher_actions_present": self.teacher_actions is not None,
            "data_sha256": sha256_hex((root / "data.npz").read_bytes()),
        }
        (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    @classmethod
    def load(cls, path: Path) -> "CombatDistillationDataset":
        """exact-key manifest と hash 一致した NPZ だけを復元する。
        object array/pickle を許可せず、未知 array や欠落 array を schema drift として停止します。
        """
        root = Path(path)
        try:
            manifest = json.loads((root / "manifest.json").read_bytes())
            raw = (root / "data.npz").read_bytes()
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read combat dataset: {exc}") from exc
        if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS or manifest["schema_version"] != DATASET_SCHEMA_VERSION:
            raise ValueError("combat dataset manifest mismatch")
        if type(manifest["teacher_actions_present"]) is not bool:
            raise ValueError("teacher_actions_present must be bool")
        if sha256_hex(raw) != manifest["data_sha256"]:
            raise ValueError("combat dataset data hash mismatch")
        with np.load(root / "data.npz", allow_pickle=False) as archive:
            expected = set(_ARRAY_NAMES) | ({"teacher_actions"} if manifest["teacher_actions_present"] else set())
            if set(archive.files) != expected:
                raise ValueError("combat dataset arrays mismatch")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        return cls(
            **{name: arrays[name] for name in _ARRAY_NAMES}, episode_ids=tuple(manifest["episode_ids"]),
            splits=tuple(manifest["splits"]), deploy_schema_hash=manifest["deploy_schema_hash"],
            burn_in_steps=manifest["burn_in_steps"], observation_source_classes=tuple(manifest["observation_source_classes"]),
            teacher_actions=arrays.get("teacher_actions"),
        )
load_combat_distillation_dataset = CombatDistillationDataset.load

"""ItemSelector package の Deployment 専用 ONNX Runtime adapter。

Training package を import せず、`Tools/Common` の共有 validator だけで manifest と
file hash を検証してから ONNX Runtime session を開く。TorchScript / pickle を一切
読まないため、model 差し替えによる任意コード実行の経路を持たない。
検証済み artifact を先に確定してから推論器を構築する点が本 module の責務である。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import onnxruntime as ort

from reinbalance_survivors_contracts.item_selector_package import (
    ITEM_SELECTOR_PACKAGE_FILES,
    ItemSelectorPackageError,
    artifact_binding_payload,
    expected_onnx_tensor_manifest,
    load_verified_ui_policy_config,
    read_item_selector_manifest,
    validate_item_selector_manifest,
    verify_item_selector_package_files,
    verify_ui_policy_binding,
)
from reinbalance_survivors_contracts.ui_policy import NonModelUiPolicyConfigV1

# ONNX graph が公開する tensor 名。Common の expected_onnx_tensor_manifest と同一契約。
ONNX_INPUT_CONTEXT = "context_features"
ONNX_INPUT_CANDIDATES = "candidate_features"
ONNX_INPUT_MASK = "candidate_mask"
ONNX_OUTPUT_LOGITS = "logits"

# ONNX session は CPU 固定。live runtime は GPU 非依存で起動できる必要がある。
_ONNX_PROVIDERS = ("CPUExecutionProvider",)


class ItemSelectorRuntimeError(ValueError):
    """ItemSelector package の検証・推論失敗を表す fail-closed 例外。

    manifest 不一致、file hash 不一致、tensor 契約違反、非有限 logit のいずれでも
    送出する。呼び出し側はこれを stop / no_op decision へ変換すること。
    """


def _positive_finite(value: Any, label: str) -> float:
    """正で有限な実数だけを受理して float へ変換する。

    temperature 系の値が 0 や NaN だと softmax が壊れるため、推論前に弾く。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ItemSelectorRuntimeError(f"{label} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ItemSelectorRuntimeError(f"{label} must be positive and finite")
    return number


def _unit_interval(value: Any, label: str) -> float:
    """[0, 1] の有限な実数だけを受理して float へ変換する。

    confidence_threshold が範囲外だと gate が常時開く / 常時閉じるため、先に弾く。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ItemSelectorRuntimeError(f"{label} must be a real number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ItemSelectorRuntimeError(f"{label} must be within [0, 1]")
    return number


def _assert_onnx_io_contract(session: ort.InferenceSession, manifest: Mapping[str, Any]) -> None:
    """ONNX session の入出力名・rank を manifest の宣言と突き合わせる。

    別 model を同名 file で差し替えられた場合に、推論前へ検出するための境界検査。
    """
    expected_inputs, expected_outputs = expected_onnx_tensor_manifest(manifest)
    actual_inputs = {tensor.name: tensor for tensor in session.get_inputs()}
    actual_outputs = {tensor.name: tensor for tensor in session.get_outputs()}

    if set(actual_inputs) != {entry["name"] for entry in expected_inputs}:
        raise ItemSelectorRuntimeError("ItemSelector ONNX input names mismatch")
    if set(actual_outputs) != {entry["name"] for entry in expected_outputs}:
        raise ItemSelectorRuntimeError("ItemSelector ONNX output names mismatch")

    for entry in list(expected_inputs) + list(expected_outputs):
        tensor = actual_inputs.get(entry["name"]) or actual_outputs[entry["name"]]
        if len(tensor.shape) != len(entry["shape"]):
            raise ItemSelectorRuntimeError(
                f"ItemSelector ONNX tensor {entry['name']!r} rank mismatch"
            )
        for actual_dim, expected_dim in zip(tensor.shape, entry["shape"]):
            # 文字列次元は dynamic axis。整数次元だけを厳密に照合する。
            if isinstance(expected_dim, int) and isinstance(actual_dim, int):
                if actual_dim != expected_dim:
                    raise ItemSelectorRuntimeError(
                        f"ItemSelector ONNX tensor {entry['name']!r} shape mismatch"
                    )


class OnnxItemSelector:
    """検証済み ItemSelector package を ONNX Runtime で推論する Deployment adapter。

    package の manifest / file hash / UI policy binding を先に確定し、その後で
    ONNX session を開く。Training の ItemSelectorArtifact とは独立に動作し、
    共有検証ロジックは `Tools/Common` の item_selector_package から再利用する。
    """

    def __init__(
        self,
        *,
        package_dir: Path,
        manifest: Mapping[str, Any],
        session: ort.InferenceSession,
        vocabulary: frozenset[str],
        ui_policy_config: NonModelUiPolicyConfigV1,
    ) -> None:
        """検証済みの構成要素だけを受け取って adapter を組み立てる。

        直接 construct せず `load()` を使うこと。`load()` を通さない構築は
        hash 検証を飛ばすため、live 経路では使用しない。
        """
        self.package_dir = Path(package_dir)
        self.manifest: dict[str, Any] = dict(manifest)
        self._session = session
        self._vocabulary = vocabulary
        self._ui_policy_config = ui_policy_config
        self._nmax = int(self.manifest["nmax"])
        self._context_dim = int(self.manifest["context_dim"])
        self._candidate_dim = int(self.manifest["candidate_dim"])
        self._temperature = _positive_finite(self.manifest["temperature"], "temperature")
        self._student_output_temperature = _positive_finite(
            self.manifest["student_output_temperature"], "student_output_temperature"
        )
        self._confidence_threshold = _unit_interval(
            self.manifest["confidence_threshold"], "confidence_threshold"
        )

    @classmethod
    def load(cls, package_dir: Path) -> "OnnxItemSelector":
        """package を全面検証してから ONNX session を開く。

        manifest schema、宣言 file 集合、file content hash、UI policy binding を
        すべて通過した package だけを推論器にする。1 つでも欠ければ起動しない。
        """
        root = Path(package_dir)
        try:
            manifest = read_item_selector_manifest(root)
            vocabulary = validate_item_selector_manifest(manifest)
            verify_item_selector_package_files(root, manifest)
            verify_ui_policy_binding(manifest)
            ui_policy_config = load_verified_ui_policy_config(root, manifest)
        except ItemSelectorPackageError as exc:
            raise ItemSelectorRuntimeError(f"ItemSelector package rejected: {exc}") from exc

        # 宣言 file 以外が同居している package は差し替えの疑いがあるため拒否する。
        actual_files = {entry.name for entry in root.iterdir()}
        if actual_files != ITEM_SELECTOR_PACKAGE_FILES | {"manifest.json"}:
            raise ItemSelectorRuntimeError("ItemSelector package directory contents mismatch")

        onnx_path = root / "model.onnx"
        if onnx_path.is_symlink():
            raise ItemSelectorRuntimeError("ItemSelector model.onnx must not be a symlink")
        try:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            session = ort.InferenceSession(
                str(onnx_path), sess_options=options, providers=list(_ONNX_PROVIDERS)
            )
        except Exception as exc:  # noqa: BLE001  # ORT は多様な例外型を送出する
            raise ItemSelectorRuntimeError(f"cannot open ItemSelector ONNX model: {exc}") from exc

        _assert_onnx_io_contract(session, manifest)
        return cls(
            package_dir=root,
            manifest=manifest,
            session=session,
            vocabulary=vocabulary,
            ui_policy_config=ui_policy_config,
        )

    @property
    def nmax(self) -> int:
        """1 回の level-up で扱える候補スロット数の上限を返す。"""
        return self._nmax

    @property
    def context_dim(self) -> int:
        """context feature vector の次元を返す。"""
        return self._context_dim

    @property
    def candidate_dim(self) -> int:
        """candidate feature vector 1 件あたりの次元を返す。"""
        return self._candidate_dim

    @property
    def feature_schema(self) -> str:
        """package が期待する feature schema 識別子を返す。"""
        return str(self.manifest["feature_schema"])

    @property
    def vocabulary(self) -> frozenset[str]:
        """package が学習済みの item id 語彙を返す。"""
        return self._vocabulary

    @property
    def ui_policy_config(self) -> NonModelUiPolicyConfigV1:
        """package に封入された non-model UI policy config を返す。"""
        return self._ui_policy_config

    @property
    def temperature(self) -> float:
        """confidence 較正用の temperature を返す。

        argmax の raw logit ではなく、この温度で割った分布を確信度とする。
        """
        return self._temperature

    @property
    def student_output_temperature(self) -> float:
        """ONNX 出力 logit に適用する student 温度を返す。"""
        return self._student_output_temperature

    @property
    def confidence_threshold(self) -> float:
        """choose_card を許可する最低確信度を返す。

        これを下回る決定は card を選ばず、安全側 (no_op / stop) に倒す。
        """
        return self._confidence_threshold

    def predict(
        self,
        context_features: np.ndarray,
        candidate_features: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> np.ndarray:
        """候補ごとの student 温度適用済み logits を返す。

        shape / dtype / mask を fail-closed で検証し、非有限値を含む出力は拒否する。
        戻り値の shape は [batch, nmax]。
        """
        context = np.ascontiguousarray(np.asarray(context_features, dtype=np.float32))
        candidates = np.ascontiguousarray(np.asarray(candidate_features, dtype=np.float32))
        mask = np.ascontiguousarray(np.asarray(candidate_mask, dtype=bool))

        if context.ndim != 2 or context.shape[1] != self._context_dim:
            raise ItemSelectorRuntimeError("context_features shape mismatch")
        if candidates.ndim != 3 or candidates.shape[2] != self._candidate_dim:
            raise ItemSelectorRuntimeError("candidate_features shape mismatch")
        if mask.ndim != 2:
            raise ItemSelectorRuntimeError("candidate_mask must be rank 2")
        if not (context.shape[0] == candidates.shape[0] == mask.shape[0]):
            raise ItemSelectorRuntimeError("ItemSelector batch sizes disagree")
        if candidates.shape[1] != mask.shape[1]:
            raise ItemSelectorRuntimeError("candidate count and mask width disagree")
        if candidates.shape[1] > self._nmax:
            raise ItemSelectorRuntimeError("candidate count exceeds artifact Nmax")
        if not bool(mask.any(axis=1).all()):
            raise ItemSelectorRuntimeError("all-masked candidate row is not allowed")
        if not np.all(np.isfinite(context)) or not np.all(np.isfinite(candidates)):
            raise ItemSelectorRuntimeError("ItemSelector inputs must be finite")

        try:
            outputs = self._session.run(
                [ONNX_OUTPUT_LOGITS],
                {
                    ONNX_INPUT_CONTEXT: context,
                    ONNX_INPUT_CANDIDATES: candidates,
                    ONNX_INPUT_MASK: mask,
                },
            )
        except Exception as exc:  # noqa: BLE001  # ORT は多様な例外型を送出する
            raise ItemSelectorRuntimeError(f"ItemSelector ONNX inference failed: {exc}") from exc

        logits = np.asarray(outputs[0], dtype=np.float32)
        if logits.shape != (candidates.shape[0], candidates.shape[1]):
            raise ItemSelectorRuntimeError("ItemSelector ONNX output shape changed")
        scaled = logits / np.float32(self._student_output_temperature)
        if not np.all(np.isfinite(scaled[mask])):
            raise ItemSelectorRuntimeError("ItemSelector produced non-finite valid logits")
        return scaled

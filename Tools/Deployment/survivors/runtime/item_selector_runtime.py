"""ItemSelector package の Deployment 専用 ONNX Runtime adapter。

Training package を import せず、`Tools/Common` の共有 validator だけで manifest と
file hash を検証してから ONNX Runtime session を開く。TorchScript / pickle を一切
読まないため、model 差し替えによる任意コード実行の経路を持たない。
検証済み artifact を先に確定してから推論器を構築する点が本 module の責務である。

やさしい説明: アイテム選択AIの箱を先に隅々まで確認し、安全なONNXだけを読み込んで
候補ごとの点数を返す。学習専用ライブラリや実行可能なpickleは読み込まない。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import onnxruntime as ort

from reinbalance_survivors_contracts.canonical_json import sha256_hex
from reinbalance_survivors_contracts.item_selector_package import (
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
_ONNX_TENSOR_TYPES = {"float32": "tensor(float)", "bool": "tensor(bool)"}
_STRICT_ONNX_DIMENSIONS = frozenset(
    {(ONNX_INPUT_CONTEXT, 1), (ONNX_INPUT_CANDIDATES, 2)}
)


class ItemSelectorRuntimeError(ValueError):
    """ItemSelector package の検証・推論失敗を表す fail-closed 例外。

    やさしい説明: manifest 不一致、file hash 不一致、tensor 契約違反、非有限 logit
    のどれでも処理を止め、呼び出し側が安全な stop / no_op decision に変換できるようにする。
    """


def _positive_finite(value: Any, label: str) -> float:
    """正で有限な実数だけを受理して float へ変換する。

    やさしい説明: temperature 系の値が 0 や NaN だと softmax が壊れるため、推論前に弾く。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ItemSelectorRuntimeError(f"{label} must be a real number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ItemSelectorRuntimeError(f"{label} must be positive and finite") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ItemSelectorRuntimeError(f"{label} must be positive and finite")
    return number


def _unit_interval(value: Any, label: str) -> float:
    """[0, 1] の有限な実数だけを受理して float へ変換する。

    やさしい説明: confidence_threshold が範囲外だと gate が常時開く / 常時閉じるため、先に弾く。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ItemSelectorRuntimeError(f"{label} must be a real number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ItemSelectorRuntimeError(f"{label} must be within [0, 1]") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ItemSelectorRuntimeError(f"{label} must be within [0, 1]")
    return number


def _assert_onnx_io_contract(session: ort.InferenceSession, manifest: Mapping[str, Any]) -> None:
    """ONNX session の入出力名・rank を manifest の宣言と突き合わせる。

    やさしい説明: 同名の別モデルへ差し替えられても、入力名・出力名・形が違えば
    推論を始める前に止める。
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
        if tensor.type != _ONNX_TENSOR_TYPES[entry["dtype"]]:
            raise ItemSelectorRuntimeError(
                f"ItemSelector ONNX tensor {entry['name']!r} dtype mismatch"
            )
        if len(tensor.shape) != len(entry["shape"]):
            raise ItemSelectorRuntimeError(
                f"ItemSelector ONNX tensor {entry['name']!r} rank mismatch"
            )
        for axis, (actual_dim, expected_dim) in enumerate(
            zip(tensor.shape, entry["shape"])
        ):
            # 候補数の軸は Nmax 以下で可変だが、feature 幅は必ず manifest と一致させる。
            # やさしい説明: カード枚数は変えられても、カード一枚の項目数は変えられない。
            strict = (entry["name"], axis) in _STRICT_ONNX_DIMENSIONS
            if strict:
                if actual_dim != expected_dim:
                    raise ItemSelectorRuntimeError(
                        f"ItemSelector ONNX tensor {entry['name']!r} shape mismatch"
                    )
                continue
            # batch 軸・候補数軸は 1 件から Nmax 件まで任意の本数を受理できるよう、
            # ONNX 側が固定値ではなく動的軸として宣言していることを要求する。
            # やさしい説明: 決め打ちの人数・枚数でしか動かないモデルは起動前に弾く。
            if isinstance(actual_dim, int):
                raise ItemSelectorRuntimeError(
                    f"ItemSelector ONNX tensor {entry['name']!r} axis {axis} must be a dynamic dimension"
                )


class OnnxItemSelector:
    """検証済み ItemSelector package を ONNX Runtime で推論する Deployment adapter。

    やさしい説明: package の manifest / file hash / UI policy binding を先に確定し、その後で
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

        やさしい説明: 直接 construct せず `load()` を使うこと。`load()` を通さない構築は
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

        やさしい説明: manifest schema、宣言 file 集合、file content hash、UI policy binding を
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

        onnx_path = root / "model.onnx"
        try:
            if onnx_path.is_symlink():
                raise ItemSelectorRuntimeError("ItemSelector model.onnx must not be a symlink")
            onnx_bytes = onnx_path.read_bytes()
        except OSError as exc:
            raise ItemSelectorRuntimeError(f"cannot read ItemSelector ONNX model: {exc}") from exc
        if sha256_hex(onnx_bytes) != manifest["onnx_model_hash"]:
            raise ItemSelectorRuntimeError("ItemSelector ONNX model hash mismatch after verification")
        try:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            session = ort.InferenceSession(
                onnx_bytes, sess_options=options, providers=list(_ONNX_PROVIDERS)
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
        """1 回の level-up で扱える候補スロット数の上限を返す。

        やさしい説明: モデルが一度に比べられるカード枚数を知らせる。
        """
        return self._nmax

    @property
    def context_dim(self) -> int:
        """context feature vector の次元を返す。

        やさしい説明: ゲーム状況を表す入力値が何個必要かを知らせる。
        """
        return self._context_dim

    @property
    def candidate_dim(self) -> int:
        """candidate feature vector 1 件あたりの次元を返す。

        やさしい説明: カード一枚を表す入力値が何個必要かを知らせる。
        """
        return self._candidate_dim

    @property
    def feature_schema(self) -> str:
        """package が期待する feature schema 識別子を返す。

        やさしい説明: どの観測項目の並びで学習したモデルかを知らせる。
        """
        return str(self.manifest["feature_schema"])

    @property
    def vocabulary(self) -> frozenset[str]:
        """package が学習済みの item id 語彙を返す。

        やさしい説明: モデルが知っているアイテム名の集合を知らせる。
        """
        return self._vocabulary

    @property
    def ui_policy_config(self) -> NonModelUiPolicyConfigV1:
        """package に封入された non-model UI policy config を返す。

        やさしい説明: モデル外で使う安全なUIルール設定を呼び出し側へ渡す。
        """
        return self._ui_policy_config

    @property
    def temperature(self) -> float:
        """confidence 較正用の temperature を返す。

        やさしい説明: argmax の raw logit ではなく、この温度で割った分布を確信度とする。
        """
        return self._temperature

    @property
    def student_output_temperature(self) -> float:
        """ONNX 出力 logit に適用する student 温度を返す。

        やさしい説明: 候補の点数差を調整するときに使う学習済みの倍率を知らせる。
        """
        return self._student_output_temperature

    @property
    def confidence_threshold(self) -> float:
        """choose_card を許可する最低確信度を返す。

        やさしい説明: これを下回る決定は card を選ばず、安全側 (no_op / stop) に倒す。
        """
        return self._confidence_threshold

    def predict(
        self,
        context_features: np.ndarray,
        candidate_features: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> np.ndarray:
        """候補ごとの student 温度適用済み logits を返す。

        やさしい説明: shape / dtype / mask を fail-closed で検証し、非有限値を含む出力は拒否する。
        戻り値の shape は [batch, 入力候補数]。
        """
        context = np.asarray(context_features)
        candidates = np.asarray(candidate_features)
        mask = np.asarray(candidate_mask)
        if context.dtype != np.float32:
            raise ItemSelectorRuntimeError("context_features dtype mismatch")
        if candidates.dtype != np.float32:
            raise ItemSelectorRuntimeError("candidate_features dtype mismatch")
        if mask.dtype != np.bool_:
            raise ItemSelectorRuntimeError("candidate_mask dtype mismatch")
        context = np.ascontiguousarray(context)
        candidates = np.ascontiguousarray(candidates)
        mask = np.ascontiguousarray(mask)

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
        # ONNX graph が candidate_mask を無視しても、masked 位置は runtime 側で確実に選択不能にする。
        # やさしい説明: モデルの中身を信用せず、外側の安全な層でマスクを掛け直す。
        scaled = np.where(mask, scaled, np.float32(-np.inf))
        return scaled

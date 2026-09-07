"""ItemSelector package 契約と ONNX runtime loader を回帰検証する。

やさしい説明: 正常な箱の推論結果に加え、壊れた manifest・hash・ONNX・symlink と
不正な logits が必ず入口で拒否されることを確認する。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from conftest import ItemSelectorPackageFixture, _UNMASKED_SELECTOR_ONNX


def _runtime_types():
    """ItemSelector runtime の公開型を遅延 import する。

    やさしい説明: TDD の RED 実行では未実装 module が各テストの失敗理由になる。
    """
    from survivors.runtime.item_selector_runtime import (
        ItemSelectorRuntimeError,
        OnnxItemSelector,
    )

    return ItemSelectorRuntimeError, OnnxItemSelector


def test_loads_manifest_contract_and_predicts_declared_action_space(
    item_selector_package: ItemSelectorPackageFixture,
) -> None:
    """manifest の feature schema・Nmax・語彙と一致する推論を行う。

    やさしい説明: 箱に書かれた入力幅と候補数が、そのまま runtime の公開値と出力になる。
    """
    _, selector_type = _runtime_types()
    selector = selector_type.load(item_selector_package.root)

    assert selector.feature_schema == "context_only_v1"
    assert selector.nmax == 4
    assert selector.context_dim == 4
    assert selector.candidate_dim == 3
    assert selector.vocabulary == frozenset({"knife", "wand", "whip"})

    context = np.zeros((1, 4), dtype=np.float32)
    candidates = np.array(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [1.0, 1.0, 1.0]]],
        dtype=np.float32,
    )
    logits = selector.predict(context, candidates, np.ones((1, 4), dtype=bool))

    np.testing.assert_array_equal(logits, [[6.0, 15.0, 24.0, 3.0]])
    assert np.all(np.isfinite(logits))


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "wrong_type",
        "artifact_identity",
        "model.pt",
        "model.onnx",
        "ui_policy_config.json",
    ],
)
def test_rejects_invalid_manifest_schema_and_hash(
    item_selector_package: ItemSelectorPackageFixture, tmp_path: Path, case: str
) -> None:
    """必須 field 欠損・型違い・file hash 不一致を load 前に拒否する。

    やさしい説明: 箱の説明書や中身を一か所でも書き換えた package は使わない。
    """
    error_type, selector_type = _runtime_types()
    broken = item_selector_package.copy_to(tmp_path / case)
    if case == "missing":
        broken.manifest.pop("nmax")
        broken.write_manifest()
    elif case == "wrong_type":
        broken.manifest["nmax"] = "4"
        broken.write_manifest()
    elif case == "artifact_identity":
        broken.manifest["artifact_identity"] = "0" * 64
        broken.write_manifest()
    else:
        (broken.root / case).write_bytes(b"different")

    with pytest.raises(error_type):
        selector_type.load(broken.root)


def test_rejects_hash_valid_but_malformed_onnx(
    item_selector_package: ItemSelectorPackageFixture, tmp_path: Path
) -> None:
    """hash を更新した壊れた ONNX でも session 作成時に拒否する。

    やさしい説明: 説明書と同じ壊れたファイルを用意しても、モデルとして読めなければ止める。
    """
    error_type, selector_type = _runtime_types()
    broken = item_selector_package.copy_to(tmp_path / "malformed")
    broken.replace_onnx(b"not-an-onnx-model", update_hashes=True)

    with pytest.raises(error_type, match="cannot open ItemSelector ONNX model"):
        selector_type.load(broken.root)


@pytest.mark.parametrize(
    ("tensor_name", "tensor_type", "tensor_shape"),
    [
        ("context_features", "tensor(double)", ["batch", 4]),
        ("logits", "tensor(double)", ["batch", "candidates"]),
        ("context_features", "tensor(float)", ["batch", "context"]),
        (
            "candidate_features",
            "tensor(float)",
            ["batch", "candidates", "candidate_dim"],
        ),
    ],
)
def test_rejects_onnx_io_dtype_and_symbolic_fixed_dimensions(
    item_selector_package: ItemSelectorPackageFixture,
    tensor_name: str,
    tensor_type: str,
    tensor_shape: list[object],
) -> None:
    """ONNX metadata の dtype と固定次元が manifest と違えば拒否する。

    やさしい説明: 名前と rank だけを似せた別モデルでも、数値型や固定幅が違えば使わない。
    """
    from survivors.runtime.item_selector_runtime import (
        ItemSelectorRuntimeError,
        _assert_onnx_io_contract,
    )

    tensors = {
        "context_features": SimpleNamespace(
            name="context_features", type="tensor(float)", shape=["batch", 4]
        ),
        "candidate_features": SimpleNamespace(
            name="candidate_features",
            type="tensor(float)",
            shape=["batch", "candidates", 3],
        ),
        "candidate_mask": SimpleNamespace(
            name="candidate_mask", type="tensor(bool)", shape=["batch", "candidates"]
        ),
        "logits": SimpleNamespace(
            name="logits", type="tensor(float)", shape=["batch", "candidates"]
        ),
    }
    tensors[tensor_name] = SimpleNamespace(
        name=tensor_name,
        type=tensor_type,
        shape=tensor_shape,
    )
    session = Mock()
    session.get_inputs.return_value = [
        tensors["context_features"],
        tensors["candidate_features"],
        tensors["candidate_mask"],
    ]
    session.get_outputs.return_value = [tensors["logits"]]

    with pytest.raises(ItemSelectorRuntimeError, match="dtype mismatch|shape mismatch"):
        _assert_onnx_io_contract(session, item_selector_package.manifest)


def test_rejects_onnx_replaced_after_package_verification(
    item_selector_package: ItemSelectorPackageFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """hash 検証後に置換された ONNX bytes を session へ渡さない。

    やさしい説明: 点検直後に同じ名前の別モデルへ差し替えられても、そのモデルは実行しない。
    """
    from survivors.runtime import item_selector_runtime as runtime_module

    original_verify = runtime_module.verify_item_selector_package_files

    def replace_after_verify(package_dir: Path, manifest: dict) -> dict[str, str]:
        """正規のhash検証直後に別の有効ONNXへ置換する競合を再現する。

        やさしい説明: 実際の検証処理は残し、検証とsession作成の隙間だけを攻撃する。
        """
        hashes = original_verify(package_dir, manifest)
        (package_dir / "model.onnx").write_bytes(_UNMASKED_SELECTOR_ONNX)
        return hashes

    monkeypatch.setattr(
        runtime_module, "verify_item_selector_package_files", replace_after_verify
    )

    with pytest.raises(runtime_module.ItemSelectorRuntimeError, match="hash mismatch"):
        runtime_module.OnnxItemSelector.load(item_selector_package.root)


def test_candidate_mask_excludes_masked_choice(
    item_selector_package: ItemSelectorPackageFixture,
) -> None:
    """最大 score の候補でも mask=False なら選択対象から除外する。

    やさしい説明: 画面に無いカードが高得点でも、その番号を選ばないことを確かめる。
    """
    _, selector_type = _runtime_types()
    selector = selector_type.load(item_selector_package.root)
    candidates = np.array(
        [[[1.0, 0.0, 0.0], [100.0, 100.0, 100.0], [2.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    mask = np.array([[True, False, True]])

    logits = selector.predict(np.zeros((1, 4), dtype=np.float32), candidates, mask)

    assert logits[0, 1] < logits[0, 0]
    assert int(np.argmax(logits[0])) == 2


@pytest.mark.parametrize("input_name", ["context_features", "candidate_features", "candidate_mask"])
def test_predict_rejects_wrong_input_dtype(
    item_selector_package: ItemSelectorPackageFixture,
    input_name: str,
) -> None:
    """推論入力三種の dtype を暗黙変換せず fail-closed で拒否する。

    やさしい説明: 整数 feature や文字列の False を別の値として誤解せず、入力ミスを止める。
    """
    error_type, selector_type = _runtime_types()
    selector = selector_type.load(item_selector_package.root)
    context = np.zeros((1, 4), dtype=np.float32)
    candidates = np.zeros((1, 3, 3), dtype=np.float32)
    mask = np.ones((1, 3), dtype=bool)
    if input_name == "context_features":
        context = context.astype(np.float64)
    elif input_name == "candidate_features":
        candidates = candidates.astype(np.int32)
    else:
        mask = np.array([["True", "False", "True"]])

    with pytest.raises(error_type, match=f"{input_name} dtype mismatch"):
        selector.predict(context, candidates, mask)


@pytest.mark.parametrize(
    ("permutation", "expected"),
    [
        ([0, 1, 2], [1.0, 2.0, 3.0]),
        ([2, 0, 1], [3.0, 1.0, 2.0]),
        ([3, 1, 0, 2], [4.0, 2.0, 1.0, 3.0]),
        ([2, 0, 3, 1], [3.0, 1.0, 4.0, 2.0]),
    ],
)
def test_target_nmax_preserves_choice_permutation_indices(
    item_selector_package: ItemSelectorPackageFixture,
    permutation: list[int],
    expected: list[float],
) -> None:
    """target-derived Nmax 以内の候補で ONNX output index を表示順へ保つ。

    やさしい説明: 三枚・四枚のカードを並べ替えても、返る点数は同じ並び替えに追従する。
    """
    _, selector_type = _runtime_types()
    selector = selector_type.load(item_selector_package.root)
    base = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    candidates = base[permutation][None, :, :]

    logits = selector.predict(
        np.zeros((1, 4), dtype=np.float32),
        candidates,
        np.ones((1, len(permutation)), dtype=bool),
    )

    assert logits.shape == (1, len(permutation))
    np.testing.assert_array_equal(logits[0], expected)


def test_rejects_non_finite_valid_logits(
    item_selector_package: ItemSelectorPackageFixture,
) -> None:
    """有限入力の演算で overflow した有効候補 logits を拒否する。

    やさしい説明: モデルが Infinity を返して選択順を決められない場合は実行を止める。
    """
    error_type, selector_type = _runtime_types()
    selector = selector_type.load(item_selector_package.root)
    candidates = np.full((1, 3, 3), np.float32(3.0e38), dtype=np.float32)

    with pytest.raises(error_type, match="non-finite valid logits"):
        selector.predict(
            np.zeros((1, 4), dtype=np.float32),
            candidates,
            np.ones((1, 3), dtype=bool),
        )


@pytest.mark.parametrize(
    "name", ["manifest.json", "model.pt", "model.onnx", "ui_policy_config.json"]
)
def test_rejects_symlink_for_every_read_file(
    item_selector_package: ItemSelectorPackageFixture, tmp_path: Path, name: str
) -> None:
    """manifest と package 本体の全 read file で symlink を対称に拒否する。

    やさしい説明: 箱の外へ差し替えられる入口を、説明書を含む四ファイルすべてで塞ぐ。
    """
    error_type, selector_type = _runtime_types()
    broken = item_selector_package.copy_to(tmp_path / f"linked-{name.replace('.', '-')}")
    target = broken.root / name
    outside = tmp_path / "outside" / name
    outside.parent.mkdir(exist_ok=True)
    shutil.move(target, outside)
    target.symlink_to(outside)

    with pytest.raises(error_type):
        selector_type.load(broken.root)


def test_rejects_symlink_package_root(
    item_selector_package: ItemSelectorPackageFixture, tmp_path: Path
) -> None:
    """package root 自体が symlink の場合も読み込み前に拒否する。

    やさしい説明: 箱全体を別の場所へ向ける root escape も、個別ファイルと同じく塞ぐ。
    """
    error_type, selector_type = _runtime_types()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(item_selector_package.root, target_is_directory=True)

    with pytest.raises(error_type):
        selector_type.load(linked_root)

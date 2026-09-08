"""`survivors.runtime.artifact_bundle` の契約・trust anchor・hardware 照合テスト。

03-05 combat package の実契約（`manifest.json` + `model.pt`）に対する round-trip、
事前固定 trust anchor への exact identity 一致、実 hardware profile 値の照合、および
非検証 deserialization の不在を検証する。

やさしい説明:
    このテストが守っているのは 4 つです。(1) 学習側が実際に出力する形の成果物を、
    実行側が正しく読めること。(2) 呼び出し元が自作した成果物だけでは本番起動できない
    こと。(3) 動かすPCのスペックが想定と違えば起動しないこと。(4) 成果物の読み込みに
    任意コード実行の経路が無いこと。

    依存方向の規則により、このテストも Tools/Training を import しません。契約の形は
    `_runtime_fixtures` が独立に組み立てたものを使います。
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch as th

from reinbalance_survivors_contracts.canonical_json import canonical_json_bytes
from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.target_action import ActionSemantics

from survivors.runtime import artifact_bundle as ab
from survivors.runtime.artifact_bundle import (
    REQUIRED_ACTION_DIM,
    REQUIRED_DECISION_HZ,
    TRUST_REGISTRY_SIGNATURE_SUFFIX,
    BundleLoadError,
    CombatGruPolicy,
    CombatPolicy,
    HostRuntimeProfile,
    RuntimeBundle,
    TrustAnchor,
    TrustAnchorError,
    load_production_trust_anchor,
)

from . import _runtime_fixtures as fx
from .conftest import FormalBundleInputs


# --- round-trip: 03-05 実契約 --------------------------------------------------


def test_load_accepts_training_shaped_package(
    formal_inputs: FormalBundleInputs, pinned_production_trust_key: str
) -> None:
    """03-05 契約どおりの package と、本番鍵で署名された registry で live bundle が成立する。

    やさしい説明: 「全部正しいときはちゃんと起動する」ことを確かめる基準テストです。
    ここが通らないと、他の拒否テストが「そもそも何も起動できないだけ」になります。
    `pinned_production_trust_key` は「発行者の鍵はこれ」という状態を作る fixture で、
    これが無いと（=本番鍵未固定なら）このテストは起動しません。
    """
    bundle = RuntimeBundle.load(**formal_inputs.as_load_kwargs())

    assert bundle.live_eligible is True
    assert bundle.development_only is False
    bundle.assert_live_eligible()

    schema = DeployObsSchema.default_v1()
    assert bundle.combat_policy.observation_dim == 3 * schema.dim
    assert bundle.action_dim == REQUIRED_ACTION_DIM
    assert bundle.combat_policy.hidden_dim == fx.DEFAULT_HIDDEN_DIM
    assert isinstance(bundle.combat_policy.model, CombatGruPolicy)
    assert bundle.deploy_schema_hash == schema.schema_hash
    assert bundle.host_profile == formal_inputs.host_profile


def test_startup_report_carries_real_hardware_values(
    formal_inputs: FormalBundleInputs, pinned_production_trust_key: str
) -> None:
    """startup report に OS / GPU / driver / CUDA / capture backend の実値が載る。

    やさしい説明: 取り違えに人が気づけるよう、起動時の環境をそのまま記録します。
    """
    report = RuntimeBundle.load(**formal_inputs.as_load_kwargs()).startup_report
    expected = formal_inputs.host_profile

    assert report["bundle_kind"] == "formal"
    assert report["os_build"] == expected.os_build
    assert report["gpu_name"] == expected.gpu_name
    assert report["driver_version"] == expected.driver_version
    assert report["cuda_version"] == expected.cuda_version
    assert report["capture_backend"] == expected.capture_backend
    assert report["key_lease_duration_ms"] == expected.key_lease_duration_ms
    assert report["decision_hz"] == REQUIRED_DECISION_HZ
    assert report["combat_model_sha256"] == formal_inputs.combat_manifest["model_sha256"]
    # report 全体が canonical JSON 化できること（監査ログへそのまま書けること）。
    assert json.loads(canonical_json_bytes(report))["live_eligible"] is True


def test_loaded_model_reproduces_fixture_weights(
    formal_inputs: FormalBundleInputs, pinned_production_trust_key: str
) -> None:
    """独立に組んだ fixture model と、ロードされた model の出力が一致する。

    やさしい説明: 「学習側の形」と「実行側の形」を別々に書いたのに、重みがぴったり
    はまり、同じ入力から同じ答えが出ることを確かめます。これが契約一致の実証です。
    """
    bundle = RuntimeBundle.load(**formal_inputs.as_load_kwargs())
    reference, _ = fx.build_combat_model()

    observation = th.arange(
        bundle.combat_policy.observation_dim, dtype=th.float32
    ).unsqueeze(0) / 100.0
    hidden = bundle.combat_policy.initial_hidden_state()

    with th.no_grad():
        logits, value, new_hidden = bundle.combat_policy.model.step(observation, hidden)
        expected_hidden = reference.recurrent(observation, hidden)
        expected_logits = reference.actor(expected_hidden)
        expected_value = reference.value(expected_hidden).squeeze(-1)

    assert th.allclose(new_hidden, expected_hidden, atol=1e-6)
    assert th.allclose(logits, expected_logits, atol=1e-6)
    assert th.allclose(value, expected_value, atol=1e-6)
    assert logits.shape == (1, REQUIRED_ACTION_DIM)


def test_item_selector_round_trip_predicts_finite_logits(
    formal_inputs: FormalBundleInputs, pinned_production_trust_key: str
) -> None:
    """bundle 経由の ItemSelector が ONNX Runtime で有限 logits を返す。

    やさしい説明: アイテム選択側も、箱を検証したうえで実際に推論できることを見ます。
    """
    bundle = RuntimeBundle.load(**formal_inputs.as_load_kwargs())
    selector = bundle.item_selector
    assert selector is not None

    context = np.zeros((1, selector.context_dim), dtype=np.float32)
    candidates = np.ones((1, selector.nmax, selector.candidate_dim), dtype=np.float32)
    mask = np.ones((1, selector.nmax), dtype=bool)

    logits = selector.predict(context, candidates, mask)
    assert logits.shape == (1, selector.nmax)
    assert np.all(np.isfinite(logits))


# --- trust anchor -------------------------------------------------------------


def test_load_does_not_accept_a_caller_supplied_trust_anchor(
    formal_inputs: FormalBundleInputs, pinned_production_trust_key: str
) -> None:
    """`RuntimeBundle.load()` は trust anchor そのものを引数として受け取らない。

    やさしい説明: このテストが今回の修正の骨格です。以前は呼び出し元が `TrustAnchor`
    を直接渡せたため、自分で鍵を作り自分で「正規リリース一覧」を署名すれば、その場で
    作った成果物でも本番起動できてしまいました。許可証を自分で発行できるなら許可証の
    意味が無いので、引数そのものを廃止しました。ここでは (1) signature に
    `trust_anchor` が存在しないこと、(2) 代わりに「場所」だけを渡す
    `trust_registry_path` があること、(3) 昔の呼び方をしても TypeError で弾かれる
    ことを確かめます。
    """
    parameters = inspect.signature(RuntimeBundle.load).parameters
    assert "trust_anchor" not in parameters
    assert "trust_registry_path" in parameters

    caller_anchor = TrustAnchor.load(
        formal_inputs.registry_path, verification_public_keys=[fx.public_key_hex()]
    )
    with pytest.raises(TypeError):
        RuntimeBundle.load(
            **formal_inputs.as_load_kwargs(), trust_anchor=caller_anchor
        )


def test_self_signed_registry_cannot_reach_live_startup(
    formal_inputs: FormalBundleInputs,
) -> None:
    """整合した成果物一式でも、本番鍵が固定されていなければ live 起動しない。

    やさしい説明: `formal_inputs` の一式は hash も系譜も完全に整合していて、テスト用の
    鍵で正しく署名された registry も揃っています。それでも「その鍵を本番の発行者として
    認める」という宣言が source 側に無い限り、起動は必ず拒否されます。
    修正前のコードは、この自己署名 anchor をそのまま受け取って `live_eligible=True` を
    返していました。
    """
    assert ab.PRODUCTION_TRUST_ANCHOR_PUBLIC_KEYS == ()
    with pytest.raises(TrustAnchorError, match="no production trust anchor public key"):
        RuntimeBundle.load(**formal_inputs.as_load_kwargs())


def test_registry_signed_by_non_production_key_cannot_reach_live_startup(
    formal_inputs: FormalBundleInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """本番鍵とは別の鍵で署名された registry は、内容が整合していても起動に使えない。

    やさしい説明: 「鍵が 1 つも無いから止まる」ではなく「知らない鍵だから止まる」ことを
    確かめます。registry の中身（release entry）は正規のものと完全に同一で、hash も
    hardware profile もすべて一致します。違うのは署名した鍵だけです。攻撃者が自前の鍵で
    どれだけ整合した一覧を作っても、source 固定鍵で検証できない以上は通りません。
    """
    monkeypatch.setattr(
        ab, "PRODUCTION_TRUST_ANCHOR_PUBLIC_KEYS", (fx.public_key_hex(),)
    )
    entry = json.loads(formal_inputs.registry_path.read_bytes())["releases"][0]
    forged_registry = fx.write_trust_registry(
        tmp_path / "attacker" / "releases.json",
        [entry],
        seed=fx.UNTRUSTED_SIGNING_KEY_SEED,
    )
    with pytest.raises(TrustAnchorError, match="does not verify against any pinned key"):
        RuntimeBundle.load(
            **formal_inputs.as_load_kwargs(trust_registry_path=forged_registry)
        )


def test_load_without_registry_path_is_rejected(
    formal_inputs: FormalBundleInputs,
    pinned_production_trust_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """registry の所在が別チャネルで与えられていなければ live 起動できない。

    やさしい説明: 「一覧を見ないで起動する」という抜け道が無いことを確かめます。
    path 未指定かつ環境変数も未設定なら、鍵が固定されていても起動しません。
    """
    monkeypatch.delenv(ab.TRUST_REGISTRY_PATH_ENV, raising=False)
    with pytest.raises(TrustAnchorError, match="registry path is not set"):
        RuntimeBundle.load(**formal_inputs.as_load_kwargs(trust_registry_path=None))


def test_caller_produced_artifacts_are_never_live_eligible(
    formal_inputs: FormalBundleInputs,
    tmp_path: Path,
    pinned_production_trust_key: str,
) -> None:
    """caller が自作した package / descriptor / store の組では live_eligible にならない。

    やさしい説明: 本テストがこの PR の中心です。呼び出し元が自分で成果物一式を作り、
    保管庫へ登録し、系譜まで正しく組み立てても、その identity が「別チャネルで配布
    された署名済み一覧」に載っていなければ起動できない、という性質を確かめます。
    つまり、その場で作れるものだけを根拠に本番起動することはできません。
    """
    forged = tmp_path / "forged"
    # 呼び出し元が独自に用意した、それ自体は整合している成果物一式。
    capability = fx.hash_of("attacker-capability")
    choice_capability = fx.hash_of("attacker-choice-capability")
    selector_manifest = fx.write_item_selector_package(
        forged / "selector", target_capability_hash=capability
    )
    model, model_config = fx.build_combat_model()
    combat_manifest = fx.write_combat_package(forged / "combat", model, model_config)
    from reinbalance_survivors_contracts.artifact_store import ArtifactStore

    store = ArtifactStore(forged / "store")
    descriptors = fx.build_formal_descriptors(
        store=store,
        combat_manifest=combat_manifest,
        item_selector_manifest=selector_manifest,
        target_capability_hash=capability,
        choice_capability_hash=choice_capability,
    )

    # 信頼 root は正規のもの（caller は署名できないうえ、anchor 自体も渡せない）。
    with pytest.raises(BundleLoadError, match="not registered in trusted release registry"):
        RuntimeBundle.load(
            combat_package_dir=forged / "combat",
            item_selector_dir=forged / "selector",
            artifact_store=store,
            descriptors=descriptors,
            target_profile=formal_inputs.target_profile,
            host_profile=formal_inputs.host_profile,
            trust_registry_path=formal_inputs.registry_path,
        )


def test_registry_signed_by_untrusted_key_is_rejected(
    formal_inputs: FormalBundleInputs, tmp_path: Path
) -> None:
    """発行者以外の鍵で署名された registry は trust anchor として読めない。

    やさしい説明: 攻撃者が自分で一覧を作って署名しても、こちらが知っている鍵と違う
    ため検証に失敗します。
    """
    entry = json.loads(formal_inputs.registry_path.read_bytes())["releases"][0]
    forged_registry = fx.write_trust_registry(
        tmp_path / "forged" / "releases.json",
        [entry],
        seed=fx.UNTRUSTED_SIGNING_KEY_SEED,
    )
    with pytest.raises(TrustAnchorError, match="does not verify against any pinned key"):
        TrustAnchor.load(forged_registry, verification_public_keys=[fx.public_key_hex()])


def test_tampered_registry_is_rejected(
    formal_inputs: FormalBundleInputs, tmp_path: Path
) -> None:
    """署名後に内容を書き換えた registry は拒否される。"""
    entry = json.loads(formal_inputs.registry_path.read_bytes())["releases"][0]
    tampered = fx.write_trust_registry(
        tmp_path / "tampered" / "releases.json", [entry], tamper=True
    )
    with pytest.raises(TrustAnchorError, match="does not verify against any pinned key"):
        TrustAnchor.load(tampered, verification_public_keys=[fx.public_key_hex()])


def test_non_canonical_registry_bytes_are_rejected(
    formal_inputs: FormalBundleInputs, tmp_path: Path
) -> None:
    """署名は通るが canonical JSON でない registry は拒否される。

    やさしい説明: 署名した byte 列と、読み取った中身が 1 対 1 に対応しない状態を
    防ぎます。空白や並び順で見た目を変えられると、署名の意味が薄れるためです。
    """
    document = json.loads(formal_inputs.registry_path.read_bytes())
    payload = json.dumps(document, indent=2).encode("utf-8")  # canonical ではない
    path = tmp_path / "pretty" / "releases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.with_name(path.name + TRUST_REGISTRY_SIGNATURE_SUFFIX).write_text(
        fx.signing_key().sign(payload).hex(), encoding="utf-8"
    )
    with pytest.raises(TrustAnchorError, match="not canonical JSON"):
        TrustAnchor.load(path, verification_public_keys=[fx.public_key_hex()])


def test_trust_anchor_requires_verification_keys(
    formal_inputs: FormalBundleInputs,
) -> None:
    """検証鍵を渡さない trust anchor 読み込みは拒否される。"""
    with pytest.raises(TrustAnchorError, match="no trust anchor verification key"):
        TrustAnchor.load(formal_inputs.registry_path, verification_public_keys=[])


def test_production_trust_anchor_is_fail_closed(tmp_path: Path) -> None:
    """本番鍵が未固定の状態では production trust anchor を作れない。

    やさしい説明: 鍵をまだ配っていないのに本番起動できてしまうと危険なので、既定では
    必ず失敗します。テスト用の鍵が本番鍵に紛れ込んでいないことも併せて確認します。
    """
    assert ab.PRODUCTION_TRUST_ANCHOR_PUBLIC_KEYS == ()
    assert fx.public_key_hex() not in ab.PRODUCTION_TRUST_ANCHOR_PUBLIC_KEYS
    with pytest.raises(TrustAnchorError, match="no production trust anchor public key"):
        load_production_trust_anchor(tmp_path / "releases.json")


@pytest.mark.parametrize(
    "swapped_key",
    [
        "combat_student_release_identity_hash",
        "item_selector_release_identity_hash",
        "perception_final_verdict_identity_hash",
    ],
)
def test_registered_parent_identity_swap_is_rejected(
    formal_inputs: FormalBundleInputs,
    tmp_path: Path,
    swapped_key: str,
    pinned_production_trust_key: str,
) -> None:
    """registry が固定した parent identity と descriptor がずれれば起動しない。

    やさしい説明: runtime bundle だけが一覧に載っていればよい、とはしません。親に
    あたる 3 つの成果物も、一覧に書かれたものと同一であることを要求します。
    """
    entry = dict(json.loads(formal_inputs.registry_path.read_bytes())["releases"][0])
    entry[swapped_key] = fx.hash_of("some-other-artifact")
    registry = fx.write_trust_registry(tmp_path / "swap" / "releases.json", [entry])

    with pytest.raises(BundleLoadError, match="identity_hash mismatch"):
        RuntimeBundle.load(**formal_inputs.as_load_kwargs(trust_registry_path=registry))


def test_duplicate_release_entries_are_rejected(
    formal_inputs: FormalBundleInputs, tmp_path: Path
) -> None:
    """同一 runtime identity が 2 回登録された registry は曖昧なので拒否する。"""
    entry = json.loads(formal_inputs.registry_path.read_bytes())["releases"][0]
    registry = fx.write_trust_registry(tmp_path / "dup" / "releases.json", [entry, entry])
    with pytest.raises(TrustAnchorError, match="duplicate trusted release entry"):
        TrustAnchor.load(registry, verification_public_keys=[fx.public_key_hex()])


def test_unsupported_registry_schema_version_is_rejected(
    formal_inputs: FormalBundleInputs, tmp_path: Path
) -> None:
    """未知の schema_version を持つ registry は読み込まない。"""
    entry = json.loads(formal_inputs.registry_path.read_bytes())["releases"][0]
    registry = fx.write_trust_registry(
        tmp_path / "schema" / "releases.json",
        [entry],
        schema_version="survivors.trusted_release_registry.v99",
    )
    with pytest.raises(TrustAnchorError, match="unsupported trusted release registry"):
        TrustAnchor.load(registry, verification_public_keys=[fx.public_key_hex()])


# --- hardware / target profile 実値照合 ---------------------------------------


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    [
        ("os_build", "Windows 10 Pro 10.0.19045"),
        ("gpu_name", "NVIDIA GeForce RTX 3060"),
        ("driver_version", "552.22"),
        ("cuda_version", "12.1"),
        ("capture_backend", "obs-virtualcam"),
        ("key_lease_duration_ms", 500),
    ],
)
def test_host_profile_value_mismatch_blocks_startup(
    formal_inputs: FormalBundleInputs,
    field_name: str,
    wrong_value: Any,
    pinned_production_trust_key: str,
) -> None:
    """実 hardware 値が信頼済みリリースの期待値と 1 項目でも違えば起動しない。

    やさしい説明: 「値が入っているか」ではなく「同じ値か」を比べていることを確かめ
    ます。エラー文には、どの項目が、何を期待して、実際は何だったのかが出ます。
    """
    mismatched = fx.default_host_profile(**{field_name: wrong_value})

    with pytest.raises(BundleLoadError) as excinfo:
        RuntimeBundle.load(**formal_inputs.as_load_kwargs(host_profile=mismatched))

    message = str(excinfo.value)
    assert "host profile does not match" in message
    assert field_name in message
    assert repr(wrong_value) in message
    assert repr(getattr(formal_inputs.host_profile, field_name)) in message


def test_host_profile_mismatch_lists_every_differing_field(
    formal_inputs: FormalBundleInputs, pinned_production_trust_key: str
) -> None:
    """複数項目が違う場合は、そのすべてが報告される。"""
    mismatched = fx.default_host_profile(gpu_name="other-gpu", cuda_version="11.8")
    differences = mismatched.mismatches(formal_inputs.host_profile)
    assert {item.field_name for item in differences} == {"gpu_name", "cuda_version"}

    with pytest.raises(BundleLoadError) as excinfo:
        RuntimeBundle.load(**formal_inputs.as_load_kwargs(host_profile=mismatched))
    message = str(excinfo.value)
    assert "gpu_name" in message and "cuda_version" in message


def test_host_profile_rejects_wrong_decision_hz() -> None:
    """decision_hz は 15 Hz 固定であり、それ以外は構築時点で拒否される。"""
    with pytest.raises(BundleLoadError, match="decision_hz must be 15"):
        fx.default_host_profile(decision_hz=30)


@pytest.mark.parametrize("field_name", ["os_build", "gpu_name", "capture_backend"])
def test_host_profile_rejects_empty_values(field_name: str) -> None:
    """空文字の hardware 値は「未設定」なので受理しない。"""
    with pytest.raises(BundleLoadError, match="must be a non-empty string"):
        fx.default_host_profile(**{field_name: ""})


def test_host_profile_wire_round_trip() -> None:
    """host profile は exact field 集合で wire 往復できる。"""
    profile = fx.default_host_profile()
    assert HostRuntimeProfile.from_wire(profile.to_wire()) == profile
    with pytest.raises(BundleLoadError, match="host profile fields mismatch"):
        HostRuntimeProfile.from_wire({**profile.to_wire(), "extra": 1})


def test_target_profile_ref_hash_mismatch_is_rejected(
    formal_inputs: FormalBundleInputs, pinned_production_trust_key: str
) -> None:
    """target identity 参照が信頼済みリリースと違えば起動しない。"""
    from reinbalance_survivors_contracts.target_action import TargetProfileRef

    other = TargetProfileRef(
        build_id="vs-1.12.000",
        canonical_save_hash=fx.hash_of("other-save"),
        hardware_profile_id="other-win64",
    )
    with pytest.raises(BundleLoadError, match="target_profile_ref_hash mismatch"):
        RuntimeBundle.load(**formal_inputs.as_load_kwargs(target_profile=other))


# --- combat package manifest schema -------------------------------------------


def _mutated_package(
    formal_inputs: FormalBundleInputs, tmp_path: Path, mutate: Any
) -> Path:
    """combat package を複製し、manifest へ変更を加えた directory を返す。"""
    package = formal_inputs.copy_combat_package(tmp_path / "combat")
    manifest = json.loads((package / "manifest.json").read_bytes())
    mutate(manifest)
    fx.rewrite_combat_manifest(package, manifest)
    return package


@pytest.mark.parametrize(
    ("case", "mutate"),
    [
        ("missing key", lambda m: m.pop("checkpoint_sha256")),
        ("extra key", lambda m: m.update({"vecnormalize_sha256": fx.hash_of("nope")})),
        ("schema version", lambda m: m.update({"schema_version": "survivors.x.v2"})),
        ("development only", lambda m: m.update({"development_only": True})),
        ("not eligible", lambda m: m.update({"formal_student_eligible": False})),
        ("files list", lambda m: m.update({"files": ["manifest.json", "policy.zip"]})),
        ("model hash", lambda m: m.update({"model_sha256": fx.hash_of("wrong-model")})),
        ("short hash", lambda m: m.update({"checkpoint_sha256": "abc"})),
        (
            "model_config keys",
            lambda m: m.update({"model_config": {**m["model_config"], "lstm_hidden_size": 4}}),
        ),
        (
            "model_config value",
            lambda m: m.update({"model_config": {**m["model_config"], "hidden_dim": 0}}),
        ),
        (
            "dependency identities",
            lambda m: m.update({"formal_dependency_identities": {"fidelity_verdict": fx.hash_of("f")}}),
        ),
    ],
)
def test_invalid_combat_manifest_is_rejected(
    formal_inputs: FormalBundleInputs, tmp_path: Path, case: str, mutate: Any
) -> None:
    """欠損・余分・不整合な manifest はすべて package ロード時点で拒否される。

    やさしい説明: 説明書が少しでも契約と違えば、モデルを読む前に止めます。
    `vecnormalize_sha256` や `lstm_hidden_size` のような 03-05 契約に存在しない項目を
    足した場合も、余分なキーとして拒否されます。
    """
    package = _mutated_package(formal_inputs, tmp_path, mutate)
    with pytest.raises(BundleLoadError):
        ab._load_combat_package(package)


def test_manifest_rejection_surfaces_through_bundle_load(
    formal_inputs: FormalBundleInputs, tmp_path: Path, pinned_production_trust_key: str
) -> None:
    """manifest 改変は `RuntimeBundle.load()` 経由でも live 起動を止める。"""
    package = _mutated_package(
        formal_inputs, tmp_path, lambda m: m.update({"development_only": True})
    )
    with pytest.raises(BundleLoadError, match="development_only"):
        RuntimeBundle.load(**formal_inputs.as_load_kwargs(combat_package_dir=package))


def test_valid_manifest_with_wrong_content_hash_is_rejected(
    formal_inputs: FormalBundleInputs, tmp_path: Path, pinned_production_trust_key: str
) -> None:
    """manifest は正しい形でも、registry が固定した manifest hash と違えば拒否する。

    やさしい説明: 書式が正しいだけでは足りません。「まさにこの説明書」であることを、
    署名済み一覧の指紋と突き合わせて確かめます。
    """
    package = _mutated_package(
        formal_inputs,
        tmp_path,
        lambda m: m.update({"checkpoint_sha256": fx.hash_of("another-checkpoint")}),
    )
    # package 単体としては契約を満たす。
    ab._load_combat_package(package)
    with pytest.raises(BundleLoadError, match="combat package manifest hash"):
        RuntimeBundle.load(**formal_inputs.as_load_kwargs(combat_package_dir=package))


def test_extra_file_in_package_directory_is_rejected(
    formal_inputs: FormalBundleInputs, tmp_path: Path
) -> None:
    """宣言外の file が同居する package は差し替えの疑いがあるため拒否する。"""
    package = formal_inputs.copy_combat_package(tmp_path / "combat")
    (package / "vecnormalize.pkl").write_bytes(b"unexpected")
    with pytest.raises(BundleLoadError, match="directory contents mismatch"):
        ab._load_combat_package(package)


def test_model_file_hash_mismatch_is_rejected(
    formal_inputs: FormalBundleInputs, tmp_path: Path
) -> None:
    """model.pt の中身が manifest の指紋と違えば deserialize 前に止まる。"""
    package = formal_inputs.copy_combat_package(tmp_path / "combat")
    (package / "model.pt").write_bytes(b"replaced payload")
    with pytest.raises(BundleLoadError, match="hash mismatch"):
        ab._load_combat_package(package)


def test_model_payload_structure_mismatch_is_rejected(tmp_path: Path) -> None:
    """`model.pt` が `{model_config, model_state_dict}` 以外なら拒否する。"""
    package = tmp_path / "combat"
    model, model_config = fx.build_combat_model()
    fx.write_combat_package(package, model, model_config)
    th.save({"model_state_dict": model.state_dict()}, package / "model.pt")
    manifest = json.loads((package / "manifest.json").read_bytes())
    manifest["model_sha256"] = fx.sha256_hex((package / "model.pt").read_bytes())
    fx.rewrite_combat_manifest(package, manifest)

    with pytest.raises(BundleLoadError, match="model.pt structure mismatch"):
        ab._load_combat_package(package)


def test_wrong_model_topology_is_rejected(tmp_path: Path) -> None:
    """submodule 構成が違う state_dict は strict ロードで拒否される。

    やさしい説明: 部品名が `recurrent` / `actor` / `value` でないモデルを入れても、
    「一致しないもの」として読み込みを断ります。別モデルの混入をここで止めます。
    """
    package = tmp_path / "combat"
    model, model_config = fx.build_combat_model()
    fx.write_combat_package(package, model, model_config)

    class _WrongTopology(th.nn.Module):
        """recurrent/actor/value を持たない別構成のモデル。"""

        def __init__(self) -> None:
            super().__init__()
            self.lstm_actor = th.nn.LSTM(
                model_config["observation_dim"], model_config["hidden_dim"]
            )

    th.save(
        {"model_config": dict(model_config), "model_state_dict": _WrongTopology().state_dict()},
        package / "model.pt",
    )
    manifest = json.loads((package / "manifest.json").read_bytes())
    manifest["model_sha256"] = fx.sha256_hex((package / "model.pt").read_bytes())
    fx.rewrite_combat_manifest(package, manifest)

    with pytest.raises(BundleLoadError, match="state_dict incompatible"):
        ab._load_combat_package(package)


def test_model_config_conflict_between_manifest_and_payload_is_rejected(
    tmp_path: Path,
) -> None:
    """manifest と model.pt の model_config が食い違えば拒否する。"""
    package = tmp_path / "combat"
    model, model_config = fx.build_combat_model()
    fx.write_combat_package(package, model, model_config)
    manifest = json.loads((package / "manifest.json").read_bytes())
    manifest["model_config"] = {**model_config, "hidden_dim": model_config["hidden_dim"] + 1}
    fx.rewrite_combat_manifest(package, manifest)

    with pytest.raises(BundleLoadError, match="does not match the manifest"):
        ab._load_combat_package(package)


def test_missing_package_directory_is_rejected(tmp_path: Path) -> None:
    """存在しない package directory は明示的に拒否する。"""
    with pytest.raises(BundleLoadError, match="combat package directory not found"):
        ab._load_combat_package(tmp_path / "absent")


# --- DAG / store / capability -------------------------------------------------


def test_missing_descriptors_are_rejected(
    formal_inputs: FormalBundleInputs, pinned_production_trust_key: str
) -> None:
    """descriptor が空の bundle は起動しない。"""
    with pytest.raises(BundleLoadError, match="requires artifact descriptors"):
        RuntimeBundle.load(**formal_inputs.as_load_kwargs(descriptors=[]))


def test_failed_perception_verdict_is_rejected(
    formal_inputs: FormalBundleInputs, tmp_path: Path, pinned_production_trust_key: str
) -> None:
    """passed=False の perception final verdict では起動しない。"""
    from reinbalance_survivors_contracts.artifact_store import ArtifactStore

    store = ArtifactStore(tmp_path / "store")
    descriptors = fx.build_formal_descriptors(
        store=store,
        combat_manifest=formal_inputs.combat_manifest,
        item_selector_manifest=formal_inputs.selector_manifest,
        target_capability_hash=formal_inputs.target_capability_hash,
        choice_capability_hash=formal_inputs.choice_capability_hash,
        verdict_passed=False,
    )
    with pytest.raises(BundleLoadError, match="formal runtime DAG rejected"):
        RuntimeBundle.load(
            **formal_inputs.as_load_kwargs(descriptors=descriptors, artifact_store=store)
        )


def test_action_semantics_hash_mismatch_is_rejected(
    formal_inputs: FormalBundleInputs, pinned_production_trust_key: str
) -> None:
    """action semantics が信頼済みリリースと違えば起動しない。"""
    other = ActionSemantics(
        actions=tuple(f"move_variant_{index}" for index in range(REQUIRED_ACTION_DIM))
    )
    assert other.num_actions == REQUIRED_ACTION_DIM
    with pytest.raises(BundleLoadError, match="action_semantics_hash mismatch"):
        RuntimeBundle.load(**formal_inputs.as_load_kwargs(action_semantics=other))


def test_wrong_argument_types_are_rejected(
    formal_inputs: FormalBundleInputs, pinned_production_trust_key: str
) -> None:
    """store / target profile / host profile の型を取り違えたら起動しない。

    やさしい説明: これらの型検査は trust anchor を確立した **あと** に走ります。
    信頼 root の確立が最初なので、鍵が固定されていなければ型の話に進む前に止まります。
    """
    with pytest.raises(BundleLoadError, match="artifact_store must be"):
        RuntimeBundle.load(**formal_inputs.as_load_kwargs(artifact_store=object()))
    with pytest.raises(BundleLoadError, match="target_profile must be"):
        RuntimeBundle.load(**formal_inputs.as_load_kwargs(target_profile=object()))
    with pytest.raises(BundleLoadError, match="host_profile must be"):
        RuntimeBundle.load(**formal_inputs.as_load_kwargs(host_profile=object()))


# --- golden fixture -----------------------------------------------------------


def test_golden_fixture_bundle_refuses_live_startup() -> None:
    """golden fixture bundle は development_only で、正式起動を拒否する。

    やさしい説明: 正式な成果物が無くてもテストは回せますが、その bundle で本番を
    起動しようとすると必ず止まります。
    """
    model, model_config = fx.build_combat_model()
    policy = CombatPolicy(
        model=CombatGruPolicy(**model_config),
        observation_dim=model_config["observation_dim"],
        action_dim=model_config["action_dim"],
        hidden_dim=model_config["hidden_dim"],
    )
    bundle = RuntimeBundle.from_golden_fixture(policy)

    assert bundle.development_only is True
    assert bundle.live_eligible is False
    assert bundle.host_profile is None
    assert bundle.item_selector is None
    assert bundle.startup_report["bundle_kind"] == "golden_fixture"
    with pytest.raises(BundleLoadError, match="not live_eligible"):
        bundle.assert_live_eligible()


# --- 非検証 deserialization の不在 --------------------------------------------


_RUNTIME_PACKAGE_DIR = Path(ab.__file__).resolve().parent
_FORBIDDEN_PATTERNS = (
    r"\bpickle\.loads?\s*\(",
    r"\bimport\s+pickle\b",
    r"\bmarshal\.loads?\s*\(",
    r"\bshelve\.",
    r"\byaml\.load\s*\((?![^)]*SafeLoader)",
    r"(?<![\w.])eval\s*\(",
    r"(?<![\w.])exec\s*\(",
    r"weights_only\s*=\s*False",
)


@pytest.mark.parametrize(
    "source_path", sorted(_RUNTIME_PACKAGE_DIR.glob("*.py")), ids=lambda p: p.name
)
def test_runtime_sources_have_no_unverified_deserialization(source_path: Path) -> None:
    """runtime package の source に任意コード実行につながる読み込みが無いことを確認する。

    やさしい説明: 成果物の読み込み経路に pickle や eval が紛れ込むと、ファイルを
    差し替えるだけで任意のコードを実行されてしまいます。そうした呼び出しが 1 つも
    無いことを、ソースそのものを走査して確かめます。
    """
    source = source_path.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_PATTERNS:
        assert not re.search(pattern, source), (
            f"{source_path.name} contains forbidden deserialization pattern {pattern!r}"
        )


@pytest.mark.parametrize(
    "source_path", sorted(_RUNTIME_PACKAGE_DIR.glob("*.py")), ids=lambda p: p.name
)
def test_every_torch_load_pins_weights_only(source_path: Path) -> None:
    """`torch.load` はすべて `weights_only=True` で呼ばれている。

    やさしい説明: PyTorch のファイル読み込みは、既定では中に書かれた任意の
    オブジェクトを復元してしまいます。テンソルと素の値だけを読む設定に固定されて
    いることを確かめます。
    """
    source = source_path.read_text(encoding="utf-8")
    for match in re.finditer(r"\b(?:th|torch)\.load\s*\(", source):
        call = source[match.end() : match.end() + 240]
        assert "weights_only=True" in call, (
            f"{source_path.name}: torch.load at offset {match.start()} must pin weights_only=True"
        )


def test_runtime_sources_use_lf_line_endings() -> None:
    """新規 runtime source / test はすべて LF 改行である。"""
    roots = (_RUNTIME_PACKAGE_DIR, Path(__file__).resolve().parent)
    for root in roots:
        for path in sorted(root.glob("*.py")):
            assert b"\r\n" not in path.read_bytes(), f"{path} must use LF line endings"

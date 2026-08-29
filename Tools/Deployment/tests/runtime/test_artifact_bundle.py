"""artifact bundle: golden fixture loader の hash / schema / eligibility 境界を検証する。

formal artifact なしでも全テストを実行でき、development_only bundle が正式起動を拒否する
ことを確認する。
"""
import pytest

from reinbalance_survivors_contracts.deploy_obs import DeployObsSchema
from reinbalance_survivors_contracts.ui_policy import NonModelUiPolicyConfigV1
from survivors.runtime.artifact_bundle import (
    BundleLoadError,
    RuntimeBundle,
    _CombatGRU,
    BUNDLE_DEVELOPMENT_SENTINEL,
)


def _minimal_model(obs_dim: int = 4, action_dim: int = 9, hidden_dim: int = 8) -> _CombatGRU:
    """テスト用の最小 _CombatGRU を返す。"""
    return _CombatGRU(obs_dim, action_dim, hidden_dim)


class TestCombatGRUInit:
    def test_valid_dims_create_model(self):
        model = _minimal_model()
        assert model.observation_dim == 4
        assert model.action_dim == 9
        assert model.hidden_dim == 8

    def test_zero_obs_dim_raises(self):
        with pytest.raises(BundleLoadError, match="observation_dim"):
            _CombatGRU(0, 9, 8)

    def test_float_dim_raises(self):
        with pytest.raises(BundleLoadError, match="observation_dim"):
            _CombatGRU(4.0, 9, 8)  # type: ignore[arg-type]

    def test_step_returns_logit_and_hidden(self):
        import torch
        model = _minimal_model(obs_dim=4, action_dim=9, hidden_dim=8)
        obs = torch.zeros(1, 4)
        hidden = torch.zeros(1, 8)
        logit, new_hidden = model.step(obs, hidden)
        assert logit.shape == (1, 9)
        assert new_hidden.shape == (1, 8)


class TestGoldenFixtureBundle:
    def test_development_only_is_true(self):
        bundle = RuntimeBundle.from_golden_fixture(_minimal_model())
        assert bundle.development_only is True

    def test_live_eligible_is_false(self):
        bundle = RuntimeBundle.from_golden_fixture(_minimal_model())
        assert bundle.live_eligible is False

    def test_startup_report_kind_is_sentinel(self):
        bundle = RuntimeBundle.from_golden_fixture(_minimal_model())
        assert bundle.startup_report["bundle_kind"] == BUNDLE_DEVELOPMENT_SENTINEL

    def test_assert_live_eligible_raises(self):
        bundle = RuntimeBundle.from_golden_fixture(_minimal_model())
        with pytest.raises(BundleLoadError, match="development_only"):
            bundle.assert_live_eligible()

    def test_item_selector_defaults_to_none(self):
        bundle = RuntimeBundle.from_golden_fixture(_minimal_model())
        assert bundle.item_selector is None

    def test_deploy_schema_is_populated(self):
        bundle = RuntimeBundle.from_golden_fixture(_minimal_model())
        assert isinstance(bundle.deploy_schema, DeployObsSchema)
        assert bundle.deploy_schema_hash == bundle.deploy_schema.schema_hash

    def test_custom_deploy_schema_accepted(self):
        schema = DeployObsSchema.default_v1()
        bundle = RuntimeBundle.from_golden_fixture(_minimal_model(), deploy_schema=schema)
        assert bundle.deploy_schema.schema_hash == schema.schema_hash

    def test_ui_policy_config_defaults_to_installed(self):
        bundle = RuntimeBundle.from_golden_fixture(_minimal_model())
        assert isinstance(bundle.ui_policy_config, NonModelUiPolicyConfigV1)

    def test_custom_ui_policy_config_accepted(self):
        config = NonModelUiPolicyConfigV1.load_default()
        bundle = RuntimeBundle.from_golden_fixture(_minimal_model(), ui_policy_config=config)
        assert bundle.ui_policy_config is config

    def test_non_gru_model_raises(self):
        with pytest.raises(BundleLoadError, match="_CombatGRU"):
            RuntimeBundle.from_golden_fixture(object())  # type: ignore[arg-type]

    def test_golden_item_selector_stored(self):
        """item_selector に任意 object を渡せる (golden mode)。"""
        sentinel = object()
        bundle = RuntimeBundle.from_golden_fixture(_minimal_model(), item_selector=sentinel)
        assert bundle.item_selector is sentinel


class TestFormalBundleNotAvailableWithoutArtifacts:
    """formal package なしでは load() は失敗する — hash 境界テスト。"""

    def test_load_requires_sha256_perception_verdict(self, tmp_path):
        with pytest.raises((BundleLoadError, ValueError), match="SHA-256|sha256|hash"):
            RuntimeBundle.load(
                tmp_path,
                tmp_path,
                perception_verdict_hash="not-a-sha256",
                target_capability_hash="a" * 64,
            )

    def test_load_requires_sha256_capability_hash(self, tmp_path):
        with pytest.raises((BundleLoadError, ValueError), match="SHA-256|sha256|hash"):
            RuntimeBundle.load(
                tmp_path,
                tmp_path,
                perception_verdict_hash="a" * 64,
                target_capability_hash="bad",
            )

    def test_load_missing_manifest_raises(self, tmp_path):
        with pytest.raises((BundleLoadError, ValueError, OSError)):
            RuntimeBundle.load(
                tmp_path,
                tmp_path,
                perception_verdict_hash="a" * 64,
                target_capability_hash="b" * 64,
            )

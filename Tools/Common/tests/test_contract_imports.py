"""共有 contract パッケージの import 衛生テスト。

パッケージが、現在の作業ディレクトリに依存しない安定した module identity を持つ
インストール済みパッケージとして import でき、かつ副作用として Training/Deployment
専用の重い依存を巻き込まないことを保証する。
"""

import json
import subprocess
import sys
from pathlib import Path

import reinbalance_survivors_contracts as contracts

# 共有パッケージが決して import してはならない、重い／consumer 専用の module。
FORBIDDEN_MODULES = [
    "torch",
    "stable_baselines3",
    "sb3_contrib",
    "gymnasium",
    "cv2",
    "dxcam",
    "onnxruntime",
    "requests",
]


def _probe(cwd, code):
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_version_and_all_exports():
    assert contracts.__version__ == "0.1.0"
    for name in ("UiIntentV1", "decide_non_model_ui_intent", "canonical_hash"):
        assert name in contracts.__all__


def test_fingerprint_is_deterministic():
    assert contracts.contract_fingerprint() == contracts.contract_fingerprint()


def test_fingerprint_expected_keys():
    fp = contracts.contract_fingerprint()
    assert set(fp) == {
        "package_version",
        "ui_intent_v1_qualname",
        "non_model_ui_policy_config_hash",
        "deploy_obs_v1_schema_hash",
        "action_semantics_v1_hash",
    }
    assert fp["ui_intent_v1_qualname"] == (
        "reinbalance_survivors_contracts.ui_intent.UiIntentV1"
    )


def test_same_module_and_fingerprint_across_cwds(tmp_path):
    """インストール済みパッケージは cwd に関係なく同一に解決される（cwd シャドウなし）。"""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    code = (
        "import json, reinbalance_survivors_contracts as m;"
        "print(json.dumps({'file': m.__file__, 'fp': m.contract_fingerprint()}))"
    )
    ra = _probe(dir_a, code)
    rb = _probe(dir_b, code)
    assert ra["file"] == rb["file"]
    assert ra["fp"] == rb["fp"]


def test_no_forbidden_modules_imported():
    code = (
        "import json, sys;"
        "import reinbalance_survivors_contracts;"
        f"forbidden={FORBIDDEN_MODULES!r};"
        "print(json.dumps(sorted(set(sys.modules) & set(forbidden))))"
    )
    leaked = _probe(".", code)
    assert leaked == [], f"shared package imported forbidden modules: {leaked}"


def _tools_dir():
    for parent in Path(__file__).resolve().parents:
        if (parent / "Common").is_dir() and (parent / "Training").is_dir():
            return parent
    raise AssertionError("could not locate the Tools/ directory")


def test_no_duplicate_contract_module_in_consumers():
    """consumer はインストール済みパッケージを import しなければならず、ローカル複製を使ってはならない。"""
    tools = _tools_dir()
    for consumer in ("Training", "Deployment"):
        root = tools / consumer
        if not root.is_dir():
            continue
        dupes = [
            p
            for p in root.rglob("reinbalance_survivors_contracts*")
            if p.suffix in ("", ".py") and "tests" not in p.parts
        ]
        assert not dupes, f"duplicate contract module under {consumer}: {dupes}"

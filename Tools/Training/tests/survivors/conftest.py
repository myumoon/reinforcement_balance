"""Survivors test collection の optional training dependency 境界を定義する。

Common 契約だけを入れた最小 venv では SB3・Gymnasium・PyYAML・Deployment package が
ありません。それらを import 時または実行時に必要とする既存 test file だけを収集対象外にし、
残る契約 test は実行します。
このディレクトリを sys.path へ追加することで、value_scorer_fixtures などのローカルヘルパーを
テストファイルから直接インポートできるようにする。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_OPTIONAL_DEPENDENCY_TESTS = {
    "stable_baselines3": {
        "test_bootstrap_eval.py",
        "test_bootstrap_gate_eval_callback.py",
        "test_survivors_run_supervisor_callback.py",
        "test_task_cell_sampler_callback.py",
        "test_weapon_phase_exposure_guard.py",
    },
    "gymnasium": {
        "test_perception_error_wrapper.py",
    },
    "survivors": {
        "test_deploy_obs_wrapper.py",
        "test_perception_error_wrapper.py",
    },
    "yaml": {
        "test_content_manifest.py",
        "test_content_scenario_coverage.py",
    },
}

collect_ignore = sorted(
    {
        test_file
        for package_name, test_files in _OPTIONAL_DEPENDENCY_TESTS.items()
        if importlib.util.find_spec(package_name) is None
        for test_file in test_files
    }
)

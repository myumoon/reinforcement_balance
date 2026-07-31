"""共有 Survivors 契約に対する Deployment 側の smoke テスト。

Deployment が（独自の cwd / テストルート / lock を持ちつつ）Common/Training と
まったく同じインストール済み contract パッケージを import し、golden な UI policy
JSONL を byte 単位で再現し、かつ requirements lock が完全に pin されていることを検証する。
"""

import json
import subprocess
import sys
from pathlib import Path

import reinbalance_survivors_contracts as contracts
from reinbalance_survivors_contracts import (
    NonModelUiPolicyConfigV1,
    UiPolicyInputV1,
    canonical_json_bytes,
    decide_non_model_ui_intent,
    sha256_hex,
)


def _find_shared_fixture() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "Common" / "tests" / "fixtures" / "ui_policy_cases_v1.json"
        if cand.is_file():
            return cand
    raise FileNotFoundError("shared ui_policy_cases_v1.json not found")


def test_shared_contract_importable():
    assert contracts.__version__ == "0.1.0"
    fp = contracts.contract_fingerprint()
    assert fp["ui_intent_v1_qualname"] == (
        "reinbalance_survivors_contracts.ui_intent.UiIntentV1"
    )


def test_cross_package_ui_policy_jsonl_matches_golden():
    data = json.loads(_find_shared_fixture().read_text(encoding="utf-8"))
    cfg = NonModelUiPolicyConfigV1.default_config()
    assert cfg.config_hash == data["config_hash"]
    lines = []
    for case in data["cases"]:
        inp = UiPolicyInputV1.from_wire(case["input"])
        intent = decide_non_model_ui_intent(inp, cfg)
        expected = intent.to_wire() if intent is not None else None
        lines.append(canonical_json_bytes({"case": case["name"], "intent": expected}))
    assert sha256_hex(b"\n".join(lines)) == data["expected_jsonl_sha256"]


def test_import_is_cwd_independent(tmp_path):
    code = (
        "import json, reinbalance_survivors_contracts as m;"
        "print(json.dumps({'file': m.__file__, 'fp': m.contract_fingerprint()}))"
    )

    def probe(cwd):
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=str(cwd), capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    assert probe(dir_a) == probe(dir_b)


def _parse_lock(path: Path) -> list[str]:
    reqs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            reqs.append(line)
    return reqs


def _is_pinned(req: str) -> bool:
    """要件が pin されているのは、``==`` をちょうど1つ使い、範囲演算子を含まない場合のみ。"""
    if "==" not in req:
        return False
    if any(op in req for op in (">=", "<=", "~=", "!=", ">", "<")):
        return False
    name, _, version = req.partition("==")
    return bool(name.strip()) and bool(version.strip())


def test_requirements_lock_is_fully_pinned():
    lock = Path(__file__).resolve().parents[1] / "requirements.lock"
    reqs = _parse_lock(lock)
    assert reqs, "requirements.lock has no requirements"
    for req in reqs:
        assert _is_pinned(req), f"unpinned / ranged requirement: {req!r}"


def test_lock_parser_rejects_unpinned():
    assert not _is_pinned("numpy>=1.26.4")
    assert not _is_pinned("numpy")
    assert not _is_pinned("numpy~=1.26")
    assert not _is_pinned("numpy!=1.0")
    assert _is_pinned("numpy==1.26.4")


def test_pip_check_passes():
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr

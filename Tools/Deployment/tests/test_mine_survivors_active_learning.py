"""mine_survivors_active_learning の class 数導出のテスト。

class_probs が無い候補の一様分布が、リテラル 12 ではなく class map の num_classes から作られることを確かめる。
"""
from __future__ import annotations

import json
import math

from mine_survivors_active_learning import entropy_score, main, mine_top_k


def test_missing_class_probs_uses_class_map_num_classes(tmp_path):
    """CLI は既定の class map v2（16 クラス）の一様分布で entropy を計算する。

    一様分布の entropy は log(クラス数) に比例するので、12 と 16 の取り違えを検出できる。
    """
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps({"annotations": [{"category_id": 2}]}), encoding="utf-8")
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps([{"image_id": 1}]), encoding="utf-8")
    out = tmp_path / "out.json"
    assert main(["--annotations", str(ann), "--candidate-frames", str(cand), "--output", str(out)]) == 0
    entropy = json.loads(out.read_text(encoding="utf-8"))[0]["components"]["entropy"]
    expected = mine_top_k([{"image_id": 1}], {2: 1}, 1, 1, num_classes=16)[0].components["entropy"]
    assert math.isclose(entropy, expected, rel_tol=1e-6)
    assert not math.isclose(entropy, entropy_score([1.0 / 12] * 12), rel_tol=1e-6)

# Survivors simulator content coverage

03-03 は `/content_schema` を content identity の canonical input とし、`Logic → schema → Python audit` の一方向で coverage を判定する。YAML は ID の照合キー、target relevance、scenario、evidence だけを持ち、名称、最大レベル、XP 値、進化 prerequisite を複製しない。

## 5 gates

- `implemented`: effect handler または意図的な no-combat handler が存在する。
- `reachable`: starting 以外を含む取得・強化・進化経路が存在する。
- `observed`: policy observation に content の結果が現れる。
- `trained`: deterministic scenario cell が訓練 evidence に関連付く。
- `evaluated`: scenario に判定可能な eval assertion がある。

`audit_survivors_content.py --schema <captured-content-schema.json>` は全行を利用直前に再検証し、各 gate の blocking 件数を JSON で出力する。ID の追加・削除、未知 key、重複 ID、非有限値、未実装 level、effect/obs/scenario 欠落は fail-closed で拒否する。

## Intentional exclusions

Pentagram (12)、Laurel (15)、Gorgeous Moon (27) は starting weapon のみから除外する。Pentagram は継続的な starting XP を保証せず、Laurel は防御専用、Gorgeous Moon は進化取得だからである。この除外は acquisition、effect、observation、training、evaluation の免除ではない。YAML には理由と alternative coverage が必須で、5 gates のどれかを落とすと監査は失敗する。

Stone Mask (14) は最大レベル 5 で取得・強化・slot 観測を行う。金貨は現行 combat simulator の目的変数でないため effect handler を `intentional_no_combat_gold_only` と明示し、暗黙の未実装として扱わない。

Enemy 0–10 の canonical default table は `FSurvivorsGameLogic` が所有する。現行 `/content_schema` shape と 00-05 exact-key consumer を変更せず、LLT が全 type の spawn、HP、damage、XP、boss/resistance flag、finite observation/type encoding を表駆動で検証する。

## Verification boundaries

Python tests と audit CLI は Linux/WSL で実行できる。UE5 Editor build と Low Level Tests は Windows UE5.4 環境で実行する。content/constants/Logic の変更は 00-05 producer closure が hash へ取り込むため、既存 fidelity verdict は stale になり、01-05 formal 収集または 03-04 FR4 より前に current-hash integration verdict の再発行が必要である。

# Survivors Artifact Identity / Store Contract

## 目的

Survivors の source model から final campaign evidence までを、Git 外の
content-addressed object store と不変 descriptor DAG で追跡する。descriptor は
artifact 内容、ordered parent refs、stable metadata だけで identity hash を計算し、
timestamp、operator、hostname、local/absolute path、validation 結果は identity へ含めない。

## Identity DAG

概念 edge は常に parent source から child artifact へ向かう。

```text
source_descriptor
  -> teacher_validation_verdict
  -> choice_dataset_release
  -> item_selector_release / combat_student_release
  -> runtime_bundle
  -> replay_shadow_verdict
  -> canary_campaign
  -> goal_evidence
```

保存形式では child descriptor だけが immutable parent hash を `parents` に保持する。
parent descriptor から descendant を参照しない。source descriptor、validation verdict、
dataset manifest は in-place 更新せず、変更が必要な場合は新しい descriptor node を作る。

## Object Store

- Object URI は `artifact://sha256/<64-hex>`。
- primary store は CLI の `--store-root` で必ず明示する。workspace 内 fallback は持たない。
- object は `objects/sha256/<prefix>/<hash>` に atomic temp write から rename する。
- `logical_id` は絶対 path、drive separator、`..` を含めない。
- 同一 `logical_id` と同一 hash の put は idempotent。異なる bytes への上書きは拒否する。
- release/canary の final evidence は primary と backup の resolved path/device が異なることを gate する。

## License / Privacy Classification

| 種別 | 例 | 既定分類 | 公開可否 |
|---|---|---:|---|
| save/config/model/dataset | canonical save hash, JSON/JSONL/NPZ, ONNX | `internal` | project 内のみ |
| telemetry/log | replay/shadow/canary metrics | `internal` | secret 除去後のみ |
| actual game frame/icon/video | screenshots, frame dumps, mp4 | `private` | 既定で公開 export から除外 |
| third-party/source critic | teacher corpus, source model | `private` | 再配布不可として扱う |

public export は `private` / `restricted` / `secret` object bytes を既定で ZIP に含めない。
manifest には license と privacy classification を残すが、`non_identity_metadata` は空にして
local path、absolute path、operator、hostname、secret 値を出さない。

## Retention

- final bundle、C0-C4 telemetry/video refs、goal evidence: goal supersede 後も最低 365 日。
- source model、teacher corpus、selector/student release: 最低 365 日。
- failed pilot/intermediate shards: 最低 90 日。
- 削除時は object を即時削除せず、tombstone descriptor と parent impact report を残す。
  impact report は対象 hash、descendant logical id、release/campaign への影響を列挙する。

## Backup / Restore Gate

release 前に以下を完了する。

1. primary store から deterministic ZIP64 bundle を作る。
2. backup root が primary と別 resolved path/device であることを確認する。
3. 空の backup store へ import する。
4. full manifest verify、または quarterly restore test で定義された random sample verify を実行する。
5. `restore_test_verdict` descriptor を作り、subject bundle hash、manifest hash、verify mode、
   checked object count、pass/blocking reasons を記録する。

hash だけ残って object 実体が欠落、corrupt、または retention due の場合、release gate は fail closed とする。

## Disk / Failure / Migration Runbook

- disk 使用量は store root ごとに `objects/sha256` と `index/logical` を別集計する。
  80% 超過で cleanup 候補を出すが、retention 未満または release descendant を持つ object は削除しない。
- backup 失敗時は release を止め、primary manifest audit、backup import log、missing/corrupt URI を記録する。
- store 移行時は manifest export/import 後に full verify を実行し、移行元と移行先の manifest hash を一致させる。
- tombstone から復元する場合は、impact report の descendant を全て再 audit し、必要な restore verdict を再発行する。

# Agent Runtime

## 概要

`Tools/Deployment/survivors/runtime/` は、本家 Vampire Survivors のリアルタイム画面から typed `AgentDecision` を返す推論パイプラインを提供する。OS への input は一切送信しない。

## アーキテクチャ

```
PerceptionSnapshot → AgentRuntime → AgentDecision
                         ├── CombatSession (gameplay)
                         ├── ItemSession (level_up_items)
                         └── NonModelUiPolicy (fallback / chest / confirm)
```

## モジュール構成

| ファイル | 責務 |
|---|---|
| `artifact_bundle.py` | golden fixture / formal package の hash・schema・capability fail-closed 検証とロード |
| `combat_session.py` | GRU 隠れ状態を episode 単位で保持する RecurrentPPO 推論セッション |
| `item_session.py` | ItemSelectorArtifact を使ってレベルアップ候補を採点し CHOOSE_CARD UiIntentV1 を返す |
| `decision_scheduler.py` | 15 Hz 固定 cadence のデシジョンタイミング管理 |
| `agent_runtime.py` | screen_state 別に combat / item / non-model UI policy を振り分けるオーケストレータ |

## 実行境界

- runtime は typed `AgentDecision` を返すだけで、OS input は送信しない（05-03 が所有する）
- `AgentDecision.kind` は `"move"` / `"ui"` / `"no_op"` / `"stop"` の 4 種のみ
- UiPresentationSnapshotV1 の ROI は ItemSelector / combat モデル feature に漏洩しない
- fallback / meta / ack / confirm 決定は共有 `NonModelUiPolicyV1` から生成し、05-03 は UiIntent を生成しない

## Bundle の種類

### Golden Fixture Bundle (development only)

```python
bundle = RuntimeBundle.from_golden_fixture(combat_model)
assert bundle.development_only is True
assert bundle.live_eligible is False
```

- formal artifact なしで全 loader / session / scheduler テストが実行できる
- `assert_live_eligible()` で必ず拒否される
- `startup_report["bundle_kind"] == "golden_fixture"`

### Formal Bundle (live eligible)

```python
bundle = RuntimeBundle.load(
    combat_package_dir,
    item_selector_dir,
    perception_verdict_hash="<04-10 final verdict SHA-256>",
    target_capability_hash="<02-03 / 03-05 capability SHA-256>",
)
```

**正式 bundle 発行条件（すべて必須）:**
- 02-03 formal ItemSelector package
- 03-05 `D03-DEPLOY-STUDENT-RELEASE`
- 04-10 `D04-PERCEPTION-FINAL` とその subject hashes

いずれかが stale / missing の場合は `BundleLoadError` で起動を拒否する。

## AgentDecision フィールド

| フィールド | 説明 |
|---|---|
| `kind` | `"move"` / `"ui"` / `"no_op"` / `"stop"` |
| `decision_id` | UUID4 (cross-frame tracking 用) |
| `ui_intent` | `UiIntentV1` (kind="ui" のみ non-None) |
| `move_action_index` | 0–8 の action index (kind="move" のみ non-None) |
| `no_op_reason` | 安全停止理由 (kind="no_op"/"stop" のみ non-None) |
| `source_snapshot_id` | 元 PerceptionSnapshot の snapshot_id |
| `inference_started_ns` / `inference_finished_ns` | monotonic ns (latency 計測用) |

## テスト

```bash
bash Tools/run-pytest.sh Tools/Deployment/tests/runtime -q -rs
```

## Stale / Invalid 対応

| 状態 | runtime の応答 |
|---|---|
| 型が PerceptionSnapshot でない | `stop` |
| 未知の screen_state | `no_op` |
| combat: 非有限 obs / shape 不一致 | `no_op` |
| item session error | `stop` |
| non-model UI policy が ContractValidationError | `stop` |
| non-model UI policy が None を返す | `no_op` |

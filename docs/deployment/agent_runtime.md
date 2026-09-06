# Agent Runtime

## 概要

`Tools/Deployment/survivors/runtime/` は、本家 Vampire Survivors のリアルタイム画面から typed `AgentDecision` を返す推論パイプラインを提供する。OS への input は一切送信しない。

combat policy は Training 側の architecture を Deployment で再定義せず、保存済み SB3 RecurrentPPO をそのままロードする。ItemSelector は ONNX Runtime で推論し、Training package を import しない。

## アーキテクチャ

```
PerceptionSnapshot → AgentRuntime → AgentDecision
                         ├── DecisionScheduler (15 Hz cadence / age / timeout gate)
                         ├── CombatSession       (ScreenState.GAMEPLAY)
                         ├── ItemSession         (ScreenState.LEVEL_UP)
                         └── NonModelUiPolicy    (FALLBACK / CHEST / CONFIRM)
```

経路の分岐は raw `screen_state` 文字列の重複表ではなく、assembler が型付けした `snapshot.ui_policy_input.screen_state`（`ScreenState` enum）を正本にする。raw 側の `target_reached_transition` は assembler が `ScreenState.CONFIRM` へ写像するため、runtime は raw 名を知らなくてよい。

## モジュール構成

| ファイル | 責務 |
|---|---|
| `artifact_bundle.py` | golden fixture / formal package の hash・DAG・restore・action・capability を fail-closed 検証してロードする |
| `item_selector_runtime.py` | 共有 `item_selector_package` 検証 + ONNX Runtime による ItemSelector adapter（Training 非依存） |
| `combat_session.py` | SB3 RecurrentPPO の actor LSTM state を episode 単位で保持する推論セッション |
| `item_session.py` | 較正済み confidence gate と typed target 検証を通して CHOOSE_CARD UiIntentV1 を返す |
| `decision_scheduler.py` | 15 Hz cadence、backlog skip、snapshot age gate、inference timeout gate |
| `agent_runtime.py` | 上記を束ね、typed screen_state 別に decision を生成するオーケストレータ |

## 実行境界

- runtime は typed `AgentDecision` を返すだけで、OS input は送信しない（05-03 が所有する）
- `AgentDecision.kind` は `"move"` / `"ui"` / `"no_op"` / `"stop"` の 4 種のみ
- UiPresentationSnapshotV1 の ROI は ItemSelector / combat モデル feature に漏洩しない
- fallback / meta / ack / confirm 決定は共有 `NonModelUiPolicyV1` から生成し、05-03 は UiIntent を生成しない

## 時計とゲート

`captured_ns` と runtime 内部の時刻は、capture 層と同じ単調高分解能時計 `time.perf_counter_ns()` を使う。Windows の `time.monotonic_ns()` は分解能が約 15.6 ms しかなく、1 tick ぶんの推論時間を測れないため使用しない。

| ゲート | 条件 | 未達時 |
|---|---|---|
| cadence | `scheduler.should_decide(now_ns)` | `no_op`（off-cadence、推論しない） |
| snapshot age | `0 <= now_ns - captured_ns <= 100 ms` かつ `captured_ns > 0` | `no_op` |
| global validity | 観測可能 field の validity 平均 `>= 0.5` | `no_op` |
| perception 停止 | 観測可能 field の age が全て 1.0 | `no_op` |
| inference timeout | 推論 `<= 20 ms` | `no_op`（結果を破棄） |

`unobservable` な field（`enemy_hp` / `cooldown`）は契約上 validity=0 / age=1 で固定されるため、validity gate の分母から除外する。`temporal_inferred` な field は episode 先頭で age=1.0 になるのが正常なので、「全 field が 1.0」のときだけ perception 停止と判定する。

## Episode 境界

raw `screen_state` が `death` / `result` / `unknown` になった時点、および `ui_policy_input.screen_state` が `ScreenState.UNKNOWN` の場合、runtime は判断より先に `CombatSession.reset_episode()` を呼び actor LSTM state を破棄する。`paused` は待機のみで state を保持する。

## Bundle の種類

### Golden Fixture Bundle (development only)

```python
bundle = RuntimeBundle.from_golden_fixture(combat_policy)
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
    artifact_store=store,             # restore 検証済み ArtifactStore
    descriptors=descriptors,          # runtime_bundle + 全 ancestor descriptor
    target_profile=target_profile,    # TargetProfileRef
)
```

`perception_verdict_hash` は引数で受け取らない。検証済みの `perception_final_verdict` descriptor から導出するため、64 桁文字列を渡すだけで startup report へ複製されることはない。

**gate 順序（すべて必須）:**

1. `validate_formal_runtime_dag()` — immutable parent（02-03 ItemSelector release / 03-05 combat student release / 04-10 perception final verdict）と 04-10 exact subject hashes、verdict の `passed` / `development_only`
2. `ArtifactStore.verify()` — 全 descriptor file の restore 検証
3. package manifest の canonical hash が、対応する release descriptor の `identity_metadata["package_manifest_hash"]` と一致
4. hardware / action / time / capability — `target_profile_ref_hash`、`action_semantics_hash`、`decision_hz == 15`、`target_capability_hash`、`choice_capability_hash`

いずれかが stale / missing の場合は `BundleLoadError` で起動を拒否する。

### combat package の形式

```
manifest.json / policy.zip / vecnormalize.pkl
```

- `policy.zip` は SB3 RecurrentPPO、`vecnormalize.pkl` は deploy VecNormalize
- ロード順序は **manifest 検証 → file content hash 照合 → deserialize** を厳守する。hash 照合を通らない file を torch / pickle へ渡さないことが、package 差し替えによる任意コード実行を防ぐ唯一の境界である
- VecNormalize は `training=False` / `norm_reward=False` で復元し、観測統計だけを使う（gym env 不要）
- `model_config.action_dim` は 9 固定。`action_semantics_hash` は共有 `ActionSemantics.default_v1().semantics_hash` と一致しなければならない

## AgentDecision フィールド

plan 05-01 の契約に一致させ、05-03 が同じ wire を消費できるようにする。

| フィールド | 説明 |
|---|---|
| `decision_id` | UUID4 (cross-frame tracking 用) |
| `kind` | `"move"` / `"ui"` / `"no_op"` / `"stop"` |
| `action_index` | 0–8 の action index (kind="move" のみ non-None) |
| `ui_intent` | `UiIntentV1` (kind="ui" のみ non-None) |
| `confidence` | `[0, 1]`。combat は policy 確率、item は較正済み確率、non-model policy は 1.0 |
| `reason` | 決定理由 / 安全停止理由（常に非空） |
| `source_snapshot_id` / `source_frame_id` / `source_content_hash` | 元 PerceptionSnapshot への binding |
| `snapshot_timestamp_ns` | `PerceptionSnapshot.captured_ns` |
| `inference_started_ns` / `inference_finished_ns` | `perf_counter_ns`（latency 計測用） |

canonical wire は `to_wire()` / `canonical_bytes()` / `decision_hash()`、JSONL は `decisions_to_jsonl()` で得る。`AgentDecision.from_wire()` で replay 用に復元できる。

## ItemSelector confidence gate

`ItemSession` は argmax をそのまま `choose_card` にしない。

1. artifact の `student_output_temperature` を適用した logits を得る
2. masked（padding）slot を除外し、artifact の `temperature` で較正した softmax 確率を計算する
3. 最大確率が `confidence_threshold` 未満なら intent を作らず `no_op`
4. 勝者を `UiPresentationSnapshotV1` の typed target へ解決する。`validity=True`、`semantic_kind == "item_card"`、かつ一意に対応する場合だけ intent を作る。0 件 / 複数件 / invalid / fallback semantic はいずれも `stop`

## テスト

```bash
bash Tools/run-pytest.sh Tools/Deployment/tests/runtime -q -rs
```

`Tools/Deployment/tests/runtime/conftest.py` が session scope で golden RecurrentPPO と formal package 一式を 1 度だけ構築するため、正式 Artifact なしで全テストが動く。

## Stale / Invalid 対応

| 状態 | runtime の応答 |
|---|---|
| 型が PerceptionSnapshot でない | `stop`（LSTM state も破棄） |
| cadence 外の呼び出し | `no_op`（推論しない） |
| `captured_ns == 0` / stale / 未来 | `no_op` |
| global validity gate 未達 | `no_op` |
| inference timeout 超過 | `no_op` |
| `death` / `result` / `unknown` | `no_op`（LSTM state を破棄） |
| `paused` | `no_op`（state は保持） |
| `ui_policy_input` が None | `no_op`（LSTM state を破棄） |
| combat: 非有限 obs / shape 不一致 | `no_op` |
| combat: 想定外の model error | `stop` |
| item: confidence gate 未達 | `no_op` |
| item: target 未解決 / invalid / 非一意 | `stop` |
| non-model UI policy が ContractValidationError | `stop` |
| non-model UI policy が stop intent を返す | `stop` |
| non-model UI policy が None を返す | `no_op` |

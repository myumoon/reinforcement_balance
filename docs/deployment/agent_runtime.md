# Agent Runtime

`agent_runtime` は perception snapshot を入力に、combat の recurrent GRU actor と ONNX ItemSelector を
15 Hz cadence で束ね、typed `AgentDecision` を返す 05-01 の recurrent runtime です。
OS input は一切送信せず、effect は 05-03 gameplay UI state machine が所有します。

## モジュール

| モジュール | 役割 |
|---|---|
| `survivors/runtime/agent_runtime.py` | perception snapshot から `AgentDecision` を返す中心オーケストレータ。schema/validity gate、episode 境界の LSTM 破棄を担う |
| `survivors/runtime/artifact_bundle.py` | 03-05 combat package と 02-03 ItemSelector package を trust anchor と descriptor hash で検証してからロードする |
| `survivors/runtime/combat_session.py` | 検証済み combat actor (GRU) の recurrent state を episode 単位で管理し、argmax で action index を返す |
| `survivors/runtime/decision_scheduler.py` | 固定 15 Hz cadence・backlog skip・snapshot age gate・inference timeout を強制する |
| `survivors/runtime/item_selector_runtime.py` | ItemSelector package 用の Deployment 専用 ONNX Runtime adapter（TorchScript/pickle 不使用） |
| `survivors/runtime/item_session.py` | level-up 候補を ONNX ItemSelector で採点し、confidence gate 通過時のみ `choose_card` UiIntentV1 を返す |

## 境界

- combat 推論は現時点で **CPU 専用**です。GPU device 配置対応は本 PR の範囲外とし、
  別 PR での追加を提案しています（詳細は下記 Formality 節）。
- `AgentRuntime.decide()` は OS input を送信しません。move / ui / no_op / stop の
  typed decision を返すだけで、click や key の実発火は 05-03 が担います。
- death / result / unknown gap を跨ぐと combat actor の LSTM state を破棄し、
  前 run の記憶を次 run へ持ち越しません。
- `AgentDecision.decision_id` は呼び出しごとに新規生成される trace id (`uuid4`) で、
  `inference_started_ns` / `inference_finished_ns` は壁時計です。同じ入力 sequence を
  再生しても、これら 3 field は再現しません。decision の中身（kind / action_index /
  confidence / reason 等）だけが決定的です。
- `decisions_to_jsonl()` の決定性は「同じ `AgentDecision` オブジェクト列から常に
  byte-identical な JSONL を作れる」ことを保証するものであり、独立な 2 回の
  `runtime.decide()` 呼び出し列が同一 trace id を持つことは保証しません。

## CLI

```bash
# Task4 性能・30分 soak テストのみ実行
USER=neko bash Tools/run-pytest.sh \
  Tools/Deployment/tests/runtime/test_agent_runtime_performance.py -q -rs

# 30分 soak を除いた高速サブセットだけ実行
USER=neko bash Tools/run-pytest.sh \
  Tools/Deployment/tests/runtime/test_agent_runtime_performance.py -q -rs -k "not ThirtyMinuteSoak"
```

## Formality

このモジュールは 05-01 PR3 (Recurrent Agent Runtime) の成果物です。
Task4 の性能検証は CPU 推論のみを対象とし、次の budget をすべて満たすことを
`Tools/Deployment/tests/runtime/test_agent_runtime_performance.py` で確認済みです。

| 指標 | budget | 結果 |
|---|---|---|
| combat inference p95 (CPU) | <= 8 ms | PASS |
| item selector inference p95 | <= 5 ms | PASS |
| combined `runtime.decide()` p99 | <= 20 ms | PASS |
| 30分 synthetic schedule (27,000 tick) | memory growth / episode reset / NaN・範囲外 action = 0 | PASS |

ロードマップの元記述にある「GPU combat p95<=8ms」は、本 PR では GPU device 配置対応が
未実装のため CPU 推論として計測しています。GPU 対応（`th.device("cuda")` への
model/tensor 配置、CPU/GPU 双方での budget 再計測）は、別 PR の task として切り出す
ことを提案します。

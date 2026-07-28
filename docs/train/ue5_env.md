# UE5 Survivors 訓練環境 — HTTP API 仕様

Python（SB3）と UE5 エディタ（PIE）間の HTTP 通信仕様と、UE5 側の挙動を記述する。

UE5 側の実装は以下を参照:
```
ReinBalance/Source/ReinBalanceEditor/Private/Training/SurvivorsHttpEnvService.cpp
ReinBalance/Source/ReinBalanceEditor/Public/Training/SurvivorsHttpEnvService.h
```

---

## エンドポイント一覧

| メソッド | パス | 説明 |
|---|---|---|
| GET | `/obs_schema` | 観測空間の定義を取得（起動時 1 回） |
| POST | `/reset` | エピソードをリセットし初期観測を返す |
| POST | `/step` | アクションを送信し観測・報酬・done を返す |
| POST | `/params` | ゲームパラメータを動的に更新する |
| POST | `/level_up_choice` | external mode の保留候補を exactly-once で適用する |

---

## `/params` エンドポイントの挙動

### 即時反映（リセット待ちなし）

HTTP worker は要求を MPSC キューへ積み、次の game-thread Tick でゲームへ反映する。
エピソードのリセットは不要だが、HTTP worker から Game object へ直接アクセスしない。

```cpp
// SurvivorsHttpEnvService.cpp の HandleParams より
if (JsonObj->TryGetNumberField(TEXT("MinActiveEnemies"), MinActiveEnemies))
    Game->MinActiveEnemies = FMath::Clamp(MinActiveEnemies, 0, 600);
```

```
Python が /params を送信
  → UE5 の MPSC キューを game thread が取り出して Game フィールドを上書き
  → 以降のステップは新しいパラメータで動き続ける（エピソード継続中でも）
```

### エピソード途中でも適用される

`/params` の送信タイミングはエピソードの区切りと無関係。
エピソード途中に送信すると前半と後半で難易度が変わる。
Python 側はこの挙動を前提に実装すること（→ [`impl_notes.md`](impl_notes.md) の通信設計原則を参照）。

---

## 外部 level-up decision

`/params` に `{"item_selection_mode":"external"}` を送ると、XP 閾値を越えた時点で
候補を自動適用せず保留する。既定値の `"auto"` では従来どおり seeded
weighted/uniform 選択を同じステップ内で自動適用する。未知の mode は既存設定を
変更せず `400` で拒否する。

保留中の `/step` は同じ raw observation、`reward=0`、同じ decision ID を返す。
physics time、spawn、collision、weapon tick は進まない。`info` には既存の
`spawn_debug` と次の field が同居する。

```json
{
  "level_up_pending": true,
  "level_up_decision_id": "level-up-3-1-2",
  "level_up_player_level": 2,
  "level_up_backlog": 1,
  "level_up_choices": [
    {
      "choice_id": "choice-0",
      "type": "weapon_upgrade",
      "item_kind": "weapon",
      "item_id": 1,
      "slot_index": 0,
      "new_level": 2
    }
  ]
}
```

適用要求:

```http
POST /level_up_choice
Content-Type: application/json

{"decision_id":"level-up-3-1-2","choice_id":"choice-0"}
```

成功時は `status="applied"`、要求 ID、適用直後の `obs`、`obs_schema_hash`、
次の pending 状態を含む `info` を返す。通信タイムアウト時は同じ payload を
再送でき、直前に受理した同一要求には現在の `item_selection_mode` に関係なく
最初と同じ `200` response を返す。

### XP overflow 遷移

| event | player level / XP | pending | 次の処理 |
|---|---|---|---|
| 最初の閾値を越える | level を 1 だけ増加、累積 XP を保持 | level N の decision を生成 | N+1 は進めず backlog に保持 |
| pending 中の `/step` | 変化なし | 同じ ID と候補 | time/reward/spawn も変化なし |
| valid choice | item を 1 回適用 | current を原子的に解除 | backlog の閾値を 1 つだけ評価 |
| 次の閾値も超過済み | level をさらに 1 だけ増加 | 新しい decision ID | 再び停止 |
| 候補 pool 枯渇かつ次の閾値を超過済み | level を 1 だけ増加 | `type="no_upgrade"` の非空候補 | no-upgrade の受理まで停止し、残り backlog も同様に1つずつ処理 |
| `/reset` | level/XP を初期化 | なし | backlog と idempotency 履歴も消去 |

### HTTP エラー

| status | 条件 | state mutation |
|---:|---|---|
| `400` | 空 body、invalid JSON、未知 field を含む choice request、型不正、未知の item/weapon pool/starting mode | なし |
| `409` | stale/unknown decision ID、受理済み decision に異なる choice、候補にない choice ID | なし |
| `200` | valid choice、または直前の同一 choice の duplicate retry | valid は 1 回だけ適用、duplicate は再適用なし |

Python では `SurvivorsUE5Env` の実クラスである `SurvivorsEnv`（Monitor wrapper
では `SurvivorsMonitor`）の
`choose_level_up(decision_id, choice_id)` を使用する。

---

## ゲームパラメータ定義

定義・デフォルト値・説明コメントは以下を Source of Truth とすること:

```
ReinBalance/Source/ReinBalance/Public/Survivors/Logic/SurvivorsGame.h
```

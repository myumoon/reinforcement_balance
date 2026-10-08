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

## deploy_raw（opt-in）

DeployObs v2 の観測を Training 側で作るための raw state。既定は無効で、無効のときの `/reset`・`/step` 応答、flat obs、`obs_schema_hash` は従来と同一（キー自体を出さない）。

- 有効化: `POST /params {"deploy_raw": true}`。JSON bool 以外は `{"error":"deploy_raw must be bool"}` で拒否し、同じリクエストの他の項目も更新しない。`false` で無効化。reset では解除されない。
- 有効時の `/reset` 応答: `{"obs":[...],"obs_schema_hash":"...","deploy_raw":{...}}`
- 有効時の `/step` 応答: `info` に `deploy_raw` キーが増える（既存キーはそのまま）。単体経路（`ProcessStep`）と並列経路（`SurvivorsParallelSetupActor` → `BuildInfoJson`）のどちらも `CompleteStep` で1回だけ付与するので、並列 map でも同じ形になる。
- Training 側の読み手は `Tools/Training/games/survivors/deploy_raw_env.py`（`DeployRawEnv`）。

`deploy_raw` のキー（順序固定・全キー必須）:

| キー | 型 | 内容 |
|---|---|---|
| `schema_version` | str | `"survivors_deploy_raw.v1"` |
| `elapsed_s` | float | 経過時間（秒）。Python 側の timestamp はこれを ns にしたもの |
| `camera` | object | `center_x` / `center_y`（= 自機位置）、`half_width` 400、`half_height` 225、`cull_margin` 100（u） |
| `player` | object | `world_x` / `world_y`、`hp_ratio`（0〜1）、`level`（int） |
| `duration_mult` | float | パッシブによる持続時間倍率 |
| `weapon_slots` / `passive_slots` | list × 6 | `{"index", "type_id", "level"}`。`type_id` は C++ `EWeaponType` / `EPassiveItemType` の値（空き = 0、level 0） |
| `entities` | list | 下表 |

`entities` の各要素:

| キー | 型 | 内容 |
|---|---|---|
| `entity_id` | int | 同じ物体なら tick をまたいで不変、生成のたびに新しい id。上位ビット（`>> 40`）が id 空間（1 敵 / 2 ジェム / 3 projectile / 4 zone / 5 orbit / 6 aura） |
| `class_name` | str | `enemy_normal` / `enemy_boss` / `gem_blue` / `gem_green` / `gem_red` / `weapon_projectile` / `weapon_zone` / `weapon_orbit` / `weapon_aura` |
| `world_x` / `world_y` | float | 世界座標 |
| `radius_world` | float | 敵は当たり判定半径、ジェムは 0、武器エフェクトは `GetProjectileObsView()` の半径 |
| `slot` | int \| null | 武器エフェクトだけ。出した武器のスロット |
| `ttl_true_s` | float \| null | 武器エフェクトだけ。sim の真の残り時間（oracle 診断専用。release では使わない） |
| `warning` | bool | 武器エフェクトの予兆表示中（Santa Water など）。敵・ジェムは false |

- entity はカメラ範囲＋余白（`|dx| <= 400+100`、`|dy| <= 225+100`、境界を含む）で除外済み。最終的な可視判定（中心が画面内）は Python 側で行う。
- 武器エフェクトの範囲は `GetProjectileObsView()` と同じ（orbit は King Bible / Unholy Vespers、aura は Garlic / Soul Eater）。King Bible の本は周期ごとに新しい id になる。
- sim に遮蔽は無いので `occluded` は出さない（Python 側で常に false）。
- Python テストの正は、実 UE5 Editor PIE の HTTP 応答を保存した `Tools/Training/tests/survivors/fixtures/deploy_raw_pie_v1.json`。`deploy_raw_llt_v1.json` は C++ LLT の `[fixture]` テスト専用。
- PIE fixture の再取得（既定 port 8767）: `python Tools/Training/capture_survivors_deploy_raw_fixture.py --output Tools/Training/tests/survivors/fixtures/deploy_raw_pie_v1.json`。別 port は `--port` で指定する。
- fixture の再生成: LLT `SurvivorsDeployRawTests.cpp` の `[fixture]` テストは毎回、現在の C++ 出力と fixture が完全一致することを確認する。C++ 側を意図して変えたときだけ、短いドライブ（例: `subst W: <worktree>`）から `REINBALANCE_WRITE_DEPLOY_RAW_FIXTURE=1 ./ReinBalance/Binaries/Win64/ReinBalanceLogicTests/ReinBalanceLogicTests.exe -r console "[fixture]"` で書き直す。

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

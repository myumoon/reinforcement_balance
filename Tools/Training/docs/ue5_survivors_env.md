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

---

## `/params` エンドポイントの挙動

### 即時反映（リセット待ちなし）

受信した瞬間にゲームオブジェクトのフィールドへ直接代入する。
エピソードのリセットや次ステップまで待機する処理は一切ない。

```cpp
// SurvivorsHttpEnvService.cpp の HandleParams より
if (JsonObj->TryGetNumberField(TEXT("MinActiveEnemies"), MinActiveEnemies))
    Game->MinActiveEnemies = FMath::Clamp(MinActiveEnemies, 0, 600);
```

```
Python が /params を送信
  → UE5 が即座に Game フィールドを上書き
  → 以降のステップは新しいパラメータで動き続ける（エピソード継続中でも）
```

### エピソード途中でも適用される

`/params` の送信タイミングはエピソードの区切りと無関係。
エピソード途中に送信すると前半と後半で難易度が変わる。
Python 側はこの挙動を前提に実装すること（→ [`training_impl_notes.md`](training_impl_notes.md) の通信設計原則を参照）。

---

## ゲームパラメータ定義

定義・デフォルト値・説明コメントは以下を Source of Truth とすること:

```
ReinBalance/Source/ReinBalance/Public/Survivors/Logic/SurvivorsGame.h
```
